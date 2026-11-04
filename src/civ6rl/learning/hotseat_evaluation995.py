"""Portable journal for a bounded, frozen-actor Hotseat integration diagnostic.

The caller must use NativeHotseatActionSession (which validates native handoffs)
and a checkpoint-verified, evaluation_only worker. A declared v6 schema and
fixed version here do not prove that weights were loaded or frozen. No host or
Torch module is imported by this journal.

Each record retains before/choice/immediate after without rewriting them. The
after response can belong to another player: these are NOT PPO transitions,
and this module computes no reward, GAE or bootstrap. Replays contain only a
player's decision-time observation and selected action, never the successor's
view or the audit-only victory payload. Fourteen global decisions alone do
not demonstrate control of four players if the actor never ends a turn.
"""
from __future__ import annotations

import copy
import json
import math
import re

from .action_receipt import selected_action_and_assignments, verify_applied_action
from .encoding import ContentVocabulary, encode_response
from .task_context import encode_task_response, scenario_digest, SKILL_SCENARIO_SCHEMA


SCHEMA = "civ6rl.hotseat_evaluation995.v1"
ACTOR_SCHEMA = "civ6rl.recurrent_policy.v6"
HOTSEAT_SCHEMA = "civ6rl.hotseat_action_session923.v1"
STOP_REASONS = frozenset(("terminal", "native_cap", "decision_cap",
                          "unsupported_menu", "time_cap", "action_failed"))
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX32 = re.compile(r"^[0-9a-f]{32}$")
_RESPONSE_KEYS = frozenset(("stage", "schema", "nonce", "episode_id", "game_generation",
    "sequence", "operation", "observation", "rounds", "outcome", "automatic_choices_applied",
    "action_interface_complete", "strategic_training_ready", "decisions", "applied_action"))
_OUTCOME_KEYS = frozenset(("terminated", "truncated", "reason", "native_victory",
                         "global_engine_terminal", "strategic_training_ready"))
# Wire source pins used by the native validator. This is a join check, not a
# replacement for that validator's on-disk source/engine verification.
_VICTORY_PROOF = "b7de76fcb45fde2ba96958e18c71ccbcc48dae6837beb7fead6c4e44aaeb694b"
_FACTS_PROOF = "ecca6cc2b9910769f584851573016cbdad72d3f6a1e613d4cbd5b8ad52713232"
_CHOICE_REQUIRED = frozenset(("episode_id", "player_id", "sequence", "candidate_id", "log_prob",
    "value", "entropy", "inference_seconds", "model_status", "training_eligible", "policy_version",
    "memory_reset", "sampling_mode", "scenario_sha256", "task_completed_decisions"))
_CHOICE_OPTIONAL = frozenset(("assignments", "game_generation"))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _json_copy(value):
    # Refuse non-wire values (NaN, tuples, non-string object keys, subclasses)
    # instead of normalizing them into apparently unchanged evidence.
    def check(item):
        if type(item) is dict:
            _require(all(type(k) is str for k in item), "wire keys must be strings")
            for child in item.values():
                check(child)
        elif type(item) is list:
            for child in item:
                check(child)
        elif type(item) is float:
            _require(math.isfinite(item), "wire number must be finite")
        else:
            _require(item is None or type(item) in (str, int, bool), "non-JSON wire value")
    check(value)
    json.dumps(value, allow_nan=False)
    return copy.deepcopy(value)


def _same_wire(left, right):
    # Python equality alone conflates true with 1 and 1 with 1.0.
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


class HotseatEvaluationJournal:
    """One journal per fresh reset; counts are *verified applied* decisions.

    ``scenarios`` is None or a player-id map of public scenarios/None. Missing
    players are explicitly unconditioned. The scenario and vocabulary are
    snapshotted at construction. A failed record closes collection immediately:
    inference may already have advanced recurrent state, so retry is unsafe.
    ``unsupported_menu`` and ``time_cap`` are caller-reported stops, not native
    outcomes. ``record(..., after=None)`` audits a missing/failed action receipt.
    """

    def __init__(self, *, vocabulary, player_ids=(0, 1, 2, 3), decision_limit=14,
                 scenarios=None, actor_schema=ACTOR_SCHEMA):
        _require(type(vocabulary) is ContentVocabulary, "frozen ContentVocabulary required")
        _require(type(player_ids) in (tuple, list) and bool(player_ids)
                 and all(type(pid) is int and 0 <= pid <= 63 for pid in player_ids)
                 and len(set(player_ids)) == len(player_ids), "invalid configured players")
        _require(type(decision_limit) is int and 1 <= decision_limit <= 4096,
                 "invalid explicit decision limit")
        _require(actor_schema == ACTOR_SCHEMA, "only declared actor v6 is supported")
        _require(scenarios is None or (type(scenarios) is dict
                 and all(type(pid) is int and pid in player_ids for pid in scenarios)),
                 "scenarios must be keyed by configured players")
        self._vocabulary = ContentVocabulary.from_serialized(vocabulary.serialize())
        self._players = tuple(player_ids)
        self._limit = decision_limit
        self._scenarios = {pid: _json_copy((scenarios or {}).get(pid)) for pid in self._players}
        self._digests = {}
        for pid, scenario in self._scenarios.items():
            _require(scenario is None or type(scenario) is dict, "scenario must be an object or None")
            self._digests[pid] = None if scenario is None else scenario_digest(scenario)
            if scenario is not None:
                # Current skill993 puts the perspective in its validated task;
                # legacy economy scenarios put it at the scenario root.
                perspective = (scenario["skill_task"]["player_id"] if scenario["schema"] == SKILL_SCENARIO_SCHEMA
                               else scenario["player_id"])
                _require(type(perspective) is int and perspective == pid,
                         "scenario belongs to a different player")
        self._counts = dict.fromkeys(self._players, 0)
        self._records = []
        self._current = None
        self._initial = None
        self._version = None
        self._failed = False
        self._finished = None
        self._handoffs = 0

    @property
    def counts(self):
        return dict(self._counts)

    @property
    def audit_records(self):
        return copy.deepcopy(self._records)

    def replay_rows(self, player_id):
        _require(type(player_id) is int and player_id in self._players, "unconfigured replay player")
        return [copy.deepcopy({"observation": row["before"]["observation"],
                               "action": row["after"]["applied_action"]})
                for row in self._records if row["verified_applied"]
                and row["before"]["observation"]["player_id"] == player_id]

    def _response(self, response):
        _require(type(response) is dict and set(response) == _RESPONSE_KEYS,
                 "complete native response envelope required")
        _require(response["schema"] == HOTSEAT_SCHEMA and response["stage"] == "official_session_response",
                 "explicit Hotseat923 response required")
        _require(type(response["nonce"]) is str and _HEX32.fullmatch(response["nonce"]), "invalid native nonce")
        for key in ("episode_id", "game_generation"):
            _require(type(response[key]) is str and bool(response[key]), "missing native identity")
        for key in ("sequence", "rounds"):
            _require(type(response[key]) is int and response[key] >= 0, "invalid native counter")
        _require(response["automatic_choices_applied"] is False
                 and response["action_interface_complete"] is False
                 and response["strategic_training_ready"] is False, "unverified diagnostic boundary")
        outcome = response["outcome"]
        _require(type(outcome) is dict and set(outcome) == _OUTCOME_KEYS, "complete native outcome required")
        _require(type(outcome["terminated"]) is bool and type(outcome["truncated"]) is bool
                 and not (outcome["terminated"] and outcome["truncated"])
                 and outcome["global_engine_terminal"] is None
                 and outcome["strategic_training_ready"] is False, "invalid native outcome flags")
        native = outcome["native_victory"]
        _require(type(native) is dict and native.get("schema") == "civ6rl.native_victory_episode.observation.v1"
                 and native.get("episode_id") == response["episode_id"]
                 and native.get("game_generation") == response["game_generation"]
                 and native.get("policy_id") == "first-official-native-victory-v1"
                 and native.get("source_proof_sha256") == _VICTORY_PROOF
                 and native.get("facts_source_proof_sha256") == _FACTS_PROOF,
                 "native outcome provenance differs")
        _require(type(native.get("player_id")) is int and native["player_id"] in self._players
                 and type(native.get("current_game_turn")) is int and native["current_game_turn"] >= 0,
                 "invalid native actor boundary")
        _require(native.get("episode_terminated") is outcome["terminated"]
                 and native.get("winner_present") is outcome["terminated"]
                 and native.get("observation_only") is True
                 and native.get("eligible_for_training") is False
                 and native.get("exposed_to_learner") is False
                 and native.get("global_engine_terminal") is None
                 and native.get("engine_halted") is None,
                 "invalid native victory flags")
        expected_reason = ("official_native_victory" if outcome["terminated"] else
                           "diagnostic_round_limit" if outcome["truncated"] else None)
        _require(outcome["reason"] == expected_reason, "unexpected native stop reason")
        observation, decisions = response["observation"], response["decisions"]
        if observation is None:
            _require(outcome["terminated"] and decisions is None,
                     "only native terminal may omit observation and menu")
        else:
            _require(type(observation) is dict and type(observation.get("player_id")) is int
                     and type(observation.get("turn")) is int
                     and observation["player_id"] == native["player_id"]
                     and observation["turn"] == native["current_game_turn"],
                     "native and public boundaries differ")
            if decisions is None:
                # The host permits a final public observation without a menu.
                # It is audit-only; do not invent a synthetic empty catalogue
                # to encode it and do not expose it in a player's replay.
                _require(outcome["terminated"] and observation.get("schema") == "civ6rl-player-observation-v1",
                         "only native terminal may omit its menu")
            else:
                if outcome["terminated"] or outcome["truncated"]:
                    _require(type(decisions) is dict and decisions.get("candidates") == [],
                             "ended native response must close its candidate menu")
                encode_response(response, vocabulary=self._vocabulary)
        return native

    def start(self, response, policy_version):
        _require(self._initial is None and self._finished is None, "journal already started")
        _require(type(policy_version) is str and _HEX64.fullmatch(policy_version), "invalid policy version")
        raw = _json_copy(response)
        self._response(raw)
        _require(raw["operation"] == "reset" and raw["sequence"] == 0 and raw["rounds"] == 0
                 and raw["applied_action"] is None, "fresh native reset required")
        self._initial = raw
        self._current = copy.deepcopy(raw)
        self._version = policy_version
        return copy.deepcopy(raw)

    def _choice(self, before, choice):
        _require(type(choice) is dict and _CHOICE_REQUIRED <= set(choice)
                 and set(choice) <= _CHOICE_REQUIRED | _CHOICE_OPTIONAL,
                 "unsupported actor-only choice fields")
        pid = before["observation"]["player_id"]
        _require(choice["episode_id"] == before["episode_id"]
                 and type(choice["player_id"]) is int and choice["player_id"] == pid
                 and type(choice["sequence"]) is int and choice["sequence"] == before["sequence"]
                 and ("game_generation" not in choice or choice["game_generation"] == before["game_generation"]),
                 "choice identity or sequence differs")
        _require(choice["policy_version"] == self._version, "frozen policy version changed")
        _require(choice["memory_reset"] is (self._counts[pid] == 0), "incorrect per-player memory reset")
        _require(choice["training_eligible"] is False
                 and choice["sampling_mode"] == "unconditioned_policy_sample"
                 and choice["model_status"] in ("untrained", "diagnostic_updated"),
                 "unsupported diagnostic choice declaration")
        for key in ("log_prob", "value", "entropy", "inference_seconds"):
            _require(type(choice[key]) in (int, float) and math.isfinite(choice[key]), "invalid actor statistic")
        _require(choice["log_prob"] <= 0 and choice["entropy"] >= 0 and choice["inference_seconds"] >= 0,
                 "invalid actor statistic range")
        scenario = self._scenarios[pid]
        completed = None if scenario is None else self._counts[pid]
        _require(choice["scenario_sha256"] == self._digests[pid]
                 and (choice["task_completed_decisions"] is None if completed is None else
                      type(choice["task_completed_decisions"]) is int
                      and choice["task_completed_decisions"] == completed),
                 "scenario or per-player task count differs")
        frame = encode_task_response(before, vocabulary=self._vocabulary,
                                     scenario=scenario, completed_decisions=completed)
        _require(choice["candidate_id"] in frame["candidate_ids"], "choice absent from published menu")
        selected_action_and_assignments(before, choice)

    def record(self, before, choice, after):
        _require(self._initial is not None and self._finished is None and not self._failed,
                 "journal is not collecting")
        _require(sum(self._counts.values()) < self._limit, "decision cap reached")
        _require(not self._current["outcome"]["terminated"] and not self._current["outcome"]["truncated"],
                 "native episode already ended")
        row = None
        try:
            raw_before, raw_choice, raw_after = map(_json_copy, (before, choice, after))
            row = {"before": raw_before, "choice": raw_choice, "after": raw_after,
                   "verified_applied": False, "failure": None}
            _require(_same_wire(raw_before, self._current), "before differs from exact previous native response")
            self._choice(raw_before, raw_choice)
            if raw_after is None:
                row["failure"] = "missing_action_receipt"
                self._failed = True
                self._records.append(row)
                return copy.deepcopy(row)
            native = self._response(raw_after)
            _require(all(raw_after[k] == raw_before[k] for k in ("nonce", "episode_id", "game_generation")),
                     "foreign action receipt")
            _require(raw_after["operation"] == "act" and raw_after["sequence"] == raw_before["sequence"] + 1,
                     "action receipt sequence differs")
            _require(raw_before["rounds"] <= raw_after["rounds"] <= raw_before["rounds"] + 1
                     and native["current_game_turn"] == self._initial["outcome"]["native_victory"]["current_game_turn"]
                     + raw_after["rounds"], "native turn and round counters differ")
            verify_applied_action(raw_before, raw_choice, raw_after)
            # Verify exact wire types too: Python dictionary equality alone
            # would consider an applied index True equivalent to index 1.
            expected, _ = selected_action_and_assignments(raw_before, raw_choice)
            _require(_same_wire(expected, raw_after["applied_action"]), "applied action wire types differ")
            row["verified_applied"] = True
            pid = raw_before["observation"]["player_id"]
            self._counts[pid] += 1
            self._handoffs += int(native["player_id"] != pid)
            self._current = raw_after
            self._records.append(row)
            return copy.deepcopy(row)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self._failed = True
            if row is not None:
                row["failure"] = str(exc)
                self._records.append(row)
            raise ValueError(str(exc)) from exc

    def finish(self, explicit_stop_reason):
        _require(self._initial is not None and self._finished is None, "journal is not active")
        _require(type(explicit_stop_reason) is str and explicit_stop_reason in STOP_REASONS,
                 "explicit known stop reason required")
        outcome = self._current["outcome"]
        required = ("action_failed" if self._failed else "terminal" if outcome["terminated"] else
                    "native_cap" if outcome["truncated"] else "decision_cap"
                    if sum(self._counts.values()) == self._limit else None)
        if required is not None:
            _require(explicit_stop_reason == required, "stop reason differs from recorded boundary")
        else:
            _require(explicit_stop_reason in ("unsupported_menu", "time_cap"),
                     "stop reason has no recorded evidence")
        self._finished = {
            "schema": SCHEMA, "scope": "frozen_actor_wiring_diagnostic",
            "episode_id": self._initial["episode_id"], "game_generation": self._initial["game_generation"],
            "actor_schema_declared": ACTOR_SCHEMA, "policy_version": self._version,
            "checkpoint_and_freeze_validation": "required_of_caller",
            "explicit_stop_reason": explicit_stop_reason,
            "stop_source": "caller_reported" if required is None else "validated_journal_boundary",
            "verified_decisions": sum(self._counts.values()), "decision_limit": self._limit,
            "attempted_decisions": len(self._records), "counts": self.counts,
            "players_with_verified_actions": [pid for pid in self._players if self._counts[pid]],
            "native_handoffs_observed": self._handoffs,
            "native_terminal_observed": outcome["terminated"], "native_cap_observed": outcome["truncated"],
            "scenario_sha256_by_player": dict(self._digests),
            "training_eligible": False, "strategic_training_ready": False,
        }
        return copy.deepcopy(self._finished)


__all__ = ["HotseatEvaluationJournal", "SCHEMA", "ACTOR_SCHEMA", "STOP_REASONS"]
