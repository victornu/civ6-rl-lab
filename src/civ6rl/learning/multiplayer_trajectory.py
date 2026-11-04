"""Player-scoped diagnostic transitions for alternating native decisions.

The response immediately following an action is retained only as an action
receipt. A transition's next state is the next decision response for the same
player, never the response for another player. Rewards remain undefined.
"""
from __future__ import annotations

import copy
import json
import math

from .encoding import encode_response


class MultiPlayerDiagnosticTrajectory:
    """Collect transitions independently for a fixed set of players.

    ``append(before, choice, after)`` consumes one action. ``before`` is the
    complete response that supplied the candidate menu, and ``after`` is the
    immediate native action response. For a nonterminal action, ``after`` must
    contain the next decision response; it becomes the next ``before`` on the
    following call. A native truncation can retain the observed player's
    bootstrap state. A true match end requires explicit results for every
    configured player; final observations may be absent.

    Records are diagnostic only. This class never derives a reward and always
    marks its records ineligible for PPO.
    """

    def __init__(self, *, policy_identity: dict, vocabulary, player_ids, public_continuity_sources=None, actor_schema=None):
        if (type(policy_identity) is not dict or not policy_identity
                or not isinstance(policy_identity.get("policy_id"), str)
                or not policy_identity["policy_id"]
                or not isinstance(policy_identity.get("version"), (str, int))
                or isinstance(policy_identity.get("version"), bool)):
            raise ValueError("policy identity requires policy_id and version provenance")
        if type(player_ids) not in (list, tuple) or not player_ids:
            raise ValueError("a nonempty fixed player_ids sequence is required")
        if (any(type(pid) is not int or not 0 <= pid <= 63 for pid in player_ids)
                or len(set(player_ids)) != len(player_ids)):
            raise ValueError("player_ids must be unique native player IDs")
        if actor_schema is not None:
            from .encoding import actor_feature_names
            actor_feature_names(actor_schema)  # Reject unknown layouts, never derive them from a free policy label.
        self.actor_schema = actor_schema
        self.policy_identity = copy.deepcopy(policy_identity)
        self.vocabulary = vocabulary
        self.player_ids = tuple(player_ids)
        from .skill_task991 import validate_public_sources
        self.public_continuity_sources = (validate_public_sources(public_continuity_sources)
            if public_continuity_sources is not None else None)
        self.episode_id: str | None = None
        self.game_generation: str | None = None
        self.records: list[dict] = []
        self._pending: dict[int, dict] = {}
        self._last_response: dict | None = None
        self._last_encoded_binding: dict | None = None
        self._closed = False

    @staticmethod
    def _identity(response: dict) -> tuple[str, str, int, int]:
        if type(response) is not dict:
            raise ValueError("native response must be an object")
        try:
            episode, generation = response["episode_id"], response["game_generation"]
            sequence = response["sequence"]
            observation = response["observation"]
            player = observation.get("player_id") if isinstance(observation, dict) else None
        except (KeyError, TypeError) as exc:
            raise ValueError("native response is missing episode/player/sequence identity") from exc
        if (not isinstance(episode, str) or not episode or not isinstance(generation, str)
                or not generation or type(sequence) is not int or sequence < 0
                or (player is not None and (type(player) is not int or not 0 <= player <= 63))):
            raise ValueError("invalid native response identity")
        return episode, generation, sequence, player

    def _bind(self, response: dict, *, _owned=False) -> tuple[int, dict]:
        episode, generation, _sequence, player = self._identity(response)
        if player is None or player not in self.player_ids:
            raise ValueError("response player is outside the configured player set")
        if self.episode_id is not None and (episode != self.episode_id or generation != self.game_generation):
            raise ValueError("episode or game generation changed; call reset explicitly")
        cache = self._last_encoded_binding
        cache_hit = (cache is not None and response == cache['response']
                and self.vocabulary is cache['vocabulary']
                and self.vocabulary.fingerprint == cache['vocabulary_fingerprint']
                and self.actor_schema == cache['actor_schema'])
        if cache_hit:
            # The exact detached snapshot already passed observation parsing
            # and encoding at its original action boundary.
            encoded = cache['encoded'] if _owned else copy.deepcopy(cache['encoded'])
        else:
            try:
                from ..native_player_observation import parse_player_observation
                parse_player_observation(json.dumps(response["observation"], allow_nan=False), player)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("public observation is invalid or has foreign perspective") from exc
            encoded = encode_response(response, vocabulary=self.vocabulary, actor_schema=self.actor_schema)
        if encoded["player_id"] != player:
            raise ValueError("encoded perspective differs from native identity")
        if self.episode_id is None:
            self.episode_id, self.game_generation = episode, generation
        return player, encoded

    def _validate_choice(self, before: dict, encoded: dict, choice: dict) -> tuple[str, int]:
        if type(choice) is not dict:
            raise ValueError("worker choice must be an object")
        player = encoded["player_id"]
        for key, expected in (("episode_id", self.episode_id), ("sequence", before["sequence"]),
                              ("player_id", player)):
            if choice.get(key) != expected:
                raise ValueError("worker choice has stale or foreign identity")
        if "game_generation" in choice and choice["game_generation"] != self.game_generation:
            raise ValueError("worker choice belongs to another game generation")
        if choice.get("training_eligible") is not False:
            raise ValueError("worker choice is not explicitly diagnostic")
        candidate_id = choice.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id not in encoded["candidate_ids"]:
            raise ValueError("worker choice is outside the player's published menu")
        for key in ("log_prob", "value", "entropy"):
            value = choice.get(key)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("worker choice has invalid behavior-policy statistics")
        if choice["entropy"] < 0:
            raise ValueError("worker entropy must be non-negative")
        return candidate_id, encoded["candidate_ids"].index(candidate_id)

    @staticmethod
    def _action_receipt(after: dict) -> dict:
        # Audit receipt only: it intentionally excludes any next observation.
        # In particular, an opponent's applied action is never an encoded input.
        outcome = after.get("outcome")
        if type(outcome) is not dict or type(outcome.get("terminated")) is not bool or type(outcome.get("truncated")) is not bool:
            raise ValueError("native action response needs explicit terminal flags")
        return {key: copy.deepcopy(after[key]) for key in (
            "episode_id", "game_generation", "sequence", "operation", "applied_action",
            "automatic_choices_applied") if key in after} | {"outcome": copy.deepcopy(outcome)}

    def _make_record(self, pending: dict, *, next_response=None, next_encoded=None,
                     terminal_result=None, terminal_observation=None,
                     terminated=False, truncated=False, truncation_reason=None,
                     transfer_pending=False):
        before = pending["before"]
        choice = pending["choice"]
        own = (lambda key: pending[key]) if transfer_pending else (lambda key: copy.deepcopy(pending[key]))
        receipt = own("action_receipt")
        record = {
            "schema": "civ6rl.multiplayer_diagnostic_transition.v1",
            "policy_identity": copy.deepcopy(self.policy_identity),
            "vocabulary_sha256": self.vocabulary.fingerprint,
            "episode_id": before["episode_id"], "game_generation": before["game_generation"],
            "player_id": before["observation"]["player_id"], "sequence": before["sequence"],
            "candidate_id": pending["candidate_id"], "action_index": pending["action_index"],
            "choice": own("choice"),
            "assignments": own("assignments"),
            "before_response": own("before"), "encoded_before": own("encoded_before"),
            "action_receipt": receipt,
            "next_response": copy.deepcopy(next_response) if next_response is not None else None,
            "encoded_next": copy.deepcopy(next_encoded) if next_encoded is not None else None,
            "terminal_result": copy.deepcopy(terminal_result),
            "terminal_observation": copy.deepcopy(terminal_observation),
            "behavior_log_prob": choice["log_prob"], "behavior_value": choice["value"],
            "behavior_entropy": choice["entropy"], "reward": None,
            "terminated": terminated, "truncated": truncated,
            # A closed native menu retains public state but cannot use the
            # normal policy forward (its Categorical needs candidates).
            "public_successor_available": next_encoded is not None,
            "bootstrap_available": (not terminated and next_encoded is not None
                                    and bool(next_encoded["candidate_ids"])),
            "training_eligible": False, "reward_contract": None,
        }
        if truncation_reason:
            record["truncation_reason"] = truncation_reason
        if self.public_continuity_sources is not None:
            record["public_witnesses"] = own("public_witnesses")
        return record

    def _resolve(self, player: int, response: dict, encoded: dict):
        pending = self._pending.pop(player, None)
        if pending is not None:
            record = self._make_record(pending, next_response=response, next_encoded=encoded,
                                       transfer_pending=True)
            self.records.append(record)
            return record
        return None

    @staticmethod
    def _terminal_observation(observation: dict, player: int, turn: int | None) -> dict:
        try:
            from ..native_player_observation import parse_player_observation
            observation = parse_player_observation(json.dumps(observation, allow_nan=False), player)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("terminal observation is invalid or has foreign perspective") from exc
        if turn is not None and observation["turn"] != turn:
            raise ValueError("terminal observation is stale relative to the terminal response turn")
        return copy.deepcopy(observation)

    def _validate_terminal(self, after: dict, terminal: dict) -> tuple[dict, dict]:
        if type(terminal) is not dict or set(terminal) not in ({"results"}, {"results", "observations"}):
            raise ValueError("episode end requires explicit per-player results")
        results, observations = terminal["results"], terminal.get("observations", {})
        expected = set(self.player_ids)
        if (type(results) is not dict or any(type(pid) is not int for pid in results)
                or set(results) != expected):
            raise ValueError("terminal results must cover every configured player")
        if (type(observations) is not dict or any(type(pid) is not int for pid in observations)
                or not set(observations) <= expected):
            raise ValueError("terminal observations have foreign player keys")
        terminated = after["outcome"]["terminated"]
        truncated = after["outcome"]["truncated"]
        if terminated == truncated:
            raise ValueError("episode end must be exactly terminated or truncated")
        if truncated and set(observations) != expected:
            raise ValueError("legacy truncation results and observations must cover every configured player")
        after_observation = after.get("observation")
        terminal_turn = after_observation.get("turn") if isinstance(after_observation, dict) else None
        if terminal_turn is None:
            victory = after["outcome"].get("native_victory")
            if isinstance(victory, dict):
                terminal_turn = victory.get("current_game_turn")
        if terminal_turn is not None and (type(terminal_turn) is not int or terminal_turn < 0):
            raise ValueError("terminal turn must be a non-negative integer")
        clean_results, clean_observations = {}, {}
        for pid in self.player_ids:
            result = results[pid]
            required = {"episode_id", "game_generation", "player_id", "terminated", "truncated", "result"}
            if (type(result) is not dict or set(result) != required
                    or result["episode_id"] != self.episode_id or result["game_generation"] != self.game_generation
                    or type(result["player_id"]) is not int or result["player_id"] != pid
                    or result["result"] is None or type(result["terminated"]) is not bool
                    or type(result["truncated"]) is not bool or result["terminated"] is not terminated
                    or result["truncated"] is not truncated):
                raise ValueError("terminal result identity/flags are missing or inconsistent")
            clean_results[pid] = copy.deepcopy(result)
            if pid in observations:
                clean_observations[pid] = self._terminal_observation(observations[pid], pid, terminal_turn)
        return clean_results, clean_observations

    def append(self, before: dict, choice: dict, after: dict, *, terminal: dict | None = None,
               public_witnesses=None) -> dict | None:
        """Consume one verified action, resolving only the same player's prior step.

        Returns the newly completed record, if the player has a prior pending
        action. True terminal responses require results for every player.
        Truncations can finish without results. Neither path invents reward,
        observations, or an eliminated-player result from a missing view.
        """
        if self._closed:
            raise ValueError("trajectory is closed")
        player, encoded = self._bind(before, _owned=True)
        if before["outcome"].get("terminated") or before["outcome"].get("truncated"):
            raise ValueError("cannot act after episode end")
        if self._last_response is not None and before != self._last_response:
            raise ValueError("native response stream is stale, reordered, or has a gap")
        candidate_id, action_index = self._validate_choice(before, encoded, choice)
        episode, generation, before_sequence, before_player = self._identity(before)
        after_episode, after_generation, after_sequence, after_player = self._identity(after)
        if (after_episode != episode or after_generation != generation
                or after_sequence != before_sequence + 1 or after.get("operation") != "act"
                or after.get("automatic_choices_applied") is not False):
            raise ValueError("native action response is stale, reordered, or foreign")
        if after_player is not None and after_player not in self.player_ids:
            raise ValueError("native response player is outside the configured player set")
        from .action_receipt import verify_applied_action
        assignments = verify_applied_action(before, choice, after)
        receipt = self._action_receipt(after)
        outcome = receipt["outcome"]
        if outcome["terminated"] and outcome["truncated"]:
            raise ValueError("episode end must be exactly terminated or truncated")
        ended = outcome["terminated"] or outcome["truncated"]
        immediate_resolved = None
        results, observations = {}, {}
        after_encoded = None
        if ended:
            if outcome["terminated"]:
                if terminal is None:
                    raise ValueError("terminal action requires per-player results")
                if after.get("decisions") is not None:
                    raise ValueError("terminal response must not publish another decision menu")
            if terminal is not None:
                results, observations = self._validate_terminal(after, terminal)
            if outcome["truncated"] and after.get("decisions") is not None:
                menu = after["decisions"]
                if type(menu) is not dict or type(menu.get("candidates")) is not list or menu["candidates"]:
                    raise ValueError("native truncation requires a closed empty candidate menu")
                after_player, after_encoded = self._bind(after, _owned=True)
            elif after["observation"] is not None:
                # Preserve the legacy complete terminal mapping contract; a
                # bare observation without a menu cannot supply encoded bootstrap.
                if terminal is None:
                    raise ValueError("truncation observation needs a closed decision menu")
                if after_player != before_player:
                    raise ValueError("terminal response observation has a foreign player perspective")
                observation = self._terminal_observation(after["observation"], after_player, None)
                if after_player in observations and observation != observations[after_player]:
                    raise ValueError("terminal observation differs from the native response")
                observations[after_player] = observation
        else:
            if terminal is not None:
                raise ValueError("terminal result data supplied before episode end")
            if after.get("decisions") is None:
                raise ValueError("nonterminal action needs the next published decision response")
            # Validate the successor, but do not use it as the pending player's
            # next state unless that player is the one making the next decision.
            after_player, after_encoded = self._bind(after, _owned=True)

        # Validate the complete sidecar before changing pending transitions.
        # These narrow owned-state views never enter encoding or memory.
        clean_witnesses = {}
        if self.public_continuity_sources is None:
            if public_witnesses is not None:
                raise ValueError("public witnesses require declared continuity sources")
        else:
            from .skill_task991 import validate_public_witness, project_public_observation
            required_players = set(self._pending) | {player}
            if type(public_witnesses) is not list or len(public_witnesses) != len(required_players):
                raise ValueError("public witnesses must cover every pending player after each action")
            for witness in public_witnesses:
                pid = witness.get("player_id") if type(witness) is dict else None
                if type(pid) is not int or pid not in required_players or pid in clean_witnesses:
                    raise ValueError("public witnesses have duplicate or foreign players")
                clean = validate_public_witness(witness, sources=self.public_continuity_sources,
                    episode_id=episode, game_generation=generation, player_id=pid, sequence=after_sequence)
                boundary_turn = (after["observation"]["turn"] if after.get("observation") is not None
                    else outcome.get("native_victory",{}).get("current_game_turn"))
                if type(boundary_turn) is not int or clean["observation"]["turn"] != boundary_turn:
                    raise ValueError("public witness turn differs from native receipt boundary")
                if pid == after_player and after.get("observation") is not None:
                    if clean["observation"] != project_public_observation(after["observation"]):
                        raise ValueError("public witness differs from the active player's published observation")
                prior = self._pending.get(pid)
                if pid == player:
                    previous = before["observation"]
                elif prior["public_witnesses"]:
                    previous = prior["public_witnesses"][-1]["observation"]
                else:
                    previous = prior["before"]["observation"]
                observed = clean["observation"]
                if (observed["turn"] < previous["turn"] or
                        (observed["width"],observed["height"]) != (previous["width"],previous["height"])):
                    raise ValueError("public witness geometry or turn differs from pending player")
                clean_witnesses[pid] = clean

        resolved = self._resolve(player, before, encoded)
        self._pending[player] = {
            "before": copy.deepcopy(before), "choice": copy.deepcopy(choice),
            "candidate_id": candidate_id, "action_index": action_index,
            "assignments": assignments,
            "encoded_before": copy.deepcopy(encoded), "action_receipt": receipt,
        }
        if self.public_continuity_sources is not None:
            self._pending[player]["public_witnesses"] = []
            for pid,pending in self._pending.items():
                pending["public_witnesses"].append(clean_witnesses[pid])

        if ended:
            for pid in tuple(self._pending):
                pending = self._pending.pop(pid)
                record = self._make_record(
                    pending, terminal_result=results.get(pid), terminal_observation=observations.get(pid),
                    next_response=after if outcome["truncated"] and pid == after_player and after_encoded is not None else None,
                    next_encoded=after_encoded if outcome["truncated"] and pid == after_player else None,
                    terminated=outcome["terminated"], truncated=outcome["truncated"],
                    truncation_reason="native_terminal" if outcome["terminated"] else "native_truncation",
                    transfer_pending=True)
                self.records.append(record)
                if pending["before"]["sequence"] == before_sequence and pending["before"]["observation"]["player_id"] == player:
                    resolved = record
            self._closed = True
            self._last_response = None
            self._last_encoded_binding = None
        else:
            self._last_response = copy.deepcopy(after)
            self._last_encoded_binding = dict(response=self._last_response,
                encoded=after_encoded, vocabulary=self.vocabulary,
                vocabulary_fingerprint=self.vocabulary.fingerprint, actor_schema=self.actor_schema)
            # Resolve as soon as the next same-player decision arrives. This
            # preserves a valid bootstrap state even if the caller then stops
            # the run before submitting that player's next action.
            immediate_resolved = self._resolve(after_player, after, after_encoded)
        self.records.sort(key=lambda row: (row["episode_id"], row["game_generation"], row["sequence"]))
        return immediate_resolved or resolved

    def pending_public_records(self) -> list[dict]:
        """Read-only pending intervals for public task stopping, never PPO admission.

        The caller can classify a terminal before the same player returns.
        Nonterminal rows still lack a decision successor and cannot bootstrap.
        """
        if self.public_continuity_sources is None:
            raise ValueError("public pending intervals require declared continuity sources")
        return [self._make_record(pending) for _,pending in sorted(self._pending.items())]

    def finish_budget(self) -> list[dict]:
        """Mark outstanding steps as diagnostic-budget truncations without a next state."""
        if not self._closed:
            # Defensive fallback for callers that stop after receiving a
            # same-player successor but before another append call.
            if self._last_response is not None:
                last_player = self._last_response.get("observation", {}).get("player_id")
                if last_player in self._pending:
                    _, encoded = self._bind(self._last_response, _owned=True)
                    self._resolve(last_player, self._last_response, encoded)
            for pid in tuple(self._pending):
                pending = self._pending.pop(pid)
                self.records.append(self._make_record(pending, truncated=True,
                    truncation_reason="diagnostic_budget", transfer_pending=True))
            self._last_response = None
            self._last_encoded_binding = None
            self._closed = True
            self.records.sort(key=lambda row: (row["episode_id"], row["game_generation"], row["sequence"]))
        return self.records

    def reset(self, *, episode_id: str, game_generation: str) -> list[dict]:
        """Start an explicitly identified episode; abandon old pending steps as truncations."""
        if not isinstance(episode_id, str) or not episode_id or not isinstance(game_generation, str) or not game_generation:
            raise ValueError("reset requires new episode and generation identities")
        if episode_id == self.episode_id and game_generation == self.game_generation:
            raise ValueError("reset must change episode or game generation")
        for pid in tuple(self._pending):
            pending = self._pending.pop(pid)
            self.records.append(self._make_record(pending, truncated=True,
                truncation_reason="episode_reset", transfer_pending=True))
        self.episode_id, self.game_generation = episode_id, game_generation
        self._last_response = None
        self._last_encoded_binding = None
        self._closed = False
        self.records.sort(key=lambda row: (row["episode_id"], row["game_generation"], row["sequence"]))
        return self.records

    def training_records(self):
        raise RuntimeError("No qualified native reward/action-space contract: diagnostic transitions are not PPO samples")


__all__ = ["MultiPlayerDiagnosticTrajectory"]
