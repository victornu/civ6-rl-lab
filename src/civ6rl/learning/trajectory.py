"""Lossless player-scoped diagnostic transition capture for recurrent learning.

Undefined curriculum rewards remain None, never an invented zero. These
records cannot be fed to PPO until a scenario/reward contract is qualified.
This first collector deliberately refuses a change of player perspective;
hotseat requires pending transitions resolved on that player's next decision.
"""
from __future__ import annotations
import copy
import math


class DiagnosticTrajectory:
    def __init__(self, *, policy_identity: dict, vocabulary):
        if not policy_identity or not isinstance(policy_identity, dict):
            raise ValueError("policy provenance required")
        self.policy_identity = copy.deepcopy(policy_identity)
        self.vocabulary = vocabulary
        self.records = []
        self._successor = None
        self.finished = False

    def append(self, before, choice, after):
        from .encoding import encode_response
        if self.finished:
            raise ValueError("trajectory is already closed")
        if self._successor is not None and before != self._successor:
            raise ValueError("non-contiguous trajectory")
        frame = encode_response(before, vocabulary=self.vocabulary)
        for key in ("episode_id", "game_generation"):
            if before[key] != after[key]:
                raise ValueError("reset/generation boundary inside transition")
        if before["observation"]["player_id"] != after["observation"]["player_id"]:
            raise ValueError("hotseat needs a next-decision observation for the same player")
        if after["sequence"] != before["sequence"] + 1 or after["operation"] != "act":
            raise ValueError("native successor is not the selected action's boundary")
        if (choice.get("episode_id") != before["episode_id"] or
                choice.get("sequence") != before["sequence"] or choice.get("player_id") != frame["player_id"]):
            raise ValueError("policy decision has a stale identity")
        candidate_id = choice.get("candidate_id")
        if candidate_id not in frame["candidate_ids"]:
            raise ValueError("decision outside published menu")
        from .action_receipt import verify_applied_action
        assignments = verify_applied_action(before, choice, after)
        for key in ("log_prob", "value", "entropy"):
            if type(choice.get(key)) not in (int, float) or not math.isfinite(choice[key]):
                raise ValueError("nonfinite behavior-policy statistic")
        outcome = after.get("outcome", {})
        if type(outcome.get("terminated")) is not bool:
            raise ValueError("missing native terminal flag")
        record = {"schema": "civ6rl.diagnostic_transition970.v1", "player_id": frame["player_id"],
                  "episode_id": frame["episode_id"], "sequence": frame["sequence"],
                  "policy_identity": self.policy_identity, "vocabulary_sha256": self.vocabulary.fingerprint,
                  "before": copy.deepcopy(before), "after": copy.deepcopy(after),
                  "candidate_id": candidate_id, "action_index": frame["candidate_ids"].index(candidate_id),
                  "assignments": assignments,
                  "behavior_log_prob": choice["log_prob"], "behavior_value": choice["value"],
                  "behavior_entropy": choice["entropy"], "reward": None,
                  "terminated": outcome["terminated"], "truncated": False,
                  "training_eligible": False, "reward_contract": None}
        self.records.append(record)
        self._successor = copy.deepcopy(after)
        self.finished = outcome["terminated"]
        return record

    def finish_budget(self):
        if self.records and not self.finished:
            self.records[-1]["truncated"] = True
            self.records[-1]["truncation_reason"] = "diagnostic_budget"
        self.finished = True
        return self.records

    def training_records(self):
        raise RuntimeError("No qualified native scenario/reward contract: diagnostic transitions are not PPO samples")
