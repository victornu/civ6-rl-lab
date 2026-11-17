"""The GameCore can publish the deadline turn before the player is ready."""
import hashlib
import json
import unittest

from civ6rl.learning.lan_rollout import task_step
from civ6rl.learning.skill_context993 import build_skill_scenario, skill_digest
from civ6rl.learning.skill_task991 import DEFENSE, build_task


def view(turn, *, ready=True, city=True, warrior=True, truncated=False):
    return {
        "observation": dict(
            schema="civ6rl-player-observation-v1", player_id=0, turn=turn,
            width=10, height=10, cities=[dict(id=1, owner=0, x=1, y=1)] if city else [],
            units=[dict(id=2, owner=0, type=0, x=1, y=2)] if warrior else [],
        ),
        "decisions": {"turn_controls": dict(active=ready, active_complete=ready)},
        "outcome": dict(terminated=False, truncated=truncated, reason=None),
    }


class DefenseDeadlineTests(unittest.TestCase):
    def setUp(self):
        self.before = view(22)
        task = build_task(
            task_id="defense-deadline-test", player_id=0, start_observation=self.before["observation"],
            objective=dict(kind=DEFENSE, defended_city=dict(id=1, x=1, y=1),
                           force_types=["UNIT_WARRIOR"], minimum_force=1),
            deadline_turn=23, unit_types=[dict(index=0, type="UNIT_WARRIOR")],
        )
        catalogue = hashlib.sha256(json.dumps(task["unit_types"], sort_keys=True, ensure_ascii=True,
                                              separators=(",", ":")).encode("ascii")).hexdigest()
        self.scenario = build_skill_scenario(
            scenario_id=task["task_id"], skill_task=task, max_decisions=4, max_turns=1,
            identity=dict(save_sha256="a"*64, gamecore_sha256="b"*64, application_config_sha256="c"*64,
                          final_database_sha256="d"*64, gameplay_scripts_sha256="e"*64, lineage_sha256="f"*64),
            catalogue_source=dict(unit_types_sha256=catalogue, gameplay_database_sha256="d"*64,
                                  index_order_source_sha256="e"*64),
        )
        self.passive = view(23, ready=False)

    def test_scores_the_completed_turn(self):
        result = task_step(self.scenario, self.before, [self.passive, view(23)], decision_count=1)
        self.assertEqual(result["reward"], 1.0)
        self.assertFalse(result["bootstrap"])
        self.assertEqual(result["scenario_sha256"], skill_digest(self.scenario))

    def test_defender_lost_while_turn_is_becoming_ready(self):
        result = task_step(self.scenario, self.before, [self.passive, view(23, warrior=False)], decision_count=1)
        self.assertEqual(result["reason"], "public_deadline_unmet")
        self.assertEqual(result["reward"], -1.0)

    def test_collection_limit_cannot_confirm_a_passive_deadline(self):
        with self.assertRaisesRegex(ValueError, "deadline requires a ready"):
            task_step(self.scenario, self.before, [view(23, ready=False, truncated=True)], decision_count=1)

    def test_observed_city_loss_still_ends_immediately(self):
        result = task_step(self.scenario, self.before, [view(22, ready=False, city=False)], decision_count=1)
        self.assertEqual(result["reason"], "protected_city_lost")
        self.assertEqual(result["reward"], -1.0)
