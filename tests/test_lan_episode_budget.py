"""Bounded LAN integration on retained native I/O and fixture behavior scores.

No native process or genuine on-policy training collection is created here.
"""
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_lan_policy_learner as fixtures
from tests.test_lan_budget_transport import budget


class RetainedClient:
    def __init__(self, rows):
        self.rows = iter(deepcopy(rows))
        self.initial = next(self.rows)["view"]
        self.view_calls = 0
        self.receipt = None
        self.session_state = dict(human_player_ids=[0])

    def view(self):
        self.view_calls += 1
        return self.initial

    def act(self, player, decision_id, candidate_id, *, parameters):
        row = next(self.rows)
        assert row["operation"] == "act"
        assert row["receipt"]["player_id"] == player
        selection = row["receipt"]["selection"]
        assert (selection["decision_id"], selection["candidate_id"]) == (decision_id, candidate_id)
        assert selection.get("parameters") == parameters
        self.receipt = row["receipt"]
        return row["view"]

    def last_command_receipt(self, player):
        assert self.receipt["player_id"] == player
        return self.receipt

    def poll(self, player):
        row = next(self.rows)
        assert row["operation"] == "poll" and row["view"]["observation"]["player_id"] == player
        return row["view"]


class ScriptedWorker:
    def __init__(self, records, contract):
        self.records = records
        self.diagnostic_collection = deepcopy(contract)
        self.collection_seconds_remaining = contract["wall_seconds"] - contract["request_seconds"] - 30

    def infer(self, view, *, scenario, completed_decisions):
        record = self.records[completed_decisions]
        assert record["before"] == view
        return deepcopy(record["choice"])


@unittest.skipIf(fixtures.torch is None, "PyTorch is an optional learning dependency")
class LanEpisodeBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixtures.LanJointPolicyLearnerTests.setUpClass()
        cls.helper = fixtures.LanJointPolicyLearnerTests()
        raw = (cls.helper.root / "work/bbg751-government/progress/public-replay.jsonl").read_bytes()
        if sha256(raw).hexdigest() != "5a92aecad628364e24b6d88ffb93e7a843cb8e73e63e099352076855f66b7a13":
            raise ValueError("Retained native progress transcript changed")
        cls.rows = [json.loads(line) for line in raw.splitlines()]
        cls.native_records = []
        before = cls.rows[0]["view"]
        for row in cls.rows[1:]:
            if row["operation"] == "close":
                break
            if row["operation"] == "act":
                selection = row["receipt"]["selection"]
                choice = dict(candidate_id=selection["candidate_id"])
                if "parameters" in selection:
                    choice["assignments"] = selection["parameters"]["assignments"]
                cls.native_records.append(dict(before=before, choice=choice,
                    command_receipt=row["receipt"], successors=[row["view"]]))
            elif row["operation"] == "poll":
                cls.native_records[-1]["successors"].append(row["view"])
            else:
                raise ValueError("Unexpected retained native boundary")
            before = row["view"]
        if len(cls.native_records) != 18:
            raise ValueError("Expected the existing eighteen-decision native interval")

    def fixture(self):
        return self.helper.fixture(native_records=self.native_records, max_turns=30, max_decisions=64)

    def test_eighteen_native_intervals_collect_without_redefining_the_task_horizon(self):
        from civ6rl.learning.lan_collection import collect_episode
        _, records, scenario, _ = self.fixture()
        contract = budget(18)
        client = RetainedClient(self.rows)
        summary = collect_episode(client, ScriptedWorker(records, contract), scenario,
                                  max_decisions=18, diagnostic_collection=contract)
        self.assertEqual(summary["records"], records)
        self.assertEqual(summary["polls"], 72)
        self.assertEqual(summary["collection"], contract)
        self.assertEqual(summary["task_result"]["reason"], "explicit_global_collection_boundary")
        self.assertTrue(summary["task_result"]["truncated"])
        self.assertTrue(summary["task_result"]["bootstrap"])
        self.assertFalse(summary["task_result"]["terminated"])
        self.assertEqual(scenario["horizon"], dict(max_decisions=64, max_turns=30))
        self.assertEqual(next(client.rows)["operation"], "close")

    def test_round_boundary_truncates_at_the_real_ready_successor_and_requires_handshake(self):
        from civ6rl.learning.lan_collection import collect_episode
        _, records, scenario, _ = self.fixture()
        contract = budget(18)
        contract["max_rounds"] = 1
        client = RetainedClient(self.rows)
        summary = collect_episode(client, ScriptedWorker(records, contract), scenario,
                                  max_decisions=18, diagnostic_collection=contract)
        self.assertEqual(summary["task_result"]["reason"], "explicit_collection_round_boundary")
        self.assertTrue(summary["records"][-1]["successors"][-1]["decisions"]["turn_controls"]["active_complete"])
        self.assertTrue(summary["task_result"]["bootstrap"])
        client = RetainedClient(self.rows)
        wrong = dict(contract, max_rounds=2)
        with self.assertRaisesRegex(ValueError, "must match the worker handshake"):
            collect_episode(client, ScriptedWorker(records, contract), scenario,
                            max_decisions=18, diagnostic_collection=wrong)
        self.assertEqual(client.view_calls, 0)
        expired = ScriptedWorker(records, contract)
        expired.collection_seconds_remaining = 0
        with self.assertRaisesRegex(ValueError, "no remaining collection time"):
            collect_episode(client, expired, scenario, max_decisions=18, diagnostic_collection=contract)
        self.assertEqual(client.view_calls, 0)

    def test_long_update_requires_explicit_budget_before_any_optimizer_change(self):
        from civ6rl.learning.learner import update_native_diagnostic
        torch = fixtures.torch
        model, records, scenario, options = self.fixture()
        optimizer = torch.optim.Adam(model.parameters())
        with self.assertRaisesRegex(ValueError, "decision bound"):
            update_native_diagnostic(model, optimizer, self.helper.vocabulary, records, scenario,
                                     **options, epochs=1)
        with self.assertRaisesRegex(ValueError, "declared collection budget"):
            update_native_diagnostic(model, optimizer, self.helper.vocabulary, records, scenario,
                                     **options, epochs=1, diagnostic_collection=budget(16))
        self.assertFalse(optimizer.state)

    def test_worker_updates_and_resumes_all_eighteen_records_with_budget_provenance(self):
        from civ6rl.learning.model import PlayerMemory
        from civ6rl.learning.training_state import load_training_bytes
        from civ6rl.learning.worker import _update_diagnostic
        torch = fixtures.torch
        model, records, scenario, options = self.fixture()
        contract = budget(18)
        model._sample_audit = options["sample_audit"]
        model._policy_version = options["expected_policy_version"]
        model._diagnostic_updates = 0
        model._closed_episodes = set()
        model._diagnostic_collection = contract
        model._diagnostic_collection_updates = 0
        runtime = (torch, self.helper.vocabulary, model, PlayerMemory(), "cpu")
        request = dict(op="update_diagnostic", records=records, scenario=scenario,
                       policy_version=model._policy_version, burn_in=0, epochs=1)
        current = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            os.chdir(temporary)
            try:
                report = _update_diagnostic(request, runtime)
                self.assertEqual(report["collection"], contract)
                self.assertEqual(report["optimized_decisions"], 18)
                self.assertLess(report["behavior_log_prob_max_error"], 1e-5)
                reference = report["learner_checkpoint"]
                restored = load_training_bytes(Path(reference["path"]).read_bytes(), sha256=reference["sha256"])
                self.assertEqual(restored[2]._last_update_context["lan_collection"], contract)
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, restored[2].state_dict()[key], rtol=0, atol=0)
                self.assertEqual(model._diagnostic_collection_updates, 1)
                with self.assertRaisesRegex(ValueError, "update budget exhausted"):
                    _update_diagnostic(request, runtime)
            finally:
                os.chdir(current)

    def test_extended_route_reuses_the_existing_checkpointed_grouped_optimizer(self):
        from civ6rl.learning import learner
        model, records, scenario, options = self.fixture()
        # Routing fixture only: the repeated rows are never replayed or optimized.
        rows = (records * 4)[:65]
        group = {"fixture": True}
        prepared = dict(ppo_group=group, report=dict(critic_mode="player_only"))
        with patch.object(learner, "_prepare_native_diagnostic", return_value=prepared) as prepare:
            with patch("civ6rl.learning.ppo.update_grouped", return_value={}) as update:
                report = learner.update_native_diagnostic(model, object(), self.helper.vocabulary,
                    rows, scenario, **options, epochs=1, diagnostic_collection=budget(256))
        self.assertEqual(prepare.call_args.kwargs["decision_limit"], 256)
        self.assertEqual(prepare.call_args.kwargs["sequence_limit"], 256)
        self.assertEqual(update.call_args.args[2], [group])
        self.assertEqual(update.call_args.kwargs["maximum_steps"], 256)
        self.assertEqual(report["collection"]["max_decisions"], 256)


if __name__ == "__main__":
    unittest.main()
