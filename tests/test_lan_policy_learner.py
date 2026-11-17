"""Learner regressions on retained native v29 I/O, with fixture behavior scores.

The native choices were scripted. The scores/audit below are teacher-forced
test metadata, not evidence of unconditioned native collection or training.
"""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class LanJointPolicyLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from civ6_host.runtime.paths import resource_root
        from civ6rl.learning.content_vocabulary import load_public_content_vocabulary

        cls.root = resource_root()
        proof_path = cls.root / "evidence/host-bbg751-native-policy-reconfirmation.json"
        if not proof_path.is_file():
            raise unittest.SkipTest("Retained native v29 delivery required")
        proof_bytes = proof_path.read_bytes()
        if sha256(proof_bytes).hexdigest() != "2da83940019da172e8295805fdaef6680281caa5352a4b8998cfddcc3606c6e8":
            raise ValueError("Native v29 proof changed")
        proof = json.loads(proof_bytes)

        def verified(reference):
            path = cls.root / reference["path"]
            raw = path.read_bytes()
            if sha256(raw).hexdigest() != reference["sha256"]:
                raise ValueError("Retained native reference changed")
            return raw

        coverage = json.loads(verified(proof["consumer"]["interval_coverage"]))
        rows = [json.loads(raw) for raw in verified(coverage["transcript"]).splitlines()]
        cls.native_records = []
        before = rows[0]["view"]
        for row in rows[1:]:
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
                raise ValueError("Unexpected native replay boundary")
            before = row["view"]
        if len(cls.native_records) != 2 or "assignments" not in cls.native_records[-1]["choice"]:
            raise ValueError("Missing retained joint policy action")
        capture = json.loads((cls.root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json").read_bytes())
        database = capture["database_copies"]["gameplay"]
        cls.vocabulary = load_public_content_vocabulary(database["path"],
            expected_sha256=database["sha256"],
            gameinfo_evidence_path=cls.root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)
        cls.catalogue_scenario = json.loads((cls.root / "work/bbg751-defense/scenario.json").read_bytes())
        cls.checkpoint_refs = proof["consumer"]["source_only_checkpoint_resumes"]

    def fixture(self, *, native_records=None, max_turns=5, max_decisions=None):
        from civ6rl.learning.artifacts import policy_version
        from civ6rl.learning.encoding import ACTOR_V15, GLOBAL_FEATURES, MAP_CHANNELS, actor_feature_names
        from civ6rl.learning.model import PolicyConfig, RecurrentPolicy, collate
        from civ6rl.learning.skill_context993 import build_skill_scenario, skill_digest
        from civ6rl.learning.skill_task991 import DEVELOPMENT, build_task
        from civ6rl.learning.task_context import encode_task_response

        records = deepcopy(self.native_records if native_records is None else native_records)
        first = records[0]["before"]["observation"]
        task = build_task(task_id="joint-policy-learner-regression", player_id=first["player_id"],
            start_observation=first, objective=dict(kind=DEVELOPMENT, unit_type="UNIT_BUILDER",
                target_count=4096, gold_reserve=0), deadline_turn=first["turn"] + max_turns,
            unit_types=self.catalogue_scenario["skill_task"]["unit_types"])
        scenario = build_skill_scenario(scenario_id=task["task_id"], skill_task=task,
            max_decisions=len(records) if max_decisions is None else max_decisions, max_turns=max_turns,
            identity={key: None for key in self.catalogue_scenario["identity"]},
            catalogue_source=self.catalogue_scenario["catalogue_source"])
        entities, actions = actor_feature_names(ACTOR_V15)
        torch.manual_seed(31)
        model = RecurrentPolicy(PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES),
            len(entities), len(actions), self.vocabulary.vocabulary_size,
            hidden_size=32, spatial_size=8, actor_schema=ACTOR_V15))
        version = policy_version(model, self.vocabulary)
        audit = {}
        state = None
        for i, record in enumerate(records):
            view, choice = record["before"], record["choice"]
            frame = encode_task_response(view, vocabulary=self.vocabulary, scenario=scenario,
                                         completed_decisions=i, actor_schema=ACTOR_V15)
            self.assertEqual(frame["candidate_ids"], [r["candidate_id"] for r in view["decisions"]["candidates"]])
            with torch.no_grad():
                logp, value, state, _ = model.evaluate_actions(collate([frame]),
                    torch.tensor([frame["candidate_ids"].index(choice["candidate_id"])]),
                    [choice.get("assignments")], state)
            # These declarations exercise learner validation in this fixture;
            # they do not relabel the retained scripted native collection.
            choice.update(episode_id=view["episode_id"], game_generation=view["game_generation"],
                player_id=view["observation"]["player_id"], sequence=view["sequence"],
                policy_version=version, sampling_mode="unconditioned_policy_sample",
                memory_reset=i == 0, scenario_sha256=skill_digest(scenario),
                task_completed_decisions=i, log_prob=logp.item(), value=value.item())
            key = (view["episode_id"], view["game_generation"], choice["player_id"], view["sequence"])
            audit[key] = dict(before=view, choice=choice)
        return model, records, scenario, dict(sample_audit=audit, expected_policy_version=version)

    def test_native_joint_hashes_reach_replay_gae_and_policy_decoder_gradients(self):
        from civ6rl.learning.learner import _prepare_native_diagnostic, update_native_diagnostic
        model, records, scenario, options = self.fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        prepared = _prepare_native_diagnostic(model, optimizer, self.vocabulary, records, scenario,
            **options, lan=True, collection_cut=True, epochs=1)
        self.assertEqual(prepared["ppo_group"]["assignments"],
                         [[None], [records[-1]["choice"]["assignments"]]])
        self.assertLess(prepared["report"]["behavior_log_prob_max_error"], 1e-5)
        self.assertTrue(prepared["report"]["task_steps"][-1]["truncated"])
        self.assertTrue(torch.isfinite(prepared["ppo_group"]["advantages"]).all())
        before = model.policy_score.weight.detach().clone()
        report = update_native_diagnostic(model, optimizer, self.vocabulary, records, scenario,
                                          **options, epochs=1)
        self.assertEqual(report["optimized_decisions"], 2)
        self.assertFalse(report["training_eligible"])
        self.assertTrue(torch.isfinite(model.policy_score.weight.grad).all())
        self.assertGreater(float((model.policy_score.weight.detach() - before).abs().max()), 0.)

    def test_missing_or_different_joint_hashes_fail_before_optimizer_changes(self):
        from civ6rl.learning.learner import update_native_diagnostic
        model, records, scenario, options = self.fixture()
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        wrong = deepcopy(records)
        wrong[-1]["choice"].pop("assignments")
        with self.assertRaisesRegex(ValueError, "requires unsigned joint"):
            update_native_diagnostic(model, optimizer, self.vocabulary, wrong, scenario, **options, epochs=1)
        wrong = deepcopy(records)
        cards = wrong[-1]["choice"]["assignments"]
        cards[0], cards[1] = cards[1], cards[0]
        with self.assertRaisesRegex(ValueError, "did not complete the sampled action"):
            update_native_diagnostic(model, optimizer, self.vocabulary, wrong, scenario, **options, epochs=1)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(optimizer.state)

    def test_completed_reference_checkpoints_preserve_learned_state_with_explicit_transitions(self):
        from civ6rl.learning.training_state import load_training_bytes, training_bytes

        def exact(left, right):
            self.assertIs(type(left), type(right))
            if isinstance(left, torch.Tensor):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
            elif isinstance(left, dict):
                self.assertEqual(set(left), set(right))
                for key in left:
                    exact(left[key], right[key])
            elif isinstance(left, (list, tuple)):
                self.assertEqual(len(left), len(right))
                for a, b in zip(left, right):
                    exact(a, b)
            else:
                self.assertEqual(left, right)

        for entry in self.checkpoint_refs:
            reference = entry["checkpoint"]
            raw = (self.root / reference["path"]).read_bytes()
            self.assertEqual(sha256(raw).hexdigest(), reference["sha256"])
            from civ6rl.learning.training_state import exact_source_transition
            parent = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
            profile = exact_source_transition(parent['implementation_sha256'])
            runtime = load_training_bytes(raw, sha256=reference["sha256"],
                expected_vocabulary_fingerprint=self.vocabulary.fingerprint,
                implementation_migration=profile)
            parent = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
            restored = torch.load(io.BytesIO(training_bytes(runtime)), weights_only=True, map_location="cpu")
            self.assertEqual(set(parent), set(restored))
            for key in parent:
                if key != "implementation_sha256":
                    exact(parent[key], restored[key])

    def test_v29_transition_accepts_only_the_reviewed_prior_source_pins(self):
        from civ6rl.learning.training_state import (
            LAN_JOINT_POLICY_LEARNER_V29, exact_source_transition, implementation_hashes,
        )
        prior = implementation_hashes()
        prior.update(LAN_JOINT_POLICY_LEARNER_V29)
        self.assertEqual(exact_source_transition(prior), "lan_joint_policy_learner_v29")
        prior["lan_rollout.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "no reviewed exact source transition"):
            exact_source_transition(prior)


if __name__ == "__main__":
    unittest.main()
