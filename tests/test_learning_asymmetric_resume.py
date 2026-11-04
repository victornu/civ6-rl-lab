"""Checkpoint tests for an actor and an isolated privileged value critic."""
import hashlib
import io
import random
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class AsymmetricResumeTests(unittest.TestCase):
    def setUp(self):
        from tests.test_learning_encoding import response
        from civ6rl.learning.artifacts import policy_version
        from civ6rl.learning.encoding import (ACTION_FEATURES, ENTITY_FEATURES,
            GLOBAL_FEATURES, MAP_CHANNELS, ContentVocabulary, encode_response)
        from civ6rl.learning.model import PolicyConfig, PlayerMemory, RecurrentPolicy, collate
        from civ6rl.learning.ppo import update

        torch.manual_seed(817)
        torch.set_num_threads(1)
        raw = response(candidates=[
            {"candidate_id": "end", "action": {"kind": "end_turn"}},
            {"candidate_id": "skip", "action": {"kind": "skip_unit", "unit_id": 101}},
        ])
        self.vocabulary = ContentVocabulary.fit(raw)
        config = PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES), len(ENTITY_FEATURES),
                              len(ACTION_FEATURES), self.vocabulary.vocabulary_size,
                              hidden_size=32, spatial_size=8)
        self.model = RecurrentPolicy(config)
        self.actor_optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)
        # Exercise the existing public PPO update before adding the sidecar.
        batch = collate([encode_response(raw, vocabulary=self.vocabulary)])
        with torch.no_grad():
            distribution, value, _ = self.model(batch)
            actions = torch.tensor([[0]], dtype=torch.long)
            old_log_prob = distribution.log_prob(actions[0])[None, :]
        update(self.model, self.actor_optimizer, [batch], actions, old_log_prob,
               torch.ones((1, 1)), value.detach()[None, :] + 0.25, epochs=1)
        self.model._optimizer = self.actor_optimizer
        self.model._diagnostic_updates = 1
        self.model._sample_audit = {}
        self.model._closed_episodes = {("episode-1", "generation-1")}
        self.model._last_update_context = {
            "scenario_sha256": "a" * 64,
            "records_sha256": "b" * 64,
            "parent_policy_version": "c" * 64,
            "optimized_decisions": 2,
            "epochs": 1,
        }
        self.model._policy_version = policy_version(self.model, self.vocabulary)
        self.memory = PlayerMemory()
        self.runtime = (torch, self.vocabulary, self.model, self.memory, "cpu")

    def attach_critic(self):
        from civ6rl.learning.privileged_critic import (
            PrivilegedCritic, PrivilegedCriticConfig, critic_bytes, critic_version,
            update_critic,
        )

        config = PrivilegedCriticConfig(
            privileged_schema="test-hidden-state-v1",
            privileged_features=("global_gold_total", "hidden_unit_count"),
            actor_hidden_dim=self.model.config.hidden_size,
            hidden_dim=8,
        )
        critic = PrivilegedCritic(config)
        optimizer = critic.make_optimizer(learning_rate=3e-4)
        values = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
        mask = torch.tensor([[True, False]])
        hidden, cell = self.model.initial_state(1)
        target = torch.tensor([2.0], dtype=torch.float32)
        update_critic(critic, optimizer, values, mask, hidden, cell, target)
        version = critic_version(critic)
        self.model._privileged_state = {"critic": critic, "optimizer": optimizer,
                                        "version": version}
        self.model._last_update_context.update({
            "privileged_sha256": hashlib.sha256(critic_bytes(critic)).hexdigest(),
            "parent_critic_version": "d" * 64,
        })
        return critic

    def test_player_only_roundtrip_preserves_legacy_context_shape_and_v1_is_rejected(self):
        from civ6rl.learning.training_state import training_bytes, load_training_bytes

        blob = training_bytes(self.runtime)
        restored = load_training_bytes(blob, sha256=hashlib.sha256(blob).hexdigest(), device="cpu")
        self.assertIsNone(restored[2]._privileged_state)
        self.assertEqual(self.model._last_update_context, restored[2]._last_update_context)
        self.assertEqual(set(self.model._last_update_context), {
            "scenario_sha256", "records_sha256", "parent_policy_version",
            "optimized_decisions", "epochs"})

        legacy = io.BytesIO()
        torch.save({"schema": "civ6rl.learner_boundary.v1"}, legacy)
        legacy_blob = legacy.getvalue()
        with self.assertRaisesRegex(ValueError, "v1/v2 snapshots"):
            load_training_bytes(legacy_blob, sha256=hashlib.sha256(legacy_blob).hexdigest(), device="cpu")
        legacy_v2 = io.BytesIO()
        torch.save({"schema": "civ6rl.learner_boundary.v2"}, legacy_v2)
        legacy_v2_blob = legacy_v2.getvalue()
        with self.assertRaisesRegex(ValueError, "v1/v2 snapshots"):
            load_training_bytes(legacy_v2_blob, sha256=hashlib.sha256(legacy_v2_blob).hexdigest(), device="cpu")

    def test_history_only_checkpoint_keeps_separate_critic_without_private_provenance(self):
        from civ6rl.learning.artifacts import policy_bytes
        from civ6rl.learning.training_state import training_bytes, load_training_bytes

        actor_before = policy_bytes(self.model, self.vocabulary)
        self.attach_critic()
        self.model._critic_mode = "history_only_diagnostic"
        self.model._last_update_context.pop("privileged_sha256")
        self.assertEqual(actor_before, policy_bytes(self.model, self.vocabulary))
        blob = training_bytes(self.runtime)
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        self.assertEqual("history_only_diagnostic", payload["critic_mode"])
        self.assertIsNotNone(payload["privileged_critic"])
        self.assertNotIn("privileged_sha256", payload["last_update_context"])
        restored = load_training_bytes(blob, sha256=hashlib.sha256(blob).hexdigest(), device="cpu")
        model = restored[2]
        self.assertEqual("history_only_diagnostic", model._critic_mode)
        self.assertIsNotNone(model._privileged_state)
        self.assertEqual(self.model._privileged_state["version"], model._privileged_state["version"])
        self.assertEqual(self.model._last_update_context, model._last_update_context)
        self.assertEqual({"scenario_sha256", "records_sha256", "parent_policy_version",
                          "parent_critic_version", "optimized_decisions", "epochs"},
                         set(model._last_update_context))

        altered = dict(payload)
        altered["critic_mode"] = "privileged_diagnostic"
        stream = io.BytesIO()
        torch.save(altered, stream)
        altered_blob = stream.getvalue()
        with self.assertRaisesRegex(ValueError, "learner update provenance"):
            load_training_bytes(altered_blob, sha256=hashlib.sha256(altered_blob).hexdigest(), device="cpu")

        # Switching modes without the corresponding provenance or adding private
        # snapshot provenance to a history-only update is rejected.
        self.model._critic_mode = "privileged_diagnostic"
        with self.assertRaisesRegex(ValueError, "learner update provenance"):
            training_bytes(self.runtime)
        self.model._critic_mode = "history_only_diagnostic"
        self.model._last_update_context["privileged_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "learner update provenance"):
            training_bytes(self.runtime)

    def test_asymmetric_roundtrip_restores_isolated_critic_and_optimizer(self):
        from civ6rl.learning.artifacts import policy_bytes
        from civ6rl.learning.training_state import training_bytes, load_training_bytes
        from civ6rl.learning.privileged_critic import critic_bytes, critic_version

        actor_artifact_before = policy_bytes(self.model, self.vocabulary)
        critic = self.attach_critic()
        actor_artifact_after = policy_bytes(self.model, self.vocabulary)
        self.assertEqual(actor_artifact_before, actor_artifact_after)
        self.assertFalse(any("privileged" in name for name in self.model.state_dict()))

        blob = training_bytes(self.runtime)
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        expected_python = random.Random()
        expected_python.setstate(payload["rng"]["python"])
        expected_python_value = expected_python.random()
        expected_generator = torch.Generator(device="cpu")
        expected_generator.set_state(payload["rng"]["torch_cpu"])
        expected_torch = torch.rand(4, generator=expected_generator)
        restored = load_training_bytes(blob, sha256=hashlib.sha256(blob).hexdigest(), device="cpu",
                                      expected_vocabulary_fingerprint=self.vocabulary.fingerprint)
        restored_model = restored[2]
        state = restored_model._privileged_state
        self.assertEqual(set(state), {"critic", "optimizer", "version"})
        self.assertEqual(critic_version(critic), state["version"])
        self.assertEqual(critic_bytes(critic), critic_bytes(state["critic"]))
        self.assertEqual(actor_artifact_before, policy_bytes(restored_model, self.vocabulary))
        self.assertEqual(self.model._last_update_context, restored_model._last_update_context)
        self.assertEqual([name for name, _ in critic.named_parameters()],
                         [name for name, _ in state["critic"].named_parameters()])
        for original, resumed in ((self.actor_optimizer, restored_model._optimizer),
                                  (self.model._privileged_state["optimizer"], state["optimizer"])):
            before, after = original.state_dict(), resumed.state_dict()
            self.assertEqual(before["param_groups"], after["param_groups"])
            self.assertEqual(before["state"].keys(), after["state"].keys())
            for index, moments in before["state"].items():
                for key, value in moments.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, after["state"][index][key])
                    else:
                        self.assertEqual(value, after["state"][index][key])
        self.assertFalse(set(id(p) for p in restored_model.parameters()) &
                         set(id(p) for group in state["optimizer"].param_groups for p in group["params"]))
        self.assertEqual(expected_python_value, random.random())
        torch.testing.assert_close(expected_torch, torch.rand(4))

    def test_critic_payload_tampering_and_optimizer_sharing_are_rejected(self):
        from civ6rl.learning.training_state import training_bytes, load_training_bytes

        self.attach_critic()
        blob = training_bytes(self.runtime)
        payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
        payload["privileged_critic"]["bytes"] += b"tampered"
        stream = io.BytesIO()
        torch.save(payload, stream)
        tampered = stream.getvalue()
        with self.assertRaisesRegex(ValueError, "digest or version"):
            load_training_bytes(tampered, sha256=hashlib.sha256(tampered).hexdigest(), device="cpu")

        from civ6rl.learning.privileged_critic import critic_version
        critic_state = self.model._privileged_state
        critic_state["optimizer"] = torch.optim.Adam(
            list(critic_state["critic"].parameters()) + [next(self.model.parameters())], lr=3e-4)
        critic_state["version"] = critic_version(critic_state["critic"])
        with self.assertRaisesRegex(ValueError, "share parameters|owns actor parameters|parameter order"):
            training_bytes(self.runtime)


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class AsymmetricContinuationTests(unittest.TestCase):
    @staticmethod
    def _continued_episode(first):
        import copy
        response = copy.deepcopy(first)
        response["episode_id"] = "resume-episode-2"
        response["game_generation"] = "resume-generation-2"
        response["decisions"]["episode_id"] = response["episode_id"]
        response["decisions"]["game_generation"] = response["game_generation"]
        return response

    @staticmethod
    def _rng(device):
        import random
        torch.mps.synchronize() if device == "mps" else None
        return (random.getstate(), torch.get_rng_state().clone(),
                torch.mps.get_rng_state().clone() if device == "mps" else None)

    @staticmethod
    def _optimizer_copy(optimizer):
        import copy
        return copy.deepcopy(optimizer.state_dict())

    def _run_branch(self, runtime, first, scenario, device):
        from tests.test_learning_asymmetric_ppo import collect, request_for
        from civ6rl.learning.worker import _update_diagnostic

        records = collect(runtime, self._continued_episode(first), scenario)
        if runtime[2]._critic_mode == "history_only_diagnostic":
            request = dict(records=records, scenario=scenario,
                           policy_version=runtime[2]._policy_version,
                           burn_in=1, epochs=2,
                           critic_version=runtime[2]._privileged_state["version"])
        else:
            request = request_for(runtime, scenario, records)
        report = _update_diagnostic(request, runtime)
        model = runtime[2]
        return {
            "records": records,
            "report": report,
            "actor": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "critic": {key: value.detach().cpu().clone()
                       for key, value in model._privileged_state["critic"].state_dict().items()},
            "actor_optimizer": self._optimizer_copy(model._optimizer),
            "critic_optimizer": self._optimizer_copy(model._privileged_state["optimizer"]),
            "policy_version": model._policy_version,
            "critic_version": model._privileged_state["version"],
            "critic_mode": model._critic_mode,
            "update_context": dict(model._last_update_context),
            "updates": model._diagnostic_updates,
            "closed_episodes": set(model._closed_episodes),
            "rng": self._rng(device),
        }

    def _assert_same_branch(self, left, right):
        import json

        def canonical_hash(value):
            encoded = json.dumps(value, sort_keys=True, ensure_ascii=True,
                                 allow_nan=False, separators=(",", ":")).encode()
            return hashlib.sha256(encoded).hexdigest()

        for first_record, second_record in zip(left["records"], right["records"]):
            a, b = first_record["choice"], second_record["choice"]
            for key in ("candidate_id", "assignments", "policy_version", "critic_version", "critic_mode"):
                self.assertEqual(a.get(key), b.get(key))
            for key in ("log_prob", "value", "entropy"):
                self.assertAlmostEqual(a[key], b[key], places=5)
        self.assertEqual(left["policy_version"], right["policy_version"])
        self.assertEqual(left["critic_version"], right["critic_version"])
        self.assertEqual(left["critic_mode"], right["critic_mode"])
        self.assertEqual(left["update_context"]["records_sha256"], canonical_hash(left["records"]))
        self.assertEqual(right["update_context"]["records_sha256"], canonical_hash(right["records"]))
        left_context = {key: value for key, value in left["update_context"].items()
                        if key != "records_sha256"}
        right_context = {key: value for key, value in right["update_context"].items()
                         if key != "records_sha256"}
        self.assertEqual(left_context, right_context)
        self.assertEqual(left["updates"], right["updates"])
        self.assertEqual(left["closed_episodes"], right["closed_episodes"])
        self.assertEqual(left["actor"].keys(), right["actor"].keys())
        self.assertEqual(left["critic"].keys(), right["critic"].keys())
        for weights in ("actor", "critic"):
            for key in left[weights]:
                torch.testing.assert_close(left[weights][key], right[weights][key], rtol=1e-5, atol=1e-7)
        for label in ("actor_optimizer", "critic_optimizer"):
            a, b = left[label], right[label]
            self.assertEqual(a["param_groups"], b["param_groups"])
            self.assertEqual(a["state"].keys(), b["state"].keys())
            for index, moments in a["state"].items():
                for name, value in moments.items():
                    if isinstance(value, torch.Tensor):
                        torch.testing.assert_close(value, b["state"][index][name], rtol=1e-5, atol=1e-7)
                    else:
                        self.assertEqual(value, b["state"][index][name])
        self.assertEqual(left["rng"][0], right["rng"][0])
        torch.testing.assert_close(left["rng"][1], right["rng"][1], rtol=0, atol=0)
        if left["rng"][2] is not None:
            torch.testing.assert_close(left["rng"][2], right["rng"][2], rtol=0, atol=0)

    def _check_continuation(self, device, critic_mode="privileged_diagnostic"):
        import os
        import tempfile
        from pathlib import Path
        from tests.test_learning_asymmetric_ppo import fixture, request_for, collect
        from civ6rl.learning.worker import _update_diagnostic, _build_policy
        from civ6rl.learning.training_state import load_training_bytes
        from civ6rl.learning.critic_modes import mode_of

        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp:
            os.chdir(temp)
            try:
                runtime, first, scenario, records = fixture(device)
                if critic_mode == "history_only_diagnostic":
                    runtime = _build_policy(runtime[1].serialize(), device, 978,
                                            critic_mode=critic_mode)
                    records = collect(runtime, first, scenario)
                    first_request = dict(records=records, scenario=scenario,
                        policy_version=runtime[2]._policy_version, burn_in=1, epochs=2,
                        critic_version=runtime[2]._privileged_state["version"])
                else:
                    first_request = request_for(runtime, scenario, records)
                first_report = _update_diagnostic(first_request, runtime)
                resume_ref = first_report["learner_checkpoint"]
                checkpoint = Path(resume_ref["path"]).read_bytes()
                saved = torch.load(io.BytesIO(checkpoint), map_location="cpu", weights_only=True)
                self.assertEqual(critic_mode, saved["critic_mode"])
                if critic_mode == "history_only_diagnostic":
                    self.assertNotIn("privileged_sha256", saved["last_update_context"])

                continued = self._run_branch(runtime, first, scenario, device)
                restored = load_training_bytes(checkpoint, sha256=resume_ref["sha256"], device=device,
                    expected_vocabulary_fingerprint=runtime[1].fingerprint)
                resumed = self._run_branch(restored, first, scenario, device)
                self._assert_same_branch(continued, resumed)

                public = _build_policy(runtime[1].serialize(), device, 978,
                                        checkpoint=first_report["checkpoint"])
                self.assertIsNone(getattr(public[2], "_privileged_state", None))
                self.assertEqual("player_only", mode_of(public[2]))
                self.assertEqual(public[2]._policy_version, first_report["policy_version"])
            finally:
                os.chdir(original_cwd)

    def test_cpu_asymmetric_update_continues_identically_after_resume(self):
        self._check_continuation("cpu")

    @unittest.skipUnless(torch is not None and hasattr(torch.backends, "mps")
                         and torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_asymmetric_update_continues_identically_after_resume(self):
        self._check_continuation("mps")

    def test_cpu_history_only_update_continues_identically_after_resume(self):
        self._check_continuation("cpu", "history_only_diagnostic")

    @unittest.skipUnless(torch is not None and hasattr(torch.backends, "mps")
                         and torch.backends.mps.is_available(), "MPS unavailable")
    def test_mps_history_only_update_continues_identically_after_resume(self):
        self._check_continuation("mps", "history_only_diagnostic")


if __name__ == "__main__":
    unittest.main()
