"""Worker bridge compatibility for retained native v32 governor decisions.

These checks exercise checkpoint loading and policy inference only. The
teacher-forced native trace is not an episode collected by this policy and is
not evidence of learning or training admission.
"""
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class LanGovernorWorkerBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from civ6_host.runtime.paths import resource_root
        from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
        from civ6rl.learning.governors import extend_governor_vocabulary

        root = resource_root()
        replay = root / "work/bbg751-governors/native/public-replay.jsonl"
        capture = root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json"
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest("Retained native governor trace and pinned gameplay database required")
        with replay.open(encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream]
        cls.views = [row["view"] for row in rows]
        if len(cls.views) < 5 or any(
                view.get("decisions", {}).get("schema") !=
                "civ6-host.network-capability-decisions.v32" for view in cls.views[:5]):
            raise ValueError("Retained governor trace does not contain five consecutive v32 views")
        expected_actions = (
            "consider_governor_titles", "appoint_governor",
            "assign_governor", "promote_governor",
        )
        actual_actions = []
        for index in range(1, 5):
            selected = rows[index]["receipt"]["selection"]["candidate_id"]
            previous = rows[index - 1]["view"]["decisions"]["candidates"]
            action = next(row["action"] for row in previous
                          if row["candidate_id"] == selected)
            actual_actions.append(action["kind"])
        actual_actions = tuple(actual_actions)
        if actual_actions != expected_actions:
            raise ValueError("Retained native governor action sequence differs")

        capture_data = json.loads(capture.read_bytes())
        database = capture_data["database_copies"]["gameplay"]
        database_path = Path(database["path"])
        if sha256(database_path.read_bytes()).hexdigest() != database["sha256"]:
            raise RuntimeError("Pinned public gameplay database changed")
        parent = load_public_content_vocabulary(
            database_path, expected_sha256=database["sha256"],
            gameinfo_evidence_path=root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)
        cls.vocabulary = extend_governor_vocabulary(parent, [cls.views[0]])

    def make_runtime(self, actor_schema):
        from civ6rl.learning.encoding import (
            GLOBAL_FEATURES, MAP_CHANNELS, actor_feature_names,
        )
        from civ6rl.learning.model import PolicyConfig, RecurrentPolicy
        from civ6rl.learning.artifacts import policy_bytes
        from civ6rl.learning.worker import _build_policy

        entities, actions = actor_feature_names(actor_schema)
        config = PolicyConfig(
            map_channels=len(MAP_CHANNELS), global_features=len(GLOBAL_FEATURES),
            entity_features=len(entities), action_features=len(actions),
            vocabulary_size=self.vocabulary.vocabulary_size,
            actor_schema=actor_schema)
        model = RecurrentPolicy(config)
        blob = policy_bytes(model, self.vocabulary)
        digest = sha256(blob).hexdigest()
        from civ6_host.runtime.paths import resource_root
        artifact_root = resource_root() / "work/bbg751-governors/actor-v16"
        artifact_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifact_root, prefix="worker-checkpoint-") as directory:
            artifact_dir = Path(directory) / "learner-artifacts"
            artifact_dir.mkdir()
            (artifact_dir / (digest + ".pt")).write_bytes(blob)
            previous_cwd = Path.cwd()
            try:
                os.chdir(directory)
                return _build_policy(
                    self.vocabulary.serialize(), "cpu", 1601,
                    checkpoint={"path": "learner-artifacts/" + digest + ".pt",
                                "sha256": digest})
            finally:
                os.chdir(previous_cwd)

    def test_v16_checkpoint_infers_consecutive_native_governor_views_with_memory(self):
        from civ6rl.learning.encoding import ACTOR_V16
        from civ6rl.learning.model import actor_schema_for_config
        from civ6rl.learning.worker import _infer

        runtime = self.make_runtime(ACTOR_V16)
        self.assertEqual(actor_schema_for_config(runtime[2].config), ACTOR_V16)
        choices = [_infer(view, runtime) for view in self.views[:4]]
        self.assertEqual([choice["memory_reset"] for choice in choices],
                         [True, False, False, False])
        self.assertTrue(all(choice["candidate_id"] in {
            row["candidate_id"] for row in view["decisions"]["candidates"]
        } for choice, view in zip(choices, self.views[:4])))
        self.assertEqual(len(runtime[3]._states), 1)
        key = (self.views[0]["game_generation"], self.views[0]["episode_id"])
        self.assertIn((key, self.views[0]["observation"]["player_id"]), runtime[3]._states)

    def test_v15_checkpoint_refuses_v32_governor_catalogue(self):
        from civ6rl.learning.encoding import ACTOR_V15
        from civ6rl.learning.worker import _infer

        runtime = self.make_runtime(ACTOR_V15)
        with self.assertRaises(ValueError):
            _infer(self.views[0], runtime)


if __name__ == "__main__":
    unittest.main()
