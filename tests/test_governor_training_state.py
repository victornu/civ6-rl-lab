"""Input-width migration preserves learned Adam state without a native PPO update."""
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import unittest

import torch

from civ6rl.learning.encoding import (
    ACTOR_V15, ACTOR_V16, ContentVocabulary, MAP_CHANNELS, GLOBAL_FEATURES,
    actor_feature_names,
)
from civ6rl.learning.model import PolicyConfig, RecurrentPolicy
from civ6rl.learning.training_state import _migrate_v15_actor_and_optimizer


class GovernorTrainingStateTests(unittest.TestCase):
    def fixture(self):
        original = ContentVocabulary(['a:old', 'z:old'])
        extended = ContentVocabulary(['a:old', 'g:new', 'z:old'])
        entities, actions = actor_feature_names(ACTOR_V15)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(96)
            model = RecurrentPolicy(PolicyConfig(
                map_channels=len(MAP_CHANNELS), global_features=len(GLOBAL_FEATURES),
                entity_features=len(entities), action_features=len(actions),
                vocabulary_size=original.vocabulary_size, hidden_size=16,
                spatial_size=4, category_size=4, actor_schema=ACTOR_V15))
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        state = optimizer.state_dict()
        for index, (_, parameter) in enumerate(model.named_parameters()):
            state['state'][index] = dict(step=torch.tensor(3.),
                exp_avg=torch.full_like(parameter, .125),
                exp_avg_sq=torch.full_like(parameter, .5))
        return original, extended, model, state

    def test_adam_numeric_columns_shared_tail_and_token_rows_are_preserved(self):
        original, extended, parent, source = self.fixture()
        before = deepcopy(source)
        prior_rng = torch.get_rng_state().clone()
        child, state, receipt = _migrate_v15_actor_and_optimizer(
            parent, original, extended, source, source_learner_sha256='a'*64)
        self.assertTrue(torch.equal(prior_rng, torch.get_rng_state()))
        self.assertEqual(receipt['target_actor_schema'], ACTOR_V16)
        self.assertFalse(receipt['retained_trajectories_reusable'])
        names = [name for name, _ in parent.named_parameters()]
        for name, old_features, new_features in (
                ('entity.0.weight', actor_feature_names(ACTOR_V15)[0], actor_feature_names(ACTOR_V16)[0]),
                ('candidate.0.weight', actor_feature_names(ACTOR_V15)[1], actor_feature_names(ACTOR_V16)[1])):
            index = names.index(name)
            for moment in ('exp_avg', 'exp_avg_sq'):
                old, new = source['state'][index][moment], state['state'][index][moment]
                self.assertTrue(torch.equal(old[:, len(old_features):], new[:, len(new_features):]))
                for column, feature in enumerate(old_features):
                    self.assertTrue(torch.equal(old[:, column], new[:, new_features.index(feature)]))
                for column, feature in enumerate(new_features):
                    if feature not in old_features:
                        self.assertEqual(torch.count_nonzero(new[:, column]).item(), 0)
        index = names.index('categories.weight')
        for moment in ('exp_avg', 'exp_avg_sq'):
            old, new = source['state'][index][moment], state['state'][index][moment]
            self.assertTrue(torch.equal(old[0], new[0]))
            for row, token in enumerate(original.tokens, 1):
                self.assertTrue(torch.equal(old[row], new[extended.tokens.index(token)+1]))
            self.assertEqual(torch.count_nonzero(new[extended.tokens.index('g:new')+1]).item(), 0)
        for index, values in before['state'].items():
            for key, value in values.items():
                self.assertTrue(torch.equal(value, source['state'][index][key]))
        self.assertEqual(before['param_groups'], source['param_groups'])
        for index, values in source['state'].items():
            self.assertTrue(torch.equal(values['step'], state['state'][index]['step']))
        self.assertEqual(parent.config.hidden_size, child.config.hidden_size)

    def test_bad_parent_moment_shape_refuses_without_mutation(self):
        original, extended, parent, state = self.fixture()
        state['state'][0]['exp_avg'] = torch.zeros(1)
        before = torch.get_rng_state().clone()
        with self.assertRaises(ValueError):
            _migrate_v15_actor_and_optimizer(parent, original, extended, state,
                                             source_learner_sha256='a'*64)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_migrated_v16_adam_runs_task_conditioned_recurrent_ppo_fixture(self):
        """Synthetic transitions check optimizer integration; they are not native PPO data.

        The starting policy and optimizer are the retained, migrated governor
        checkpoint. Decisions are sampled by that V16 policy from the retained
        public V32 menus. Successors are deliberately synthesized to close a
        bounded learner fixture; no scripted native receipt is relabelled as
        on-policy experience.
        """
        from civ6_host.runtime.paths import resource_root
        from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
        from civ6rl.learning.encoding import ACTOR_V15, ACTOR_V16, actor_feature_names
        from civ6rl.learning.governors import extend_governor_vocabulary
        from civ6rl.learning.lan_rollout import _ready
        from civ6rl.learning.model import PlayerMemory, actor_schema_for_config
        from civ6rl.learning.artifacts import load_policy_bytes, policy_version
        from civ6rl.learning.skill_context993 import build_skill_scenario
        from civ6rl.learning.skill_task991 import DEVELOPMENT, build_task
        from civ6rl.learning.task_context import encode_task_response
        from civ6rl.learning.training_state import _migrate_v15_actor_and_optimizer
        from civ6rl.learning.worker import _infer
        from civ6rl.learning.learner import update_native_diagnostic
        from civ6rl.learning.action_receipt import selected_action_and_assignments

        saved_rng = torch.get_rng_state().clone()
        self.addCleanup(torch.set_rng_state, saved_rng)
        torch.manual_seed(3216)

        root = resource_root()
        evidence_path = root / "evidence/agent-bbg751-governor-inputs.json"
        evidence_raw = evidence_path.read_bytes()
        self.assertEqual(sha256(evidence_raw).hexdigest(),
                         "ecaae3c5c745d47a5c606b3cdd1218cb9424d2b5a30ef9ec14b079ff8103d7d5")
        evidence = json.loads(evidence_raw)
        replay_ref = evidence["validation"]["public_replay"]
        replay_raw = (root / replay_ref["path"]).read_bytes()
        self.assertEqual(sha256(replay_raw).hexdigest(), replay_ref["sha256"])
        replay_rows = [json.loads(line) for line in replay_raw.splitlines()]
        views = [row["view"] for row in replay_rows[:5]]
        self.assertEqual(len(views), 5)
        self.assertTrue(all(view["decisions"]["schema"] ==
                            "civ6-host.network-capability-decisions.v32" for view in views))

        capture = json.loads((root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json").read_bytes())
        database = capture["database_copies"]["gameplay"]
        vocabulary = load_public_content_vocabulary(
            database["path"], expected_sha256=database["sha256"],
            gameinfo_evidence_path=root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)
        vocabulary = extend_governor_vocabulary(vocabulary, [views[0]])

        checkpoint_ref = next(
            row for row in evidence["checkpoint_adaptation"]["artifacts"]
            if row["path"].endswith("904c34797991e6ad74718027bcfc33f045cb59fc093e7c69bf2211ee04f8e293.learner.pt"))
        checkpoint_raw = (root / checkpoint_ref["path"]).read_bytes()
        self.assertEqual(sha256(checkpoint_raw).hexdigest(), checkpoint_ref["sha256"])
        parent = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
        parent_model, parent_vocabulary = load_policy_bytes(
            parent["policy"], sha256=parent["policy_sha256"], device="cpu")
        model, optimizer_state, migration = _migrate_v15_actor_and_optimizer(
            parent_model, parent_vocabulary, vocabulary, parent["optimizer"],
            source_learner_sha256=checkpoint_ref["sha256"])
        self.assertEqual(migration["id"], "public_governors_v16")
        self.assertEqual(migration["source_actor_schema"], ACTOR_V15)
        self.assertEqual(migration["target_actor_schema"], ACTOR_V16)
        self.assertEqual(migration["source_learner_sha256"], checkpoint_ref["sha256"])
        self.assertTrue(migration["preserved"]["actor_optimizer_by_parameter_name"])
        self.assertFalse(migration["retained_trajectories_reusable"])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        optimizer.load_state_dict(optimizer_state)
        model.eval()
        model._policy_version = policy_version(model, vocabulary)
        self.assertEqual(model._policy_version, migration["target_policy_sha256"])
        model._diagnostic_updates = parent["updates"]
        model._sample_audit = {}
        model._closed_episodes = set(map(tuple, parent["closed_episodes"]))
        model._critic_mode = parent["critic_mode"]
        model._privileged_state = None
        runtime = (torch, vocabulary, model, PlayerMemory(), "cpu")
        loaded_vocabulary = vocabulary
        self.assertEqual(actor_schema_for_config(model.config), ACTOR_V16)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertGreater(model._diagnostic_updates, 0)

        # These are actual parent-checkpoint Adam moments after explicit V15→V16 mapping.
        shared_parameter = dict(model.named_parameters())["entity.0.weight"]
        parent_parameter_id = parent["optimizer_parameter_names"].index("entity.0.weight")
        old_entity_features, _ = actor_feature_names(ACTOR_V15)
        new_entity_features, _ = actor_feature_names(ACTOR_V16)
        shared_feature = "kind_unit"
        torch.testing.assert_close(
            optimizer.state[shared_parameter]["exp_avg"][:, new_entity_features.index(shared_feature)],
            parent["optimizer"]["state"][parent_parameter_id]["exp_avg"][:, old_entity_features.index(shared_feature)],
            rtol=0, atol=0)
        moment_before = {name: value.detach().clone() for name, value in optimizer.state[shared_parameter].items()
                         if isinstance(value, torch.Tensor)}
        self.assertGreater(float(moment_before["exp_avg"].abs().sum()), 0.0)
        step_before = float(moment_before["step"])

        decision_views = views[2:5]
        start = decision_views[0]
        unit_types = json.loads((root / "work/bbg751-defense/scenario.json").read_bytes())["skill_task"]["unit_types"]
        unit_catalogue = {row["index"]: row["type"] for row in unit_types}
        builder_count = sum(unit_catalogue.get(unit["type"]) == "UNIT_BUILDER"
                            for unit in start["observation"]["units"])
        task = build_task(task_id="synthetic-v16-governor-update", player_id=start["observation"]["player_id"],
            start_observation=start["observation"],
            objective=dict(kind=DEVELOPMENT, unit_type="UNIT_BUILDER",
                           target_count=builder_count + 500, gold_reserve=0),
            deadline_turn=start["observation"]["turn"] + 3, unit_types=unit_types)
        scenario = build_skill_scenario(scenario_id=task["task_id"], skill_task=task,
            max_decisions=2, max_turns=3,
            identity={key: None for key in json.loads((root / "work/bbg751-defense/scenario.json").read_bytes())["identity"]},
            catalogue_source=json.loads((root / "work/bbg751-defense/scenario.json").read_bytes())["catalogue_source"])

        records = []
        sample_audit = {}
        previous = None
        for index in range(2):
            before = deepcopy(decision_views[index]) if previous is None else previous
            self.assertTrue(_ready(before))
            frame = encode_task_response(before, vocabulary=loaded_vocabulary, scenario=scenario,
                                         completed_decisions=index, actor_schema=ACTOR_V16)
            self.assertEqual(frame["actor_schema"], ACTOR_V16)
            entity_names, _ = actor_feature_names(ACTOR_V16)
            governor_column = entity_names.index("kind_governor")
            self.assertTrue(any(row[governor_column] == 1.0 for row in frame["entities"]))
            choice = _infer(before, runtime, scenario=scenario, completed_decisions=index)
            action, assignments = selected_action_and_assignments(before, choice)
            selection = dict(decision_id=before["decisions"]["decision_id"],
                             candidate_id=choice["candidate_id"])
            if assignments is not None:
                selection["parameters"] = dict(assignments=assignments)
            after = deepcopy(decision_views[index + 1])
            after["applied_action"] = action
            receipt = dict(schema="civ6-host.command-receipt.v1", episode_id=before["episode_id"],
                game_generation=before["game_generation"], player_id=before["observation"]["player_id"],
                sequence=before["sequence"] + 1, decision_turn=before["observation"]["turn"],
                response_turn=after["observation"]["turn"], selection=selection,
                command_result=dict(status="completed", reason=None, requested_action=action))
            record = dict(before=before, choice=choice, command_receipt=receipt, successors=[after])
            records.append(record)
            key = (before["episode_id"], before["game_generation"],
                   before["observation"]["player_id"], before["sequence"])
            sample_audit[key] = dict(before=before, choice=choice)
            previous = after

        before_params = [value.detach().clone() for value in model.parameters()]
        report = update_native_diagnostic(model, optimizer, loaded_vocabulary, records, scenario,
            sample_audit=sample_audit, expected_policy_version=model._policy_version,
            epochs=1)
        self.assertEqual(report["optimized_decisions"], 2)
        self.assertFalse(report["training_eligible"])
        self.assertTrue(torch.isfinite(torch.tensor(report["metrics"]["loss"])))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))
        parameters = dict(model.named_parameters())
        candidate_gradient = parameters["candidate.0.weight"].grad
        self.assertIsNotNone(candidate_gradient)
        self.assertTrue(torch.isfinite(candidate_gradient).all())
        self.assertGreater(float(candidate_gradient.abs().sum()), 0.0)
        _, action_features = actor_feature_names(ACTOR_V16)
        assign_column = action_features.index("kind:assign_governor")
        self.assertTrue(any(any(candidate["action"]["kind"] == "assign_governor"
                                for candidate in view["decisions"]["candidates"])
                            for view in decision_views[:2]))
        governor_gradient = candidate_gradient[:, assign_column]
        self.assertTrue(torch.isfinite(governor_gradient).all())
        self.assertGreater(float(governor_gradient.abs().sum()), 0.0)
        self.assertTrue(any(not torch.equal(old, new) for old, new in
                            zip(before_params, model.parameters())))
        self.assertGreater(float(optimizer.state[shared_parameter]["step"]), step_before)
        self.assertTrue(torch.isfinite(optimizer.state[shared_parameter]["exp_avg"]).all())


if __name__ == '__main__':
    unittest.main()
