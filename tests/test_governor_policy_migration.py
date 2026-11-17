"""Explicit V15→V16 governor-policy migration contracts."""
import hashlib
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is an optional learning dependency')
class GovernorPolicyMigrationTests(unittest.TestCase):
    def _v15(self, vocabulary):
        from civ6rl.learning.encoding import (
            ACTOR_V15, MAP_CHANNELS, GLOBAL_FEATURES, actor_feature_names,
            GLOBAL_CATEGORICAL_FIELDS,
        )
        from civ6rl.learning.model import PolicyConfig, RecurrentPolicy, FAMILY_PRIOR
        from civ6rl.learning.task_context import TASK_FEATURES
        entities, actions = actor_feature_names(ACTOR_V15)
        return RecurrentPolicy(PolicyConfig(
            map_channels=len(MAP_CHANNELS), global_features=len(GLOBAL_FEATURES),
            entity_features=len(entities), action_features=len(actions),
            vocabulary_size=vocabulary.vocabulary_size, hidden_size=8,
            spatial_size=4, category_size=3,
            global_categories=len(GLOBAL_CATEGORICAL_FIELDS),
            task_features=len(TASK_FEATURES), actor_schema=ACTOR_V15,
            candidate_family_prior=FAMILY_PRIOR))

    def _vocabularies(self):
        from civ6rl.learning.encoding import ContentVocabulary
        original = ContentVocabulary(['action.kind:end_turn', 'entity.type:UNIT_SCOUT',
                                     'z:retained_token'])
        child = ContentVocabulary(sorted(['a:new_governor_token', *original.tokens,
                                          'm:city_byte_token']))
        return original, child

    def test_named_columns_shared_category_block_and_token_rows_migrate(self):
        from civ6rl.learning.encoding import ACTOR_V15, ACTOR_V16, actor_feature_names
        from civ6rl.learning.migrate_actor import migrate_v15_model
        original, vocabulary = self._vocabularies()
        parent = self._v15(original)
        with torch.no_grad():
            for parameter in parent.parameters():
                parameter.copy_(torch.arange(parameter.numel(), dtype=parameter.dtype)
                                .reshape(parameter.shape) / max(1, parameter.numel()))
        child, report = migrate_v15_model(parent, original, vocabulary)
        old_entities, old_actions = actor_feature_names(ACTOR_V15)
        new_entities, new_actions = actor_feature_names(ACTOR_V16)
        self.assertEqual(report['parent_schema'], ACTOR_V15)
        self.assertEqual(report['child_schema'], ACTOR_V16)
        self.assertFalse(report['training_eligible'])
        self.assertFalse(report['retained_rollouts_reusable_for_update'])
        self.assertFalse(report['optimizer_state_migrated'])
        self.assertFalse(report['recurrent_state_migrated'])
        self.assertEqual(set(report['parameter_mappings']), set(parent.state_dict()))

        for name, previous, current in (('entity.0.weight', old_entities, new_entities),
                                         ('candidate.0.weight', old_actions, new_actions)):
            mapping = report['parameter_mappings'][name]
            self.assertEqual(mapping['axis'], 1)
            numeric_indices = mapping['indices'][:len(previous)]
            self.assertEqual(numeric_indices, [current.index(feature) for feature in previous])
            self.assertEqual(mapping['indices'][len(previous):],
                             list(range(len(current), len(current) + parent.config.category_size)))
            old_weight, new_weight = parent.state_dict()[name], child.state_dict()[name]
            for old_index, new_index in enumerate(mapping['indices']):
                torch.testing.assert_close(new_weight[:, new_index], old_weight[:, old_index])
            torch.testing.assert_close(new_weight[:, len(current):],
                                       old_weight[:, len(previous):])
            unmapped = set(range(new_weight.shape[1])) - set(mapping['indices'])
            if unmapped:
                self.assertEqual(float(new_weight[:, sorted(unmapped)].abs().sum()), 0.0)

        embedding_map = report['parameter_mappings']['categories.weight']
        self.assertEqual(embedding_map['axis'], 0)
        self.assertEqual(embedding_map['indices'][0], 0)
        self.assertEqual(embedding_map['indices'][1:],
                         [vocabulary.tokens.index(token) + 1 for token in original.tokens])
        old_embedding, new_embedding = (parent.state_dict()['categories.weight'],
                                        child.state_dict()['categories.weight'])
        for old_index, new_index in enumerate(embedding_map['indices']):
            torch.testing.assert_close(new_embedding[new_index], old_embedding[old_index])
        unmapped_rows = set(range(new_embedding.shape[0])) - set(embedding_map['indices'])
        if unmapped_rows:
            self.assertEqual(float(new_embedding[sorted(unmapped_rows)].abs().sum()), 0.0)
        for name, mapping in report['parameter_mappings'].items():
            if mapping is None:
                torch.testing.assert_close(child.state_dict()[name], parent.state_dict()[name])

    def test_optimizer_tensor_mapping_zero_fills_new_axis_and_requires_identity_shape(self):
        from civ6rl.learning.migrate_actor import migrate_v15_parameter_tensor
        source = torch.tensor([[1., 2.], [3., 4.]])
        result = migrate_v15_parameter_tensor(source, (2, 4),
            {'axis': 1, 'indices': [0, 2]})
        torch.testing.assert_close(result, torch.tensor([[1., 0., 2., 0.],
                                                         [3., 0., 4., 0.]]))
        copied = migrate_v15_parameter_tensor(source, source.shape, None)
        torch.testing.assert_close(copied, source)
        self.assertIsNot(copied, source)
        with self.assertRaises(ValueError):
            migrate_v15_parameter_tensor(source, (2, 3), None)
        with self.assertRaises(ValueError):
            migrate_v15_parameter_tensor(source, (2, 4), {'axis': 1, 'indices': [1, 1]})

    def test_explicit_v15_bytes_migrate_and_v16_artifact_round_trips(self):
        from civ6rl.learning.artifacts import policy_bytes, load_policy_bytes
        from civ6rl.learning.encoding import ACTOR_V16
        from civ6rl.learning.migrate_actor import migrate_policy_bytes
        original, vocabulary = self._vocabularies()
        parent = self._v15(original)
        payload = policy_bytes(parent, original)
        restored_parent, restored_vocabulary = load_policy_bytes(payload,
            sha256=hashlib.sha256(payload).hexdigest())
        self.assertEqual(restored_vocabulary.fingerprint, original.fingerprint)
        for name, value in parent.state_dict().items():
            torch.testing.assert_close(restored_parent.state_dict()[name], value)
        migrated, report = migrate_policy_bytes(payload,
            sha256=hashlib.sha256(payload).hexdigest(), vocabulary=vocabulary,
            target_schema=ACTOR_V16)
        child, loaded_vocabulary = load_policy_bytes(migrated,
            sha256=hashlib.sha256(migrated).hexdigest())
        self.assertEqual(report['parent_schema'], 'civ6rl.recurrent_policy.v15')
        self.assertEqual(report['child_schema'], ACTOR_V16)
        self.assertEqual(loaded_vocabulary.fingerprint, vocabulary.fingerprint)
        self.assertEqual(child.config.actor_schema, ACTOR_V16)
        self.assertFalse(report['training_eligible'])

    def test_v16_family_prior_selects_kind_columns_by_name_even_when_scattered(self):
        from civ6rl.learning.encoding import ACTOR_V16
        from civ6rl.learning.model import _family_prior_features
        names = ('kind:appoint_governor', 'numeric:price', 'kind:promote_governor',
                 'typed:identity', 'kind:assign_governor', 'kind:consider_governor_titles')
        candidates = torch.tensor([[[1., 5., 0., 4., 0., 0.],
                                    [0., 3., 0., 2., 1., 0.],
                                    [0., 9., 0., 1., 0., 1.]]])
        selected = _family_prior_features(candidates, names, ACTOR_V16)
        torch.testing.assert_close(selected, torch.tensor([[[1., 0., 0., 0.],
                                                             [0., 0., 1., 0.],
                                                             [0., 0., 0., 1.]]]))


if __name__ == '__main__':
    unittest.main()
