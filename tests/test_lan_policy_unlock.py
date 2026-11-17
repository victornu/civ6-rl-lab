"""The frozen actor's explicit mapping for the distinct host unlock gesture."""
import unittest

from civ6rl.learning.encoding import ACTOR_V15, _action_vector, actor_feature_names


class PolicyUnlockFeaturesTests(unittest.TestCase):
    def test_unlock_keeps_its_policy_family_cost_and_fixed_tensor_width(self):
        action = dict(kind='policy_unlock', cost=37, currency='YIELD_GOLD')
        vector = _action_vector(action, dict(width=4, height=4), {}, actor_schema=ACTOR_V15)
        _, features = actor_feature_names(ACTOR_V15)
        self.assertEqual(len(vector), len(features))
        self.assertEqual(vector[features.index('kind:policies')], 1.0)
        self.assertEqual(sum(value for value, name in zip(vector, features) if name.startswith('kind:')), 1.0)
        self.assertEqual(vector[features.index('cost')], 37.0)
        self.assertEqual(vector[features.index('cost_known')], 1.0)
        self.assertEqual(action['kind'], 'policy_unlock')

    def test_unknown_hidden_fields_and_invented_currency_are_rejected(self):
        for action in (dict(kind='policy_unlock', cost=37, currency='YIELD_GOLD', balance=100),
                       dict(kind='policy_unlock', cost=37, currency='GOLD')):
            with self.assertRaises(ValueError):
                _action_vector(action, dict(width=4, height=4), {}, actor_schema=ACTOR_V15)
