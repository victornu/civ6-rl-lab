"""V32 public governor features, backed by the retained teacher-forced trace."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

from civ6_host.runtime.paths import resource_root
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import (
    ACTOR_V15, ACTOR_V16, actor_feature_names, encode_response,
)
from civ6rl.learning.governors import (
    ACTION_FEATURES as GOVERNOR_ACTION_FEATURES,
    ACTION_CATEGORICAL_FIELDS as GOVERNOR_ACTION_CATEGORICAL_FIELDS,
    action_features as governor_action_features,
    extend_governor_vocabulary,
)


class GovernorEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = resource_root()
        replay = root / 'work/bbg751-governors/native/public-replay.jsonl'
        capture = root / 'work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json'
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest('Retained v32 governor trace and pinned gameplay database required')
        with replay.open(encoding='utf-8') as stream:
            records = [json.loads(line) for line in stream]
        cls.views = [record['view'] for record in records[:5]]
        if len(cls.views) < 5 or any(
                view.get('decisions', {}).get('schema') !=
                'civ6-host.network-capability-decisions.v32' for view in cls.views):
            raise ValueError('Retained actor replay does not contain five public v32 frames')
        capture_data = json.loads(capture.read_bytes())
        database = capture_data['database_copies']['gameplay']
        database_path = Path(database['path'])
        if sha256(database_path.read_bytes()).hexdigest() != database['sha256']:
            raise RuntimeError('Pinned public gameplay database changed')
        parent = load_public_content_vocabulary(
            database_path, expected_sha256=database['sha256'],
            gameinfo_evidence_path=root / 'evidence/phase41-gameinfo-maps.json',
            include_action_identities=True)
        cls.parent_vocabulary = parent
        cls.vocabulary = extend_governor_vocabulary(parent, cls.views)

    def test_v32_explicitly_encodes_all_candidates_in_v16(self):
        from civ6rl.learning.governors import CONTRACT

        found_kinds = set()
        for view in self.views:
            frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V16)
            self.assertEqual(frame['candidate_ids'], [
                row['candidate_id'] for row in view['decisions']['candidates']])
            self.assertEqual(len(frame['candidate_ids']), len(frame['candidates']))
            self.assertEqual(len(frame['candidate_ids']), len(frame['action_mask']))
            self.assertTrue(all(frame['action_mask']))
            self.assertEqual(frame['governors_contract'], CONTRACT)
            entity_names, action_names = actor_feature_names(ACTOR_V16)
            self.assertEqual(frame['entity_features'], list(entity_names))
            self.assertEqual(frame['action_features'], list(action_names))
            self.assertEqual(len(frame['entities'][0]), len(entity_names))
            self.assertTrue(all(len(row) == len(action_names) for row in frame['candidates']))
            for candidate in view['decisions']['candidates']:
                kind = candidate['action']['kind']
                if kind not in ('appoint_governor', 'promote_governor',
                                'assign_governor', 'consider_governor_titles'):
                    continue
                found_kinds.add(kind)
                row = frame['candidate_ids'].index(candidate['candidate_id'])
                self.assertEqual(frame['candidates'][row][action_names.index('kind:' + kind)], 1.0)
            self.assertEqual(len(frame['candidate_source_rows']), len(frame['candidate_ids']))
        self.assertEqual(found_kinds, {
            'appoint_governor', 'promote_governor', 'assign_governor',
            'consider_governor_titles'})

    def test_typed_governor_and_promotion_identities_are_distinct(self):
        view = next(view for view in self.views if any(
            row['action']['kind'] == 'promote_governor'
            for row in view['decisions']['candidates']))
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V16)
        action_names = frame['action_categorical_fields']
        self.assertEqual(action_names[-len(GOVERNOR_ACTION_CATEGORICAL_FIELDS):],
                         list(GOVERNOR_ACTION_CATEGORICAL_FIELDS))
        candidate = next(row for row in view['decisions']['candidates']
                         if row['action']['kind'] == 'promote_governor')
        action = candidate['action']
        row = frame['candidate_ids'].index(candidate['candidate_id'])
        categories = frame['action_categorical'][row]
        for field, namespace in (('governor_index', 'governor.index'),
                                 ('governor_hash', 'governor.hash'),
                                 ('promotion_index', 'promotion.index'),
                                 ('promotion_hash', 'promotion.hash')):
            self.assertEqual(categories[action_names.index(field)],
                             self.vocabulary.lookup(namespace, action[field]))
        self.assertNotEqual(self.vocabulary.lookup('governor.index', action['governor_index']),
                            self.vocabulary.lookup('promotion.index', action['promotion_index']))
        self.assertLessEqual(len(self.parent_vocabulary.tokens), len(self.vocabulary.tokens))
        self.assertTrue(set(self.parent_vocabulary.tokens) <= set(self.vocabulary.tokens))

    def test_assignment_city_features_require_exact_own_owner_and_id_join(self):
        from civ6rl.learning.governors import city_id_bytes

        view = self.views[0]
        catalogue = deepcopy(view['decisions']['governors'])
        own = view['observation']['cities'][0]
        governor = catalogue['governors'][0]
        own_target = {'owner': view['observation']['player_id'], 'city_id': own['id'],
                      'can_assign': True}
        governor['assignment_targets'].append(own_target)
        own_action = {'kind': 'assign_governor', 'governor_index': governor['index'],
                      'governor_hash': governor['hash'], 'city_owner': own_target['owner'],
                      'city_id': own_target['city_id']}
        own_features = governor_action_features(
            own_action, catalogue, observation=view['observation'], width=view['observation']['width'],
            height=view['observation']['height'])
        names = list(GOVERNOR_ACTION_FEATURES)
        self.assertEqual(own_features[names.index('governor_target_city_is_own')], 1.0)
        self.assertEqual(own_features[names.index('governor_target_city_x_known')], 1.0)
        self.assertEqual(own_features[names.index('governor_target_city_y_known')], 1.0)
        self.assertEqual(own_features[names.index('governor_target_city_population')], own['population'])

        foreign_owner = (own['owner'] + 1) % 64
        foreign_target = {'owner': foreign_owner, 'city_id': own['id'], 'can_assign': True}
        governor['assignment_targets'].append(foreign_target)
        foreign_action = dict(own_action, city_owner=foreign_owner)
        foreign_features = governor_action_features(
            foreign_action, catalogue, observation=view['observation'], width=view['observation']['width'],
            height=view['observation']['height'])
        self.assertEqual(foreign_features[names.index('governor_target_city_is_own')], 0.0)
        for field, value_field in (('x', 'x_norm'), ('y', 'y_norm'),
                                   ('population', 'population'), ('food_yield', 'food_yield'),
                                   ('production_yield', 'production_yield'), ('gold_yield', 'gold_yield'),
                                   ('science_yield', 'science_yield'), ('culture_yield', 'culture_yield'),
                                   ('faith_yield', 'faith_yield')):
            self.assertEqual(foreign_features[names.index('governor_target_city_' + field + '_known')], 0.0)
            self.assertEqual(foreign_features[names.index('governor_target_city_' + value_field)], 0.0)
        self.assertEqual(city_id_bytes(0x12345678), (0x78, 0x56, 0x34, 0x12))

    def test_v15_keeps_its_exact_input_contract_and_refuses_v32(self):
        from civ6rl.learning.lan_inputs import validate

        entities15, actions15 = actor_feature_names(ACTOR_V15)
        entities16, actions16 = actor_feature_names(ACTOR_V16)
        self.assertNotEqual(entities15, entities16)
        self.assertNotEqual(actions15, actions16)
        with self.assertRaises(ValueError):
            encode_response(self.views[0], vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertFalse(validate(self.views[0]))

    def test_v16_projects_retained_v31_menu_with_unknown_governor_channels(self):
        from civ6rl.learning.governors import (
            ACTION_CATEGORICAL_FIELDS as GOVERNOR_ACTION_CATEGORICAL_FIELDS,
            ACTION_FEATURES as GOVERNOR_ACTION_FEATURES,
            ENTITY_FEATURES as GOVERNOR_ENTITY_FEATURES,
        )
        from civ6rl.learning.lan_inputs import validate

        root = resource_root()
        replay_path = root / 'work/bbg751-turn-source-scope/swap-units/native/public-replay.jsonl'
        raw = replay_path.read_bytes()
        self.assertEqual(sha256(raw).hexdigest(),
                         'fc093b9c6d237bdcf0d82e39b228498863b1befc9c84f3046228ac8e3e64f27f')
        rows = [json.loads(line) for line in raw.splitlines()]
        view = rows[0]['view']
        self.assertEqual(view['decisions']['schema'], 'civ6-host.network-capability-decisions.v31')
        self.assertNotIn('governors', view['decisions'])
        self.assertFalse(validate(view))

        default_frame = encode_response(view, vocabulary=self.vocabulary)
        explicit_v15 = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertEqual(default_frame, explicit_v15)
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V16)

        entity_names, action_names = actor_feature_names(ACTOR_V16)
        self.assertEqual(frame['actor_schema'], ACTOR_V16)
        self.assertEqual(frame['candidate_ids'], [row['candidate_id']
                                                   for row in view['decisions']['candidates']])
        self.assertEqual(len(frame['candidates']), len(view['decisions']['candidates']))
        self.assertEqual(len(frame['action_mask']), len(frame['candidate_ids']))
        self.assertTrue(all(frame['action_mask']))
        self.assertTrue(all(len(row) == len(action_names) for row in frame['candidates']))
        self.assertTrue(all(len(row) == len(entity_names) for row in frame['entities']))

        entity_start = len(entity_names) - len(GOVERNOR_ENTITY_FEATURES)
        self.assertTrue(all(all(value == 0.0 for value in row[entity_start:])
                            for row in frame['entities']))
        governor_entity_known = [entity_names.index(name) for name in GOVERNOR_ENTITY_FEATURES
                                 if name.endswith('_known')]
        self.assertTrue(all(row[index] == 0.0 for row in frame['entities']
                            for index in governor_entity_known))

        governor_action_indices = [action_names.index(name) for name in GOVERNOR_ACTION_FEATURES]
        governor_action_indices.extend(action_names.index('kind:' + kind) for kind in (
            'appoint_governor', 'promote_governor', 'assign_governor',
            'consider_governor_titles'))
        self.assertTrue(all(row[index] == 0.0 for row in frame['candidates']
                            for index in governor_action_indices))
        action_categories = frame['action_categorical_fields']
        categorical_indices = [action_categories.index(field)
                               for field in GOVERNOR_ACTION_CATEGORICAL_FIELDS
                               if field in ('governor_index', 'governor_hash',
                                            'promotion_index', 'promotion_hash')]
        self.assertTrue(all(row[index] == 0 for row in frame['action_categorical']
                            for index in categorical_indices))

    def test_v32_actor_input_rejects_privileged_or_admin_payloads(self):
        from civ6rl.learning.lan_inputs import validate

        for destination, field in (('view', 'privileged_snapshot'),
                                   ('decisions', 'admin_state')):
            with self.subTest(destination=destination):
                view = deepcopy(self.views[0])
                target = view if destination == 'view' else view['decisions']
                target[field] = {'hidden_enemy_count': 7, 'private_score': 9001}
                with self.assertRaises(ValueError):
                    validate(view)
                with self.assertRaises(ValueError):
                    encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V16)


if __name__ == '__main__':
    unittest.main()
