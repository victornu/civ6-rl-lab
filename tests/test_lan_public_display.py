"""Display consumer regressions; BBG display labels below are fixtures.

The raw v30 view and stock display views are retained native evidence. Applying
fixture labels to the BBG view does not establish a BBG formatter execution.
"""
from copy import deepcopy
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import unittest

from civ6_host.contracts import player_display
from civ6_host.runtime.paths import resource_root
from civ6rl.learning import lan_inputs, lan_rollout
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import ACTOR_V15, GLOBAL_FEATURES, actor_feature_names, encode_response
from civ6rl.learning.lan_critic import encode_lan_privileged_snapshot


def display_fixture(raw, text=None):
    if text is None:
        # Fixture data only, not a replacement for the native formatter.
        text = {}
        for name in ('science', 'culture', 'faith'):
            value = raw['observation']['economy'][name]
            sign = '+' if name != 'faith' and value > 0 else ''
            text[name] = sign + str(math.trunc(value * 10) / 10).replace('.', ',')
    return dict(raw, schema=player_display.PUBLIC_VIEW_SCHEMA, locale='fr_FR',
        observation=player_display.project_observation(raw['observation'], text),
        display_source_sha256=player_display.SOURCE_SHA256,
        display_scope=deepcopy(player_display.SCOPE), top_panel_text=deepcopy(text))


class LanPublicDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = resource_root()

        def read(reference):
            path = cls.root / reference['path']
            data = path.read_bytes()
            if sha256(data).hexdigest() != reference['sha256']:
                raise ValueError('Retained display/input evidence changed')
            return data

        path = cls.root / 'evidence/host-bbg751-v30-native-query.json'
        if not path.is_file():
            raise unittest.SkipTest('Retained v30 native query required')
        if sha256(path.read_bytes()).hexdigest() != '9fb3f965b8a98a41e009c0c2ea08836900eaaa738e277c5bf03f671789e5af96':
            raise ValueError('Native v30 proof changed')
        proof = json.loads(path.read_bytes())
        cls.raw = json.loads(read(proof['consumer']['public_replay']).splitlines()[0])['view']
        capture = json.loads((cls.root / 'work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json').read_bytes())
        db = capture['database_copies']['gameplay']
        cls.vocabulary = load_public_content_vocabulary(db['path'], expected_sha256=db['sha256'],
            gameinfo_evidence_path=cls.root / 'evidence/phase41-gameinfo-maps.json',
            include_action_identities=True)
        cls.read_reference = staticmethod(read)

    def encode(self, view):
        return encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)

    def test_public_numbers_and_city_yields_reach_fixed_actor_layout(self):
        original = deepcopy(self.raw)
        public = display_fixture(self.raw, dict(science='+7,6', culture='+6,9', faith='0,9'))
        raw_frame, frame = self.encode(self.raw), self.encode(public)
        self.assertEqual(public['observation']['economy'], dict(science=7.6, culture=6.9, faith=0.9, gold=381))
        for name in ('science', 'culture', 'faith'):
            index = GLOBAL_FEATURES.index(name)
            self.assertEqual(raw_frame['global']['mask'][index], 0)
            self.assertEqual(frame['global']['mask'][index], 1)
            self.assertEqual(frame['global']['values'][index], public['observation']['economy'][name])
        entity_names, _ = actor_feature_names(ACTOR_V15)
        # Own city rows follow own units in the existing entity layout.
        for i, city in enumerate(public['observation']['cities'], start=len(public['observation']['units'])):
            for name, value in city['yields'].items():
                self.assertEqual(frame['entities'][i][entity_names.index(name + '_yield')], value)
                self.assertEqual(frame['entities'][i][entity_names.index(name + '_yield_known')], 1)
        for name in ('candidate_ids', 'action_mask', 'candidates', 'action_categorical', 'policy_catalog'):
            self.assertEqual(frame[name], raw_frame[name])
        self.assertEqual(len(frame['global']['values']), len(raw_frame['global']['values']))
        self.assertEqual(self.raw, original)

    def test_unknown_scope_precision_and_private_fields_are_refused(self):
        public = display_fixture(self.raw)
        mutations = (
            lambda v: v.update(display_source_sha256='0' * 64),
            lambda v: v['display_scope'].update(tooltip_precision_qualified=True),
            lambda v: v.update(locale='en_US'),
            lambda v: v['observation'].update(schema=player_display.RAW_OBSERVATION_SCHEMA),
            lambda v: v['observation']['economy'].update(gold=381.44921875),
            lambda v: v['observation']['economy'].update(science=7.65234375),
            lambda v: v['observation']['cities'][0]['yields'].update(production=8.1015625),
            lambda v: v.update(privileged_snapshot={'players': []}),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = deepcopy(public)
                mutate(bad)
                with self.assertRaises(ValueError):
                    self.encode(bad)

    def test_retained_stock_public_receipts_keep_their_existing_scope(self):
        proof = json.loads((self.root / 'evidence/host-xp2-player-display.json').read_bytes())
        for sample in proof['samples']:
            public = json.loads(self.read_reference(sample['view']))
            player_display.validate_public(public)
            self.assertEqual(public['observation']['economy'], sample['public_economy'])
            # Old catalogues are not silently relabelled as v24-v30.
            with self.assertRaises(ValueError):
                self.encode(public)

    def snapshot(self, view):
        obs = view['observation']; pid = obs['player_id']; turn = obs['turn']
        pins = {name: str(i) * 64 for i, name in enumerate(
            ('core_sha256', 'host_sha256', 'closure_sha256', 'collector_source_sha256'), 1)}
        event = dict(stage='privileged_critic_snapshot977', schema='civ6rl.native_privileged_critic_snapshot977.v1',
            **{name: view[name] for name in ('episode_id', 'game_generation', 'sequence')},
            player_id=pid, turn=turn, **pins, snapshot=dict(schema='civ6rl-native-privileged-payload977-v1',
                local_player_before=pid, observer_before=pid, turn_before=turn,
                local_player_after=pid, observer_after=pid, turn_after=turn,
                players=[dict(player_id=pid, gold=self.raw['observation']['economy']['gold']),
                         dict(player_id=1, gold=10000.25)],
                units=[{name: unit[name] for name in ('owner', 'id', 'type', 'x', 'y')} for unit in obs['units']]))
        return event, pins

    def test_critic_joins_public_floor_but_retains_private_precision(self):
        public = display_fixture(self.raw)
        snapshot, pins = self.snapshot(public)
        unchanged = deepcopy(snapshot)
        features = encode_lan_privileged_snapshot(snapshot, public, expected_context=pins)
        self.assertAlmostEqual(features['values'][1], math.log1p(self.raw['observation']['economy']['gold']))
        self.assertEqual(snapshot, unchanged)
        with self.assertRaisesRegex(ValueError, 'own gold differs'):
            bad = deepcopy(public)
            bad['observation']['economy']['gold'] -= 1
            encode_lan_privileged_snapshot(snapshot, bad, expected_context=pins)
        # Raw observations still require exact own-gold agreement.
        raw_bad = deepcopy(self.raw)
        raw_bad['observation']['economy']['gold'] = math.floor(raw_bad['observation']['economy']['gold'])
        with self.assertRaisesRegex(ValueError, 'own gold differs'):
            encode_lan_privileged_snapshot(snapshot, raw_bad, expected_context=pins)
        actor = self.encode(public)
        changed = deepcopy(snapshot)
        changed['snapshot']['players'][1]['gold'] = -999.75
        other = encode_lan_privileged_snapshot(changed, public, expected_context=pins)
        self.assertNotEqual(other, features)
        self.assertEqual(self.encode(public), actor)
        for name in ('episode_id', 'game_generation', 'sequence', 'turn', 'player_id'):
            bad = deepcopy(snapshot)
            bad[name] = bad[name] + 1 if type(bad[name]) is int else bad[name] + '-foreign'
            with self.assertRaises(ValueError):
                encode_lan_privileged_snapshot(bad, public, expected_context=pins)

    def test_public_projection_survives_existing_action_and_poll_interval(self):
        proof = json.loads((self.root / 'evidence/host-bbg751-native-policy-reconfirmation.json').read_bytes())
        coverage = json.loads(self.read_reference(proof['consumer']['interval_coverage']))
        rows = [json.loads(line) for line in self.read_reference(coverage['transcript']).splitlines()]
        records = []; before = rows[0]['view']
        for row in rows[1:]:
            if row['operation'] == 'close':
                break
            if row['operation'] == 'act':
                selection = row['receipt']['selection']
                choice = dict(candidate_id=selection['candidate_id'])
                if 'parameters' in selection:
                    choice['assignments'] = selection['parameters']['assignments']
                records.append(dict(before=display_fixture(before), choice=choice,
                    command_receipt=row['receipt'], successors=[display_fixture(row['view'])]))
            else:
                self.assertEqual(row['operation'], 'poll')
                records[-1]['successors'].append(display_fixture(row['view']))
            before = row['view']
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertEqual(lan_rollout.transition(record), record['successors'][-1])
            self.encode(record['before'])
            self.assertTrue(lan_inputs.policy_reconfirmation_allowed(record['before']))

    def test_real_checkpoints_score_fixture_with_existing_cnn_lstm_and_masks(self):
        import torch
        from civ6rl.learning.model import collate
        from civ6rl.learning.task_context import encode_task_response
        from civ6rl.learning.training_state import exact_source_transition, load_training_bytes

        proof = json.loads((self.root / 'evidence/host-bbg751-v30-native-query.json').read_bytes())
        public = display_fixture(self.raw)
        frame = encode_task_response(public, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        for entry in proof['consumer']['source_only_checkpoint_resumes']:
            reference = entry['checkpoint']
            data = self.read_reference(reference)
            payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
            profile = exact_source_transition(payload['implementation_sha256'])
            self.assertEqual(profile, 'lan_player_only_envoys_v31_from_v24'
                if entry['migration'].startswith('lan_player_only')
                else 'lan_asymmetric_envoys_v31_from_v24')
            runtime = load_training_bytes(data, sha256=reference['sha256'],
                expected_vocabulary_fingerprint=self.vocabulary.fingerprint,
                implementation_migration=profile)
            with torch.no_grad():
                distribution, value, state = runtime[2](collate([frame]))
            for tensor in (distribution.probs, value, *state):
                self.assertTrue(torch.isfinite(tensor).all())
            self.assertAlmostEqual(distribution.probs.sum().item(), 1.0, places=5)
            self.assertEqual(distribution.probs.shape[-1], len(frame['candidate_ids']))

    def test_skill_task_collection_and_bootstrap_consume_public_projection(self):
        from tests import test_lan_policy_learner as fixtures
        from tests.test_lan_episode_budget import RetainedClient, ScriptedWorker
        from tests.test_lan_budget_transport import budget
        from civ6rl.learning.lan_collection import collect_episode
        from civ6rl.learning.skill_task991 import project_public_observation
        from civ6rl.learning.task_context import encode_task_response

        fixtures.LanJointPolicyLearnerTests.setUpClass()
        helper = fixtures.LanJointPolicyLearnerTests()
        records = deepcopy(helper.native_records)
        for record in records:
            record['before'] = display_fixture(record['before'])
            record['successors'] = [display_fixture(v) for v in record['successors']]
        _, records, scenario, _ = helper.fixture(native_records=records)
        rows = [dict(operation='reset', view=records[0]['before'])]
        for record in records:
            rows.append(dict(operation='act', view=record['successors'][0], receipt=record['command_receipt']))
            rows.extend(dict(operation='poll', view=view) for view in record['successors'][1:])
        contract = budget(len(records))
        summary = collect_episode(RetainedClient(rows), ScriptedWorker(records, contract), scenario,
            max_decisions=len(records), diagnostic_collection=contract)
        self.assertEqual(summary['records'], records)
        self.assertTrue(summary['task_result']['bootstrap'])
        frame = encode_task_response(records[0]['before'], vocabulary=self.vocabulary,
            actor_schema=ACTOR_V15, scenario=scenario, completed_decisions=0)
        self.assertEqual(frame['global']['mask'][GLOBAL_FEATURES.index('science')], 1)
        obs = records[0]['before']['observation']
        self.assertEqual(project_public_observation(obs)['economy']['gold'], obs['economy']['gold'])


if __name__ == '__main__':
    unittest.main()
