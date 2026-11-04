"""Synthetic transition tests; no claim that these fixtures simulate Civ6."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None


def private_snapshot(response):
    from tests.test_learning_privileged_encoding import hashes
    obs = response['observation']
    player, turn = obs['player_id'], obs['turn']
    return dict(stage='privileged_critic_snapshot977',
        schema='civ6rl.native_privileged_critic_snapshot977.v1',
        episode_id=response['episode_id'], game_generation=response['game_generation'],
        sequence=response['sequence'], player_id=player, turn=turn, **hashes(),
        snapshot=dict(schema='civ6rl-native-privileged-payload977-v1',
            local_player_before=player, local_player_after=player, observer_before=player,
            observer_after=player, turn_before=turn, turn_after=turn,
            players=[{'player_id':player,'gold':obs['economy']['gold']}, {'player_id':1,'gold':50}],
            units=[{k:u[k] for k in ('id','owner','type','x','y')} for u in obs['units']] +
                  [dict(id=42,owner=1,type=5,x=1,y=1)]))


def collect(runtime, first, scenario):
    from civ6rl.learning.worker import _infer
    from civ6rl.learning.action_receipt import selected_action_and_assignments
    records = []
    before = copy.deepcopy(first)
    for i in range(4):
        choice = _infer(before, runtime, scenario=scenario, completed_decisions=i)
        after = copy.deepcopy(before)
        after['sequence'] += 1
        after['decisions']['sequence'] = after['sequence']
        after['operation'] = 'act'
        after['applied_action'] = selected_action_and_assignments(before, choice)[0]
        after['observation']['economy']['gold'] += 1
        records.append(dict(before=before, choice=choice, after=after))
        before = after
    return records


def fixture(device='cpu'):
    from tests.test_learning_encoding import response
    from civ6rl.learning.encoding import ContentVocabulary
    from civ6rl.learning.worker import _build_policy
    from civ6rl.learning.scenario import build_economy_gold_scenario
    torch.set_num_threads(2)
    first = response(progress_schema='civ6rl.saved_decisions972.v1', candidates=[
        {'candidate_id':'end','action':{'kind':'end_turn'}},
        {'candidate_id':'skip','action':{'kind':'skip_unit','unit_id':101}}])
    first['outcome'] = dict(terminated=False, truncated=False)
    vocabulary = ContentVocabulary.fit(first)
    runtime = _build_policy(vocabulary.serialize(), device, 978, critic_mode='privileged_diagnostic')
    scenario = build_economy_gold_scenario(scenario_id='asymmetric-fixture978', player_id=0,
        start_observation=first['observation'],target_gold=99,max_decisions=4,max_turns=4)
    records = collect(runtime, first, scenario)
    return runtime, first, scenario, records


def request_for(runtime, scenario, records):
    from tests.test_learning_privileged_encoding import hashes
    return dict(records=records,scenario=scenario,policy_version=runtime[2]._policy_version,
        burn_in=1,epochs=2,critic_version=runtime[2]._privileged_state['version'],
        privileged_context=hashes(),privileged_snapshots=[private_snapshot(r['before']) for r in records] +
            [None if records[-1]['after']['outcome']['terminated'] else private_snapshot(records[-1]['after'])])


@unittest.skipIf(torch is None, 'PyTorch optional on native host')
class AsymmetricPPOTests(unittest.TestCase):
    def update(self, runtime, scenario, records, **overrides):
        from civ6rl.learning.learner import update_native_diagnostic
        request = request_for(runtime, scenario, records)
        request.update(overrides)
        model = runtime[2]
        return update_native_diagnostic(model, torch.optim.Adam(model.parameters(), lr=3e-4),
            runtime[1], request['records'], request['scenario'], sample_audit=model._sample_audit,
            expected_policy_version=request['policy_version'], burn_in=1, epochs=2,
            privileged_state=model._privileged_state, privileged_snapshots=request['privileged_snapshots'],
            privileged_context=request['privileged_context'], expected_critic_version=request['critic_version'])

    def check_update(self, device):
        from civ6rl.learning.artifacts import policy_version
        from civ6rl.learning.privileged_critic import critic_version
        runtime, _, scenario, records = fixture(device)
        model = runtime[2]
        actor_version = model._policy_version
        prior_critic = model._privileged_state['version']
        value_head = {k:v.clone() for k,v in model.value.state_dict().items()}
        report = self.update(runtime, scenario, records)
        self.assertEqual(report['critic_mode'], 'privileged_diagnostic')
        self.assertEqual(report['behavior_value_max_error'], 0)
        self.assertLess(report['behavior_log_prob_max_error'], 1e-5)
        self.assertEqual(report['metrics']['value_loss'], 0)
        self.assertNotEqual(policy_version(model,runtime[1]), actor_version)
        self.assertNotEqual(critic_version(model._privileged_state['critic']), prior_critic)
        self.assertEqual(report['critic_version'], critic_version(model._privileged_state['critic']))
        self.assertTrue(report['task_steps'][-1]['truncated'])
        self.assertNotEqual(report['bootstrap_value'], 0.)
        for name, value in model.value.state_dict().items():
            torch.testing.assert_close(value, value_head[name], rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in model.value.parameters()))
        self.assertFalse(any('privileged' in name for name in model.state_dict()))
        return report

    def test_cpu_separate_critic_updates_with_no_actor_value_loss(self):
        self.check_update('cpu')

    @unittest.skipUnless(torch is not None and hasattr(torch.backends,'mps') and torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps_separate_critic_updates_with_no_actor_value_loss(self):
        self.check_update('mps')

    def test_critic_creation_does_not_change_actor_parameters_or_sampling_rng(self):
        from civ6rl.learning.worker import _build_policy
        private, first, scenario, private_records = fixture()
        public = _build_policy(private[1].serialize(),'cpu',978)
        public_records = collect(public,first,scenario)
        self.assertEqual(public[2]._policy_version,private[2]._policy_version)
        for a,b in zip(public_records,private_records):
            for key in ('candidate_id','log_prob','value','entropy'):
                self.assertEqual(a['choice'][key],b['choice'][key])

    def test_stale_or_misjoined_private_data_is_refused_before_parameter_change(self):
        from civ6rl.learning.artifacts import policy_version
        from civ6rl.learning.privileged_critic import critic_version
        for fault in ('version','join','missing','source'):
            runtime, _, scenario, records = fixture()
            request = request_for(runtime,scenario,records)
            if fault=='version':request['critic_version']='f'*64
            if fault=='join':request['privileged_snapshots'][1]['sequence']+=1
            if fault=='missing':request['privileged_snapshots'].pop()
            if fault=='source':request['privileged_context']['core_sha256']='f'*64
            actor=policy_version(runtime[2],runtime[1]);critic=critic_version(runtime[2]._privileged_state['critic'])
            with self.subTest(fault=fault), self.assertRaises(ValueError):
                self.update(runtime,scenario,records,**{k:request[k] for k in ('critic_version','privileged_snapshots','privileged_context')})
            self.assertEqual(policy_version(runtime[2],runtime[1]),actor)
            self.assertEqual(critic_version(runtime[2]._privileged_state['critic']),critic)

    def test_terminal_uses_zero_bootstrap_without_fabricated_private_successor(self):
        runtime, _, scenario, records = fixture()
        records[-1]['after']['outcome']['terminated']=True
        records[-1]['after']['decisions']['candidates']=[]
        result=self.update(runtime,scenario,records)
        self.assertEqual(result['bootstrap_value'],0.)
        self.assertTrue(result['task_steps'][-1]['terminated'])

    def test_task_success_ignores_available_native_successor_value(self):
        runtime, _, scenario, records = fixture()
        records[-1]['after']['observation']['economy']['gold']=scenario['objective']['target']
        result=self.update(runtime,scenario,records)
        self.assertEqual(result['bootstrap_value'],0.)
        self.assertTrue(result['task_steps'][-1]['task_success'])
        self.assertFalse(result['task_steps'][-1]['native_game_ended'])

    def test_inference_protocol_refuses_private_payload(self):
        from civ6rl.learning.worker import _handle
        runtime, first, _, _ = fixture()
        with self.assertRaisesRegex(ValueError,'invalid_infer_fields'):
            _handle(dict(id=2,op='infer',response=first,privileged_snapshot=private_snapshot(first)),[runtime])


if __name__=='__main__':
    unittest.main()
