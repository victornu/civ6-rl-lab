import copy
import os
import tempfile
from pathlib import Path
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch is an optional learning dependency')
class NativeSequenceLearnerTests(unittest.TestCase):
    def setUp(self):
        from tests.test_learning_encoding import response
        from civ6rl.learning.encoding import ContentVocabulary
        from civ6rl.learning.worker import _build_policy,_infer
        from civ6rl.learning.scenario import build_economy_gold_scenario
        from civ6rl.learning.action_receipt import selected_action_and_assignments
        torch.set_num_threads(2)
        self.first=response(candidates=[{'candidate_id':'end','action':{'kind':'end_turn'}},
                                        {'candidate_id':'skip','action':{'kind':'skip_unit','unit_id':101}}])
        self.first['outcome']={'terminated':False,'truncated':False}
        self.vocab=ContentVocabulary.fit(self.first)
        self.runtime=_build_policy(self.vocab.serialize(),'cpu',974)
        self.model=self.runtime[2];self.version=self.model._policy_version
        self.scenario=build_economy_gold_scenario(scenario_id='diagnostic-fixture',player_id=0,
           start_observation=self.first['observation'],target_gold=99,max_decisions=4,max_turns=4)
        self.records=[];before=self.first
        for _ in range(4):
            choice=_infer(before,self.runtime,scenario=self.scenario,completed_decisions=len(self.records))
            after=copy.deepcopy(before);after['sequence']+=1;after['decisions']['sequence']=after['sequence']
            after['operation']='act';after['applied_action']=selected_action_and_assignments(before,choice)[0]
            after['observation']['economy']['gold']+=1
            self.records.append(dict(before=before,choice=choice,after=after));before=after

    def update(self,records=None,**kwargs):
        from civ6rl.learning.learner import update_native_diagnostic
        return update_native_diagnostic(self.model,torch.optim.Adam(self.model.parameters(),lr=.001),self.vocab,
            self.records if records is None else records,self.scenario,sample_audit=self.model._sample_audit,
            expected_policy_version=self.version,**kwargs)

    def test_replay_burn_in_and_truncation_bootstrap_reach_optimizer(self):
        from civ6rl.learning.artifacts import policy_version
        report=self.update(burn_in=1,epochs=2)
        self.assertEqual(report['optimized_decisions'],3)
        self.assertLess(report['behavior_log_prob_max_error'],1e-5)
        self.assertLess(report['behavior_value_max_error'],1e-5)
        self.assertTrue(report['task_steps'][-1]['truncated'])
        self.assertFalse(report['task_steps'][-1]['terminated'])
        self.assertNotEqual(report['bootstrap_value'],0.)
        self.assertNotEqual(policy_version(self.model,self.vocab),self.version)
        self.assertFalse(report['training_eligible'])

    def test_changed_actor_statistics_and_coverage_seed_cannot_be_admitted(self):
        from civ6rl.learning.artifacts import policy_version
        wrong=copy.deepcopy(self.records);wrong[0]['choice']['log_prob']+=.1
        with self.assertRaises(ValueError):self.update(wrong)
        wrong=copy.deepcopy(self.records);wrong[0]['choice']['sampling_mode']='coverage_selected_seed'
        with self.assertRaises(ValueError):self.update(wrong)
        self.assertEqual(policy_version(self.model,self.vocab),self.version)

    def test_another_player_or_spliced_sequence_is_refused(self):
        wrong=copy.deepcopy(self.records);wrong[1]['before']['observation']['player_id']=1
        with self.assertRaises(ValueError):self.update(wrong)
        with self.assertRaises(ValueError):self.update(self.records[1:])

    def test_native_terminal_has_no_bootstrap_and_does_not_need_next_menu(self):
        records=copy.deepcopy(self.records)
        records[-1]['after']['outcome']['terminated']=True
        records[-1]['after']['decisions']['candidates']=[]
        report=self.update(records,epochs=1)
        self.assertEqual(report['bootstrap_value'],0.)
        self.assertTrue(report['task_steps'][-1]['native_game_ended'])
        self.assertTrue(report['task_steps'][-1]['terminated'])
        self.assertFalse(report['task_steps'][-1]['truncated'])

    def test_stale_model_weights_are_detected_before_optimization(self):
        with torch.no_grad():self.model.value.bias.add_(.3)
        with self.assertRaisesRegex(ValueError,'behavior differs'):self.update()

    def test_snapshot_preserves_features_vocabulary_and_probabilities(self):
        from civ6rl.learning.artifacts import policy_bytes,policy_version,load_policy_bytes
        from civ6rl.learning.model import collate
        from civ6rl.learning.encoding import encode_response
        blob=policy_bytes(self.model,self.vocab)
        self.assertEqual(policy_version(self.model,self.vocab),self.version)
        loaded,vocab=load_policy_bytes(blob,sha256=self.version)
        self.assertEqual(vocab.fingerprint,self.vocab.fingerprint)
        batch=collate([encode_response(self.first,vocabulary=self.vocab)])
        torch.testing.assert_close(loaded(batch)[0].probs,self.model(batch)[0].probs)
        with self.assertRaises(ValueError):load_policy_bytes(blob,sha256='0'*64)

    def test_worker_update_saves_version_and_requires_new_episode(self):
        from civ6rl.learning.worker import _update_diagnostic,_infer,_build_policy
        cwd=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                request=dict(records=self.records,scenario=self.scenario,policy_version=self.version,burn_in=1,epochs=1)
                report=_update_diagnostic(request,self.runtime)
                self.assertEqual(report['model_status'],'diagnostic_updated')
                self.assertTrue(Path(report['checkpoint']['path']).is_file())
                with self.assertRaisesRegex(ValueError,'new_episode'):_infer(self.records[-1]['after'],self.runtime)
                loaded=_build_policy(self.vocab.serialize(),'cpu',974,report['checkpoint'])
                self.assertEqual(loaded[2]._policy_version,report['policy_version'])
                fresh=copy.deepcopy(self.first);fresh['episode_id']='new-episode';fresh['decisions']['episode_id']='new-episode'
                fresh['game_generation']='new-generation';fresh['decisions']['game_generation']='new-generation'
                result=_infer(fresh,loaded)
                self.assertEqual(result['model_status'],'diagnostic_updated')
                self.assertTrue(result['memory_reset'])
            finally:os.chdir(cwd)


if __name__=='__main__':unittest.main()
