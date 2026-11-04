import copy
import hashlib
import io
import os
import random
import tempfile
from pathlib import Path
import unittest
try:
    import torch
except ImportError:
    torch=None


@unittest.skipIf(torch is None,'PyTorch is an optional learning dependency')
class LearnerBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.cwd=Path.cwd()
        self.tmp=tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(os.chdir,self.cwd)

    def prepare(self,device='cpu'):
        from tests.test_learning_learner import NativeSequenceLearnerTests
        from civ6rl.learning.worker import _build_policy,_infer,_update_diagnostic
        from civ6rl.learning.action_receipt import selected_action_and_assignments
        fixture=NativeSequenceLearnerTests();fixture.setUp()
        runtime=_build_policy(fixture.vocab.serialize(),device,975)
        records=[];before=copy.deepcopy(fixture.first)
        for _ in range(4):
            choice=_infer(before,runtime,scenario=fixture.scenario,completed_decisions=len(records))
            after=copy.deepcopy(before);after['sequence']+=1;after['decisions']['sequence']=after['sequence']
            after['operation']='act';after['applied_action']=selected_action_and_assignments(before,choice)[0]
            after['observation']['economy']['gold']+=1
            records.append(dict(before=before,choice=choice,after=after));before=after
        request=dict(records=records,scenario=fixture.scenario,policy_version=runtime[2]._policy_version,burn_in=1,epochs=1)
        report=_update_diagnostic(request,runtime)
        return fixture,runtime,report

    def continue_episode(self,fixture,runtime):
        from civ6rl.learning.worker import _infer,_update_diagnostic
        from civ6rl.learning.action_receipt import selected_action_and_assignments
        before=copy.deepcopy(fixture.first)
        for obj in (before,before['decisions']):
            obj['episode_id']='continuation';obj['game_generation']='continuation-generation'
        records=[]
        for _ in range(4):
            choice=_infer(before,runtime,scenario=fixture.scenario,completed_decisions=len(records))
            after=copy.deepcopy(before);after['sequence']+=1;after['decisions']['sequence']=after['sequence']
            after['operation']='act';after['applied_action']=selected_action_and_assignments(before,choice)[0]
            after['observation']['economy']['gold']+=1
            records.append(dict(before=before,choice=choice,after=after));before=after
        report=_update_diagnostic(dict(records=records,scenario=fixture.scenario,policy_version=runtime[2]._policy_version,burn_in=1,epochs=1),runtime)
        return records,report

    def check_continuation(self,device):
        from civ6rl.learning.training_state import load_training_bytes
        fixture,runtime,report=self.prepare(device)
        reference=report['learner_checkpoint'];data=Path(reference['path']).read_bytes()
        saved_payload=torch.load(io.BytesIO(data),weights_only=True)
        uninterrupted_records,uninterrupted=self.continue_episode(fixture,runtime)
        next_python=random.random();next_torch=torch.rand(4,device=device).cpu()
        original_params={k:v.clone() for k,v in runtime[2].state_dict().items()}
        original_optimizer=copy.deepcopy(runtime[2]._optimizer.state_dict())
        loaded=load_training_bytes(data,sha256=reference['sha256'],device=device)
        self.assertEqual(loaded[2]._diagnostic_updates,1)
        self.assertEqual(loaded[2]._last_update_context,saved_payload['last_update_context'])
        restored_records,restored=self.continue_episode(fixture,loaded)
        self.assertEqual(restored['completed_updates'],2)
        self.assertEqual([r['choice']['candidate_id'] for r in uninterrupted_records],
                         [r['choice']['candidate_id'] for r in restored_records])
        for a,b in zip(uninterrupted_records,restored_records):
            self.assertAlmostEqual(a['choice']['log_prob'],b['choice']['log_prob'],places=6)
            self.assertAlmostEqual(a['choice']['value'],b['choice']['value'],places=6)
        for k,v in loaded[2].state_dict().items():
            torch.testing.assert_close(v,original_params[k],rtol=1e-6,atol=1e-7)
        restored_optimizer=loaded[2]._optimizer.state_dict()
        self.assertEqual(restored_optimizer['param_groups'],original_optimizer['param_groups'])
        for index,state in original_optimizer['state'].items():
            for name,tensor in state.items():
                torch.testing.assert_close(restored_optimizer['state'][index][name],tensor,rtol=1e-6,atol=1e-7)
        self.assertEqual(random.random(),next_python)
        torch.testing.assert_close(torch.rand(4,device=device).cpu(),next_torch,rtol=0,atol=0)
        self.assertEqual(loaded[2]._closed_episodes,runtime[2]._closed_episodes)
        self.assertFalse(restored['training_eligible'])
        return dict(device=device,choices_equal=True,optimizer_and_parameters_close=True,
                    policy_digest_equal=restored['policy_version']==uninterrupted['policy_version'],
                    counters_preserved=True,rng_equal=True)

    def test_cpu_interruption_preserves_next_samples_and_optimizer_update(self):
        self.assertTrue(self.check_continuation('cpu')['policy_digest_equal'])

    @unittest.skipUnless(torch is not None and hasattr(torch.backends,'mps') and torch.backends.mps.is_available(),'MPS not available')
    def test_mps_interruption_preserves_next_samples_and_optimizer_update(self):
        self.check_continuation('mps')

    def test_pending_sample_cannot_be_hidden_in_a_boundary_snapshot(self):
        from civ6rl.learning.training_state import training_bytes
        from civ6rl.learning.worker import _infer
        fixture,runtime,_=self.prepare()
        fresh=copy.deepcopy(fixture.first)
        for obj in (fresh,fresh['decisions']):
            obj['episode_id']='pending';obj['game_generation']='pending'
        _infer(fresh,runtime)
        with self.assertRaisesRegex(ValueError,'empty completed boundary'):training_bytes(runtime)

    def test_modified_implementation_optimizer_or_vocab_is_refused_without_rng_mutation(self):
        from civ6rl.learning.training_state import load_training_bytes
        _,_,report=self.prepare()
        raw=Path(report['learner_checkpoint']['path']).read_bytes()
        for field in ('implementation','optimizer','vocabulary','collector'):
            payload=torch.load(io.BytesIO(raw),weights_only=True)
            kwargs={}
            if field=='implementation':payload['implementation_sha256']['model.py']='0'*64
            if field=='optimizer':
                first=next(iter(payload['optimizer']['state'].values()))
                first['exp_avg'].fill_(float('nan'))
            if field=='vocabulary':kwargs['expected_vocabulary_fingerprint']='0'*64
            if field=='collector':payload['collector']['pending_samples']=1
            stream=io.BytesIO();torch.save(payload,stream);data=stream.getvalue()
            py=random.getstate();rng=torch.get_rng_state().clone()
            with self.assertRaises(ValueError):
                load_training_bytes(data,sha256=hashlib.sha256(data).hexdigest(),**kwargs)
            self.assertEqual(random.getstate(),py);self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_resume_protocol_keeps_closed_episodes_and_cumulative_budget(self):
        from civ6rl.learning.worker import _handle,_infer
        fixture,_,report=self.prepare()
        box=[None]
        result=_handle(dict(id=1,op='init',vocabulary=fixture.vocab.serialize(),device='cpu',seed=123,
                            resume=report['learner_checkpoint']),box)
        self.assertTrue(result['learner_resumed']);self.assertEqual(result['completed_updates'],1)
        with self.assertRaisesRegex(ValueError,'new_episode'):_infer(fixture.first,box[0])

    def test_failed_update_discards_runtime_instead_of_leaving_partial_weights(self):
        from civ6rl.learning.worker import _handle
        fixture,runtime,_=self.prepare();box=[runtime]
        with self.assertRaises(ValueError):
            _handle(dict(id=2,op='update_diagnostic',records=[],scenario=fixture.scenario,
                         policy_version=runtime[2]._policy_version,burn_in=0,epochs=1),box)
        self.assertIsNone(box[0])


if __name__=='__main__':unittest.main()
