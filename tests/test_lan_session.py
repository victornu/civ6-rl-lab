"""Grouped lifecycle checks on retained native public city/policy exchanges.

Commands and observations are historical native I/O. The task and behavior
scores are fixtures, not newly sampled actions, native runs, or PPO updates.
"""
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
import json
from math import floor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from civ6rl.learning.lan_monitor import _decisions
from civ6rl.learning.lan_session import collect_session


class ReplayClient:
    def __init__(self, rows):
        self.rows = deepcopy(rows)
        self.index = -1
        self.aborted = False
        self.closed = False
        self.session_state = {'human_player_ids': [0]}

    def reset(self):
        assert self.index == -1
        self.index = 0
        assert self.rows[0]['operation'] == 'reset'
        return self.view()

    def view(self):
        return deepcopy(self.rows[self.index]['view'])

    def act(self, player, decision, candidate, parameters=None):
        self.index += 1
        row = self.rows[self.index]
        assert row['operation'] == 'act'
        selection = dict(decision_id=decision, candidate_id=candidate)
        if parameters is not None:
            selection['parameters'] = parameters
        assert row['receipt']['selection'] == selection
        assert row['receipt']['player_id'] == player
        return self.view()

    def poll(self, player):
        self.index += 1
        assert self.rows[self.index]['operation'] == 'poll'
        assert self.view()['observation']['player_id'] == player
        return self.view()

    def last_command_receipt(self, player):
        row = self.rows[self.index]['receipt']
        assert row['player_id'] == player
        return deepcopy(row)

    def close(self, player):
        self.index += 1
        assert self.rows[self.index]['operation'] == 'close'
        assert self.index == len(self.rows)-1
        self.closed = True
        return self.view()

    def abort(self):
        self.aborted = True


class LanSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from civ6_host.runtime.paths import resource_root
        cls.root = resource_root()
        proof = cls.root / 'evidence/host-bbg751-city-controls-native.json'
        if not proof.is_file():
            raise unittest.SkipTest('Private native city/policy receipts are required')
        if sha256(proof.read_bytes()).hexdigest() != 'fd5c9da810a0b193a96d06475118320b7af3e27cdb5c272549dad483091db149':
            raise ValueError('Retained city delivery changed')
        receipt = json.loads(proof.read_bytes())['native']['lifetimes'][0]['public_replay']
        raw = (cls.root / receipt['path']).read_bytes()
        if sha256(raw).hexdigest() != receipt['sha256']:
            raise ValueError('Retained city exchange changed')
        cls.city = [json.loads(line) for line in raw.splitlines()]
        policy_proof = cls.root / 'evidence/host-bbg751-native-policy-reconfirmation.json'
        if sha256(policy_proof.read_bytes()).hexdigest() != '2da83940019da172e8295805fdaef6680281caa5352a4b8998cfddcc3606c6e8':
            raise ValueError('Retained policy delivery changed')
        coverage = json.loads(policy_proof.read_bytes())['consumer']['interval_coverage']
        raw = (cls.root / coverage['path']).read_bytes()
        if sha256(raw).hexdigest() != coverage['sha256']:
            raise ValueError('Retained policy coverage changed')
        transcript = json.loads(raw)['transcript']
        raw = (cls.root / transcript['path']).read_bytes()
        if sha256(raw).hexdigest() != transcript['sha256']:
            raise ValueError('Retained policy exchange changed')
        policy_sessions = [json.loads(line) for line in raw.splitlines()]
        # The receipt also retains a second, query-only saved-state reload.
        # Exercise the first COMPLETE action episode; never splice generations.
        first_close = next(i for i, row in enumerate(policy_sessions) if row['operation'] == 'close')
        cls.policy = policy_sessions[:first_close+1]
        cls.template = json.loads((cls.root / 'work/bbg751-defense/scenario.json').read_bytes())

    def scenario(self, rows):
        scenario = deepcopy(self.template)
        observation = rows[0]['view']['observation']
        start = observation['turn']
        scenario['scenario_id'] = 'session-fixture-not-admitted'
        scenario['horizon'] = dict(max_decisions=2, max_turns=5)
        # This task is only a collection fixture. Never invent provenance.
        scenario['identity'] = {key: None for key in scenario['identity']}
        scenario['skill_task'].update(task_id='session-fixture-not-admitted',
            start_turn=start, deadline_turn=start+5,
            objective=dict(kind='development_capacity_and_reserve',
                unit_type='UNIT_BUILDER', target_count=1, gold_reserve=100000),
            initial_metrics=dict(public_gold=floor(observation['economy']['gold']), unit_count=0))
        return scenario

    def worker_factory(self, client, *, fail=False):
        owner = self
        state = dict(started=False, closed=False)

        @contextmanager
        def factory():
            owner.assertEqual(client.index, 0, 'Worker started before the native reset')
            state['started'] = True

            class Worker:
                def infer(self, view, *, scenario, completed_decisions):
                    if fail:
                        raise RuntimeError('fixture actor failed')
                    row = client.rows[client.index+1]
                    selection = row['receipt']['selection']
                    result = dict(episode_id=view['episode_id'], game_generation=view['game_generation'],
                        player_id=view['observation']['player_id'], sequence=view['sequence'],
                        candidate_id=selection['candidate_id'], training_eligible=False,
                        inference_seconds=0., entropy=0., model_status='fixture')
                    if 'parameters' in selection:
                        result['assignments'] = selection['parameters']['assignments']
                    return result

            try:
                yield Worker()
            finally:
                state['closed'] = True

        return factory, state

    def test_city_and_joint_policy_replay_keep_exact_choices(self):
        for rows in (self.city, self.policy):
            with self.subTest(family=rows[1]['view']['applied_action']['kind']), TemporaryDirectory() as tmp:
                client = ReplayClient(rows)
                factory, state = self.worker_factory(client)
                out = Path(tmp) / 'collection'
                result = collect_session(client, factory, self.scenario(rows), out, max_decisions=2)
                self.assertTrue(client.closed and state['closed'])
                self.assertFalse(client.aborted)
                self.assertEqual(result['decisions'], 2)
                self.assertTrue(result['task_result']['truncated'])
                exchange = [json.loads(line) for line in (out / 'exchange.jsonl').read_bytes().splitlines()]
                self.assertEqual(len(list(_decisions(exchange))), 2)
                for original, actual in zip(rows, [r for r in exchange if r['stage'] != 'actor']):
                    self.assertEqual(original['view'], actual['view'])
                saved = [json.loads(line) for line in (out / 'records.jsonl').read_bytes().splitlines()]
                self.assertEqual(result['records'], saved)
                summary = json.loads((out / 'result.json').read_bytes())
                self.assertFalse(summary['native_lifecycle_verified'])
                self.assertFalse(summary['training_ready'])
                self.assertEqual(summary['completed_commands'], 2)
                self.assertEqual(json.loads((out / 'progress.json').read_bytes())['status'], 'collected')

    def test_actor_failure_aborts_host_and_closes_worker_without_task_loss(self):
        with TemporaryDirectory() as tmp:
            client = ReplayClient(self.city)
            factory, state = self.worker_factory(client, fail=True)
            out = Path(tmp) / 'failed'
            with self.assertRaisesRegex(RuntimeError, 'fixture actor failed'):
                collect_session(client, factory, self.scenario(self.city), out, max_decisions=2)
            self.assertTrue(client.aborted and state['closed'])
            status = json.loads((out / 'progress.json').read_bytes())
            self.assertEqual(status['status'], 'failed')
            self.assertNotIn('task_result', status)
            self.assertFalse((out / 'result.json').exists())

    def test_existing_output_and_bad_budget_refused_before_reset(self):
        with TemporaryDirectory() as tmp:
            client = ReplayClient(self.city)
            factory, state = self.worker_factory(client)
            with self.assertRaises(FileExistsError):
                collect_session(client, factory, self.scenario(self.city), tmp, max_decisions=2)
            with self.assertRaisesRegex(ValueError, 'scenario horizon'):
                collect_session(client, factory, self.scenario(self.city), Path(tmp)/'bad', max_decisions=3)
            with self.assertRaisesRegex(ValueError, 'collection time'):
                collect_session(client, factory, self.scenario(self.city), Path(tmp)/'bad-time',
                                max_decisions=2, max_collection_seconds=float('nan'))
            self.assertEqual(client.index, -1)
            self.assertFalse(state['started'])

    def test_worker_startup_and_host_close_failures_preserve_cleanup(self):
        with TemporaryDirectory() as tmp:
            client = ReplayClient(self.city)

            @contextmanager
            def failed_start():
                raise RuntimeError('fixture worker init failed')
                yield

            with self.assertRaisesRegex(RuntimeError, 'worker init failed'):
                collect_session(client, failed_start, self.scenario(self.city), Path(tmp)/'init',
                                max_decisions=2)
            self.assertTrue(client.aborted)

            client = ReplayClient(self.city)
            factory, state = self.worker_factory(client)
            def failed_close(player):
                raise RuntimeError('fixture native close failed')
            def failed_abort():
                client.aborted = True
                raise RuntimeError('fixture abort also failed')
            client.close = failed_close
            client.abort = failed_abort
            with self.assertRaisesRegex(RuntimeError, 'native close failed'):
                collect_session(client, factory, self.scenario(self.city), Path(tmp)/'close',
                                max_decisions=2)
            self.assertTrue(client.aborted and state['closed'])
            status = json.loads((Path(tmp)/'close/progress.json').read_bytes())
            self.assertIn('abort also failed', status['cleanup_error'])
            self.assertFalse((Path(tmp)/'close/result.json').exists())

    def test_episode_budget_acknowledgement_and_time_bound(self):
        budget = dict(schema='civ6rl.bounded_episode_budget.v1', max_decisions=2,
                      max_updates=1, max_rounds=1, wall_seconds=120, request_seconds=20)
        with TemporaryDirectory() as tmp:
            client = ReplayClient(self.city)
            factory, state = self.worker_factory(client)

            @contextmanager
            def budgeted_worker():
                with factory() as worker:
                    worker.diagnostic_collection = deepcopy(budget)
                    worker.collection_seconds_remaining = 70.
                    yield worker

            with self.assertRaisesRegex(ValueError, 'collection time'):
                collect_session(client, budgeted_worker, self.scenario(self.city),
                    Path(tmp)/'invalid', max_decisions=2, max_collection_seconds=71,
                    diagnostic_collection=budget)
            self.assertEqual(client.index, -1)
            result = collect_session(client, budgeted_worker, self.scenario(self.city),
                Path(tmp)/'bounded', max_decisions=2, diagnostic_collection=budget)
            self.assertEqual(result['collection'], budget)
            self.assertTrue(state['closed'] and client.closed)


if __name__ == '__main__':
    unittest.main()
