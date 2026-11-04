import copy
import unittest
from civ6rl.learning.encoding import ContentVocabulary
from civ6rl.learning.trajectory import DiagnosticTrajectory
from civ6rl.learning.action_receipt import selected_action_and_assignments
from tests.test_learning_encoding import response


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.before = response()
        self.after = copy.deepcopy(self.before)
        self.after.update(sequence=self.before['sequence']+1, operation='act', outcome={'terminated':False},
                          applied_action=self.before['decisions']['candidates'][0]['action'])
        self.after['decisions']['sequence'] = self.after['sequence']
        self.choice = dict(episode_id=self.before['episode_id'], player_id=0, sequence=self.before['sequence'],
                           candidate_id=self.before['decisions']['candidates'][0]['candidate_id'],
                           log_prob=-.5, value=.1, entropy=.6)
        self.trace = DiagnosticTrajectory(policy_identity={'seed':42,'model_status':'untrained'},
                                          vocabulary=ContentVocabulary.fit(self.before))

    def test_diagnostic_budget_is_truncation_without_invented_reward(self):
        self.trace.append(self.before,self.choice,self.after)
        record=self.trace.finish_budget()[0]
        self.assertIsNone(record['reward']); self.assertTrue(record['truncated'])
        self.assertFalse(record['terminated']); self.assertFalse(record['training_eligible'])
        with self.assertRaises(RuntimeError): self.trace.training_records()

    def test_native_terminal_is_not_reclassified_as_truncation(self):
        self.after['outcome']['terminated']=True
        self.trace.append(self.before,self.choice,self.after)
        self.assertFalse(self.trace.finish_budget()[0]['truncated'])

    def test_substitution_and_wrong_player_successor_rejected(self):
        changed=copy.deepcopy(self.after);changed['applied_action']={'kind':'different'}
        with self.assertRaises(ValueError): self.trace.append(self.before,self.choice,changed)
        changed=copy.deepcopy(self.after);changed['observation']['player_id']=1
        with self.assertRaises(ValueError): self.trace.append(self.before,self.choice,changed)

    def test_reset_cannot_be_spliced_into_recurrent_sequence(self):
        self.trace.append(self.before,self.choice,self.after)
        with self.assertRaises(ValueError): self.trace.append(self.before,self.choice,self.after)

    def test_policy_assignments_are_recorded_after_native_materialization(self):
        before = copy.deepcopy(self.before)
        slots = [
            {"slot": 0, "prior_index": 8, "options": [
                {"index": 1, "hash": 101, "native_hash": 101, "policy_type": "POLICY_B"},
                {"index": 2, "hash": 150, "native_hash": 150, "policy_type": "POLICY_C"}]},
            {"slot": 1, "prior_index": 9, "options": [
                {"index": 3, "hash": 202, "native_hash": 202, "policy_type": "POLICY_D"},
                {"index": 4, "hash": 203, "native_hash": 203, "policy_type": "POLICY_E"}]},
        ]
        before["decisions"]["policies"] = {"player_id": 0, "turn": 7,
                                                  "reason": "available", "slots": slots}
        before["decisions"]["candidates"] = [{"candidate_id": "policy", "action": {
            "kind": "policies", "slots": slots}}]
        choice = {"episode_id": before["episode_id"], "player_id": 0,
                  "sequence": before["sequence"], "candidate_id": "policy",
                  "assignments": [101, 202], "log_prob": -0.5, "value": 0.1, "entropy": 0.6}
        after = copy.deepcopy(before)
        after.update(sequence=before["sequence"] + 1, operation="act",
                     outcome={"terminated": False}, automatic_choices_applied=False)
        after["decisions"]["sequence"] = after["sequence"]
        after["decisions"]["candidates"] = [{"candidate_id": "next", "action": {"kind": "end_turn"}}]
        after["decisions"].pop("policies")
        after["applied_action"] = selected_action_and_assignments(before, choice)[0]
        collector = DiagnosticTrajectory(policy_identity={"policy_id": "test", "version": "untrained"},
                                         vocabulary=ContentVocabulary.fit(before))
        record = collector.append(before, choice, after)
        self.assertEqual([101, 202], record["assignments"])
        self.assertFalse(record["training_eligible"])


if __name__=='__main__': unittest.main()
