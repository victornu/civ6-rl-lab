import copy
import unittest

from civ6rl.learning.encoding import ContentVocabulary
from civ6rl.learning.multiplayer_trajectory import MultiPlayerDiagnosticTrajectory
from civ6rl.learning.action_receipt import selected_action_and_assignments, verify_applied_action
from tests.test_learning_encoding import response


EPISODE = "episode-multi"
GENERATION = "generation-a"


def native_response(player_id, sequence, *, episode=EPISODE, generation=GENERATION):
    value = response()
    value["episode_id"] = episode
    value["game_generation"] = generation
    value["sequence"] = sequence
    value["observation"]["player_id"] = player_id
    for row in value["observation"]["units"] + value["observation"]["cities"]:
        if row.get("owner") == 0:
            row["owner"] = player_id
    for row in value["observation"]["plots"]:
        if row.get("owner") == 0:
            row["owner"] = player_id
    value["decisions"]["episode_id"] = episode
    value["decisions"]["game_generation"] = generation
    value["decisions"]["sequence"] = sequence
    value["decisions"]["player_id"] = player_id
    value["decisions"]["player_map"]["player_id"] = player_id
    value["decisions"]["production"]["player_id"] = player_id
    for row in value["decisions"]["player_map"]["plots"]:
        if row.get("owner") == 0:
            row["owner"] = player_id
    for row in value["decisions"]["combat_cities"]:
        if row.get("owner") == 0:
            row["owner"] = player_id
    return value


def action_result(before, after, *, terminated=False, truncated=False):
    result = copy.deepcopy(after)
    result.update(sequence=before["sequence"] + 1, operation="act",
                  applied_action=copy.deepcopy(before["decisions"]["candidates"][0]["action"]),
                  automatic_choices_applied=False,
                  outcome={"terminated": terminated, "truncated": truncated})
    return result


def choice(before, *, candidate_id=None):
    return {"episode_id": before["episode_id"], "player_id": before["observation"]["player_id"],
            "sequence": before["sequence"], "candidate_id": candidate_id or before["decisions"]["candidates"][0]["candidate_id"],
            "log_prob": -0.25, "value": 0.5, "entropy": 0.6, "training_eligible": False}


def terminal_payload(episode=EPISODE, generation=GENERATION, *, omit_player=None):
    results, observations = {}, {}
    for pid in (0, 1):
        if pid == omit_player:
            continue
        results[pid] = {"episode_id": episode, "game_generation": generation, "player_id": pid,
                        "terminated": True, "truncated": False, "result": {"standing": "unknown"}}
        observations[pid] = copy.deepcopy(native_response(pid, 99)["observation"])
    return {"results": results, "observations": observations}


class MultiPlayerTrajectoryTests(unittest.TestCase):
    def setUp(self):
        initial = native_response(0, 1)
        self.vocabulary = ContentVocabulary.fit(initial)
        self.trace = MultiPlayerDiagnosticTrajectory(
            policy_identity={"policy_id": "rnn-policy", "version": "untrained-v1"},
            vocabulary=self.vocabulary, player_ids=(0, 1))

    def test_policy_assignment_receipt_matches_native_choose_shape(self):
        before = native_response(0, 1)
        slots = [{"slot": 0, "prior_index": 0, "options": [
            {"index": 0, "hash": 100, "native_hash": 100, "policy_type": "POLICY_A"},
            {"index": 1, "hash": 0xFFFFFFFF, "native_hash": -1, "policy_type": "POLICY_B"}]},
            {"slot": 1, "prior_index": -1, "options": [
                {"index": 2, "hash": 202, "native_hash": 202, "policy_type": "POLICY_C"}]}]
        before["decisions"]["candidates"] = [{"candidate_id": "policy", "action": {
            "kind": "policies", "slots": slots}}]
        learner_choice = {"candidate_id": "policy", "assignments": [0xFFFFFFFF, 202]}
        expected, assignments = selected_action_and_assignments(before, learner_choice)
        self.assertEqual([0xFFFFFFFF, 202], assignments)
        self.assertEqual({"kind": "policies", "candidate": {
            "status": "candidate", "player_id": 0, "turn": 7, "slot_count": 2,
            "assignments": [[0, -1], [1, 202]], "prior_indices": [0, -1]}}, expected)
        after = {"applied_action": copy.deepcopy(expected), "automatic_choices_applied": False}
        self.assertEqual(assignments, verify_applied_action(before, learner_choice, after))
        substituted = copy.deepcopy(after)
        substituted["applied_action"]["candidate"]["assignments"][0][1] = 100
        with self.assertRaisesRegex(ValueError, "differs from the selected candidate"):
            verify_applied_action(before, learner_choice, substituted)
        with self.assertRaisesRegex(ValueError, "non-policy action"):
            selected_action_and_assignments(native_response(0, 1), {
                "candidate_id": native_response(0, 1)["decisions"]["candidates"][0]["candidate_id"],
                "assignments": [1]})

    def test_multiplayer_collector_records_materialized_joint_assignment(self):
        before = native_response(0, 1)
        slots = [{"slot": 0, "prior_index": 8, "options": [
            {"index": 1, "hash": 101, "native_hash": 101, "policy_type": "POLICY_B"},
            {"index": 2, "hash": 150, "native_hash": 150, "policy_type": "POLICY_C"}]},
            {"slot": 1, "prior_index": 9, "options": [
                {"index": 3, "hash": 202, "native_hash": 202, "policy_type": "POLICY_D"},
                {"index": 4, "hash": 203, "native_hash": 203, "policy_type": "POLICY_E"}]}]
        before["decisions"]["policies"] = {"player_id": 0, "turn": 7,
                                                  "reason": "available", "slots": slots}
        before["decisions"]["candidates"] = [{"candidate_id": "policy", "action": {
            "kind": "policies", "slots": slots}}]
        learner_choice = choice(before)
        learner_choice.update(candidate_id="policy", assignments=[101, 202])
        after = action_result(before, native_response(0, 2))
        after["applied_action"] = selected_action_and_assignments(before, learner_choice)[0]
        collector = MultiPlayerDiagnosticTrajectory(
            policy_identity={"policy_id": "rnn-policy", "version": "untrained-v1"},
            vocabulary=ContentVocabulary.fit(before), player_ids=(0, 1))
        record = collector.append(before, learner_choice, after)
        self.assertEqual([101, 202], record["assignments"])
        self.assertEqual(0, record["player_id"])
        self.assertFalse(record["training_eligible"])

    def test_alternating_players_resolve_only_on_their_own_next_decision(self):
        p0 = native_response(0, 1)
        p1 = action_result(p0, native_response(1, 2))
        p0_again = action_result(p1, native_response(0, 3))
        p1_again = action_result(p0_again, native_response(1, 4))
        self.assertIsNone(self.trace.append(p0, choice(p0), p1))
        record0 = self.trace.append(p1, choice(p1), p0_again)
        self.assertEqual(0, record0["player_id"])
        self.assertEqual({1}, set(self.trace._pending))
        self.assertEqual(0, record0["before_response"]["observation"]["player_id"])
        self.assertEqual(0, record0["next_response"]["observation"]["player_id"])
        self.assertEqual(0, record0["encoded_next"]["player_id"])
        self.assertNotIn("observation", record0["action_receipt"])
        self.assertIsNone(record0["reward"])
        self.assertFalse(record0["training_eligible"])
        self.assertEqual("untrained-v1", record0["policy_identity"]["version"])
        self.assertEqual(GENERATION, record0["game_generation"])
        record1 = self.trace.append(p0_again, choice(p0_again), p1_again)
        self.assertEqual(1, record1["player_id"])
        self.assertEqual(1, record1["next_response"]["observation"]["player_id"])

    def test_budget_keeps_known_same_player_bootstrap_but_not_opponent_view(self):
        p0 = native_response(0, 1)
        p0_next = action_result(p0, native_response(0, 2))
        self.trace.append(p0, choice(p0), p0_next)
        self.assertEqual(1, len(self.trace.records))
        records = self.trace.finish_budget()
        record = records[0]
        self.assertEqual(0, record["next_response"]["observation"]["player_id"])
        self.assertEqual(0, record["encoded_next"]["player_id"])
        self.assertFalse(record["truncated"])

        second = MultiPlayerDiagnosticTrajectory(
            policy_identity={"policy_id": "rnn-policy", "version": "untrained-v1"},
            vocabulary=self.vocabulary, player_ids=(0, 1))
        p0_first = native_response(0, 1)
        opponent = action_result(p0_first, native_response(1, 2))
        second.append(p0_first, choice(p0_first), opponent)
        other = second.finish_budget()[0]
        self.assertTrue(other["truncated"])
        self.assertIsNone(other["next_response"])
        self.assertIsNone(other["encoded_next"])

    def test_candidate_membership_player_identity_and_stream_order_are_enforced(self):
        p0 = native_response(0, 1)
        p1 = action_result(p0, native_response(1, 2))
        foreign = choice(p0, candidate_id="not-published")
        with self.assertRaises(ValueError):
            self.trace.append(p0, foreign, action_result(p0, p1))
        stale = choice(p0)
        stale["player_id"] = 1
        with self.assertRaises(ValueError):
            self.trace.append(p0, stale, action_result(p0, p1))
        substituted = action_result(p0, p1)
        substituted["applied_action"] = {"kind": "different"}
        with self.assertRaisesRegex(ValueError, "differs from the selected candidate"):
            self.trace.append(p0, choice(p0), substituted)
        self.trace.append(p0, choice(p0), p1)
        reordered = native_response(1, 4)
        with self.assertRaises(ValueError):
            self.trace.append(reordered, choice(reordered), action_result(reordered, p0))

    def test_episode_reset_flushes_pending_as_truncated_without_a_fake_next_state(self):
        p0 = native_response(0, 1)
        p1 = action_result(p0, native_response(1, 2))
        self.trace.append(p0, choice(p0), p1)
        records = self.trace.reset(episode_id="episode-b", game_generation="generation-b")
        self.assertEqual(1, len(records))
        self.assertEqual("episode_reset", records[0]["truncation_reason"])
        self.assertTrue(records[0]["truncated"])
        self.assertIsNone(records[0]["next_response"])
        self.assertIsNone(records[0]["reward"])
        fresh = native_response(1, 0, episode="episode-b", generation="generation-b")
        fresh_after = native_response(0, 1, episode="episode-b", generation="generation-b")
        self.trace.append(fresh, choice(fresh), action_result(fresh, fresh_after))
        self.assertEqual("episode-b", self.trace._pending[1]["before"]["episode_id"])

    def test_terminal_requires_per_player_results_and_observations_and_never_fills_reward(self):
        p0 = native_response(0, 1)
        p1 = action_result(p0, native_response(1, 2))
        self.trace.append(p0, choice(p0), p1)
        terminal_after = action_result(p1, p1, terminated=True)
        terminal_after["observation"] = None
        terminal_after["decisions"] = None
        incomplete = terminal_payload(omit_player=1)
        with self.assertRaisesRegex(ValueError, "cover every configured player"):
            self.trace.append(p1, choice(p1), terminal_after, terminal=incomplete)
        self.assertIn(0, self.trace._pending)
        records = terminal_payload()
        result = self.trace.append(p1, choice(p1), terminal_after, terminal=records)
        self.assertTrue(self.trace._closed)
        self.assertEqual({0, 1}, {row["player_id"] for row in self.trace.records})
        by_player = {row["player_id"]: row for row in self.trace.records}
        for pid, row in by_player.items():
            self.assertTrue(row["terminated"])
            self.assertIsNone(row["reward"])
            self.assertEqual(pid, row["terminal_observation"]["player_id"])
            self.assertEqual(pid, row["terminal_result"]["player_id"])
            self.assertIsNone(row["encoded_next"])
        self.assertEqual(1, result["player_id"])

    def test_wrong_generation_and_cross_player_terminal_observation_rejected(self):
        p0 = native_response(0, 1)
        p1 = action_result(p0, native_response(1, 2))
        self.trace.append(p0, choice(p0), p1)
        other_gen = native_response(0, 3, generation="generation-other")
        with self.assertRaises(ValueError):
            self.trace.append(p1, choice(p1), action_result(p1, other_gen))
        terminal_after = action_result(p1, p1, terminated=True)
        terminal_after["observation"] = None
        terminal_after["decisions"] = None
        terminal = terminal_payload()
        terminal["observations"][0]["player_id"] = 1
        with self.assertRaisesRegex(ValueError, "foreign perspective"):
            self.trace.append(p1, choice(p1), terminal_after, terminal=terminal)

        terminal = terminal_payload()
        terminal["observations"][0]["turn"] -= 1
        terminal_after["outcome"]["native_victory"] = {"current_game_turn": 7}
        with self.assertRaisesRegex(ValueError, "stale relative to the terminal response turn"):
            self.trace.append(p1, choice(p1), terminal_after, terminal=terminal)


if __name__ == "__main__":
    unittest.main()
