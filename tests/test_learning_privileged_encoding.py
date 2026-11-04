"""Portable synthetic contract tests for the privileged sidecar encoder."""
import copy
import math
import unittest

from civ6rl.learning.privileged_encoding import (
    PLAYER_SLOTS,
    PRIVILEGED_FEATURES,
    PRIVILEGED_SCHEMA,
    SNAPSHOT_SCHEMA,
    encode_privileged_snapshot,
)


def hashes():
    return {
        "core_sha256": "a" * 64,
        "host_sha256": "b" * 64,
        "closure_sha256": "c" * 64,
        "collector_source_sha256": "d" * 64,
    }


def response():
    episode = "episode-synthetic-1"
    generation = "native-episode-synthetic-1"
    observation = {
        "schema": "civ6rl-player-observation-v1",
        "player_id": 0,
        "turn": 12,
        "width": 5,
        "height": 4,
        "economy": {"gold": 10.5},
        "units": [
            {"id": 7, "type": 21, "x": 2, "y": 1, "damage": 0,
             "moves": 2, "max_moves": 2, "owner": 0},
        ],
    }
    decisions = {
        "schema": "civ6rl.saved_decisions972.v1",
        "episode_id": episode,
        "game_generation": generation,
        "sequence": 3,
        "player_id": 0,
        "turn": 12,
    }
    return {
        "stage": "official_session_response",
        "schema": "civ6rl.action_session920.v1",
        "episode_id": episode,
        "game_generation": generation,
        "sequence": 3,
        "observation": observation,
        "decisions": decisions,
    }


def snapshot(response_value=None):
    response_value = response() if response_value is None else response_value
    pins = hashes()
    return {
        "stage": "privileged_critic_snapshot977",
        "schema": SNAPSHOT_SCHEMA,
        "episode_id": response_value["episode_id"],
        "game_generation": response_value["game_generation"],
        "player_id": 0,
        "sequence": 3,
        "turn": 12,
        **pins,
        "snapshot": {
            "schema": "civ6rl-native-privileged-payload977-v1",
            "local_player_before": 0,
            "observer_before": 0,
            "turn_before": 12,
            "local_player_after": 0,
            "observer_after": 0,
            "turn_after": 12,
            "players": [
                {"player_id": 0, "gold": 10.5},
                {"player_id": 2, "gold": -2.0},
                {"player_id": 3},  # gold omitted is unknown, not known zero
                {"player_id": 63, "gold": 0},
            ],
            "units": [
                {"id": 7, "owner": 0, "type": 21, "x": 2, "y": 1},
                {"id": 7, "owner": 2, "type": 21, "x": 4, "y": 3},
                {"id": 8, "owner": 3, "type": 25, "x": 3, "y": 2},
            ],
        },
    }


def encode(snapshot_value=None, response_value=None, context=None):
    response_value = response() if response_value is None else response_value
    return encode_privileged_snapshot(
        snapshot() if snapshot_value is None else snapshot_value,
        response_value,
        expected_context=hashes() if context is None else context,
    )


class PrivilegedEncodingTests(unittest.TestCase):
    def test_fixed_slot_shape_explicit_masks_and_signed_scaling(self):
        result = encode()
        self.assertEqual(result["schema"], PRIVILEGED_SCHEMA)
        self.assertEqual(result["features"], PRIVILEGED_FEATURES)
        self.assertEqual(len(result["values"]), PLAYER_SLOTS * 3)
        self.assertEqual(len(result["known"]), PLAYER_SLOTS * 3)
        self.assertTrue(all(math.isfinite(value) for value in result["values"]))

        # Slot zero: alive, signed-log gold, one known unit.
        self.assertEqual(result["values"][:3], [1.0, math.log1p(10.5), math.log1p(1)])
        self.assertEqual(result["known"][:3], [True, True, True])
        # Absent player 1 is known not alive; gold and unit count stay unknown.
        self.assertEqual(result["values"][3:6], [0.0, 0.0, 0.0])
        self.assertEqual(result["known"][3:6], [True, False, False])
        # Player 2 is alive with negative gold and a distinct owner-local unit ID.
        self.assertEqual(result["values"][6], 1.0)
        self.assertEqual(result["values"][7], -math.log1p(2.0))
        self.assertEqual(result["values"][8], math.log1p(1))
        # Player 3 has a unit count but no published gold field.
        self.assertEqual(result["values"][9:12], [1.0, 0.0, math.log1p(1)])
        self.assertEqual(result["known"][9:12], [True, False, True])
        # Private unit IDs/coordinates are not emitted as features.
        self.assertEqual(len(result), 4)

    def test_join_requires_every_identity_and_source_pin(self):
        good_snapshot = snapshot()
        good_response = response()
        for field, changed in (
            ("episode_id", "episode-other"),
            ("game_generation", "native-other"),
            ("sequence", 4),
            ("player_id", 1),
            ("turn", 13),
            ("core_sha256", "f" * 64),
        ):
            bad = copy.deepcopy(good_snapshot)
            bad[field] = changed
            with self.subTest(field=field), self.assertRaises(ValueError):
                encode(bad, good_response)
        context = hashes()
        context["closure_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "source pin differs"):
            encode(good_snapshot, good_response, context)
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            encode(good_snapshot, good_response, dict(context, extra="x"))

    def test_payload_boundary_player_and_public_own_state_are_rechecked(self):
        base_snapshot, base_response = snapshot(), response()
        mutations = []
        bad = copy.deepcopy(base_snapshot);bad["snapshot"]["turn_after"] += 1;mutations.append(bad)
        bad = copy.deepcopy(base_snapshot);bad["snapshot"]["observer_before"] = 1;mutations.append(bad)
        bad = copy.deepcopy(base_snapshot);bad["snapshot"]["players"].append({"player_id": 2});mutations.append(bad)
        bad = copy.deepcopy(base_snapshot);bad["snapshot"]["units"][1]["owner"] = 10;mutations.append(bad)
        bad = copy.deepcopy(base_snapshot);bad["snapshot"]["units"][0]["x"] = 5;mutations.append(bad)
        for bad in mutations:
            with self.subTest(snapshot=bad), self.assertRaises(ValueError):
                encode(bad, base_response)

        bad_response = copy.deepcopy(base_response)
        bad_response["observation"]["economy"]["gold"] += 1
        with self.assertRaisesRegex(ValueError, "gold differs"):
            encode(base_snapshot, bad_response)
        bad_response = copy.deepcopy(base_response)
        bad_response["observation"]["units"][0]["type"] += 1
        with self.assertRaisesRegex(ValueError, "own units differ"):
            encode(base_snapshot, bad_response)

    def test_duplicate_ids_are_owner_namespaced_and_same_player_duplicates_refuse(self):
        state = snapshot()["snapshot"]
        state["units"].append({"id": 7, "owner": 2, "type": 21, "x": 2, "y": 1})
        with self.assertRaisesRegex(ValueError, "duplicate privileged unit"):
            encode(snapshot_value={**snapshot(), "snapshot": state})
        # ID 7 is valid in both owner namespaces, as in the normal fixture.
        self.assertTrue(encode()["known"][0])

    def test_duplicate_public_own_id_with_different_coordinates_refuses(self):
        response_value = response()
        response_value["observation"]["units"].append(
            {"id": 7, "type": 21, "x": 3, "y": 1, "damage": 0,
             "moves": 1, "max_moves": 2, "owner": 0})
        with self.assertRaisesRegex(ValueError, "duplicate public own unit"):
            encode(response_value=response_value)

    def test_unknown_gold_is_masked_and_negative_gold_is_not_saturated(self):
        value = encode()
        self.assertEqual(value["known"][10], False)
        self.assertEqual(value["values"][7], -math.log1p(2.0))
        self.assertLess(value["values"][7], 0)
        self.assertEqual(value["values"][1], math.log1p(10.5))


if __name__ == "__main__":
    unittest.main()
