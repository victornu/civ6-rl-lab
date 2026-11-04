import copy
import unittest

from civ6rl.learning.encoding import (
    ACTION_FEATURES,
    ENTITY_FEATURES,
    GLOBAL_FEATURES,
    GLOBAL_CATEGORICAL_FIELDS,
    MAP_CHANNELS,
    ACTION_CATEGORICAL_FIELDS,
    ContentVocabulary,
    encode_response,
)


def response(*, candidates=None, policies=False, policy_catalog=None, price="130", production_progress=34,
             attacks_remaining=1, progress_schema=None, empty_progress=False,
             decisions_schema=None, treasury=None):
    pid, width, height, turn = 0, 3, 2, 7
    obs = {
        "schema": "civ6rl-player-observation-v1", "player_id": pid, "turn": turn,
        "width": width, "height": height,
        "plots": [{"index": 0, "x": 0, "y": 0, "visible": True, "revealed": True,
                   "terrain_type": 2, "feature_type": -1, "owner": -1, "water": False,
                   "impassable": False},
                  {"index": 5, "x": 2, "y": 1, "visible": True, "revealed": True,
                   "terrain_type": 1, "feature_type": -1, "owner": pid, "water": False,
                   "impassable": False}],
        "units": [{"id": 101, "type": 5, "x": 0, "y": 0, "damage": 0,
                   "moves": 2, "max_moves": 2, "owner": pid}],
        "visible_enemy_units": [],
        "cities": [{"id": 9, "x": 2, "y": 1, "population": 4, "owner": pid,
                    "yields": {"food": 3, "production": 2, "gold": 1,
                               "science": 1, "culture": 0, "faith": 0}}],
        "economy": {"science": 4, "culture": 3, "gold": 20, "faith": 0},
    }
    player_map = {
        "schema": "civ6rl.saved_player_map933.v1", "player_id": pid, "turn": turn,
        "width": width, "height": height,
        "plots": [{"index": 0, "x": 0, "y": 0, "revealed": True, "visible": True,
                   "owner": -1, "terrain_type": 2, "feature_type": -1,
                   "water": False, "impassable": False},
                  {"index": 5, "x": 2, "y": 1, "revealed": True, "visible": True,
                   "owner": pid, "terrain_type": 1, "feature_type": -1,
                   "water": False, "impassable": False}],
        "visible_observation_id": "a" * 64,
        "source": "official_plot_tooltip_revealed_live_getters",
        "historical_memory_proven": False, "graphical_differential_proven": False,
        "complete_player_observation": False,
    }
    if candidates is None:
        candidates = [{"candidate_id": "candidate-1", "action": {"kind": "end_turn"}}]
    elif candidates == "purchase":
        candidates = [{"candidate_id": "purchase-1", "action": {"kind": "purchase", "city_id": 9,
            "category": "unit", "index": 1, "hash": 801, "item_type": "UNIT_BUILDER",
            "currency": "YIELD_GOLD", "price_token": price}}]
    decisions = {
        "schema": "civ6rl.saved_decisions968.v1", "episode_id": "episode",
        "game_generation": "generation", "sequence": 1, "player_id": pid, "turn": turn,
        "player_map": player_map, "candidates": candidates,
        "research": {"research": 14, "civic": 11},
        "production": {"player_id": pid, "game_turn": turn, "cities": [{
            "city_id": 9, "owner_id": pid, "queue_size": 1, "current_production_hash": 777,
            "previous_production_hash": 0, "production_yield": 9, "has_production_progress": True,
            "turns_left": 4, "kind": "P", "item_index": 15, "cost": 65,
            "progress": production_progress, "formation": None}]},
        "combat_units": [{"unit_id": 101, "unit_type": 5, "x": 0, "y": 0, "damage": 0,
                           "moves": 2, "attacks_remaining": attacks_remaining}],
        "combat_cities": [{"owner": pid, "city_id": 9, "district_id": 9, "plot_id": 5,
            "x": 2, "y": 1, "garrison_damage": 2, "garrison_max": 200,
            "outer_damage": 0, "outer_max": 0, "defense_strength": 20}],
    }
    if policy_catalog is not None:
        decisions["policies"] = copy.deepcopy(policy_catalog)
    selected_schema = decisions_schema or progress_schema
    if selected_schema is not None:
        decisions["schema"] = selected_schema
    if selected_schema in ("civ6rl.saved_decisions972.v1", "civ6rl.saved_decisions980.v1"):
        if empty_progress:
            decisions["research"] = {"research": None, "civic": None}
        decisions["research_progress"] = {
            "schema": "civ6rl.native_saved_research_progress971.v1",
            "player_id": pid, "turn": turn,
            "technology": ({"index": None, "cost": None, "progress": None, "remaining": None}
                           if empty_progress else
                           {"index": 14, "cost": 50, "progress": 12, "remaining": 38}),
            "civic": ({"index": None, "cost": None, "progress": None, "remaining": None}
                       if empty_progress else
                       {"index": 11, "cost": 40, "progress": 9, "remaining": 31}),
        }
    if selected_schema == "civ6rl.saved_decisions980.v1":
        decisions["treasury"] = treasury or {
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": pid, "turn": turn,
            "gold_balance": 20, "gold_yield": 5.5, "total_maintenance": 7.25,
            "net_gold_yield": -1.75,
            "public_display": {"schema": "civ6rl.native_treasury_display980.v1",
                "gold_balance": 20, "gold_yield": "+5.5", "gold_expense": "-7.3",
                "net_gold_yield": "-1.8"},
        }
    if policies:
        if "policies" not in decisions:
            decisions["policies"] = {
                "player_id": pid, "turn": turn, "reason": "available",
                "slots": [{"slot": 0, "prior_index": 0, "options": [
                    {"index": 1, "hash": 101, "native_hash": 101, "policy_type": "POLICY_A"},
                    {"index": 2, "hash": 102, "native_hash": 102, "policy_type": "POLICY_B"},
                ]}],
            }
        decisions["candidates"] = [{"candidate_id": "policy", "action": {
            "kind": "policies", "slots": copy.deepcopy(decisions["policies"]["slots"])}}]
    return {
        "stage": "official_session_response", "schema": "civ6rl.action_session920.v1",
        "nonce": "nonce", "episode_id": "episode", "game_generation": "generation",
        "sequence": 1, "operation": "observe", "observation": obs, "rounds": 0,
        "outcome": {}, "automatic_choices_applied": False,
        "action_interface_complete": False, "strategic_training_ready": False,
        "decisions": decisions, "applied_action": None,
    }


def encode(value, vocabulary=None):
    return encode_response(value, vocabulary=vocabulary or ContentVocabulary.fit(value))


class LearningEncodingTests(unittest.TestCase):
    def test_shape_masks_hex_parity_and_candidate_padding_contract(self):
        encoded = encode(response(candidates=[
            {"candidate_id": "a", "action": {"kind": "end_turn"}},
            {"candidate_id": "b", "action": {"kind": "skip_unit", "unit_id": 101}},
        ]))
        self.assertEqual(len(encoded["map"]), len(MAP_CHANNELS))
        self.assertTrue(all(len(channel) == 2 and all(len(row) == 3 for row in channel)
                            for channel in encoded["map"]))
        self.assertEqual(len(encoded["global"]["values"]), len(GLOBAL_FEATURES))
        self.assertEqual(len(encoded["global_categorical"]), len(GLOBAL_CATEGORICAL_FIELDS))
        self.assertEqual(len(encoded["entities"][0]), len(ENTITY_FEATURES))
        self.assertEqual(len(encoded["candidates"][0]), len(ACTION_FEATURES))
        self.assertEqual(encoded["candidate_ids"], ["a", "b"])
        self.assertEqual(encoded["action_mask"], [True, True])
        self.assertEqual(encoded["map"][MAP_CHANNELS.index("odd_row")][1], [1.0] * 3)
        self.assertEqual(encoded["map"][MAP_CHANNELS.index("revealed")][0], [1.0, 0.0, 0.0])
        self.assertEqual(encoded["map"][MAP_CHANNELS.index("terrain_known")][0], [1.0, 0.0, 0.0])
        self.assertFalse(encoded["training_eligible"])

    def test_ephemeral_entity_and_candidate_ids_never_change_features(self):
        first = response()
        second = copy.deepcopy(first)
        second["observation"]["units"][0]["id"] = 88001
        second["decisions"]["combat_units"][0]["unit_id"] = 88001
        second["decisions"]["candidates"][0]["candidate_id"] = "new-ephemeral-id"
        vocabulary = ContentVocabulary.fit(first)
        a, b = encode(first, vocabulary), encode(second, vocabulary)
        self.assertEqual(a["entities"], b["entities"])
        self.assertEqual(a["candidates"], b["candidates"])
        self.assertNotEqual(a["candidate_ids"], b["candidate_ids"])

    def test_fog_unknowns_remain_masked_and_unofficial_rows_are_rejected(self):
        value = encode(response())
        self.assertEqual(value["map"][MAP_CHANNELS.index("revealed")][0][1:], [0.0, 0.0])
        self.assertEqual(value["map"][MAP_CHANNELS.index("resource_known")][0], [0.0, 0.0, 0.0])
        bad = response()
        bad["decisions"]["player_map"]["plots"].append({"index": 1, "x": 1, "y": 0,
            "revealed": True, "visible": False, "owner": 0, "terrain_type": 1,
            "feature_type": -1, "water": False, "impassable": False,
            "enemy_unit": True})
        with self.assertRaises(ValueError):
            encode(bad)

    def test_assignment_actions_and_unknown_action_fields_fail_closed(self):
        encoded = encode(response(policies=True))
        self.assertEqual(encoded["policy_action_index"], 0)
        with self.assertRaises(NotImplementedError):
            encode(response(candidates=[{"candidate_id": "x", "action": {
                "kind": "end_turn", "hidden_automation": True}}]))
        with self.assertRaises(ValueError):
            encode({"schema": "diagnostic-shortcut"})

    def test_policy_catalog_tokens_preserve_menu_order_and_all_candidates(self):
        slots = [{"slot": 0, "prior_index": 0, "options": [
            {"index": 1, "hash": 101, "native_hash": 101, "policy_type": "POLICY_A"},
            {"index": 2, "hash": 102, "native_hash": 102, "policy_type": "POLICY_B"},
            {"index": 3, "hash": 103, "native_hash": 103, "policy_type": "POLICY_C"},
        ]}]
        catalog = {"player_id": 0, "turn": 7, "reason": "available", "slots": slots}
        frame = response(policy_catalog=catalog, candidates=[
            {"candidate_id": "before", "action": {"kind": "end_turn"}},
            {"candidate_id": "set-policies", "action": {"kind": "policies", "slots": copy.deepcopy(slots)}},
            {"candidate_id": "after", "action": {"kind": "skip_unit", "unit_id": 101}},
        ])
        vocabulary = ContentVocabulary.fit(frame)
        encoded = encode(frame, vocabulary)
        self.assertEqual(encoded["candidate_ids"], ["before", "set-policies", "after"])
        self.assertEqual(encoded["action_mask"], [True, True, True])
        self.assertEqual(encoded["policy_action_index"], 1)
        self.assertEqual(encoded["policy_catalog"]["slots"], slots)
        self.assertEqual(encoded["policy_catalog"]["prior_tokens"], [
            vocabulary.lookup("policy.index", 0)])
        tokens = encoded["policy_catalog"]["option_tokens"][0]
        expected = [[vocabulary.lookup("policy.index", i), vocabulary.lookup("policy.hash", h),
                     vocabulary.lookup("policy.type", t)]
                    for i, h, t in ((1, 101, "POLICY_A"), (2, 102, "POLICY_B"),
                                    (3, 103, "POLICY_C"))]
        self.assertEqual(tokens, expected)
        self.assertEqual(len(tokens), 3)
        self.assertFalse(encoded["training_eligible"])

    def test_policy_hashes_are_content_tokens_not_native_submit_ids(self):
        first = response(policies=True)
        option_hash = 0xFFFFFF65
        for slots in (first["decisions"]["policies"]["slots"],
                      first["decisions"]["candidates"][0]["action"]["slots"]):
            for option in slots[0]["options"]:
                option["hash"] = option_hash + option["index"]
                option["native_hash"] = option["hash"]
        second = copy.deepcopy(first)
        for slot in second["decisions"]["policies"]["slots"]:
            for option in slot["options"]:
                option["native_hash"] = option["hash"] - (1 << 32)
        for slot in second["decisions"]["candidates"][0]["action"]["slots"]:
            for option in slot["options"]:
                option["native_hash"] = option["hash"] - (1 << 32)
        vocab = ContentVocabulary.fit(first)
        a, b = encode(first, vocab), encode(second, vocab)
        self.assertEqual(a["policy_catalog"]["option_tokens"], b["policy_catalog"]["option_tokens"])
        self.assertEqual(a["candidates"], b["candidates"])
        self.assertEqual(a["action_categorical"], b["action_categorical"])

    def test_policy_candidate_requires_exact_single_available_catalog_and_frozen_vocab(self):
        base = response(policies=True)
        vocab = ContentVocabulary.fit(base)
        multiple = copy.deepcopy(base)
        duplicate = copy.deepcopy(multiple["decisions"]["candidates"][0])
        duplicate["candidate_id"] = "policy-second"
        multiple["decisions"]["candidates"].append(duplicate)
        with self.assertRaises(ValueError):
            encode(multiple, vocab)
        mismatch = copy.deepcopy(base)
        mismatch["decisions"]["candidates"][0]["action"]["slots"][0]["options"].pop()
        with self.assertRaises(ValueError):
            encode(mismatch, vocab)
        unknown = copy.deepcopy(base)
        option = unknown["decisions"]["policies"]["slots"][0]["options"][0]
        option.update(index=4, hash=104, native_hash=104, policy_type="POLICY_D")
        unknown["decisions"]["candidates"][0]["action"]["slots"] = copy.deepcopy(
            unknown["decisions"]["policies"]["slots"])
        with self.assertRaises(ValueError):
            encode(unknown, vocab)
        missing = copy.deepcopy(base)
        del missing["decisions"]["policies"]
        with self.assertRaises(ValueError):
            encode(missing, ContentVocabulary.fit(missing))

    def test_closed_policy_window_keeps_priors_with_empty_options(self):
        slots = [{"slot": 0, "prior_index": 0, "options": []}]
        frame = response(policy_catalog={"player_id": 0, "turn": 7,
            "reason": "window_closed", "slots": slots})
        encoded = encode(frame)
        self.assertEqual(encoded["policy_catalog"]["slots"], slots)
        self.assertEqual(len(encoded["policy_catalog"]["prior_tokens"]), 1)
        self.assertEqual(encoded["policy_catalog"]["option_tokens"], [[]])
        self.assertIsNone(encoded["policy_action_index"])

    def test_stable_public_action_categories_are_vocabularized_and_unknown_refuses(self):
        frame = response(candidates=[
            {"candidate_id": "r1", "action": {"kind": "research", "index": 3, "hash": 701}},
            {"candidate_id": "r2", "action": {"kind": "research", "index": 4, "hash": 702}},
            {"candidate_id": "p1", "action": {"kind": "production", "city_id": 9,
                 "category": "unit", "index": 2, "hash": 801, "item_type": 14}},
            {"candidate_id": "p2", "action": {"kind": "production", "city_id": 9,
                 "category": "unit", "index": 3, "hash": 802, "item_type": 15}},
        ])
        vocabulary = ContentVocabulary.fit(frame)
        encoded = encode(frame, vocabulary)
        self.assertEqual(len(encoded["action_categorical"]), 4)
        hash_slot = ACTION_CATEGORICAL_FIELDS.index("hash")
        index_slot = ACTION_CATEGORICAL_FIELDS.index("index")
        self.assertNotEqual(encoded["action_categorical"][0][hash_slot],
                            encoded["action_categorical"][1][hash_slot])
        self.assertNotEqual(encoded["action_categorical"][2][index_slot],
                            encoded["action_categorical"][3][index_slot])
        self.assertGreater(encoded["vocabulary_size"], 1)
        self.assertEqual(ContentVocabulary.from_serialized(vocabulary.serialize()).fingerprint,
                         vocabulary.fingerprint)
        unknown = copy.deepcopy(frame)
        unknown["decisions"]["candidates"][0]["action"]["hash"] = 999
        with self.assertRaises(ValueError):
            encode(unknown, vocabulary)

    def test_purchase_price_production_state_and_combat_readiness_are_features(self):
        first = response(candidates="purchase", price="130", production_progress=34,
                         attacks_remaining=1)
        second = response(candidates="purchase", price="210", production_progress=55,
                          attacks_remaining=0)
        vocabulary = ContentVocabulary.fit(first)
        a, b = encode(first, vocabulary), encode(second, vocabulary)
        city_progress = ENTITY_FEATURES.index("production_progress")
        own_attacks = ENTITY_FEATURES.index("attacks_remaining")
        purchase_cost = ACTION_FEATURES.index("cost")
        self.assertEqual(a["entities"][1][city_progress], 34.0)
        self.assertEqual(b["entities"][1][city_progress], 55.0)
        self.assertEqual(a["entities"][0][own_attacks], 1.0)
        self.assertEqual(b["entities"][0][own_attacks], 0.0)
        self.assertEqual(a["candidates"][0][purchase_cost], 130.0)
        self.assertEqual(b["candidates"][0][purchase_cost], 210.0)
        self.assertEqual(a["candidates"][0][purchase_cost + 1], 1.0)
        self.assertNotEqual(a["global_categorical"], [0, 0])
        invalid_quote = response(candidates="purchase", price="1e999")
        with self.assertRaises(ValueError):
            encode(invalid_quote, ContentVocabulary.fit(invalid_quote))
        missing_quote = response(candidates="purchase")
        del missing_quote["decisions"]["candidates"][0]["action"]["price_token"]
        with self.assertRaises(ValueError):
            encode(missing_quote, ContentVocabulary.fit(missing_quote))

    def test_optional_research_progress_is_known_masked_and_backward_compatible(self):
        legacy = encode(response())
        cost = GLOBAL_FEATURES.index("technology_cost")
        cost_known = GLOBAL_FEATURES.index("technology_cost_known")
        progress = GLOBAL_FEATURES.index("technology_progress")
        remaining = GLOBAL_FEATURES.index("technology_remaining")
        self.assertEqual(legacy["global"]["values"][cost:cost_known + 1], [0.0, 0.0])
        self.assertEqual(legacy["global"]["mask"][cost:cost_known + 1], [0.0, 1.0])

        frame = response(progress_schema="civ6rl.saved_decisions972.v1")
        encoded = encode(frame)
        self.assertEqual(encoded["global"]["values"][cost], 50.0)
        self.assertEqual(encoded["global"]["values"][progress], 12.0)
        self.assertEqual(encoded["global"]["values"][remaining], 38.0)
        self.assertEqual(encoded["global"]["mask"][cost], 1.0)

        none = encode(response(progress_schema="civ6rl.saved_decisions972.v1",
                               empty_progress=True))
        self.assertEqual(none["global"]["values"][cost:cost_known + 1], [0.0, 0.0])
        self.assertEqual(none["global"]["mask"][cost:cost_known + 1], [0.0, 1.0])
        self.assertEqual(none["global"]["mask"][remaining], 0.0)

    def test_research_progress_schema_identity_selection_and_equation_are_enforced(self):
        for mutate in (
            lambda d: d["research_progress"].update(player_id=1),
            lambda d: d["research_progress"].update(turn=8),
            lambda d: d["research_progress"]["technology"].update(index=15),
            lambda d: d["research_progress"]["civic"].update(remaining=30),
        ):
            frame = response(progress_schema="civ6rl.saved_decisions972.v1")
            mutate(frame["decisions"])
            with self.assertRaises(ValueError):
                encode(frame, ContentVocabulary.fit(frame))
        legacy_with_progress = response()
        legacy_with_progress["decisions"]["research_progress"] = {
            "schema": "civ6rl.native_saved_research_progress971.v1"}
        with self.assertRaises(ValueError):
            encode(legacy_with_progress, ContentVocabulary.fit(legacy_with_progress))

    def test_treasury_values_are_known_only_in_schema980_and_accept_negative_or_zero(self):
        for old_schema in ("civ6rl.saved_decisions968.v1", "civ6rl.saved_decisions972.v1"):
            frame = response(decisions_schema=old_schema)
            encoded = encode(frame)
            for name in ("gold_yield", "total_maintenance", "net_gold_yield"):
                index = GLOBAL_FEATURES.index(name)
                self.assertEqual(encoded["global"]["values"][index:index + 2], [0.0, 0.0])
                self.assertEqual(encoded["global"]["mask"][index:index + 2], [0.0, 1.0])

        frame = response(decisions_schema="civ6rl.saved_decisions980.v1")
        encoded = encode(frame)
        expected = {"gold_yield": 5.5, "total_maintenance": 7.3, "net_gold_yield": -1.8}
        for name, value in expected.items():
            index = GLOBAL_FEATURES.index(name)
            self.assertEqual(encoded["global"]["values"][index:index + 2], [value, 1.0])
            self.assertEqual(encoded["global"]["mask"][index:index + 2], [1.0, 1.0])

        zero = response(decisions_schema="civ6rl.saved_decisions980.v1", treasury={
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": 0, "turn": 7,
            "gold_balance": 20, "gold_yield": 0, "total_maintenance": 0,
            "net_gold_yield": 0, "public_display": {
                "schema": "civ6rl.native_treasury_display980.v1", "gold_balance": 20,
                "gold_yield": "0", "gold_expense": "0", "net_gold_yield": "0"},
        })
        zero_encoded = encode(zero)
        for name in expected:
            index = GLOBAL_FEATURES.index(name)
            self.assertEqual(zero_encoded["global"]["values"][index:index + 2], [0.0, 1.0])

    def test_schema980_treasury_requires_exact_identity_finite_balance_and_net(self):
        mutations = (
            lambda t: t.update(player_id=1),
            lambda t: t.update(turn=8),
            lambda t: t.update(schema="wrong"),
            lambda t: t.update(extra=0),
            lambda t: t.update(gold_balance=19),
            lambda t: t.update(gold_yield=float("inf")),
            lambda t: t.update(total_maintenance=True),
            lambda t: t.update(net_gold_yield=0),
            lambda t: t["public_display"].update(gold_yield="1,00.0"),
            lambda t: t["public_display"].update(gold_yield="1,234,5"),
            lambda t: t["public_display"].update(gold_yield="1 23,4"),
            lambda t: t["public_display"].update(gold_yield="1 234.5"),
            lambda t: t["public_display"].update(gold_expense="-7.25"),
            lambda t: t["public_display"].update(gold_balance=21),
        )
        for mutate in mutations:
            frame = response(decisions_schema="civ6rl.saved_decisions980.v1")
            mutate(frame["decisions"]["treasury"])
            with self.assertRaises(ValueError):
                encode(frame, ContentVocabulary.fit(frame))

        missing = response(decisions_schema="civ6rl.saved_decisions980.v1")
        del missing["decisions"]["treasury"]
        with self.assertRaises(ValueError):
            encode(missing, ContentVocabulary.fit(missing))

        legacy = response()
        legacy["decisions"]["treasury"] = {
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": 0, "turn": 7,
            "gold_balance": 20, "gold_yield": 5, "total_maintenance": 2, "net_gold_yield": 3,
            "public_display": {"schema": "civ6rl.native_treasury_display980.v1",
                "gold_balance": 20, "gold_yield": "+5", "gold_expense": "-2",
                "net_gold_yield": "+3"},
        }
        with self.assertRaises(ValueError):
            encode(legacy, ContentVocabulary.fit(legacy))

    def test_treasury_actor_inputs_follow_display_precision_not_raw_getters(self):
        a = response(decisions_schema="civ6rl.saved_decisions980.v1", treasury={
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": 0, "turn": 7,
            "gold_balance": 20.2, "gold_yield": 5.46, "total_maintenance": 7.26,
            "net_gold_yield": 5.46 - 7.26, "public_display": {
                "schema": "civ6rl.native_treasury_display980.v1", "gold_balance": 20,
                "gold_yield": "+5.5", "gold_expense": "-7.3", "net_gold_yield": "-1.8"},
        })
        b = copy.deepcopy(a)
        a["observation"]["economy"]["gold"] = 20.2
        b["observation"]["economy"]["gold"] = 20.8
        a["decisions"]["treasury"].update(gold_balance=20.2, gold_yield=5.46,
            total_maintenance=7.26, net_gold_yield=5.46 - 7.26)
        b["decisions"]["treasury"].update(gold_balance=20.8, gold_yield=5.49,
            total_maintenance=7.29, net_gold_yield=5.49 - 7.29)
        encoded_a, encoded_b = encode(a), encode(b)
        self.assertEqual(encoded_a, encoded_b)
        self.assertEqual(encoded_a["global"]["values"][GLOBAL_FEATURES.index("gold")], 20.0)

        # Locale's independently formatted net is authoritative; it need not
        # equal the subtraction of the two separately rounded display fields.
        independent = response(decisions_schema="civ6rl.saved_decisions980.v1", treasury={
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": 0, "turn": 7,
            "gold_balance": 20, "gold_yield": 1.44, "total_maintenance": 1.46,
            "net_gold_yield": 1.44 - 1.46, "public_display": {
                "schema": "civ6rl.native_treasury_display980.v1", "gold_balance": 20,
                "gold_yield": "+1.4", "gold_expense": "-1.5", "net_gold_yield": "0"},
        })
        independently_formatted = encode(independent)
        self.assertEqual(independently_formatted["global"]["values"][GLOBAL_FEATURES.index("net_gold_yield")],
                         0.0)

    def test_treasury_french_and_english_grouping_share_actor_features(self):
        english = response(decisions_schema="civ6rl.saved_decisions980.v1", treasury={
            "schema": "civ6rl.native_saved_treasury980.v1", "player_id": 0, "turn": 7,
            "gold_balance": 20, "gold_yield": 1234.56, "total_maintenance": 0,
            "net_gold_yield": 1234.56, "public_display": {
                "schema": "civ6rl.native_treasury_display980.v1", "gold_balance": 20,
                "gold_yield": "+1,234.5", "gold_expense": "0", "net_gold_yield": "+1,234.5"},
        })
        french = copy.deepcopy(english)
        french["decisions"]["treasury"]["public_display"].update(
            gold_yield="+1 234,5", gold_expense="0", net_gold_yield="+1 234,5")
        self.assertEqual(encode(english), encode(french))

    def test_gold_balance_is_floored_for_all_supported_schemas(self):
        for schema in ("civ6rl.saved_decisions968.v1", "civ6rl.saved_decisions972.v1",
                       "civ6rl.saved_decisions980.v1"):
            frame = response(decisions_schema=schema)
            frame["observation"]["economy"]["gold"] = 20.8
            if schema.endswith("980.v1"):
                frame["decisions"]["treasury"]["gold_balance"] = 20.8
                frame["decisions"]["treasury"]["public_display"]["gold_balance"] = 20
            encoded = encode(frame)
            self.assertEqual(encoded["global"]["values"][GLOBAL_FEATURES.index("gold")], 20.0)

    def test_unknown_current_production_category_refuses_entire_frame(self):
        frame = response()
        frame["decisions"]["production"]["cities"][0]["kind"] = "X"
        with self.assertRaises(ValueError):
            encode(frame, ContentVocabulary.fit(frame))

    def test_visible_enemy_city_defense_requires_currently_visible_plot(self):
        hidden = response()
        hidden["decisions"]["combat_cities"].append({"owner": 2, "city_id": 77,
            "district_id": 77, "plot_id": 4, "x": 1, "y": 1, "garrison_damage": 0,
            "garrison_max": 200, "outer_damage": 0, "outer_max": 0, "defense_strength": 10})
        with self.assertRaises(ValueError):
            encode(hidden)

    def test_foreign_city_with_colliding_ephemeral_id_remains_foreign(self):
        frame = response()
        frame["decisions"]["combat_cities"].append({"owner": 2, "city_id": 9,
            "district_id": 9, "plot_id": 0, "x": 0, "y": 0, "garrison_damage": 3,
            "garrison_max": 180, "outer_damage": 0, "outer_max": 0, "defense_strength": 12})
        encoded = encode(frame)
        city_kind = ENTITY_FEATURES.index("kind_visible_enemy_city")
        foreign = [row for row in encoded["entities"] if row[city_kind] == 1.0]
        self.assertEqual(len(foreign), 1)
        self.assertEqual(foreign[0][ENTITY_FEATURES.index("garrison_damage")], 3.0)

    def test_enemy_units_outside_visible_plots_are_rejected(self):
        frame = response()
        frame["observation"]["visible_enemy_units"] = [{"id": 202, "type": 6,
            "x": 1, "y": 0, "damage": 0, "moves": 2, "owner": 2}]
        with self.assertRaises(ValueError):
            encode(frame)


if __name__ == "__main__":
    unittest.main()
