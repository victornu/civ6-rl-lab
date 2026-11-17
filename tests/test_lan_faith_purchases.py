"""v30 faith purchase I/O reuses the frozen v15 public quote features."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unittest

from civ6_host.contracts import network_turn, plot_purchases
from civ6_host.runtime.paths import resource_root
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import (
    ACTOR_V14, ACTOR_V15, ENTITY_FEATURES_V14, PURCHASE_ENTITY_FEATURES,
    actor_feature_names, encode_response,
)


class LanFaithPurchaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = resource_root()
        replay = root / "work/bbg751-policy-reconfirmation/native/public-replay.jsonl"
        capture = root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json"
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest("Retained native v29 reply and pinned public database required")
        cls.view = None
        with replay.open(encoding="utf-8") as stream:
            for line in stream:
                view = json.loads(line)["view"]
                if (view["decisions"].get("schema") == network_turn.CAPABILITY_XP2_POLICY_RECONFIRM_SCHEMA
                        and view["decisions"].get("city_purchases")
                        and view["decisions"]["city_purchases"][0]["offers"]):
                    cls.view = view
                    break
        if cls.view is None:
            raise unittest.SkipTest("Retained native v29 city purchase catalogue unavailable")

        db = json.loads(capture.read_bytes())["database_copies"]["gameplay"]
        db_path = Path(db["path"])
        if sha256(db_path.read_bytes()).hexdigest() != db["sha256"]:
            raise RuntimeError("Pinned public gameplay database changed")
        cls.vocabulary = load_public_content_vocabulary(
            db_path, expected_sha256=db["sha256"],
            gameinfo_evidence_path=root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)

    def synthetic_v30(self):
        view = deepcopy(self.view)
        decisions = view["decisions"]
        decisions["schema"] = network_turn.CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA

        # Synthetic catalogue fixture: duplicate one retained native gold
        # quote at the same city/item identity with a distinct faith currency.
        catalog = decisions["city_purchases"][0]
        source = deepcopy(catalog["offers"][0])
        gold_action = source["action"]
        faith_action = deepcopy(gold_action)
        faith_action.update(currency="YIELD_FAITH", price_token="300")
        faith_offer = dict(action=faith_action, command_legal=True, can_afford=True)
        faith_candidate = dict(candidate_id=network_turn.digest(faith_action), action=faith_action)
        catalog["offers"].insert(1, faith_offer)

        # Keep all existing native candidate IDs and add the fixture candidate
        # immediately after its matching gold quote in purchase order.
        matching = next(i for i, row in enumerate(decisions["candidates"])
                        if row["action"] == gold_action)
        purchase_ordinal = sum(row["action"].get("kind") == "purchase"
                               for row in decisions["candidates"][:matching])
        decisions["candidates"].insert(matching + 1, faith_candidate)
        family = decisions["families"]["purchase"]
        family["candidate_ids"].insert(purchase_ordinal + 1, faith_candidate["candidate_id"])
        family["status"] = "queried_partial"
        family["unqualified"] = list(plot_purchases.CITY_FAITH_GAPS)
        decisions.pop("decision_id", None)
        decisions["decision_id"] = network_turn.digest(decisions)
        return view, gold_action, faith_action, faith_candidate["candidate_id"]

    def test_v30_faith_quotes_keep_distinct_ids_currency_cost_and_masks(self):
        view, gold, faith, faith_id = self.synthetic_v30()
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertEqual(frame["candidate_ids"].count(faith_id), 1)
        self.assertEqual(len(frame["candidate_ids"]), len(view["decisions"]["candidates"]))
        self.assertTrue(all(frame["action_mask"]))

        _, action_features = actor_feature_names(ACTOR_V15)
        currency_column = frame["action_categorical_fields"].index("currency")
        gold_id = network_turn.digest(gold)
        gold_row = frame["candidate_ids"].index(gold_id)
        faith_row = frame["candidate_ids"].index(faith_id)
        self.assertEqual(frame["candidates"][gold_row][action_features.index("cost")], float(gold["price_token"]))
        self.assertEqual(frame["candidates"][faith_row][action_features.index("cost")], 300.0)
        self.assertEqual(frame["action_categorical"][gold_row][currency_column],
                         self.vocabulary.lookup("action.currency", "YIELD_GOLD"))
        self.assertEqual(frame["action_categorical"][faith_row][currency_column],
                         self.vocabulary.lookup("action.currency", "YIELD_FAITH"))

        purchase_start = len(frame["entities"]) - sum(
            len(catalog["offers"]) for catalog in view["decisions"]["city_purchases"]
        ) - sum(len(catalog["offers"]) for catalog in view["decisions"]["plot_purchases"])
        quote_rows = frame["entities"][purchase_start:]
        quote_features = frame["entity_categorical_fields"]
        currency_index = quote_features.index("purchase_currency")
        by_currency = [row for row in zip(quote_rows, frame["entity_categorical"][purchase_start:])
                       if row[0][len(ENTITY_FEATURES_V14) + PURCHASE_ENTITY_FEATURES.index("purchase_quote_faith")] == 1.0]
        faith_rows = [(vector, categories) for vector, categories in by_currency
                      if categories[currency_index] == self.vocabulary.lookup("action.currency", "YIELD_FAITH")]
        self.assertEqual(len(faith_rows), 1)
        faith_entity = faith_rows[0][0]
        self.assertEqual(faith_entity[len(ENTITY_FEATURES_V14) + PURCHASE_ENTITY_FEATURES.index("purchase_quote_gold")],
                         0.0)
        self.assertEqual(faith_entity[len(ENTITY_FEATURES_V14) + PURCHASE_ENTITY_FEATURES.index("purchase_quote_can_afford")],
                         1.0)
        self.assertEqual(faith_entity[len(ENTITY_FEATURES_V14) + PURCHASE_ENTITY_FEATURES.index("purchase_quote_cost")],
                         300.0)
        self.assertTrue(all(frame["entity_mask"][purchase_start:]))

    def test_v29_rejects_synthetic_faith_currency(self):
        view, _, _, _ = self.synthetic_v30()
        view["decisions"]["schema"] = network_turn.CAPABILITY_XP2_POLICY_RECONFIRM_SCHEMA
        view["decisions"]["families"]["purchase"]["unqualified"] = list(plot_purchases.CITY_GAPS)
        view["decisions"].pop("decision_id", None)
        view["decisions"]["decision_id"] = network_turn.digest(view["decisions"])
        with self.assertRaises(ValueError):
            encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)

    def test_v30_inherits_v29_policy_reconfirmation_and_layouts_stay_fixed(self):
        from civ6rl.learning.encoding import _policy_reconfirm_enabled
        view, _, _, _ = self.synthetic_v30()
        self.assertTrue(_policy_reconfirm_enabled(view["decisions"]))
        entity, action = actor_feature_names(ACTOR_V15)
        self.assertEqual(len(PURCHASE_ENTITY_FEATURES), 14)
        previous_entity, previous_action = actor_feature_names(ACTOR_V14)
        self.assertEqual(entity, previous_entity + PURCHASE_ENTITY_FEATURES)
        self.assertEqual(action, previous_action)


if __name__ == "__main__":
    unittest.main()
