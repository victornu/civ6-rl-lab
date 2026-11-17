"""LAN v28 relocation uses the unchanged factorized actor on a synthetic v27 reply."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import unittest

try:
    import torch
except ImportError:
    torch = None


class LanPolicyRelocationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from civ6_host.contracts import network_turn
        from civ6_host.runtime.paths import resource_root
        from civ6rl.learning.content_vocabulary import load_public_content_vocabulary

        root = resource_root()
        replay = root / "work/bbg751-government/adoption/public-replay.jsonl"
        capture = root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json"
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest("Retained v27 reply and pinned public database required")

        active_view = None
        options_view = None
        with replay.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                view = row["view"]
                menu = view["decisions"]
                catalogue = menu.get("policies")
                if not catalogue or menu.get("schema") != "civ6-host.network-capability-decisions.v27":
                    continue
                slots = catalogue["slots"]
                if (catalogue["reason"] == "available" and len(slots) == 4
                        and all(slot["prior_index"] == -1 for slot in slots)):
                    options_view = deepcopy(view)
                if (catalogue["reason"] == "already_changed" and len(slots) == 4
                        and all(slot["prior_index"] >= 0 for slot in slots)
                        and all(not slot["options"] for slot in slots)):
                    active_view = deepcopy(view)
                if active_view is not None and options_view is not None:
                    if active_view["observation"]["turn"] == options_view["observation"]["turn"]:
                        break
                    active_view = options_view = None
        if active_view is None or options_view is None:
            raise unittest.SkipTest("Retained v27 active assignment pair unavailable")
        cls.view = active_view
        cls.options_view = options_view

        capture_data = json.loads(capture.read_bytes())
        db = capture_data["database_copies"]["gameplay"]
        db_path = Path(db["path"])
        if sha256(db_path.read_bytes()).hexdigest() != db["sha256"]:
            raise RuntimeError("Pinned public gameplay database changed")
        cls.vocabulary = load_public_content_vocabulary(
            db_path, expected_sha256=db["sha256"],
            gameinfo_evidence_path=root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)
        with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
            cls.policy_hashes = {name: code & 0xffffffff for name, code in connection.execute(
                "SELECT PolicyType,Hash FROM Policies JOIN Types ON PolicyType=Type")}

    def synthetic_v28(self):
        from civ6_host.contracts import capability_policies, network_turn

        view = deepcopy(self.view)
        decisions = view["decisions"]
        observed = decisions["policies"]
        offered = self.options_view["decisions"]["policies"]["slots"]
        prior = [row["prior_index"] for row in observed["slots"]]
        slots = deepcopy(offered)
        for index, row in enumerate(slots):
            row["prior_index"] = prior[index]

        # Synthetic fixture: the retained v27 menus show the same options in
        # these compatible slots, while the later retained reply supplies the
        # actual active-card identities. Cross-slot offers are invented only
        # for this test and do not stand in for a native v28 receipt.
        catalogue = dict(player_id=view["observation"]["player_id"],
                         turn=view["observation"]["turn"], reason="available", slots=slots)
        decisions["schema"] = network_turn.CAPABILITY_XP2_POLICY_RELOCATION_SCHEMA
        decisions["policies"] = catalogue
        decisions["policy_unlock"] = dict(
            player_id=view["observation"]["player_id"], turn=view["observation"]["turn"],
            cost=0, currency="YIELD_GOLD", reason="no_cost", can_unlock=False)

        actions = capability_policies.actions(catalogue, allow_relocation=True)
        self.assertEqual(len(actions), 1)
        policy_rows = [dict(candidate_id=network_turn.digest(action), action=action)
                       for action in actions]
        decisions["candidates"] = [row for row in decisions["candidates"]
                                   if row["action"]["kind"] not in ("policies", "policy_unlock")]
        decisions["candidates"].extend(policy_rows)
        decisions["families"]["policies"] = dict(
            status="queried_complete", candidate_ids=[row["candidate_id"] for row in policy_rows])
        decisions["families"]["policy_unlock"] = dict(status="queried_complete", candidate_ids=[])
        decisions.pop("decision_id")
        decisions["decision_id"] = network_turn.digest(decisions)
        return view, slots, policy_rows[0]["candidate_id"]

    def _frame_and_swap(self):
        from civ6rl.learning.encoding import (
            ACTOR_V15, GLOBAL_FEATURES, MAP_CHANNELS, actor_feature_names, encode_response,
        )
        from civ6rl.learning.policy_assignments import valid_next_hashes, validate_joint_assignment

        view, slots, candidate_id = self.synthetic_v28()
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        entity_features, action_features = actor_feature_names(ACTOR_V15)
        self.assertEqual(frame["actor_schema"], ACTOR_V15)
        self.assertEqual(len(frame["candidates"][0]), len(action_features))
        self.assertEqual(len(frame["entities"][0]), len(entity_features))
        self.assertEqual(len(frame["global"]["values"]), len(GLOBAL_FEATURES))
        self.assertEqual(len(frame["map"]), len(MAP_CHANNELS))
        self.assertTrue(all(frame["action_mask"]))
        self.assertIn(candidate_id, frame["candidate_ids"])
        self.assertEqual(frame["policy_action_index"], frame["candidate_ids"].index(candidate_id))

        by_index = {}
        for row in slots:
            by_index.update({option["index"]: option["hash"] for option in row["options"]})
        swap = [by_index[index] for index in (3, 2, 1, 0)]
        self.assertIn(swap[0], valid_next_hashes(slots, []))
        self.assertIn(swap[1], valid_next_hashes(slots, [swap[0]]))
        self.assertIn(swap[2], valid_next_hashes(slots, swap[:2]))
        self.assertIn(swap[3], valid_next_hashes(slots, swap[:3]))
        self.assertEqual(validate_joint_assignment(slots, swap)["policy_indices"], [3, 2, 1, 0])

        prior = [by_index[index] for index in (2, 3, 0, 1)]
        with self.assertRaises(ValueError):
            validate_joint_assignment(slots, prior)
        with self.assertRaises(ValueError):
            validate_joint_assignment(slots, [swap[0], swap[0], swap[2], swap[3]])
        return frame, slots, swap

    def synthetic_v29_unchanged(self):
        from civ6_host.contracts import capability_policies, network_turn
        view = deepcopy(self.view)
        decisions = view["decisions"]
        observed = decisions["policies"]
        offered = self.options_view["decisions"]["policies"]["slots"]
        slots = deepcopy(offered)
        for i, row in enumerate(slots):
            row["prior_index"] = observed["slots"][i]["prior_index"]
            row["options"] = [o for o in row["options"] if o["index"] == row["prior_index"]]
            self.assertEqual(len(row["options"]), 1)
        catalogue = dict(player_id=view["observation"]["player_id"],
                         turn=view["observation"]["turn"], reason="available", slots=slots)
        decisions["schema"] = network_turn.CAPABILITY_XP2_POLICY_RECONFIRM_SCHEMA
        decisions["policies"] = catalogue
        decisions["policy_unlock"] = dict(player_id=view["observation"]["player_id"],
            turn=view["observation"]["turn"], cost=0, currency="YIELD_GOLD",
            reason="no_cost", can_unlock=False)
        actions = capability_policies.actions(catalogue, allow_relocation=True, allow_unchanged=True)
        self.assertEqual(len(actions), 1)
        rows = [dict(candidate_id=network_turn.digest(a), action=a) for a in actions]
        decisions["candidates"] = [r for r in decisions["candidates"]
                                    if r["action"]["kind"] not in ("policies", "policy_unlock")] + rows
        decisions["families"]["policies"] = dict(status="queried_complete",
                                                  candidate_ids=[r["candidate_id"] for r in rows])
        decisions["families"]["policy_unlock"] = dict(status="queried_complete", candidate_ids=[])
        decisions.pop("decision_id")
        decisions["decision_id"] = network_turn.digest(decisions)
        return view, slots, rows[0]["candidate_id"]

    def test_retained_v27_reply_projects_synthetic_swap_through_v28_masks(self):
        self._frame_and_swap()

    def test_v29_unchanged_only_menu_is_enabled_only_by_validated_v29(self):
        from civ6_host.contracts import network_turn
        from civ6rl.learning.encoding import ACTOR_V15, encode_response
        from civ6rl.learning.policy_assignments import valid_next_hashes, validate_joint_assignment
        view, slots, candidate_id = self.synthetic_v29_unchanged()
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertTrue(frame["policy_catalog"]["allow_unchanged"])
        unchanged = [row["options"][0]["hash"] for row in slots]
        self.assertEqual(valid_next_hashes(slots, [], allow_unchanged=True), [unchanged[0]])
        self.assertEqual(validate_joint_assignment(slots, unchanged, allow_unchanged=True)["policy_indices"],
                         [row["prior_index"] for row in slots])
        self.assertEqual(valid_next_hashes(slots, []), [])
        with self.assertRaises(ValueError):
            validate_joint_assignment(slots, unchanged)
        from civ6rl.learning.model import PolicyConfig, RecurrentPolicy, collate
        from civ6rl.learning.encoding import GLOBAL_FEATURES, MAP_CHANNELS, actor_feature_names
        entities, actions = actor_feature_names(ACTOR_V15)
        config = PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES), len(entities), len(actions),
                              self.vocabulary.vocabulary_size, hidden_size=32, spatial_size=8,
                              actor_schema=ACTOR_V15)
        model = RecurrentPolicy(config)
        batch = collate([frame])
        self.assertEqual(batch["policy_catalogs"][0]["allow_unchanged"], True)
        sampled, _, sample_entropy = model._decode_policies(torch.zeros(32), batch["policy_catalogs"][0])
        self.assertEqual(len(sampled), len(slots))
        self.assertTrue(torch.isfinite(sample_entropy))
        _, score, entropy = model._decode_policies(torch.zeros(32), batch["policy_catalogs"][0], unchanged)
        self.assertTrue(torch.isfinite(score) and torch.isfinite(entropy))
        self.assertIn(candidate_id, frame["candidate_ids"])
        action = torch.tensor([frame["policy_action_index"]], dtype=torch.long)
        log_probs, _, _, replay_entropy = model.evaluate_actions(batch, action, [unchanged])
        self.assertTrue(torch.isfinite(log_probs).all() and torch.isfinite(replay_entropy).all())
        from civ6rl.learning.action_receipt import selected_action_and_assignments
        selected_action_and_assignments(view, {"candidate_id": candidate_id, "assignments": unchanged})
        # The same menu under v28 remains changed-only.
        decisions = view["decisions"]
        decisions["schema"] = network_turn.CAPABILITY_XP2_POLICY_RELOCATION_SCHEMA
        decisions.pop("decision_id")
        decisions["decision_id"] = network_turn.digest(decisions)
        with self.assertRaises(ValueError):
            encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        with self.assertRaises(ValueError):
            selected_action_and_assignments(view, {"candidate_id": candidate_id, "assignments": unchanged})

    @unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
    def test_factorized_sampling_and_scoring_accept_relocation(self):
        from civ6rl.learning.encoding import (
            ACTOR_V15, GLOBAL_FEATURES, MAP_CHANNELS, actor_feature_names,
        )
        from civ6rl.learning.model import PolicyConfig, RecurrentPolicy
        from civ6rl.learning.policy_assignments import validate_joint_assignment

        frame, slots, swap = self._frame_and_swap()
        entity_features, action_features = actor_feature_names(ACTOR_V15)
        torch.manual_seed(29)
        config = PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES), len(entity_features),
                              len(action_features), self.vocabulary.vocabulary_size,
                              hidden_size=32, spatial_size=8, actor_schema=ACTOR_V15)
        model = RecurrentPolicy(config)
        catalog = frame["policy_catalog"]
        hidden = torch.zeros(32)
        sampled, _, _ = model._decode_policies(hidden, catalog)
        validate_joint_assignment(slots, sampled)
        scored, log_probability, entropy = model._decode_policies(hidden, catalog, swap)
        self.assertEqual(scored, swap)
        self.assertTrue(torch.isfinite(log_probability))
        self.assertTrue(torch.isfinite(entropy))


if __name__ == "__main__":
    unittest.main()
