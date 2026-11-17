"""Synthetic v31 envoy consumer fixture, using only the frozen v15 actor layout."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

from civ6_host.contracts import network_turn
from civ6_host.runtime.paths import resource_root
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import (
    ACTOR_V15, actor_feature_names, encode_response,
)


class LanEnvoyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = resource_root()
        replay = root / "work/bbg751-city-controls/native/public-replay.jsonl"
        capture = root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json"
        if not replay.is_file() or not capture.is_file():
            raise unittest.SkipTest("Retained public LAN source and pinned gameplay database required")
        if sha256(replay.read_bytes()).hexdigest() != "5220947cf214c1c5a9a897749701ffe35843b6241557dde12d6cdd8a01c48e65":
            raise ValueError("Retained v30 public exchange changed")
        cls.base = None
        with replay.open(encoding="utf-8") as stream:
            for line in stream:
                view = json.loads(line)["view"]
                if view["decisions"].get("schema") == network_turn.CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA:
                    cls.base = view
                    break
        if cls.base is None:
            raise ValueError("Retained public v30 catalogue unavailable")
        db = json.loads(capture.read_bytes())["database_copies"]["gameplay"]
        db_path = Path(db["path"])
        if sha256(db_path.read_bytes()).hexdigest() != db["sha256"]:
            raise RuntimeError("Pinned public gameplay database changed")
        cls.vocabulary = load_public_content_vocabulary(
            db_path, expected_sha256=db["sha256"],
            gameinfo_evidence_path=root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)

    def synthetic_v31(self):
        """Build an explicitly synthetic schema fixture, never a native receipt."""
        view = deepcopy(self.base)
        decisions = view["decisions"]
        decisions["schema"] = network_turn.CAPABILITY_XP2_ENVOYS_SCHEMA
        target_player_id = (decisions["player_id"] + 1) % 64
        decisions["envoys"] = {
            "player_id": decisions["player_id"],
            "current_game_turn": decisions["turn"],
            "tokens_to_give": 1,
            "can_give_influence": True,
            "considered": False,
            "giving_tokens_blocker_present": True,
            "targets": [{"target_player_id": target_player_id, "can_give": True,
                         "tokens_received": 1}],
        }
        action = {"kind": "send_envoy", "target_player_id": target_player_id}
        candidate_id = network_turn.digest(action)
        retain = {"kind": "retain_envoys"}
        retain_id = network_turn.digest(retain)
        decisions["candidates"].extend((
            {"candidate_id": candidate_id, "action": action},
            {"candidate_id": retain_id, "action": retain},
        ))
        decisions["families"]["send_envoy"] = {
            "status": "queried_complete", "candidate_ids": [candidate_id]}
        decisions["families"]["retain_envoys"] = {
            "status": "queried_complete", "candidate_ids": [retain_id]}
        decisions.pop("decision_id", None)
        decisions["decision_id"] = network_turn.digest(decisions)
        return view, action, candidate_id

    def test_v31_envoy_actions_use_existing_v15_action_and_target_inputs(self):
        view, action, candidate_id = self.synthetic_v31()
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertEqual(frame['candidate_ids'], [r['candidate_id'] for r in view['decisions']['candidates']])
        self.assertTrue(all(frame['action_mask']))
        self.assertEqual(frame["candidate_ids"].count(candidate_id), 1)
        self.assertEqual(len(frame["candidate_ids"]), len(view["decisions"]["candidates"]))
        _, features = actor_feature_names(ACTOR_V15)
        row = frame["candidate_ids"].index(candidate_id)
        self.assertEqual(frame["candidates"][row][features.index("kind:send_envoy")], 1.0)
        target_column = frame["action_categorical_fields"].index("target_player_id")
        self.assertEqual(frame["action_categorical"][row][target_column],
                         self.vocabulary.lookup("action.target_player_id", action["target_player_id"]))
        retain_id = network_turn.digest({"kind": "retain_envoys"})
        retain_row = frame["candidate_ids"].index(retain_id)
        self.assertEqual(frame["candidates"][retain_row][features.index("kind:retain_envoys")], 1.0)

    def test_v31_keeps_all_actor_dimensions_and_retains_fixture_public_bonus(self):
        view, _, _ = self.synthetic_v31()
        decisions = view["decisions"]
        self.assertGreaterEqual(decisions["envoys"]["targets"][0]["tokens_received"], 1)
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        entities, actions = actor_feature_names(ACTOR_V15)
        self.assertEqual(frame["entity_features"], list(entities))
        self.assertEqual(frame["action_features"], list(actions))
        self.assertEqual(len(frame["candidate_ids"]), len(decisions["candidates"]))

    def test_envoy_omission_or_extra_target_is_refused_by_the_full_catalogue(self):
        from civ6rl.learning import lan_inputs
        for omit in (True,False):
            view, _, _ = self.synthetic_v31()
            decisions=view['decisions']
            if omit:
                decisions['candidates']=[r for r in decisions['candidates']
                                         if r['action']['kind']!='send_envoy']
                decisions['families']['send_envoy']['candidate_ids']=[]
            else:
                action=dict(kind='send_envoy',target_player_id=33)
                candidate_id=network_turn.digest(action)
                decisions['candidates'].append(dict(candidate_id=candidate_id,action=action))
                decisions['families']['send_envoy']['candidate_ids'].append(candidate_id)
            decisions.pop('decision_id');decisions['decision_id']=network_turn.digest(decisions)
            with self.assertRaises(ValueError):
                lan_inputs.validate(view)


if __name__ == "__main__":
    unittest.main()
