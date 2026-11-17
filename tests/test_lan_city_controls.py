"""LAN actor joins for retained BBG v30 city-control input shapes.

The BBG view and XP2 specialist click are retained native evidence. The
catalogues added to the BBG view below are explicitly synthetic: they exercise
the v30 contract/actor join, not a BBG city-control execution.
"""
from copy import deepcopy
from hashlib import sha256
import json
import unittest

from civ6_host.contracts import network_turn, player_display
from civ6_host.runtime.paths import resource_root
from civ6rl.learning.content_vocabulary import load_public_content_vocabulary
from civ6rl.learning.encoding import (
    ACTOR_V14, ACTOR_V15, ENTITY_FEATURES_V14, ENTITY_FEATURES_V15,
    actor_feature_names, encode_response,
)


def public_display_fixture(raw):
    text = {}
    for name in ("science", "culture", "faith"):
        value = raw["observation"]["economy"][name]
        sign = "+" if name != "faith" and value > 0 else ""
        text[name] = sign + str(int(value * 10) / 10).replace(".", ",")
    return dict(raw, schema=player_display.PUBLIC_VIEW_SCHEMA, locale="fr_FR",
        observation=player_display.project_observation(raw["observation"], text),
        display_source_sha256=player_display.SOURCE_SHA256,
        display_scope=deepcopy(player_display.SCOPE), top_panel_text=text)


class LanCityControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = resource_root()
        proof_path = cls.root / "evidence/host-bbg751-v30-native-query.json"
        if not proof_path.is_file():
            raise unittest.SkipTest("Retained BBG v30 query receipt is required")
        proof_bytes = proof_path.read_bytes()
        if sha256(proof_bytes).hexdigest() != "9fb3f965b8a98a41e009c0c2ea08836900eaaa738e277c5bf03f671789e5af96":
            raise ValueError("Retained BBG v30 query receipt changed")
        proof = json.loads(proof_bytes)
        replay = cls.root / proof["consumer"]["public_replay"]["path"]
        replay_bytes = replay.read_bytes()
        if sha256(replay_bytes).hexdigest() != proof["consumer"]["public_replay"]["sha256"]:
            raise ValueError("Retained BBG v30 public replay changed")
        cls.raw_view = json.loads(replay_bytes.splitlines()[0])["view"]

        specialist_proof = json.loads((cls.root / "evidence/host-xp2-city-specialist.json").read_bytes())
        acceptance_ref = specialist_proof["acceptance"]
        acceptance_path = cls.root / acceptance_ref["path"]
        acceptance_bytes = acceptance_path.read_bytes()
        if sha256(acceptance_bytes).hexdigest() != acceptance_ref["sha256"]:
            raise ValueError("Retained XP2 specialist acceptance changed")
        preparation = json.loads(acceptance_bytes)["city_specialist_preparation"]
        cls.specialist_before = preparation["specialist_before"]
        cls.specialist_after = preparation["specialist_after"]
        if (set(cls.specialist_after) != {"plot_id", "x", "y", "occupied", "locked", "workers", "capacity"}
                or cls.specialist_after["occupied"] is not True
                or cls.specialist_after["locked"] is not True
                or cls.specialist_after["workers"] is not None
                or cls.specialist_after["capacity"] is not None):
            raise ValueError("Retained XP2 specialist click shape changed")

        capture_path = cls.root / "work/official-capture/xp2-bbg751-correlated-20261104/recovered-exact-inputs.json"
        capture = json.loads(capture_path.read_bytes())
        db = capture["database_copies"]["gameplay"]
        cls.vocabulary = load_public_content_vocabulary(
            db["path"], expected_sha256=db["sha256"],
            gameinfo_evidence_path=cls.root / "evidence/phase41-gameinfo-maps.json",
            include_action_identities=True)

    def synthetic_v30_controls(self):
        view = deepcopy(self.raw_view)
        decisions = view["decisions"]
        self.assertEqual(decisions["schema"], network_turn.CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA)
        observation = view["observation"]
        pid, turn = observation["player_id"], observation["turn"]
        plots = {row["index"]: row for row in observation["plots"] if row["revealed"] is True}
        # The host requires focus/citizen catalogues in exact native city-ID
        # order; geometry ordering belongs only to the actor entity projection.
        owned = sorted(observation["cities"], key=lambda row: row["id"])
        self.assertTrue(owned)
        candidates = []
        focus_ids, citizen_ids = [], []
        focus_catalogs, citizen_catalogs = [], []
        yields = ("YIELD_FOOD", "YIELD_PRODUCTION", "YIELD_GOLD", "YIELD_SCIENCE", "YIELD_CULTURE", "YIELD_FAITH")

        for city_index, city in enumerate(owned):
            focus_rows = []
            for yield_index, yield_type in enumerate(yields):
                favored = yield_index == 1
                disfavored = yield_index == 1
                focus_rows.append(dict(yield_type=yield_type, index=yield_index,
                                       favored=favored, disfavored=disfavored))
                action = dict(kind="city_focus", city_id=city["id"], yield_type=yield_type,
                              yield_index=yield_index, favored=favored, disfavored=disfavored)
                candidate_id = network_turn.digest(action)
                candidates.append(dict(candidate_id=candidate_id, action=action))
                focus_ids.append(candidate_id)
            focus_catalogs.append(dict(schema="civ6rl.native_city_focus1000.v1",
                player_id=pid, turn=turn, city_id=city["id"], yields=focus_rows))

            city_plot = city["y"] * observation["width"] + city["x"]
            if city_plot not in plots:
                city_plot = min(plots)
            map_plot = plots[city_plot]
            # This synthetic BBG row reuses the retained XP2 specialist click
            # fields, remapped to a revealed BBG plot. Numeric meter values
            # remain unknown as in the native specialist receipt.
            specialist = dict(plot_id=city_plot, x=map_plot["x"], y=map_plot["y"],
                              occupied=self.specialist_after["occupied"],
                              locked=self.specialist_after["locked"], workers=None, capacity=None)
            if city_index == 1:
                # Synthetic public PlotInfo meter branch (>1 citizen slots).
                specialist.update(workers=2, capacity=3)
            citizen_catalogs.append(dict(schema="civ6rl.native_city_citizens.v1",
                player_id=pid, turn=turn, city_id=city["id"], plots=[specialist]))
            action = dict(kind="city_citizen", city_id=city["id"], **specialist)
            candidate_id = network_turn.digest(action)
            candidates.append(dict(candidate_id=candidate_id, action=action))
            citizen_ids.append(candidate_id)

        decisions["city_focus_catalogs"] = focus_catalogs
        decisions["citizen_catalogs"] = citizen_catalogs
        decisions["candidates"].extend(candidates)
        decisions["families"]["city_focus"] = dict(status="queried_complete", candidate_ids=focus_ids)
        decisions["families"]["city_citizen"] = dict(status="queried_complete", candidate_ids=citizen_ids)
        decisions.pop("decision_id", None)
        decisions["decision_id"] = network_turn.digest(decisions)
        return view, candidates

    def assert_controls_reach_actor(self, view, added):
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertEqual(len(frame["candidate_ids"]), len(view["decisions"]["candidates"]))
        self.assertEqual(frame["candidate_ids"][-len(added):], [row["candidate_id"] for row in added])
        self.assertTrue(all(frame["action_mask"][-len(added):]))
        entity_names, action_names = actor_feature_names(ACTOR_V15)
        focus_row = next(i for i, row in enumerate(added) if row["action"]["kind"] == "city_focus"
                         and row["action"]["yield_type"] == "YIELD_PRODUCTION")
        focus_frame_row = len(frame["candidate_ids"]) - len(added) + focus_row
        for name in ("focus_favored", "focus_disfavored", "focus_known"):
            self.assertEqual(frame["candidates"][focus_frame_row][action_names.index(name)], 1.0)

        citizen_rows = [row for row in frame["entities"]
                        if row[entity_names.index("kind_citizen_location")] == 1.0]
        self.assertEqual(len(citizen_rows), len(view["decisions"]["citizen_catalogs"]))
        by_workers = {row[entity_names.index("citizen_workers_known")]: row for row in citizen_rows}
        unknown = by_workers[0.0]
        self.assertEqual(unknown[entity_names.index("citizen_capacity_known")], 0.0)
        self.assertEqual(unknown[entity_names.index("citizen_locked")], 1.0)
        self.assertEqual(unknown[entity_names.index("citizen_occupied")], 1.0)
        metered = by_workers[1.0]
        self.assertEqual(metered[entity_names.index("citizen_workers")], 2.0)
        self.assertEqual(metered[entity_names.index("citizen_capacity")], 3.0)
        self.assertEqual(metered[entity_names.index("citizen_capacity_known")], 1.0)
        self.assertEqual(len(actor_feature_names(ACTOR_V15)[0]), len(ENTITY_FEATURES_V15))
        self.assertEqual(len(actor_feature_names(ACTOR_V14)[0]) + 14, len(ENTITY_FEATURES_V15))

    def test_v30_city_controls_join_existing_v15_actor_features(self):
        view, added = self.synthetic_v30_controls()
        self.assertIs(network_turn.validate_capability_catalogue(
            view["decisions"], type("Identity", (), {"episode_id": view["episode_id"],
            "game_generation": view["game_generation"]})(), view["sequence"],
            view["observation"]["turn"], player_id=view["observation"]["player_id"],
            observation=view["observation"], actions_closed=False), view["decisions"])
        self.assert_controls_reach_actor(view, added)

    def test_public_display_v2_keeps_same_city_actions_and_known_masks(self):
        view, added = self.synthetic_v30_controls()
        public = public_display_fixture(view)
        player_display.validate_public(public)
        self.assert_controls_reach_actor(public, added)

    def test_v30_focus_only_request_keeps_focus_and_marks_citizens_unqualified(self):
        view, added = self.synthetic_v30_controls()
        decisions = view["decisions"]
        decisions["candidates"] = [row for row in decisions["candidates"]
                                    if row["action"]["kind"] != "city_citizen"]
        decisions["citizen_catalogs"] = None
        decisions["families"]["city_citizen"] = dict(
            status="unqualified", reason="not selected in the native city-control request")
        decisions.pop("decision_id", None)
        decisions["decision_id"] = network_turn.digest(decisions)
        focus_actions = [row for row in added if row["action"]["kind"] == "city_focus"]
        identity = type("Identity", (), {"episode_id": view["episode_id"],
            "game_generation": view["game_generation"]})()
        self.assertIs(network_turn.validate_capability_catalogue(
            decisions, identity, view["sequence"], view["observation"]["turn"],
            player_id=view["observation"]["player_id"], observation=view["observation"],
            actions_closed=False), decisions)
        frame = encode_response(view, vocabulary=self.vocabulary, actor_schema=ACTOR_V15)
        self.assertEqual(frame["candidate_ids"][-len(focus_actions):],
                         [row["candidate_id"] for row in focus_actions])
        self.assertTrue(all(frame["action_mask"][-len(focus_actions):]))
        entity_names, action_names = actor_feature_names(ACTOR_V15)
        production = next(i for i, row in enumerate(focus_actions)
                          if row["action"]["yield_type"] == "YIELD_PRODUCTION")
        frame_row = len(frame["candidate_ids"]) - len(focus_actions) + production
        self.assertEqual(frame["candidates"][frame_row][action_names.index("focus_known")], 1.0)
        self.assertFalse(any(row[entity_names.index("kind_citizen_location")] == 1.0
                             for row in frame["entities"]))
        from civ6rl.learning import lan_inputs
        projection = lan_inputs.city_controls_for_actor(decisions)
        self.assertEqual(projection["schema"], "civ6rl.saved_decisions1000.v1")
        self.assertNotIn("citizen_catalogs", projection)

    def test_city_projection_does_not_relabel_or_mutate_live_lan_decisions(self):
        from civ6rl.learning import lan_inputs
        view, _ = self.synthetic_v30_controls()
        live = view["decisions"]
        schema = live["schema"]
        projected = lan_inputs.city_controls_for_actor(live)
        self.assertEqual(live["schema"], schema)
        self.assertEqual(schema, network_turn.CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA)
        self.assertEqual(projected["schema"], "civ6rl.saved_decisions.city_management.v1")
        self.assertIs(projected["candidates"], live["candidates"])
        self.assertIs(projected["city_focus_catalogs"], live["city_focus_catalogs"])
        self.assertIs(projected["citizen_catalogs"], live["citizen_catalogs"])


if __name__ == "__main__":
    unittest.main()
