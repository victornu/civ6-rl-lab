"""Conservative Python-list encoding for the native diagnostic session.

This is a transport prototype, not a claim that the current native action
space is complete. Features omit ephemeral engine IDs and opaque control tokens;
stable published content categories use the caller's frozen vocabulary.
"""
from __future__ import annotations

import math
import hashlib
import json
import re
import copy

from .policy_assignments import valid_next_hashes
from .trade_choices984 import TRADE_FEATURES, validate_and_index as validate_trade_choices
from .unit_work985 import ACTION_WORK_FEATURES, validate_and_index as validate_unit_work
from .diplomacy987 import SCHEMA as _DIPLOMACY_SCHEMA_987, validate_and_project as validate_diplomacy
from .production_info988 import (ACTION_PRODUCTION_FEATURES, PRODUCTION_KINDS,
    public_features as production_public_features, validate_and_index as validate_production_info)
from .plot_display990 import (MAP_DISPLAY_CHANNELS, MAP_CATEGORICAL_FIELDS,
    numeric_features as plot_display_features, validate_and_index as validate_plot_display)


# Channels are [channel][y][x]. Unknown plot values have a zero value and a
# zero *_known channel; actual zero values have *_known == 1.
MAP_CHANNELS = (
    "x_norm", "y_norm", "odd_row", "visible", "revealed",
    "terrain", "terrain_known", "feature", "feature_known",
    "owner_self", "owner_other", "owner_known", "water", "water_known",
    "impassable", "impassable_known", "resource_type", "resource_known",
    "resource_count", "resource_count_known",
) + MAP_DISPLAY_CHANNELS
GLOBAL_FEATURES = (
    "turn", "science", "culture", "gold", "faith",
    "technology_cost", "technology_cost_known",
    "technology_progress", "technology_progress_known",
    "technology_remaining", "technology_remaining_known",
    "civic_cost", "civic_cost_known",
    "civic_progress", "civic_progress_known",
    "civic_remaining", "civic_remaining_known",
    "gold_yield", "gold_yield_known",
    "total_maintenance", "total_maintenance_known",
    "net_gold_yield", "net_gold_yield_known",
    "faith_yield", "faith_yield_known",
    "technology_turns_left", "technology_turns_left_known",
    "technology_boosted", "technology_boosted_known",
    "civic_turns_left", "civic_turns_left_known",
    "civic_boosted", "civic_boosted_known",
)
GLOBAL_CATEGORICAL_FIELDS = ("research", "civic")
ENTITY_FEATURES = (
    "kind_unit", "kind_city", "kind_visible_enemy", "kind_visible_enemy_city", "x_norm", "y_norm",
    "damage", "damage_known", "moves", "moves_known", "max_moves",
    "max_moves_known", "population", "population_known", "food_yield",
    "food_yield_known", "production_yield", "production_yield_known",
    "gold_yield", "gold_yield_known", "science_yield", "science_yield_known",
    "culture_yield", "culture_yield_known", "faith_yield", "faith_yield_known",
    "attacks_remaining", "attacks_remaining_known",
    "production_native_yield", "production_native_yield_known",
    "production_cost", "production_cost_known", "production_progress", "production_progress_known",
    "production_turns_left", "production_turns_left_known", "production_queue_size",
    "production_queue_size_known", "production_has_progress", "production_has_progress_known",
    "garrison_damage", "garrison_damage_known", "garrison_max", "garrison_max_known",
    "outer_damage", "outer_damage_known", "outer_max", "outer_max_known",
    "defense_strength", "defense_strength_known",
)

# Fixed action vocabulary. Actions outside this explicitly limited first
# transport are refused rather than collapsed into a generic bucket.
_ACTION_KINDS = (
    "research", "civic", "production", "found_city", "skip_unit", "end_turn",
    "force_end_turn", "move", "unit_attack", "city_range_attack",
    "district_range_attack", "declare_war_minor", "government", "policies",
    "retain_government", "pantheon", "purchase", "promote_unit",
    "unit_order", "upgrade_unit", "review_envoys", "recruit_great_person",
    "resolve_captured_city", "build_improvement", "terrain_operation",
    "production_placement", "production_resume", "make_trade_route",
    "change_trade_origin",
    "queue_append", "queue_remove", "queue_swap", "send_envoy", "retain_envoys",
    "activate_prophet", "found_religion", "add_belief", "diplomacy_response",
)
ACTION_FEATURES = tuple("kind:" + kind for kind in _ACTION_KINDS) + (
    "source_x_norm", "source_y_norm", "target_x_norm", "target_y_norm",
    "source_xy_known", "target_xy_known",
    "cost", "cost_known",
    "research_turns_left", "research_turns_left_known",
    "research_boosted", "research_boosted_known",
) + TRADE_FEATURES + ACTION_WORK_FEATURES + ACTION_PRODUCTION_FEATURES
ACTION_CATEGORICAL_FIELDS = (
    "hash", "index", "item_type", "category", "mode", "operation", "input_route",
    "directive", "class", "route", "currency", "target_player_id", "yield_type",
)
ENTITY_CATEGORICAL_FIELDS = ("type", "production_kind", "production_hash", "production_index")
# Historical aliases remain v6 so old fixtures/artifacts retain their exact layout.
from .city_focus1000 import (V6 as ACTOR_V6, V7 as ACTOR_V7, DECISIONS_SCHEMA as _DECISIONS_SCHEMA_1000,
    ENTITY_FEATURES as FOCUS_ENTITY_FEATURES, ACTION_FEATURES as FOCUS_ACTION_FEATURES,
    validate_and_index as validate_city_focus, entity_features as focus_entity_features)
ENTITY_FEATURES_V7 = ENTITY_FEATURES + FOCUS_ENTITY_FEATURES
ACTION_FEATURES_V7 = (tuple("kind:" + kind for kind in _ACTION_KINDS + ("city_focus",))
                      + ACTION_FEATURES[len(_ACTION_KINDS):] + FOCUS_ACTION_FEATURES)
from .city_management import (ACTOR_SCHEMA as ACTOR_V8, DECISIONS_SCHEMA as _CITY_MANAGEMENT_SCHEMA,
    ACTION_KIND as CITIZEN_ACTION_KIND, ENTITY_FEATURES as CITIZEN_ENTITY_FEATURES,
    ACTION_FEATURES as CITIZEN_ACTION_FEATURES, validate_and_index as validate_city_management,
    location_features as citizen_location_features)
ENTITY_FEATURES_V8 = ENTITY_FEATURES_V7 + CITIZEN_ENTITY_FEATURES
ACTION_FEATURES_V8 = (tuple("kind:" + kind for kind in _ACTION_KINDS + ("city_focus", CITIZEN_ACTION_KIND))
                      + ACTION_FEATURES[len(_ACTION_KINDS):] + FOCUS_ACTION_FEATURES + CITIZEN_ACTION_FEATURES)

from .production_queue import (ACTOR_SCHEMA as ACTOR_V9, ENTITY_FEATURES as QUEUE_ENTITY_FEATURES,
    ACTION_FEATURES as QUEUE_ACTION_FEATURES, ACTION_KINDS as QUEUE_ACTION_KINDS,
    validate_and_index as validate_production_queues, entry_features as queue_entry_features)
ENTITY_FEATURES_V9 = ENTITY_FEATURES_V8 + QUEUE_ENTITY_FEATURES
ACTION_FEATURES_V9 = ACTION_FEATURES_V8 + QUEUE_ACTION_FEATURES
from .action_bindings import (DISTRICT_ENTITY_FEATURES, build_source_bindings,
    resolve_source_rows, district_features)
from .action_content import project_action_content
ACTOR_V10 = 'civ6rl.recurrent_policy.v10'
ENTITY_FEATURES_V10 = ENTITY_FEATURES_V9 + DISTRICT_ENTITY_FEATURES
ACTION_FEATURES_V10 = ACTION_FEATURES_V9 + ('source_entity_bound',)
ACTION_CATEGORICAL_FIELDS_V10 = ACTION_CATEGORICAL_FIELDS + ('target_type', 'individual')
from .movement_information import ENTITY_FEATURES as MOVEMENT_ENTITY_FEATURES, ACTION_FEATURES as MOVEMENT_ACTION_FEATURES
ACTOR_V11 = 'civ6rl.recurrent_policy.v11'
ENTITY_FEATURES_V11 = ENTITY_FEATURES_V10 + MOVEMENT_ENTITY_FEATURES
ACTION_FEATURES_V11 = ACTION_FEATURES_V10 + MOVEMENT_ACTION_FEATURES
from .world_input import ENTITY_FEATURES as WORLD_ENTITY_FEATURES, ACTION_KINDS as WORLD_ACTION_KINDS, ACTION_FEATURES as WORLD_ACTION_FEATURES
ACTOR_V12 = 'civ6rl.recurrent_policy.v12'
ENTITY_FEATURES_V12 = ENTITY_FEATURES_V11 + WORLD_ENTITY_FEATURES
_ACTION_PREFIX_V11 = len(_ACTION_KINDS) + 1 + 1
ACTION_FEATURES_V12 = (tuple('kind:' + kind for kind in _ACTION_KINDS + ('city_focus', CITIZEN_ACTION_KIND) + WORLD_ACTION_KINDS)
                       + ACTION_FEATURES_V11[_ACTION_PREFIX_V11:] + WORLD_ACTION_FEATURES)
# V13 changes only the versioned observation contract for official war-dialogue
# text. It deliberately retains every V12 tensor width and named feature.
ACTOR_V13 = 'civ6rl.recurrent_policy.v13'
ENTITY_FEATURES_V13 = ENTITY_FEATURES_V12
ACTION_FEATURES_V13 = ACTION_FEATURES_V12
# V14 retains the V13 layout while reserving unqualified resource quantities.
ACTOR_V14 = 'civ6rl.recurrent_policy.v14'
ENTITY_FEATURES_V14 = ENTITY_FEATURES_V13
ACTION_FEATURES_V14 = ACTION_FEATURES_V13
from .public_purchases import (ENTITY_FEATURES as PURCHASE_ENTITY_FEATURES,
    CAT_FIELDS as PURCHASE_CATEGORICAL_FIELDS, CONTRACT as PURCHASE_CONTRACT,
    validate_and_index as validate_public_purchases,
    entity_features as purchase_entity_features, entity_categories as purchase_entity_categories)
ACTOR_V15 = 'civ6rl.recurrent_policy.v15'
ENTITY_FEATURES_V15 = ENTITY_FEATURES_V14 + PURCHASE_ENTITY_FEATURES
ACTION_FEATURES_V15 = ACTION_FEATURES_V14
ENTITY_CATEGORICAL_FIELDS_V15 = ENTITY_CATEGORICAL_FIELDS + PURCHASE_CATEGORICAL_FIELDS


def entity_categorical_fields(actor_schema):
    actor_feature_names(actor_schema)
    return ENTITY_CATEGORICAL_FIELDS_V15 if actor_schema == ACTOR_V15 else ENTITY_CATEGORICAL_FIELDS


def action_categorical_fields(actor_schema):
    actor_feature_names(actor_schema)
    return ACTION_CATEGORICAL_FIELDS_V10 if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) else ACTION_CATEGORICAL_FIELDS


def actor_feature_names(actor_schema):
    if actor_schema == ACTOR_V6: return ENTITY_FEATURES, ACTION_FEATURES
    if actor_schema == ACTOR_V7: return ENTITY_FEATURES_V7, ACTION_FEATURES_V7
    if actor_schema == ACTOR_V8: return ENTITY_FEATURES_V8, ACTION_FEATURES_V8
    if actor_schema == ACTOR_V9: return ENTITY_FEATURES_V9, ACTION_FEATURES_V9
    if actor_schema == ACTOR_V10: return ENTITY_FEATURES_V10, ACTION_FEATURES_V10
    if actor_schema == ACTOR_V11: return ENTITY_FEATURES_V11, ACTION_FEATURES_V11
    if actor_schema == ACTOR_V12: return ENTITY_FEATURES_V12, ACTION_FEATURES_V12
    if actor_schema == ACTOR_V13: return ENTITY_FEATURES_V13, ACTION_FEATURES_V13
    if actor_schema == ACTOR_V14: return ENTITY_FEATURES_V14, ACTION_FEATURES_V14
    if actor_schema == ACTOR_V15: return ENTITY_FEATURES_V15, ACTION_FEATURES_V15
    raise ValueError("unsupported actor feature schema")


CONTENT_VOCABULARY_VERSION = "civ6rl-content-vocabulary-v1"

_PLAYER_OBS_SCHEMA = "civ6rl-player-observation-v1"
_MAP_SCHEMA = "civ6rl.saved_player_map933.v1"
_DECISIONS_SCHEMA = "civ6rl.saved_decisions968.v1"
_DECISIONS_SCHEMA_972 = "civ6rl.saved_decisions972.v1"
_DECISIONS_SCHEMA_980 = "civ6rl.saved_decisions980.v1"
_DECISIONS_SCHEMA_982 = "civ6rl.saved_decisions982.v1"
_DECISIONS_SCHEMA_983 = "civ6rl.saved_decisions983.v1"
_DECISIONS_SCHEMA_988 = "civ6rl.saved_decisions988.v1"
_DECISIONS_SCHEMA_989 = "civ6rl.saved_decisions989.v1"
_DECISIONS_SCHEMA_990 = "civ6rl.saved_decisions990.v1"
_DECISIONS_SCHEMA_999 = "civ6rl.saved_decisions999.v1"
_RESEARCH_PROGRESS_SCHEMA = "civ6rl.native_saved_research_progress971.v1"
_TREASURY_SCHEMA = "civ6rl.native_saved_treasury980.v1"
_TREASURY_DISPLAY_SCHEMA = "civ6rl.native_treasury_display980.v1"
_RESPONSE_SCHEMA = "civ6rl.action_session920.v1"
_RESPONSE_SCHEMAS = (_RESPONSE_SCHEMA, "civ6rl.hotseat_action_session923.v1")
_POLICY_REASONS = {"available", "no_policy_slots", "already_changed", "not_free", "window_closed"}
_RESPONSE_KEYS = {
    "stage", "schema", "nonce", "episode_id", "game_generation", "sequence",
    "operation", "observation", "rounds", "outcome", "automatic_choices_applied",
    "action_interface_complete", "strategic_training_ready", "decisions", "applied_action",
}
_OBS_KEYS = {"schema", "player_id", "turn", "width", "height", "plots", "units",
             "visible_enemy_units", "cities", "economy"}
_MAP_KEYS = {"schema", "player_id", "turn", "width", "height", "plots",
             "visible_observation_id", "source", "historical_memory_proven",
             "graphical_differential_proven", "complete_player_observation"}
_MAP_SOURCE = "official_plot_tooltip_revealed_live_getters"


def _token(namespace: str, value) -> str:
    if type(value) not in (str, int) or (type(value) is str and not value):
        raise ValueError(f"invalid public categorical value for {namespace}")
    return namespace + ":" + json.dumps(value, ensure_ascii=True, separators=(",", ":"))


class ContentVocabulary:
    """Frozen, deterministic namespace for published content categories.

    Fit only on the public action catalogue and player-scoped entity roster
    that the caller admits for this run. ID zero represents an absent field.
    """

    version = CONTENT_VOCABULARY_VERSION

    def __init__(self, tokens, *, manifest=None):
        if type(tokens) not in (list, tuple) or any(type(t) is not str for t in tokens):
            raise ValueError("vocabulary tokens must be strings")
        canonical = sorted(set(tokens))
        if list(tokens) != canonical:
            raise ValueError("vocabulary tokens must be unique and sorted")
        self.tokens = tuple(canonical)
        self._ids = {token: index + 1 for index, token in enumerate(self.tokens)}
        if manifest is not None and type(manifest) is not dict:
            raise ValueError("vocabulary manifest must be a dictionary")
        self.manifest = copy.deepcopy(manifest)

    @classmethod
    def fit(cls, response):
        if type(response) is not dict:
            raise ValueError("fit requires a native response")
        obs, decisions = response.get("observation"), response.get("decisions")
        if type(obs) is not dict or type(decisions) is not dict:
            raise ValueError("fit requires a player observation and action catalogue")
        if (obs.get("schema") != _PLAYER_OBS_SCHEMA or decisions.get("schema") not in
                (_DECISIONS_SCHEMA, _DECISIONS_SCHEMA_972, _DECISIONS_SCHEMA_980,
                 _DECISIONS_SCHEMA_982, _DECISIONS_SCHEMA_983, _DIPLOMACY_SCHEMA_987, _DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA)):
            raise ValueError("fit accepts only public native player/action schemas")
        tokens = set()
        candidates = decisions.get("candidates")
        if type(candidates) is not list:
            raise ValueError("catalogue candidates must be a list")
        for row in candidates:
            if type(row) is not dict or type(row.get("action")) is not dict:
                raise ValueError("malformed public candidate")
            action = row["action"]
            for field in ACTION_CATEGORICAL_FIELDS:
                if field in action:
                    tokens.add(_token("action." + field, action[field]))
        for roster_name in ("units", "cities", "visible_enemy_units"):
            roster = obs.get(roster_name)
            if type(roster) is not list:
                raise ValueError("malformed public entity roster")
            for entity in roster:
                if type(entity) is not dict:
                    raise ValueError("malformed public entity")
                if "type" in entity:
                    tokens.add(_token("entity.type", entity["type"]))
        production = decisions.get("production")
        if type(production) is dict and type(production.get("cities")) is list:
            for city in production["cities"]:
                if type(city) is not dict:
                    raise ValueError("malformed public production observation")
                kind_map = {"U": "unit", "B": "building", "D": "district", "P": "project"}
                if city.get("kind") in kind_map:
                    tokens.add(_token("action.category", kind_map[city["kind"]]))
                if city.get("current_production_hash") is not None:
                    tokens.add(_token("action.hash", city["current_production_hash"] & 0xffffffff))
                if city.get("item_index") is not None:
                    tokens.add(_token("action.index", city["item_index"]))
        queues = decisions.get('production_queues')
        if type(queues) is list:
            for queue in queues:
                if type(queue) is not dict or type(queue.get('entries')) is not list:
                    raise ValueError('malformed public queue vocabulary source')
                for entry in queue['entries']:
                    if type(entry) is not dict: raise ValueError('malformed public queue entry')
                    for name in ('category', 'hash', 'index'):
                        tokens.add(_token('action.' + name, entry[name]))
        purchases = decisions.get('purchases')
        if type(purchases) is list:
            for catalog in purchases:
                if type(catalog) is not dict or type(catalog.get('offers')) is not list:
                    raise ValueError('malformed public purchase vocabulary source')
                for offer in catalog['offers']:
                    if type(offer) is not dict or type(offer.get('action')) is not dict:
                        raise ValueError('malformed public purchase vocabulary row')
                    action = offer['action']
                    for name in ('category', 'item_type', 'currency'):
                        if name in action:
                            tokens.add(_token('action.' + name, action[name]))
        research = decisions.get("research")
        if type(research) is dict:
            for field in GLOBAL_CATEGORICAL_FIELDS:
                if research.get(field) is not None:
                    tokens.add(_token("action.index", research[field]))
        policy_catalog = decisions.get("policies")
        policy_slots = []
        if policy_catalog is not None:
            _require_policy_catalog(policy_catalog, obs.get("player_id"), obs.get("turn"))
            policy_slots = policy_catalog["slots"]
        # Include stable policy identities from the public catalogue as well
        # as candidate copies. Native hashes are validation-only values.
        policy_slot_sets = [policy_slots]
        for row in candidates:
            action = row["action"]
            if action.get("kind") == "policies":
                slots = action.get("slots")
                valid_next_hashes(slots, [])
                policy_slot_sets.append(slots)
        for slots in policy_slot_sets:
            for row in slots:
                if row["prior_index"] >= 0:
                    tokens.add(_token("policy.index", row["prior_index"]))
                for option in row["options"]:
                    tokens.update((
                        _token("policy.index", option["index"]),
                        _token("policy.hash", option["hash"]),
                        _token("policy.type", option["policy_type"]),
                    ))
        if decisions["schema"] in (_DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
            rows = validate_plot_display(decisions.get("plot_info"), obs.get("player_id"),
                                         obs.get("turn"), decisions.get("player_map"))
            for row in rows.values():
                for field in ("improvement", "district"):
                    if row[field] is not None:
                        tokens.add(_token("action.item_type", row[field]["type"]))
        elif "plot_info" in decisions:
            raise ValueError("pre-990 decisions schema cannot carry plot display")
        return cls(sorted(tokens))

    @classmethod
    def fit_many(cls, responses, *, base=None):
        """Fit dynamic public tokens across a declared response corpus.

        A database-seeded `base` is retained, including its provenance
        manifest. This is intended for runtime-only fields; the full public
        content vocabulary should come from `content_vocabulary.py`.
        """
        if type(responses) not in (list, tuple) or not responses:
            raise ValueError("fit_many requires a non-empty bounded response sequence")
        if len(responses) > 100_000:
            raise ValueError("fit_many response bound exceeded")
        if base is not None and type(base) is not cls:
            raise ValueError("fit_many base must be a ContentVocabulary")
        tokens = set(base.tokens) if base is not None else set()
        for response in responses:
            tokens.update(cls.fit(response).tokens)
            if len(tokens) > 1_000_000:
                raise ValueError("fit_many vocabulary token bound exceeded")
        manifest = None
        if base is not None:
            manifest = copy.deepcopy(base.manifest)
        elif len(responses) > 1:
            manifest = {"schema": "civ6rl.learning.content-vocabulary-fit-many.v1",
                        "responses": len(responses), "source": "public response catalogues"}
        return cls(sorted(tokens), manifest=manifest)

    @property
    def vocabulary_size(self):
        return len(self.tokens) + 1

    @property
    def fingerprint(self):
        return hashlib.sha256(self.serialize().encode("utf-8")).hexdigest()

    def lookup(self, namespace, value):
        token = _token(namespace, value)
        if token not in self._ids:
            raise ValueError(f"unknown frozen vocabulary token: {token}")
        return self._ids[token]

    def serialize(self):
        payload = {"version": self.version, "tokens": list(self.tokens)}
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        return json.dumps(payload,
                          sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_serialized(cls, serialized):
        value = json.loads(serialized)
        if (type(value) is not dict or set(value) not in ({"version", "tokens"},
                {"version", "tokens", "manifest"}) or value["version"] != cls.version):
            raise ValueError("unsupported serialized vocabulary")
        return cls(value["tokens"], manifest=value.get("manifest"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_policy_catalog(value: dict, player_id: int, turn: int) -> list[dict]:
    _require(type(value) is dict and set(value) == {"player_id", "turn", "reason", "slots"}
             and type(value.get("player_id")) is int and value["player_id"] == player_id
             and type(value.get("turn")) is int and value["turn"] == turn
             and value.get("reason") in _POLICY_REASONS and type(value.get("slots")) is list,
             "malformed public policy catalogue")
    slots = value["slots"]
    # The matching helper owns the strict native slot/option shape validation.
    valid_next_hashes(slots, [])
    _require((value["reason"] == "no_policy_slots") == (len(slots) == 0)
             and (value["reason"] == "available" or all(not row["options"] for row in slots)),
             "policy catalogue reason and menus disagree")
    return slots


def _finite(value, field: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value), f"{field} must be finite")
    return float(value)


def _display_number(value, field: str) -> float:
    from .yield_display982 import public_number
    try:
        return public_number(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a canonical public display number") from exc


def _norm_coord(value: int, extent: int) -> float:
    return 0.0 if extent <= 1 else value / (extent - 1)


def _native_purchase_price(token) -> float:
    """Mirror native_saved_purchases.price, including grammar and bounds."""
    if (type(token) is not str or len(token) > 32
            or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?", token)):
        raise ValueError("Malformed native purchase quote")
    amount = float(token)
    if not math.isfinite(amount) or not 0 <= amount <= 2**53:
        raise ValueError("Unbounded native purchase quote")
    return amount


def _plot_values(row: dict, channels: list[list[list[float]]], x: int, y: int,
                 width: int, height: int, player_id: int, *, visible: bool, revealed: bool,
                 actor_schema: str) -> None:
    at = lambda name: channels[MAP_CHANNELS.index(name)]
    at("x_norm")[y][x] = _norm_coord(x, width)
    at("y_norm")[y][x] = _norm_coord(y, height)
    # Explicit row parity preserves the offset-hex coordinate convention;
    # no rectangular-grid adjacency is inferred here.
    at("odd_row")[y][x] = float(y & 1)
    at("visible")[y][x] = float(visible)
    at("revealed")[y][x] = float(revealed)
    if not revealed:
        return
    for key, value_channel, mask_channel in (
        ("terrain_type", "terrain", "terrain_known"),
        ("feature_type", "feature", "feature_known"),
    ):
        if key in row:
            at(value_channel)[y][x] = _finite(row[key], key)
            at(mask_channel)[y][x] = 1.0
    if "owner" in row:
        owner = row["owner"]
        _require(type(owner) is int, "plot.owner must be integer")
        at("owner_self")[y][x] = float(owner == player_id)
        at("owner_other")[y][x] = float(owner >= 0 and owner != player_id)
        at("owner_known")[y][x] = 1.0
    for key, value_channel in (("water", "water"), ("impassable", "impassable")):
        if key in row:
            _require(type(row[key]) is bool, f"plot.{key} must be Boolean")
            at(value_channel)[y][x] = float(row[key])
            at(value_channel + "_known")[y][x] = 1.0
    if "resource_type" in row:
        at("resource_type")[y][x] = _finite(row["resource_type"], "resource_type")
        at("resource_known")[y][x] = 1.0
        quantity = _finite(row["resource_count"], "resource_count")
        if actor_schema not in (ACTOR_V14, ACTOR_V15):
            at("resource_count")[y][x] = quantity
            at("resource_count_known")[y][x] = 1.0


def _validate_player_map(player_map: dict, obs: dict) -> None:
    _require(type(player_map) is dict and set(player_map) == _MAP_KEYS, "malformed official player map")
    _require(player_map["schema"] == _MAP_SCHEMA and player_map["source"] == _MAP_SOURCE,
             "unsupported player-map source")
    _require(player_map["player_id"] == obs["player_id"] and player_map["turn"] == obs["turn"]
             and player_map["width"] == obs["width"] and player_map["height"] == obs["height"],
             "player map and observation boundaries differ")
    _require(player_map["historical_memory_proven"] is False
             and player_map["graphical_differential_proven"] is False
             and player_map["complete_player_observation"] is False,
             "player map claims unsupported completeness")
    pin = player_map["visible_observation_id"]
    _require(type(pin) is str and len(pin) == 64 and all(c in "0123456789abcdef" for c in pin),
             "invalid visible observation identity")


def _action_vector(action: dict, obs: dict, id_coords: dict, *, research_choice=None,
                   trade_choice=None, work_choice=None, diplomacy_choice=False,
                   production_choice=None, focus_choice=None, citizen_choice=None, queue_choice=None,
                   source_bound=False, movement_choice=None, actor_schema=ACTOR_V6) -> list[float]:
    kind = action.get("kind")
    if production_choice is not None:
        _require(kind in PRODUCTION_KINDS and "cost" not in action,
                 "production display must use its validated public quote, not generic action cost")
    action_kinds = (_ACTION_KINDS + (("city_focus",) if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) else ())
                    + ((CITIZEN_ACTION_KIND,) if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) else ())
                    + (WORLD_ACTION_KINDS if actor_schema in (ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) else ()))
    _, feature_names = actor_feature_names(actor_schema)
    _require(kind in action_kinds, f"unsupported candidate action kind: {kind!r}")
    if kind == "policies":
        _require(set(action) == {"kind", "slots"} and type(action["slots"]) is list,
                 "policy candidate must carry only its native slot assignment catalogue")
    _require((kind == "purchase") == ("price_token" in action),
             "purchase action must carry exactly one published price token")
    identity_fields = {"unit_id", "city_id", "origin_city_id", "destination_city_id",
                       "moves_token", "price_token"}
    if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and kind == 'district_range_attack':
        identity_fields.add('district_id')
    if kind == "city_focus":
        _require(type(focus_choice) is list and len(focus_choice) == len(FOCUS_ACTION_FEATURES),
                 "focus action lacks an exact public menu join")
        identity_fields.update({"yield_index", "favored", "disfavored"})
    if kind == CITIZEN_ACTION_KIND:
        _require(type(citizen_choice) is dict and set(citizen_choice) == {'source', 'target', 'features'}
                 and len(citizen_choice['features']) == len(CITIZEN_ACTION_FEATURES),
                 'Citizen action lacks its exact public native target join')
        identity_fields.update({'occupied', 'locked', 'workers', 'capacity'})
    if kind == "diplomacy_response":
        _require(diplomacy_choice is True and set(action) == {
            "kind", "session_id", "statement_sequence", "choice"},
            "diplomacy action lacks an exact validated public menu join")
        identity_fields.update({"session_id", "statement_sequence", "choice"})
    semantic_fields = {"origin_owner", "origin_city",
                       "destination_owner", "destination_city", "position", "source_position",
                       "destination_position", "charges", "selector"}
    semantic = set(action) & semantic_fields
    if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and kind in QUEUE_ACTION_KINDS:
        _require(type(queue_choice) is list and len(queue_choice) == len(QUEUE_ACTION_FEATURES),
                 'Queue action lacks its exact public native catalog join')
        semantic -= {'position', 'source_position', 'destination_position'}
        identity_fields.update({'position', 'source_position', 'destination_position'})
    permitted_trade_semantics = {
        "make_trade_route": {"origin_owner", "origin_city", "destination_owner", "destination_city"},
        "change_trade_origin": {"origin_city", "destination_city"},
    }.get(kind, set())
    semantic -= permitted_trade_semantics
    if kind in ("build_improvement", "terrain_operation"):
        semantic -= {"charges", "selector"}
    else:
        semantic -= {"charges", "selector"} & semantic
    if semantic:
        raise NotImplementedError(f"action semantics need features: {sorted(semantic)}")
    if kind in ("make_trade_route", "change_trade_origin"):
        _require(type(trade_choice) is dict and set(trade_choice) == {"source", "target", "features"},
                 "trade action lacks a validated native choice join")
    if kind in ("build_improvement", "terrain_operation"):
        _require(type(work_choice) is dict and set(work_choice) == {"source", "target", "features"},
                 "unit-work action lacks a validated native catalogue join")
    elif "charges" in action or "selector" in action:
        raise NotImplementedError("unit-work-only action fields on a non-work action")
    allowed = {"kind", "cost", "source_x", "source_y", "destination_x", "destination_y",
               "target_x", "target_y", "x", "y", "x0", "y0", "x1", "y1",
               "source_plot_id", "destination_plot_id", "plot_id", "candidate_id"} | identity_fields
    if actor_schema in (ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and kind == 'world_war_response':
        allowed.add('accept')
    allowed.update(action_categorical_fields(actor_schema))
    allowed.update(permitted_trade_semantics)
    if kind == "policies":
        allowed.add("slots")
    if kind == "build_improvement":
        allowed.update({"charges", "selector", "moves_token", "plot_id", "unit_id",
                        "index", "hash", "item_type"})
    elif kind == "terrain_operation":
        allowed.update({"charges", "moves_token", "plot_id", "unit_id", "operation", "hash"})
    unsupported = set(action) - allowed
    if unsupported:
        raise NotImplementedError(f"action fields need an encoder: {sorted(unsupported)}")
    for field, category in (("unit_id", "unit"), ("city_id", "city"),
                            ("origin_city_id", "city"), ("destination_city_id", "city")):
        if field in action and (category, action[field]) not in id_coords:
            raise NotImplementedError(f"{field} does not resolve to an observed entity")
    if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and 'district_id' in action:
        _require(('district', action['district_id']) in id_coords,
                 'district action lacks its public source geometry')
    vector = [float(name == "kind:" + kind) for name in feature_names[:len(action_kinds)]]
    coords = [None, None, None, None]
    source_x, source_y = action.get("source_x"), action.get("source_y")
    if source_x is None and "x0" in action:
        source_x, source_y = action["x0"], action.get("y0")
    target_x = action.get("target_x", action.get("destination_x", action.get("x", action.get("x1"))))
    target_y = action.get("target_y", action.get("destination_y", action.get("y", action.get("y1"))))
    if trade_choice is not None:
        # The route chooser uses x0/y0 for the destination and x1/y1 for the
        # trader origin. The validator provides canonical source/target order.
        source_x, source_y = trade_choice["source"]
        target_x, target_y = trade_choice["target"]
    if work_choice is not None:
        source_x, source_y = work_choice["source"]
        target_x, target_y = work_choice["target"]
    if citizen_choice is not None:
        source_x, source_y = citizen_choice['source']
        target_x, target_y = citizen_choice['target']
    if source_x is None and "unit_id" in action:
        source_x, source_y = id_coords[("unit", action["unit_id"])]
    if source_x is None and "city_id" in action:
        source_x, source_y = id_coords[("city", action["city_id"])]
    if source_x is None and "origin_city_id" in action:
        source_x, source_y = id_coords[("city", action["origin_city_id"])]
    if source_x is None and actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and 'district_id' in action:
        source_x, source_y = id_coords[('district', action['district_id'])]
    if target_x is None and "destination_city_id" in action:
        target_x, target_y = id_coords[("city", action["destination_city_id"])]
    if target_x is None and action.get("plot_id") is not None:
        plot_id = action["plot_id"]
        target_x, target_y = plot_id % obs["width"], plot_id // obs["width"]
    if source_x is None and action.get("source_plot_id") is not None:
        plot_id = action["source_plot_id"]
        source_x, source_y = plot_id % obs["width"], plot_id // obs["width"]
    if target_x is None and action.get("destination_plot_id") is not None:
        plot_id = action["destination_plot_id"]
        target_x, target_y = plot_id % obs["width"], plot_id // obs["width"]
    pairs = ((source_x, source_y), (target_x, target_y))
    for idx, (x, y) in enumerate(pairs):
        if x is not None or y is not None:
            _require(type(x) is int and type(y) is int and 0 <= x < obs["width"] and 0 <= y < obs["height"],
                     "action coordinate outside map")
            coords[idx * 2] = _norm_coord(x, obs["width"])
            coords[idx * 2 + 1] = _norm_coord(y, obs["height"])
    cost = _native_purchase_price(action["price_token"]) if kind == "purchase" else action.get("cost")
    if cost is not None:
        _finite(cost, "action.cost")
    vector.extend([v if v is not None else 0.0 for v in coords])
    vector.extend([float(coords[0] is not None and coords[1] is not None),
                   float(coords[2] is not None and coords[3] is not None)])
    vector.extend([_finite(cost, "action.cost") if cost is not None else 0.0, float(cost is not None)])
    research_turns, boosted = None, None
    if type(research_choice) is dict:
        cost = _finite(research_choice["cost"], kind + ".cost")
        research_turns = research_choice["turns_display"]
        boosted = research_choice["boosted"]
        # Research and civic costs occupy the existing cost channels.
        vector[-2] = cost
        vector[-1] = 1.0
    vector.extend([_finite(research_turns, "research.turns_display")
                   if research_turns is not None else 0.0,
                   float(research_turns is not None),
                   float(boosted) if boosted is not None else 0.0,
                   float(boosted is not None)])
    if trade_choice is None:
        vector.extend([0.0] * len(TRADE_FEATURES))
    else:
        features = trade_choice["features"]
        _require(type(features) is list and len(features) == len(TRADE_FEATURES),
                 "validated trade feature layout differs")
        vector.extend(_finite(value, "trade feature") for value in features)
    if work_choice is None:
        vector.extend([0.0, 0.0])
    else:
        features = work_choice["features"]
        _require(type(features) is list and len(features) == len(ACTION_WORK_FEATURES),
                 "validated unit-work feature layout differs")
        vector.extend(_finite(value, "unit-work feature") for value in features)
    vector.extend(production_public_features(production_choice) if production_choice is not None
                  else [0.0] * len(ACTION_PRODUCTION_FEATURES))
    if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        vector.extend(focus_choice if focus_choice is not None else [0.0] * len(FOCUS_ACTION_FEATURES))
    if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        vector.extend(citizen_choice['features'] if citizen_choice is not None else [0.0] * len(CITIZEN_ACTION_FEATURES))
    if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        vector.extend(queue_choice if queue_choice is not None else [0.0] * len(QUEUE_ACTION_FEATURES))
    if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        _require(type(source_bound) is bool, 'source binding flag must be boolean')
        vector.append(float(source_bound))
    if actor_schema in (ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        if kind == 'move':
            _require(type(movement_choice) is dict and set(movement_choice)=={'known','path_length','arrival_turn'}
                     and movement_choice['known'] is True
                     and type(movement_choice['path_length']) is int and movement_choice['path_length']>=2
                     and type(movement_choice['arrival_turn']) is int and movement_choice['arrival_turn']>=1,
                     'Move candidate lacks complete preview features')
            vector.extend([1.0,float(movement_choice['path_length']),float(movement_choice['arrival_turn'])])
        else:
            _require(movement_choice is None,'Non-movement candidate carries movement preview features')
            vector.extend([0.0,0.0,0.0])
    if actor_schema in (ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .world_input import action_flags
        vector.extend(action_flags(action))
    return vector


def encode_response(response: dict, *, vocabulary: ContentVocabulary, actor_schema=None) -> dict:
    """Encode one complete explicit-action session response using stdlib lists.

    `training_eligible` is always false because the native interface currently
    advertises an incomplete strategic action space.
    """
    _require(type(response) is dict and set(response) == _RESPONSE_KEYS,
             "only the complete explicit-action response envelope is accepted")
    # Both native protocols publish the current actor's observation/menu.
    # Hotseat receipts may audit the preceding actor; neither those audit
    # fields nor the envelope label are neural inputs. Keep the wire untouched.
    _require(response["stage"] == "official_session_response" and response["schema"] in _RESPONSE_SCHEMAS,
             "unsupported native response")
    _require(type(vocabulary) is ContentVocabulary, "a frozen ContentVocabulary is required")
    for key in ("episode_id", "game_generation", "nonce"):
        _require(type(response[key]) is str and bool(response[key]), f"{key} missing")
    _require(type(response["sequence"]) is int and response["sequence"] >= 0, "invalid sequence")
    _require(response["action_interface_complete"] is False
             and response["strategic_training_ready"] is False
             and response["automatic_choices_applied"] is False,
             "response is not a verified non-automatic diagnostic boundary")
    outcome = response["outcome"]
    _require(type(outcome) is dict, "response outcome must be an object")
    ended = False
    for flag in ("truncated", "terminated"):
        if flag in outcome:
            _require(type(outcome[flag]) is bool, f"outcome.{flag} must be boolean")
            ended = ended or outcome[flag]
    obs = response["observation"]
    _require(type(obs) is dict and set(obs) == _OBS_KEYS and obs["schema"] == _PLAYER_OBS_SCHEMA,
             "unsupported player observation")
    pid, width, height = obs["player_id"], obs["width"], obs["height"]
    _require(type(pid) is int and 0 <= pid <= 63 and type(width) is int and type(height) is int
             and width > 0 and height > 0 and width * height <= 16384, "invalid observation dimensions")
    _require(type(obs["turn"]) is int and obs["turn"] >= 0, "invalid observation turn")
    decisions = response["decisions"]
    _require(type(decisions) is dict and decisions.get("schema") in
             (_DECISIONS_SCHEMA, _DECISIONS_SCHEMA_972, _DECISIONS_SCHEMA_980,
              _DECISIONS_SCHEMA_982, _DECISIONS_SCHEMA_983, _DIPLOMACY_SCHEMA_987, _DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA),
             "unsupported native decisions catalogue")
    _require(decisions.get("player_id") == pid and decisions.get("turn") == obs["turn"]
             and decisions.get("episode_id") == response["episode_id"]
             and decisions.get("game_generation") == response["game_generation"]
             and decisions.get("sequence") == response["sequence"], "decisions boundary mismatch")
    has_management = decisions['schema'] == _CITY_MANAGEMENT_SCHEMA
    has_focus = decisions["schema"] in (_DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA)
    if actor_schema is None: actor_schema = ACTOR_V8 if has_management else ACTOR_V7 if has_focus else ACTOR_V6
    entity_names, action_names = actor_feature_names(actor_schema)
    _require(not has_focus or actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15),
             "schema1000 requires actor v7; dropping its focus choices is forbidden")
    _require(not has_management or actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15),
             'Grouped city management requires actor v8; dropping citizen choices is forbidden')
    _require(has_management or 'citizen_catalogs' not in decisions,
             'Legacy decisions cannot carry citizen state')
    _require(has_focus or "city_focus_catalogs" not in decisions,
             "legacy decisions cannot carry focus state")
    queue_catalogs = decisions.get('production_queues')
    queue_actions = any(type(row) is dict and type(row.get('action')) is dict
                        and row['action'].get('kind') in QUEUE_ACTION_KINDS
                        for row in decisions.get('candidates', []))
    if actor_schema not in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and type(queue_catalogs) is list:
        _require(not any(type(row) is dict and (row.get('enabled') is True or 'schema' in row) for row in queue_catalogs),
                 'Enabled production queues require explicit actor v9; dropping queue state is forbidden')
    queues = {'locations': [], 'candidates': {}}
    if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        _require(queue_catalogs is not None or not queue_actions,
                 'Actor v9 queue actions require the public ordered queue catalogs')
        if queue_catalogs is not None:
            queues = validate_production_queues(decisions, obs, ended=ended)
    focus = validate_city_focus(decisions, obs, ended=ended) if has_focus else {"cities": {}, "candidates": {}}
    is_diplomacy = decisions["schema"] == _DIPLOMACY_SCHEMA_987
    dialogue = (validate_diplomacy(decisions, obs, ended=ended,
                 allow_world_input_continuation=(actor_schema in (ACTOR_V13, ACTOR_V14, ACTOR_V15)),
                 allow_unnamed_response=(actor_schema in (ACTOR_V13, ACTOR_V14, ACTOR_V15)))
                if is_diplomacy else None)
    if "policies" in decisions:
        raw_policy_slots = _require_policy_catalog(decisions["policies"], pid, obs["turn"])
        policy_slots = copy.deepcopy(raw_policy_slots)
    else:
        # Older diagnostic fixtures predate publication of the policy menu.
        # Their empty policy frame is explicit and carries no completeness claim.
        policy_slots = []
    policy_prior_tokens = [
        vocabulary.lookup("policy.index", row["prior_index"]) if row["prior_index"] >= 0 else 0
        for row in policy_slots
    ]
    policy_option_tokens = [[[
        vocabulary.lookup("policy.index", option["index"]),
        vocabulary.lookup("policy.hash", option["hash"]),
        vocabulary.lookup("policy.type", option["policy_type"]),
    ] for option in row["options"]] for row in policy_slots]
    player_map = decisions.get("player_map")
    _validate_player_map(player_map, obs)
    management = (validate_city_management(decisions, obs, player_map, ended=ended) if has_management
                  else {'locations': [], 'candidates': {}})
    bindings = build_source_bindings(decisions, obs, ended=ended) if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) else None
    candidates = decisions.get('candidates')
    _require(type(candidates) is list, 'candidate catalogue must be a list')
    queued_goals, candidate_movement_points = ({}, [[] for _ in candidates])
    world_selected = None
    if actor_schema in (ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .movement_information import validate_and_index as validate_movement_information
        queued_goals, candidate_movement_points = validate_movement_information(
            decisions, obs, candidates, ended=ended)
    dialogue_candidate_mask = None
    if actor_schema == ACTOR_V12:
        from .world_input import validate_and_index as validate_world_input
        world_selected, world_action_mask, world_selection_known = validate_world_input(
            decisions, obs, candidates, ended=ended, dialogue=dialogue)
    elif actor_schema in (ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .war_confirmation13 import validate_and_project as validate_war_confirmation
        (world_selected, world_action_mask, world_selection_known,
         dialogue, dialogue_candidate_mask) = validate_war_confirmation(
            decisions, obs, candidates, ended=ended, dialogue=dialogue)
    else:
        world_action_mask = None
        world_selection_known = None

    channels = [[[0.0 for _ in range(width)] for _ in range(height)] for _ in MAP_CHANNELS]
    at = lambda name: channels[MAP_CHANNELS.index(name)]
    plot_rows = player_map["plots"]
    _require(type(plot_rows) is list, "player map plots must be a list")
    seen = set()
    visible_rows = {}
    for row in plot_rows:
        _require(type(row) is dict and row.get("revealed") is True, "unrevealed player-map row rejected")
        base_keys = {"index", "x", "y", "revealed", "visible", "owner", "terrain_type",
                     "feature_type", "water", "impassable"}
        _require(set(row) in (base_keys, base_keys | {"resource_type", "resource_count"}),
                 "player-map row contains unrecognized or missing fields")
        x, y, index = row.get("x"), row.get("y"), row.get("index")
        _require(type(x) is int and type(y) is int and type(index) is int and
                 0 <= x < width and 0 <= y < height and index == y * width + x and index not in seen,
                 "invalid or duplicate plot geometry")
        seen.add(index)
        if row["visible"] is True:
            visible_rows[index] = row
        _plot_values(row, channels, x, y, width, height,
                     pid,
                     visible=row.get("visible") is True, revealed=True,
                     actor_schema=actor_schema)
    visible_obs = obs["plots"]
    _require(type(visible_obs) is list, "visible plots must be a list")
    obs_by_index = {}
    for row in visible_obs:
        _require(type(row) is dict and row.get("visible") is True and row.get("revealed") is True,
                 "player observation contains a non-visible plot")
        index = row.get("index")
        _require(type(index) is int and index not in obs_by_index, "duplicate visible plot")
        obs_by_index[index] = row
    _require(set(obs_by_index) == set(visible_rows), "visible player map differs from player observation")
    for index, row in obs_by_index.items():
        mapped = visible_rows[index]
        for key in ("x", "y", "owner", "terrain_type", "feature_type", "water", "impassable"):
            _require(mapped.get(key) == row.get(key), "visible plot fields differ between official sources")
        if "resource_type" in row:
            _require(mapped.get("resource_type") == row["resource_type"]
                     and mapped.get("resource_count") == row["resource_count"],
                     "visible resource fields differ between official sources")
    for y in range(height):
        for x in range(width):
            at("odd_row")[y][x] = float(y & 1)
            at("x_norm")[y][x] = _norm_coord(x, width)
            at("y_norm")[y][x] = _norm_coord(y, height)

    # Content types are embeddings, not numeric quantities. All absent older
    # schemas keep both the new numerical known bits and categorical IDs zero.
    map_categorical = [[[0 for _ in range(width)] for _ in range(height)]
                       for _ in MAP_CATEGORICAL_FIELDS]
    if decisions["schema"] in (_DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
        rows = validate_plot_display(decisions.get("plot_info"), pid, obs["turn"], player_map)
        enabled = decisions["plot_info"]["yields_enabled"]
        for index, row in rows.items():
            x, y = index % width, index // width
            for name, value in zip(MAP_DISPLAY_CHANNELS, plot_display_features(row, enabled)):
                at(name)[y][x] = value
            for field, name in enumerate(("improvement", "district")):
                if row[name] is not None:
                    map_categorical[field][y][x] = vocabulary.lookup("action.item_type", row[name]["type"])
    else:
        _require("plot_info" not in decisions, "pre-990 decisions schema cannot carry plot display")

    _require(type(obs["economy"]) is dict and set(obs["economy"]) == {"science", "culture", "gold", "faith"},
             "unsupported economy fields")
    economy_values = {key: _finite(obs["economy"][key], "economy." + key)
                      for key in ("science", "culture", "gold", "faith")}
    global_values = [float(obs["turn"]), economy_values["science"], economy_values["culture"],
                     float(math.floor(economy_values["gold"])), economy_values["faith"]]
    # Older schemas have no published selected-research or treasury yield
    # fields. Keep the fixed vector shape while representing absence as
    # canonical zero plus an explicit false value mask.
    global_values.extend([0.0] * (len(GLOBAL_FEATURES) - len(global_values)))
    # The known-bit features themselves are known even when their value is 0;
    # the paired value feature is masked until a progress row is published.
    global_mask = [1.0] * 5 + [bit for _ in range((len(GLOBAL_FEATURES) - 5) // 2)
                               for bit in (0.0, 1.0)]
    yield_display = None
    public_city_yields = {}
    if decisions["schema"] in (_DECISIONS_SCHEMA_982, _DECISIONS_SCHEMA_983, _DIPLOMACY_SCHEMA_987, _DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
        from .yield_display982 import public_number, validate as validate_yield_display
        yield_display = validate_yield_display(decisions.get("yield_display"), pid, obs["turn"], obs)
        for name in ("science", "culture", "faith", "faith_yield"):
            index = GLOBAL_FEATURES.index(name)
            global_values[index] = public_number(yield_display["public_display"][name])
            if name == "faith_yield":
                global_values[index + 1] = 1.0
                global_mask[index] = 1.0
                global_mask[index + 1] = 1.0
        public_city_yields = {
            city["city_id"]: {name: public_number(city["public_display"][name])
                              for name in ("food", "production", "gold", "science", "culture", "faith")}
            for city in yield_display["cities"]
        }
    else:
        _require("yield_display" not in decisions,
                 "pre-982 decisions schema cannot carry schema-982 yield display")
    research = decisions.get("research")
    if not is_diplomacy:
        _require(type(research) is dict and set(research) == set(GLOBAL_CATEGORICAL_FIELDS),
                 "unsupported current research catalog")
    research_info = None
    research_choices = {"research": {}, "civic": {}}
    if decisions["schema"] in (_DECISIONS_SCHEMA_983, _DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
        from .research_info983 import validate as validate_research_info
        decision_candidates = decisions.get("candidates")
        if ended:
            _require(type(decision_candidates) is list and not decision_candidates,
                     "terminal or truncated response must have no action candidates")
        research_info = validate_research_info(
            decisions.get("research_info"), pid, obs["turn"],
            research=research,
            # Closed sessions can still observe public chooser rows, but there
            # is no live decision boundary to join them against.
            candidates=None if ended else decision_candidates,
        )
        for action_kind, topic in (("research", "technology"), ("civic", "civic")):
            research_choices[action_kind] = {
                (row["index"], row["hash"]): row for row in research_info[topic]["choices"]
            }
        for topic, prefix in (("technology", "technology"), ("civic", "civic")):
            selected = research_info[topic]["selected_info"]
            if selected is None:
                continue
            turns = selected["turns_display"]
            turns_index = GLOBAL_FEATURES.index(prefix + "_turns_left")
            if turns is not None:
                global_values[turns_index] = float(turns)
                global_values[turns_index + 1] = 1.0
                global_mask[turns_index] = 1.0
                global_mask[turns_index + 1] = 1.0
            boosted_index = GLOBAL_FEATURES.index(prefix + "_boosted")
            global_values[boosted_index] = float(selected["boosted"])
            global_values[boosted_index + 1] = 1.0
            global_mask[boosted_index] = 1.0
            global_mask[boosted_index + 1] = 1.0
    else:
        _require("research_info" not in decisions,
                 "pre-983 decisions schema cannot carry schema-983 research information")
    if decisions["schema"] == _DECISIONS_SCHEMA or is_diplomacy:
        _require("research_progress" not in decisions,
                 "legacy decisions schema cannot carry research progress")
    else:
        progress = decisions.get("research_progress")
        _require(type(progress) is dict and set(progress) == {
            "schema", "player_id", "turn", "technology", "civic"
        } and progress["schema"] == _RESEARCH_PROGRESS_SCHEMA
                 and type(progress["player_id"]) is int and progress["player_id"] == pid
                 and type(progress["turn"]) is int and progress["turn"] == obs["turn"],
                 "research progress identity or schema mismatch")
        for label, selected in (("technology", research["research"]), ("civic", research["civic"])):
            item = progress[label]
            _require(type(item) is dict and set(item) == {"index", "cost", "progress", "remaining"},
                     "malformed " + label + " progress row")
            if item["index"] is None:
                _require(selected is None and all(item[field] is None
                    for field in ("cost", "progress", "remaining")),
                    "empty " + label + " progress differs from current selection")
                continue
            _require(type(item["index"]) is int and 0 <= item["index"] <= 0xffffffff
                     and selected == item["index"],
                     label + " progress index differs from current selection")
            values = [_finite(item[field], label + "." + field)
                      for field in ("cost", "progress", "remaining")]
            _require(values[2] == values[0] - values[1],
                     label + " progress remaining differs from cost-progress")
            # Numeric positions alternate value/known mask, starting at index 5.
            base = 5 if label == "technology" else 11
            for offset, value in enumerate(values):
                global_values[base + offset * 2] = value
                global_values[base + offset * 2 + 1] = 1.0
                global_mask[base + offset * 2] = 1.0
                global_mask[base + offset * 2 + 1] = 1.0
    if decisions["schema"] in (_DECISIONS_SCHEMA_980, _DECISIONS_SCHEMA_982,
                                _DECISIONS_SCHEMA_983, _DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
        treasury = decisions.get("treasury")
        _require(type(treasury) is dict and set(treasury) == {
            "schema", "player_id", "turn", "gold_balance", "gold_yield",
            "total_maintenance", "net_gold_yield", "public_display"
        } and treasury["schema"] == _TREASURY_SCHEMA
                 and type(treasury["player_id"]) is int and treasury["player_id"] == pid
                 and type(treasury["turn"]) is int and treasury["turn"] == obs["turn"],
                 "treasury identity or schema mismatch")
        balance = _finite(treasury["gold_balance"], "treasury.gold_balance")
        raw_gold_yield = _finite(treasury["gold_yield"], "treasury.gold_yield")
        maintenance = _finite(treasury["total_maintenance"], "treasury.total_maintenance")
        raw_net_gold_yield = _finite(treasury["net_gold_yield"], "treasury.net_gold_yield")
        _require(balance == economy_values["gold"],
                 "treasury balance differs from public player observation")
        _require(raw_net_gold_yield == raw_gold_yield - maintenance,
                 "net gold yield differs from yield minus maintenance")
        display = treasury["public_display"]
        _require(type(display) is dict and set(display) == {
            "schema", "gold_balance", "gold_yield", "gold_expense", "net_gold_yield"
        } and display["schema"] == _TREASURY_DISPLAY_SCHEMA
                 and type(display["gold_balance"]) is int
                 and display["gold_balance"] == math.floor(balance),
                 "treasury public display schema or balance mismatch")
        displayed_gold_yield = _display_number(display["gold_yield"], "treasury.display.gold_yield")
        displayed_expense = _display_number(display["gold_expense"], "treasury.display.gold_expense")
        displayed_net = _display_number(display["net_gold_yield"], "treasury.display.net_gold_yield")
        # The official formatter may discard the extra fraction; do not impose
        # nearest rounding (native witness982: 1234.56 -> 1,234.5).
        tolerance = 0.100000001
        _require(abs(displayed_gold_yield - raw_gold_yield) <= tolerance
                 and abs(displayed_expense + maintenance) <= tolerance
                 and abs(displayed_net - raw_net_gold_yield) <= tolerance,
                 "treasury public display differs from raw diagnostic values")
        # These actor channels use the official formatted values. Raw getter
        # values above are cross-checks only and never enter the feature vector.
        for name, value in (("gold_yield", displayed_gold_yield),
                            ("total_maintenance", -displayed_expense),
                            ("net_gold_yield", displayed_net)):
            index = GLOBAL_FEATURES.index(name)
            global_values[index] = value
            global_values[index + 1] = 1.0
            global_mask[index] = 1.0
            global_mask[index + 1] = 1.0
    else:
        _require("treasury" not in decisions,
                 "older decisions schema cannot carry schema-980 treasury data")
    global_categorical = [
        vocabulary.lookup("action.index", research[field])
        if research is not None and research[field] is not None else 0
        for field in GLOBAL_CATEGORICAL_FIELDS
    ]
    # Diplomatic boundaries publish no ordinary production/combat sidecars.
    # Preserve observation entities, keeping those absent feature pairs unknown.
    production_by_city, attacks_by_unit, defense_by_city = {}, {}, {}
    combat_cities = []
    if not is_diplomacy:
        production = decisions.get("production")
        _require(type(production) is dict and production.get("player_id") == pid
                 and production.get("game_turn") == obs["turn"] and type(production.get("cities")) is list,
                 "production observation boundary mismatch")
        for item in production["cities"]:
            _require(type(item) is dict and type(item.get("city_id")) is int
                     and item["city_id"] not in production_by_city,
                     "malformed or duplicate city production row")
            production_by_city[item["city_id"]] = item
        own_city_ids = {city["id"] for city in obs["cities"]}
        _require(set(production_by_city) == own_city_ids, "production roster differs from own cities")

        combat_units = decisions.get("combat_units")
        _require(type(combat_units) is list, "combat unit roster missing")
        for combat_unit in combat_units:
            _require(type(combat_unit) is dict and set(combat_unit) == {
                "unit_id", "unit_type", "x", "y", "damage", "moves", "attacks_remaining"},
                "malformed published combat unit")
            uid = combat_unit["unit_id"]
            _require(type(uid) is int and uid not in attacks_by_unit, "duplicate combat unit")
            attacks_by_unit[uid] = combat_unit
        _require(set(attacks_by_unit) == {unit["id"] for unit in obs["units"]},
                 "combat roster differs from player unit observation")
        for unit in obs["units"]:
            combat_unit = attacks_by_unit[unit["id"]]
            _require((combat_unit["unit_type"], combat_unit["x"], combat_unit["y"],
                      combat_unit["damage"], combat_unit["moves"]) ==
                     (unit["type"], unit["x"], unit["y"], unit["damage"], unit["moves"]),
                     "combat unit fields differ from visible player observation")
        combat_cities = decisions.get("combat_cities")
        _require(type(combat_cities) is list, "visible combat city catalogue missing")
        for city in combat_cities:
            expected = {"owner", "city_id", "district_id", "plot_id", "x", "y", "garrison_damage",
                        "garrison_max", "outer_damage", "outer_max", "defense_strength"}
            _require(type(city) is dict and set(city) == expected, "malformed published city defense")
            key = (city["owner"], city["city_id"])
            _require(type(city["owner"]) is int and 0 <= city["owner"] <= 63
                     and type(city["x"]) is int and type(city["y"]) is int
                     and 0 <= city["x"] < width and 0 <= city["y"] < height
                     and key not in defense_by_city and type(city["plot_id"]) is int
                     and 0 <= city["plot_id"] < width * height
                     and at("visible")[city["y"]][city["x"]] == 1.0,
                     "city defense is duplicate or outside the current visible map")
            _require(city["plot_id"] == city["y"] * width + city["x"], "city defense coordinates differ")
            defense_by_city[key] = city

    entities, entity_mask, entity_categorical, entity_keys = [], [], [], []
    id_coords = {}
    for category, rows, id_field in (("unit", obs["units"], "id"),
                                    ("city", obs["cities"], "id"),
                                    ("visible_enemy", obs["visible_enemy_units"], "id")):
        _require(type(rows) is list, f"{category} roster must be a list")
        for row in rows:
            _require(type(row) is dict and type(row.get("x")) is int and type(row.get("y")) is int
                     and 0 <= row["x"] < width and 0 <= row["y"] < height, f"invalid {category} row")
            if category == "visible_enemy":
                _require(at("visible")[row["y"]][row["x"]] == 1.0,
                         "enemy unit is outside the currently visible map")
            if id_field in row:
                id_coords[(category, row[id_field])] = (row["x"], row["y"])
            vector = [float(category == "unit"), float(category == "city"),
                      float(category == "visible_enemy"), 0.0, _norm_coord(row["x"], width),
                      _norm_coord(row["y"], height)]
            for source in ("damage", "moves", "max_moves", "population"):
                known = source in row
                vector.extend([_finite(row[source], category + "." + source) if known else 0.0, float(known)])
            yields = row.get("yields", {})
            for key in ("food", "production", "gold", "science", "culture", "faith"):
                known = key in yields
                if category == "city" and yield_display is not None:
                    city_yields = public_city_yields.get(row["id"])
                    _require(city_yields is not None, "schema-982 city yield display is missing own city")
                    # The raw observation value is cross-checked by the
                    # yield-display validator above; only the displayed number
                    # enters actor inputs for schema 982.
                    vector.extend([city_yields[key], 1.0])
                else:
                    vector.extend([_finite(yields[key], category + ".yields." + key) if known else 0.0,
                                   float(known)])
            combat_unit = attacks_by_unit.get(row.get("id")) if category == "unit" else None
            prod = production_by_city.get(row.get("id")) if category == "city" else None
            defense = defense_by_city.get((pid, row.get("id"))) if category == "city" else None
            if prod is not None and prod.get("current_production_hash") is not None:
                empty = (prod.get("kind") == "N" and prod.get("current_production_hash") == 0
                         and prod.get("item_index") is None and prod.get("queue_size") == 0)
                _require(prod.get("kind") in {"U", "B", "D", "P"} or empty,
                         "unknown current production category")
            extra_values = (
                (combat_unit.get("attacks_remaining") if combat_unit else None),
                (prod.get("production_yield") if prod else None),
                (prod.get("cost") if prod else None),
                (prod.get("progress") if prod else None),
                (prod.get("turns_left") if prod else None),
                (prod.get("queue_size") if prod else None),
                (float(prod["has_production_progress"]) if prod else None),
                (defense.get("garrison_damage") if defense else None),
                (defense.get("garrison_max") if defense else None),
                (defense.get("outer_damage") if defense else None),
                (defense.get("outer_max") if defense else None),
                (defense.get("defense_strength") if defense else None),
            )
            extra_names = ("attacks_remaining", "production_native_yield", "production_cost",
                           "production_progress", "production_turns_left", "production_queue_size",
                           "production_has_progress", "garrison_damage", "garrison_max",
                           "outer_damage", "outer_max", "defense_strength")
            for name, value in zip(extra_names, extra_values):
                known = value is not None
                vector.extend([_finite(value, category + "." + name) if known else 0.0, float(known)])
            if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
                vector.extend(focus_entity_features(focus["cities"].get(row["id"]) if category == "city" else None))
            if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
                vector.extend([0.0] * len(CITIZEN_ENTITY_FEATURES))
            if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15): vector.extend([0.0] * len(QUEUE_ENTITY_FEATURES))
            entities.append(vector)
            entity_keys.append((category, row['id']) if category in ('unit', 'city') else None)
            entity_mask.append(True)
            categories = {"type": row.get("type")}
            if prod is not None:
                kind_map = {"U": "unit", "B": "building", "D": "district", "P": "project"}
                categories.update(production_kind=kind_map.get(prod.get("kind")),
                                  production_hash=(prod.get("current_production_hash") & 0xffffffff
                                                   if prod.get("current_production_hash") is not None and prod.get("kind") != "N" else None),
                                  production_index=prod.get("item_index"))
            entity_categorical.append([
                vocabulary.lookup(
                    "entity.type" if field == "type" else
                    "action.category" if field == "production_kind" else
                    "action.hash" if field == "production_hash" else "action.index",
                    categories[field])
                if categories.get(field) is not None else 0
                for field in ENTITY_CATEGORICAL_FIELDS
            ])

    # The official combat-city reader admits only currently visible city plots.
    # Add visible foreign cities as their own entities; never infer hidden cities.
    own_city_id_set = {city["id"] for city in obs["cities"]}
    for city in combat_cities:
        if city["owner"] == pid:
            _require(city["city_id"] in own_city_id_set, "own city defense identity is absent from own roster")
            own = next(row for row in obs["cities"] if row["id"] == city["city_id"])
            _require((own["x"], own["y"]) == (city["x"], city["y"]),
                     "own city defense coordinates differ from observation")
            continue
        vector = [0.0, 0.0, 0.0, 1.0, _norm_coord(city["x"], width), _norm_coord(city["y"], height)]
        for _ in range(4):
            vector.extend([0.0, 0.0])
        for _ in range(6):
            vector.extend([0.0, 0.0])
        vector.extend([0.0, 0.0])  # attacks_remaining
        for _ in range(6):
            vector.extend([0.0, 0.0])  # current production fields
        for field in ("garrison_damage", "garrison_max", "outer_damage", "outer_max", "defense_strength"):
            vector.extend([_finite(city[field], "visible_enemy_city." + field), 1.0])
        _require(len(vector) == len(ENTITY_FEATURES), "internal visible-city feature layout differs")
        if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15): vector.extend(focus_entity_features(None))
        if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15): vector.extend([0.0] * len(CITIZEN_ENTITY_FEATURES))
        if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15): vector.extend([0.0] * len(QUEUE_ENTITY_FEATURES))
        entities.append(vector)
        entity_keys.append(None)
        entity_mask.append(True)
        entity_categorical.append([0] * len(ENTITY_CATEGORICAL_FIELDS))

    if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        for location in management['locations']:
            city, row = location['city'], location['plot']
            vector = [0.0] * len(ENTITY_FEATURES_V7)
            vector[ENTITY_FEATURES.index('x_norm')] = _norm_coord(row['x'], width)
            vector[ENTITY_FEATURES.index('y_norm')] = _norm_coord(row['y'], height)
            vector.extend(citizen_location_features(city, row, width, height))
            if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15): vector.extend([0.0] * len(QUEUE_ENTITY_FEATURES))
            entities.append(vector); entity_mask.append(True)
            entity_keys.append(None)
            entity_categorical.append([0] * len(ENTITY_CATEGORICAL_FIELDS))

    if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        for location in queues['locations']:
            city, entry = location['city'], location['entry']
            vector = [0.0] * len(ENTITY_FEATURES_V8)
            vector[ENTITY_FEATURES.index('x_norm')] = _norm_coord(city['x'], width)
            vector[ENTITY_FEATURES.index('y_norm')] = _norm_coord(city['y'], height)
            vector.extend(queue_entry_features(entry))
            entities.append(vector); entity_mask.append(True)
            entity_keys.append(None)
            entity_categorical.append([0, vocabulary.lookup('action.category', entry['category']),
                vocabulary.lookup('action.hash', entry['hash']), vocabulary.lookup('action.index', entry['index'])])

    if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        for vector in entities:
            vector.extend([0.0] * len(DISTRICT_ENTITY_FEATURES))
        for district in bindings['districts']:
            vector = [0.0] * len(ENTITY_FEATURES_V9)
            vector[ENTITY_FEATURES.index('x_norm')] = _norm_coord(district['x'], width)
            vector[ENTITY_FEATURES.index('y_norm')] = _norm_coord(district['y'], height)
            vector.extend(district_features(district))
            entities.append(vector); entity_mask.append(True)
            entity_categorical.append([0] * len(ENTITY_CATEGORICAL_FIELDS))
            key = ('district', district['district_id'])
            entity_keys.append(key)
            id_coords[key] = (district['x'], district['y'])

    if actor_schema in (ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .movement_information import GOAL_ENTITY_FEATURES, POINT_ENTITY_FEATURES
        goal_start = len(ENTITY_FEATURES_V10)
        for vector, key in zip(entities, entity_keys):
            vector.extend([0.0] * len(MOVEMENT_ENTITY_FEATURES))
            if key is None or key[0] != 'unit':
                continue
            queued = queued_goals.get(key[1])
            # Ordinary catalogs cover the exact own-unit roster. A missing row
            # is unknown (as on pending diplomacy), never a fabricated no-goal.
            if key[1] not in queued_goals:
                continue
            vector[goal_start] = 1.0
            if queued is not None:
                vector[goal_start + 1] = 1.0
                vector[goal_start + 2] = _norm_coord(queued[0], width)
                vector[goal_start + 3] = _norm_coord(queued[1], height)
    if actor_schema in (ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        for vector, key in zip(entities, entity_keys):
            vector.extend([float(world_selection_known),
                           float(world_selection_known and key is not None and key[0] == 'unit'
                                 and key[1] == world_selected)])
        if (len(GOAL_ENTITY_FEATURES) != 4 or len(POINT_ENTITY_FEATURES) != 4
                or len(MOVEMENT_ENTITY_FEATURES) != len(GOAL_ENTITY_FEATURES) + len(POINT_ENTITY_FEATURES)):
            raise ValueError('Movement actor feature contract differs')

    if actor_schema == ACTOR_V15:
        purchase_quotes = validate_public_purchases(decisions, obs, ended=ended)
        for vector in entities:
            vector.extend([0.0] * len(PURCHASE_ENTITY_FEATURES))
        for categories in entity_categorical:
            categories.extend([0] * len(PURCHASE_CATEGORICAL_FIELDS))
        for location in purchase_quotes['locations']:
            city, offer = location['city'], location['offer']
            vector = [0.0] * len(ENTITY_FEATURES_V14)
            vector[ENTITY_FEATURES.index('x_norm')] = _norm_coord(city['x'], width)
            vector[ENTITY_FEATURES.index('y_norm')] = _norm_coord(city['y'], height)
            vector.extend(purchase_entity_features(city, offer, width, height))
            entities.append(vector)
            entity_keys.append(None)
            entity_mask.append(True)
            entity_categorical.append([0] * len(ENTITY_CATEGORICAL_FIELDS)
                                      + purchase_entity_categories(offer, vocabulary))

    has_trade_action = any(
        type(row) is dict and type(row.get("action")) is dict
        and row["action"].get("kind") in ("make_trade_route", "change_trade_origin")
        for row in candidates
    )
    trade_choices = {}
    if "trade_routes" in decisions or "trade_origins" in decisions or has_trade_action:
        trade_choices = validate_trade_choices(decisions, obs, ended=ended)
    has_work_action = any(
        type(row) is dict and type(row.get("action")) is dict
        and row["action"].get("kind") in ("build_improvement", "terrain_operation")
        for row in candidates
    )
    work_choices = {}
    if "unit_work" in decisions or has_work_action:
        work_choices = validate_unit_work(decisions, obs, ended=ended)
    production_choices = {}
    if decisions["schema"] in (_DECISIONS_SCHEMA_988, _DECISIONS_SCHEMA_989, _DECISIONS_SCHEMA_990, _DECISIONS_SCHEMA_999, _DECISIONS_SCHEMA_1000, _CITY_MANAGEMENT_SCHEMA):
        production_choices = validate_production_info(decisions, obs, ended=ended)
    else:
        _require("production_info" not in decisions,
                 "older decisions schema cannot carry schema-988 production information")
    candidate_ids, candidate_features, action_categorical = [], [], []
    categorical_fields = action_categorical_fields(actor_schema)
    policy_action_index = None
    for candidate in candidates:
        _require(type(candidate) is dict and set(candidate) == {"candidate_id", "action"}, "malformed candidate")
        cid = candidate["candidate_id"]
        _require(type(cid) is str and bool(cid), "candidate ID missing")
        candidate_ids.append(cid)
        action = candidate["action"]
        if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
            _require('target_type' not in action or action.get('kind') == 'upgrade_unit',
                     'Upgrade identity on a non-upgrade action')
            _require('individual' not in action or action.get('kind') == 'recruit_great_person',
                     'Great Person identity on a non-recruitment action')
            if action.get('kind') in ('upgrade_unit', 'recruit_great_person', 'add_belief'):
                manifest = vocabulary.manifest
                _require(type(manifest) is dict and type(manifest.get('action_content')) is dict
                         and type(manifest.get('database')) is dict
                         and manifest['action_content'].get('database_sha256') == manifest['database'].get('sha256'),
                         'Action identities require the exact public gameplay content catalogue')
                project_action_content(action, manifest['action_content'])
        if action.get("kind") == "policies":
            _require(policy_action_index is None, "multiple policy candidates are unsupported")
            _require("policies" in decisions and decisions["policies"]["reason"] == "available"
                     and action.get("slots") == policy_slots,
                     "policy candidate does not match the available public catalogue")
            _require(bool(valid_next_hashes(policy_slots, [])),
                     "policy candidate has no legal changed completion")
            policy_action_index = len(candidate_features)
        research_choice = None
        if research_info is not None and action.get("kind") in ("research", "civic"):
            pair = (action.get("index"), action.get("hash"))
            research_choice = research_choices[action["kind"]].get(pair)
            _require(research_choice is not None,
                     "research/civic candidate has no matching published choice row")
        movement_points = candidate_movement_points[len(candidate_features)]
        movement_choice = (dict(known=True, path_length=len(movement_points), arrival_turn=movement_points[-1][2])
                           if actor_schema in (ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15) and action.get('kind') == 'move' else None)
        candidate_features.append(_action_vector(action, obs, id_coords,
                                                 research_choice=research_choice,
                                                 trade_choice=trade_choices.get(cid),
                                                 work_choice=work_choices.get(cid),
                                                 diplomacy_choice=is_diplomacy,
                                                 production_choice=production_choices.get(cid),
                                                 citizen_choice=management['candidates'].get(cid),
                                                 queue_choice=queues['candidates'].get(cid),
                                                 source_bound=bindings['sources'][cid] is not None if bindings else False,
                                                 focus_choice=focus["candidates"].get(cid),
                                                 movement_choice=movement_choice, actor_schema=actor_schema))
        action_categorical.append([
            vocabulary.lookup("action." + field, action[field]) if field in action else 0
            for field in categorical_fields
        ])
    # Feature rows and masks keep all legal candidates, including an empty set.
    result = {
        "map": channels,
        "map_categorical": map_categorical,
        "map_categorical_fields": list(MAP_CATEGORICAL_FIELDS),
        "global": {"values": global_values, "mask": global_mask},
        "global_categorical": global_categorical,
        "global_categorical_fields": list(GLOBAL_CATEGORICAL_FIELDS),
        "entities": entities,
        "entity_mask": entity_mask,
        "candidates": candidate_features,
        "action_mask": ([True] * len(candidate_features) if world_action_mask is None else world_action_mask),
        "movement_point_features": candidate_movement_points,
        "action_categorical": action_categorical,
        "entity_categorical": entity_categorical,
        "action_categorical_fields": list(categorical_fields),
        "entity_categorical_fields": list(entity_categorical_fields(actor_schema)),
        "vocabulary_size": vocabulary.vocabulary_size,
        "vocabulary_fingerprint": vocabulary.fingerprint,
        "vocabulary_serialized": vocabulary.serialize(),
        "candidate_ids": candidate_ids,
        "policy_catalog": {"slots": policy_slots, "prior_tokens": policy_prior_tokens,
                           "option_tokens": policy_option_tokens},
        "policy_action_index": policy_action_index,
        "dialogue": dialogue,
        "player_id": pid,
        "episode_id": response["episode_id"],
        "sequence": response["sequence"],
        "training_eligible": False,
        "limitations": [
            "Diagnostic encoding only; native action_interface_complete and strategic_training_ready are false.",
            "Map contains only officially revealed tooltip fields and player-visible state; unrevealed cells remain masked.",
            "Offset-row hex coordinates are explicit; this encoder does not infer grid or hex adjacency.",
            "Terrain, feature, and resource codes are raw published numeric values with known masks; content categories use a frozen vocabulary.",
            "Ephemeral entity/candidate IDs and opaque control tokens are excluded; public action hashes, indices, current research, and current production categories are represented categorically.",
            "Schemas 972 and 980 contribute only selected technology/civic cost, progress, and remaining values; schema 968 marks them unknown, and none supplies completed research history or all chooser state.",
            "Schema 980 contributes the published player gold yield, total maintenance, and net gold yield after strict identity, balance, and arithmetic checks; schemas 968/972 mark these values unknown.",
            "Schema 980 uses only the official public one-decimal treasury display for actor yield features and floors balance; raw treasury numbers are validation-only. Other numeric fields do not yet have proven human-display precision equivalence.",
            "Schema 982 uses only official public yield display strings for global science/culture/faith, faith yield, and own-city yields; raw values are validation-only. Earlier schemas retain their historical feature interpretation and mark faith yield unknown.",
            "Schema 983 adds selected research/civic displayed turns and boost flags, plus per-choice displayed turns, boost flags, and official costs. Displayed turns preserve the native -1 no-label rule and 999 UI cap; raw research progress remains a diagnostic getter value and is not claimed human-display-equivalent.",
            "The city production yield from the official city-yield display and the separate BuildQueue production-yield getter are distinct published signals and may differ; neither is substituted for the other.",
            "Published purchase price, own-city current production cost/progress/turns/yield/queue size, own-unit attacks remaining, and visible-city defense are represented.",
            "Actions needing assignments or unsupported action fields raise NotImplementedError.",
            "Build-improvement and immediate terrain-operation candidates are joined exactly to the published UnitPanel menu and own visible unit/plot. Charges are features; UI selector and movement token are validated but excluded, and no work-menu option is pruned.",
            "Policy menus and prior occupants are transported, but policy assignment decoding is not part of this frame encoder; production queue entries, complete city yields and combat relationships/targets are not represented.",
            "Diplomacy schema 987 includes only pinned public speaker/text/reason and ordered menu text, disabled flags and tooltips. Raw statement/session IDs, AI reason/modifier, mood, choice keys and sort values are validation-only. Missing ordinary research, treasury, production and combat fields remain masked.",
            "Schema 988 joins all production, placement, resumption and queue-append choices to pinned official displayed cost/progress/turns. Unit tooltips may display static base cost when formation is nil; raw hidden getter values and progress-bar fractions are not action features. Earlier schemas mark these quote channels unknown.",
            "Schema 990 adds exact revealed PlotTooltip improvement/district types, displayed state and three groups of positive yield lines. Categories use public content embeddings; omitted yields remain unknown, never known zero. It does not add private owner/worker/branch data or historical fog memory; older schemas mark the new inputs unknown.",
            "No historical fog memory, complete player state, or complete strategic action space is claimed.",
        ],
    }
    if actor_schema in (ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        result.update(actor_schema=actor_schema, entity_features=list(entity_names), action_features=list(action_names))
        result["limitations"].append("City focus retains both public native flags and one UI cycle per yield; legacy views project unknown focus channels as zeros. No arbitrary setter or worker management is represented.")
    if actor_schema in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        result['limitations'][-1] = 'Grouped city management retains every official focus/citizen click. Citizen entities expose geometry, occupation and lock icons; worker/capacity numbers are known only when displayed by the official multi-slot meter. Historical views leave these features unknown. Tutorial gating and complete strategic admission remain separate requirements.'
    if actor_schema in (ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        result['limitations'] = [text.replace('production queue entries, complete city yields', 'complete city yields') for text in result['limitations']]
        result['limitations'].append('Queue v9 retains every observed own-city entry in native order, its public formation icons, and every offered append/remove/swap with exact positions. Availability remains diagnostic; native UI gates are not certified by encoding.')
    if actor_schema in (ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        result['candidate_source_rows'] = resolve_source_rows(bindings, entity_keys)
        result['limitations'].append('Action v10 binds each action to its own public source token, including independent districts. Upgrade, recruitment and belief identities require the same pinned public content catalogue. These joins do not certify human availability or strategic readiness.')
    if actor_schema in (ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .movement_information import CONTRACT as MOVEMENT_INFORMATION_CONTRACT
        result['movement_information_contract'] = copy.deepcopy(MOVEMENT_INFORMATION_CONTRACT)
        result['limitations'].append('Actor v11 adds each own unit’s public queued goal and complete native route preview for every offered move. Every point retains geometry, turn and order, joined only to its candidate row; the actor has no opaque unit/plot IDs or hidden path data. The new inputs change forward semantics and require a fresh optimizer; native availability and strategic readiness remain separate.')
    if actor_schema == ACTOR_V12:
        from .world_input import CONTRACT as WORLD_INPUT_CONTRACT
        result['world_input_contract'] = copy.deepcopy(WORLD_INPUT_CONTRACT)
        result['limitations'].append('Actor v12 adds public WorldInput unit selection, all-map-coordinate click candidates and modal war responses. Identifiers are join-only, map clicks are not pruned by visibility or terrain, and no preview/fog facts are added. UI decisions are represented as ordinary policy steps under the published diagnostic transition granularity.')
    if actor_schema in (ACTOR_V13, ACTOR_V14, ACTOR_V15):
        from .world_input import CONTRACT as WORLD_INPUT_CONTRACT
        from .war_confirmation13 import CONTRACT as WAR_CONFIRMATION_CONTRACT
        result['world_input_contract'] = copy.deepcopy(WORLD_INPUT_CONTRACT)
        result['war_confirmation_contract'] = copy.deepcopy(WAR_CONFIRMATION_CONTRACT)
        if actor_schema == ACTOR_V13:
            result['limitations'].append('Actor v13 uses the existing V12 text encoder for exact public war popup strings and the official decline/accept button labels. IDs and source hashes are validation-only; masked ordinary candidates retain menu index -1. The V12 layer dimensions and feature columns are unchanged, while old actor artifacts keep their original schema.')
        else:
            from .resource_quantity14 import CONTRACT as RESOURCE_QUANTITY_CONTRACT
            result['resource_quantity_contract'] = copy.deepcopy(RESOURCE_QUANTITY_CONTRACT)
            result['limitations'].append('Actor v14 keeps the V13 tensor layout and war-dialogue inputs but reserves per-plot resource quantity and known-bit channels at zero. Native counts remain validation-only; resource type and other revealed facts retain their prior interpretation. V14 is an explicit actor contract with no implicit artifact migration.')
        if dialogue_candidate_mask is not None:
            result['dialogue_candidate_mask'] = dialogue_candidate_mask
    if actor_schema == ACTOR_V15:
        result['public_purchases_contract'] = PURCHASE_CONTRACT
        result['limitations'].append('Actor v15 adds all published own-city purchase quotes as public entities, including unaffordable offers. Candidate identities and availability masks are unchanged. Prices come from native catalogues; UI admission and complete gameplay coverage remain separate.')
    return result
