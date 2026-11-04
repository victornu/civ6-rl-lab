"""Pure validation of official production chooser text and candidate joins.

This does not estimate production rules. The native reader supplies actual
getter results and the public fields selected by the pinned panel/tooltip.
Only those selected fields enter actor features; hidden raw values and bar
fractions are not encoded.
"""
from __future__ import annotations

import re


SCHEMA = "civ6rl.native_production_info988.v1"
PROOF_SHA256 = "aa71d6689507aa11cba4ad4de2d49faf6606a5a8b3a955b4f596a58630afa088"
PANEL_SHA256 = "1527d2fd8b2c82d38c228ba2dd75153b3c0bf721c934d1f7cc9cc9ddf98d337f"
TOOLTIP_SHA256 = "3929f5528f4499cf675868638e98b55b40796b922dc414e9ec34ae3d044a6ce7"
PRODUCTION_KINDS = ("production", "production_placement", "production_resume", "queue_append")
ACTION_PRODUCTION_FEATURES = (
    "production_cost", "production_cost_known", "production_cost_base",
    "production_progress", "production_progress_known", "production_turns_left",
    "production_turns_left_known", "production_never_completes", "production_never_completes_known",
)
MAX_ROWS = 4096
_KEYS = ("city_id", "category", "index", "hash", "item_type")
_CATEGORIES = {"unit", "building", "district", "project"}
_RAW_FIELDS = {"cost", "progress", "turns_left", "formation_type", "base_cost", "base_cost_visible"}
_PUBLIC_FIELDS = {"cost", "progress", "turns_left", "never_completes", "cost_kind"}


def _require(condition, message):
    if not condition:
        raise ValueError("production information: " + message)


def _integer(value, label, minimum=-(1 << 31), maximum=(1 << 31) - 1):
    _require(type(value) is int and minimum <= value <= maximum, "invalid " + label + " integer")
    return value


def quote_key(action_or_row):
    """Validated content identity; placement coordinates and queue modes differ.

    Native action modules enforce complete action field sets. Several actions
    may reference one quote, for example placing the same district on different
    tiles. This key must therefore omit those action-specific fields.
    """
    value = action_or_row
    _require(type(value) is dict and set(_KEYS) <= set(value), "missing quote identity")
    city_id = _integer(value["city_id"], "city id", 0, 0xFFFFFFFF)
    index = _integer(value["index"], "content index", 0)
    hash_value = _integer(value["hash"], "content hash", 0, 0xFFFFFFFF)
    category, item_type = value["category"], value["item_type"]
    _require(type(category) is str and category in _CATEGORIES, "invalid category")
    _require(type(item_type) is str and re.fullmatch(category.upper() + r"_[A-Z0-9_]{1,100}", item_type) is not None,
             "invalid item type")
    return city_id, category, index, hash_value, item_type


def key_for_action(action):
    _require(type(action) is dict and action.get("kind") in PRODUCTION_KINDS, "action is not quote-bearing production")
    return quote_key(action)


def _validate_row(row):
    _require(type(row) is dict and set(row) == set(_KEYS) | {"raw", "public"}, "unknown or missing row fields")
    key = quote_key(row)
    raw, public = row["raw"], row["public"]
    _require(type(raw) is dict and set(raw) == _RAW_FIELDS, "unknown or missing raw fields")
    _require(type(public) is dict and set(public) == _PUBLIC_FIELDS, "unknown or missing public fields")
    for name in ("cost", "progress", "turns_left"):
        _integer(raw[name], "raw " + name)
        if public[name] is not None:
            _integer(public[name], "public " + name)
    for name in ("formation_type", "base_cost"):
        if raw[name] is not None:
            _integer(raw[name], name)
    _require(type(raw["base_cost_visible"]) is bool, "base cost visibility must be Boolean")
    _require(type(public["never_completes"]) is bool, "never-completes flag must be Boolean")
    _require(public["cost_kind"] is None or (type(public["cost_kind"]) is str
             and public["cost_kind"] in ("base", "production")), "invalid displayed cost kind")
    is_unit = row["category"] == "unit"
    if not is_unit:
        _require(raw["formation_type"] is None and raw["base_cost"] is None
                 and raw["base_cost_visible"] is False, "non-unit has unit-only metadata")
    dynamic = not is_unit or raw["formation_type"] is not None
    if dynamic:
        _require(raw["base_cost_visible"] is False, "dynamic-cost row cannot expose fallback base cost")
        cost = raw["cost"] if raw["cost"] != 0 else None
        progress = raw["progress"] if raw["cost"] != 0 and raw["progress"] > 0 else None
        kind = "production" if cost is not None else None
    else:
        if raw["base_cost_visible"]:
            _require(raw["base_cost"] is not None and raw["base_cost"] != 0,
                     "visible fallback requires a nonzero unit base cost")
        cost = raw["base_cost"] if raw["base_cost_visible"] else None
        progress = None
        kind = "base" if cost is not None else None
    expected = {"cost": cost, "progress": progress,
                "turns_left": None if raw["turns_left"] == -1 else raw["turns_left"],
                "never_completes": raw["turns_left"] == -1, "cost_kind": kind}
    _require(public == expected, "display differs from pinned official UI projection")
    return key


def _candidate_keys(candidates):
    _require(type(candidates) is list and len(candidates) <= MAX_ROWS, "unbounded candidates")
    keys, by_id, seen = set(), {}, set()
    for row in candidates:
        _require(type(row) is dict and set(row) == {"candidate_id", "action"}, "malformed candidate envelope")
        cid, action = row["candidate_id"], row["action"]
        _require(type(cid) is str and bool(cid) and cid not in seen, "invalid or duplicate candidate ID")
        seen.add(cid)
        _require(type(action) is dict, "malformed candidate action")
        if action.get("kind") not in PRODUCTION_KINDS:
            continue
        key = key_for_action(action)
        keys.add(key)
        by_id[cid] = key
    return keys, by_id


def validate(value, player_id, turn, candidates=None, observation=None, *, ended=False):
    """Validate one native display sidecar and optional complete action join.

    An ended response may retain quote rows while its action list is empty.
    A live response supplied with candidates must have exactly the union of
    quote keys, with no invented rows and no missing displayed alternatives.
    """
    _integer(player_id, "player id", 0, 63)
    _integer(turn, "turn", 0)
    _require(type(ended) is bool, "ended must be Boolean")
    expected_fields = {"schema", "player_id", "turn", "source_sha256", "ui_sha256", "tooltip_sha256", "rows"}
    _require(type(value) is dict and set(value) == expected_fields, "unknown or missing sidecar fields")
    _require(value["schema"] == SCHEMA and type(value["player_id"]) is int and value["player_id"] == player_id
             and type(value["turn"]) is int and value["turn"] == turn, "identity or schema mismatch")
    _require(value["source_sha256"] == PROOF_SHA256 and value["ui_sha256"] == PANEL_SHA256
             and value["tooltip_sha256"] == TOOLTIP_SHA256, "source pin mismatch")
    rows = value["rows"]
    _require(type(rows) is list and len(rows) <= MAX_ROWS, "unbounded rows")
    keys = [_validate_row(row) for row in rows]
    _require(keys == sorted(set(keys)), "rows must be sorted by unique quote identity")
    if observation is not None:
        _require(type(observation) is dict and type(observation.get("player_id")) is int
                 and observation["player_id"] == player_id and type(observation.get("turn")) is int
                 and observation["turn"] == turn, "observation identity differs")
        cities = observation.get("cities")
        _require(type(cities) is list and len(cities) <= 4096, "unbounded owned cities")
        own = set()
        for city in cities:
            _require(type(city) is dict and type(city.get("owner")) is int and city["owner"] == player_id,
                     "foreign city in own observation")
            city_id = _integer(city.get("id"), "observed city id", 0, 0xFFFFFFFF)
            _require(city_id not in own, "duplicate own city")
            own.add(city_id)
        _require(all(key[0] in own for key in keys), "quote does not belong to an observed own city")
    if candidates is not None:
        offered, _ = _candidate_keys(candidates)
        if ended:
            _require(not candidates, "ended response must have empty candidates")
        else:
            _require(offered == set(keys), "live candidates do not exactly cover quoted production items")
    return value


def public_features(row):
    _validate_row(row)
    shown = row["public"]
    cost, progress, turns = (shown[name] for name in ("cost", "progress", "turns_left"))
    return [float(cost) if cost is not None else 0.0, float(cost is not None), float(shown["cost_kind"] == "base"),
            float(progress) if progress is not None else 0.0, float(progress is not None),
            float(turns) if turns is not None else 0.0, float(turns is not None),
            float(shown["never_completes"]), 1.0]


def validate_and_index(decisions, observation, *, ended=False):
    _require(type(decisions) is dict and type(observation) is dict, "missing boundary")
    candidates = decisions.get("candidates")
    _require(type(candidates) is list, "missing candidate list")
    sidecar = validate(decisions.get("production_info"), observation.get("player_id"), observation.get("turn"),
                       candidates=candidates, observation=observation, ended=ended)
    if ended:
        return {}
    _, by_id = _candidate_keys(candidates)
    quotes = {quote_key(row): row for row in sidecar["rows"]}
    return {cid: quotes[key] for cid, key in by_id.items()}
