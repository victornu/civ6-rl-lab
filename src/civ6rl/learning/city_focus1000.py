"""Player-public city focus: one official UI cycle, never an arbitrary setter."""
from __future__ import annotations

V6 = "civ6rl.recurrent_policy.v6"
V7 = "civ6rl.recurrent_policy.v7"
DECISIONS_SCHEMA = "civ6rl.saved_decisions1000.v1"
CATALOG_SCHEMA = "civ6rl.native_city_focus1000.v1"
YIELDS = tuple("YIELD_" + name for name in ("FOOD", "PRODUCTION", "GOLD", "SCIENCE", "CULTURE", "FAITH"))
ENTITY_FEATURES = tuple("focus:" + yield_type + ":" + state
                        for yield_type in YIELDS for state in ("favored", "disfavored", "known"))
ACTION_FEATURES = ("focus_favored", "focus_disfavored", "focus_known")
ACTION_FIELDS = {"kind", "city_id", "yield_type", "yield_index", "favored", "disfavored"}


def _require(ok, message):
    if not ok: raise ValueError(message)


def validate_and_index(decisions, observation, *, ended=False):
    """Require six public rows and six exact click offers per owned city.

    Both native flags may be true. Retaining them is necessary: their display
    and click precedence is source-defined, not a normalized ternary setter.
    Closed menus retain state for value bootstraps but offer no focus action.
    """
    _require(type(ended) is bool, "focus closure must be Boolean")
    _require(type(decisions) is dict and decisions.get("schema") in
             (DECISIONS_SCHEMA, 'civ6rl.saved_decisions.city_management.v1'),
             "unsupported focus decision schema")
    _require(type(observation) is dict and type(observation.get("player_id")) is int
             and 0 <= observation["player_id"] <= 63 and type(observation.get("turn")) is int,
             "focus needs public player identity")
    pid, turn = observation["player_id"], observation["turn"]
    _require(decisions.get("player_id") == pid and decisions.get("turn") == turn,
             "focus decision boundary differs")
    owned = observation.get("cities")
    _require(type(owned) is list and len(owned) <= 4096, "focus owned city roster missing")
    cities = {}
    for city in owned:
        _require(type(city) is dict and type(city.get("id")) is int and city["id"] >= 0
                 and type(city.get("owner")) is int and city["owner"] == pid
                 and city["id"] not in cities, "focus city is foreign or duplicated")
        cities[city["id"]] = city
    catalogs = decisions.get("city_focus_catalogs")
    _require(type(catalogs) is list and len(catalogs) == len(cities), "focus catalogs must match owned roster")
    by_city = {}; expected = {}; index_types = {}
    for catalog in catalogs:
        _require(type(catalog) is dict and set(catalog) == {"schema", "player_id", "turn", "city_id", "yields"}
                 and catalog["schema"] == CATALOG_SCHEMA
                 and type(catalog["player_id"]) is int and catalog["player_id"] == pid
                 and type(catalog["turn"]) is int and catalog["turn"] == turn
                 and type(catalog["city_id"]) is int and catalog["city_id"] in cities
                 and catalog["city_id"] not in by_city, "focus catalogue identity differs")
        cid = catalog["city_id"]; rows = catalog["yields"]
        _require(type(rows) is list and len(rows) == 6, "focus catalogue requires six yields")
        values = {}; indices = set()
        for yield_type, row in zip(YIELDS, rows):
            _require(type(row) is dict and set(row) == {"yield_type", "index", "favored", "disfavored"}
                     and row["yield_type"] == yield_type
                     and type(row["index"]) is int and 0 <= row["index"] <= 0x7fffffff
                     and row["index"] not in indices
                     and type(row["favored"]) is bool and type(row["disfavored"]) is bool,
                     "focus yield order, identity or flags differ")
            indices.add(row["index"])
            _require(yield_type not in index_types or index_types[yield_type] == row["index"],
                     "focus yield indices differ between cities")
            index_types[yield_type] = row["index"]
            values[yield_type] = row
            expected[(cid, yield_type)] = dict(kind="city_focus", city_id=cid, yield_type=yield_type,
                yield_index=row["index"], favored=row["favored"], disfavored=row["disfavored"])
        by_city[cid] = values
    _require(set(by_city) == set(cities), "focus roster differs")
    candidates = decisions.get("candidates")
    _require(type(candidates) is list, "focus candidates missing")
    choices = {}; seen = set()
    for candidate in candidates:
        _require(type(candidate) is dict and type(candidate.get("action")) is dict,
                 "focus candidate envelope malformed")
        action = candidate["action"]
        if action.get("kind") != "city_focus": continue
        _require(set(action) == ACTION_FIELDS and type(action["city_id"]) is int
                 and type(action["yield_index"]) is int and type(action["yield_type"]) is str
                 and type(action["favored"]) is bool and type(action["disfavored"]) is bool,
                 "focus action must be an exact UI cycle offer")
        key = (action["city_id"], action["yield_type"])
        _require(key not in seen and key in expected and action == expected[key],
                 "focus action differs from public prior state")
        cid = candidate.get("candidate_id")
        _require(type(cid) is str and bool(cid) and cid not in choices, "focus candidate identity duplicated")
        seen.add(key)
        choices[cid] = [float(action["favored"]), float(action["disfavored"]), 1.0]
    _require(not seen if ended else seen == set(expected), "focus menu must contain every offered UI cycle")
    return {"cities": by_city, "candidates": choices}


def entity_features(rows):
    if rows is None: return [0.0] * len(ENTITY_FEATURES)
    return [value for yield_type in YIELDS
            for value in (float(rows[yield_type]["favored"]), float(rows[yield_type]["disfavored"]), 1.0)]
