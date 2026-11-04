"""Public per-city purchase quotes, separate from the selectable action menu.

This module validates only the published JSON boundary. Native legality and
pricing remain facts reported by the collector; this code does not reproduce
those rules and does not alter candidates or action masks.
"""
from __future__ import annotations

import math
import re


CONTRACT = "public_native_purchase_catalog_all_visible_quotes_v1"
ENTITY_FEATURES = (
    "kind_purchase_quote", "purchase_quote_unit", "purchase_quote_building",
    "purchase_quote_plot", "purchase_quote_gold", "purchase_quote_faith",
    "purchase_quote_cost", "purchase_quote_command_legal",
    "purchase_quote_can_afford", "purchase_quote_city_x_norm",
    "purchase_quote_city_y_norm", "purchase_quote_target_x_norm",
    "purchase_quote_target_y_norm", "purchase_quote_target_known",
)
CAT_FIELDS = ("purchase_category", "purchase_item_type", "purchase_currency")

_UNIT_BUILDING_FIELDS = {
    "kind", "city_id", "category", "index", "hash", "item_type", "currency", "price_token",
}
_PLOT_FIELDS = {"kind", "city_id", "plot_id", "x", "y", "currency", "price_token"}
_CITY_FIELDS = {"id", "x", "y", "population", "owner", "yields"}
_CATALOG_FIELDS = {"player_id", "turn", "city_id", "offers"}
_OFFER_FIELDS = {"action", "command_legal", "can_afford"}
_PRICE = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e[+-][0-9]+)?\Z")
_ITEM_TYPE = re.compile(r"(?:UNIT|BUILDING)_[A-Z0-9_]{1,100}\Z")
_MAX_CITIES = 4096
_MAX_OFFERS = 4096
_MAX_MAP_PLOTS = 16384


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _uint(value, maximum, name):
    _require(type(value) is int and 0 <= value <= maximum, "invalid purchase " + name)
    return value


def _price(token):
    _require(type(token) is str and len(token) <= 32 and _PRICE.fullmatch(token) is not None,
             "malformed public purchase price")
    value = float(token)
    _require(math.isfinite(value) and 0 <= value <= 2**53 and format(value, ".17g") == token,
             "purchase price must be finite, bounded and canonical")
    return value


def _validate_action(action, city_id, width=None, height=None):
    _require(type(action) is dict and action.get("kind") == "purchase",
             "purchase offer action is malformed")
    _require(type(action.get("city_id")) is int and action["city_id"] == city_id,
             "purchase offer belongs to another city")
    _price(action.get("price_token"))
    if "plot_id" in action:
        _require(set(action) == _PLOT_FIELDS
                 and action["currency"] == "YIELD_GOLD",
                 "plot purchase offer fields or currency differ")
        if width is None or height is None:
            _uint(action["plot_id"], _MAX_MAP_PLOTS - 1, "plot id")
            x = _uint(action["x"], _MAX_MAP_PLOTS - 1, "plot x")
            y = _uint(action["y"], _MAX_MAP_PLOTS - 1, "plot y")
        else:
            plot_id = _uint(action["plot_id"], width * height - 1, "plot id")
            x = _uint(action["x"], width - 1, "plot x")
            y = _uint(action["y"], height - 1, "plot y")
            _require(plot_id == y * width + x, "plot purchase coordinates differ from its public index")
        return "plot", (x, y), action["currency"]
    _require(set(action) == _UNIT_BUILDING_FIELDS
             and action["category"] in ("unit", "building")
             and action["currency"] in ("YIELD_GOLD", "YIELD_FAITH"),
             "unit/building purchase offer fields differ")
    _uint(action["index"], 0x7fffffff, "content index")
    _uint(action["hash"], 0xffffffff, "content hash")
    _require(type(action["item_type"]) is str and _ITEM_TYPE.fullmatch(action["item_type"]) is not None
             and action["item_type"].startswith(action["category"].upper() + "_"),
             "purchase content type differs from category")
    return action["category"], action["item_type"], action["currency"]


def _identity(action):
    if "plot_id" in action:
        return ("plot", action["plot_id"], action["currency"])
    return (action["category"], action["index"], action["hash"], action["currency"])


def validate_and_index(decisions, obs, ended=False):
    """Validate all public quotes and exact selectable-candidate joins.

    Locations include unaffordable and command-illegal quotes. Candidates must
    correspond exactly to legal, affordable offers; no candidate is created or
    removed here. Only the separate pending-diplomacy menu omits this catalogue;
    an absent ordinary catalogue is an error, not an empty observation.
    """
    _require(type(ended) is bool, "purchase closure must be Boolean")
    _require(type(decisions) is dict and type(obs) is dict,
             "purchase decisions and observation must be objects")
    if "purchases" not in decisions:
        _require(decisions.get("schema") == "civ6rl.diplomacy_decisions987.v1",
                 "ordinary decisions require their public purchase catalog")
        return {"locations": []}

    pid, turn = obs.get("player_id"), obs.get("turn")
    _uint(pid, 63, "player")
    _uint(turn, 0x7fffffff, "turn")
    width, height = obs.get("width"), obs.get("height")
    _require(type(width) is int and type(height) is int and width > 0 and height > 0
             and width * height <= _MAX_MAP_PLOTS,
             "invalid public purchase map dimensions")
    _require(type(decisions.get("player_id")) is int and decisions["player_id"] == pid
             and type(decisions.get("turn")) is int and decisions["turn"] == turn,
             "purchase decisions boundary differs")

    roster = obs.get("cities")
    _require(type(roster) is list and len(roster) <= _MAX_CITIES,
             "public own-city roster missing or unbounded")
    cities = {}
    for city in roster:
        _require(type(city) is dict and set(city) == _CITY_FIELDS
                 and type(city.get("owner")) is int and city["owner"] == pid,
                 "purchase city is foreign or malformed")
        cid = _uint(city.get("id"), 0xffffffff, "city id")
        _uint(city.get("x"), width - 1, "city x")
        _uint(city.get("y"), height - 1, "city y")
        _require(cid not in cities, "duplicate public own city")
        cities[cid] = city

    catalogs = decisions["purchases"]
    _require(type(catalogs) is list and len(catalogs) == len(cities),
             "purchase catalogs must match the exact own-city roster")
    catalog_ids = []
    locations = []
    expected = {}
    total = 0
    for catalog in catalogs:
        _require(type(catalog) is dict and set(catalog) == _CATALOG_FIELDS
                 and type(catalog.get("player_id")) is int and catalog["player_id"] == pid
                 and type(catalog.get("turn")) is int and catalog["turn"] == turn,
                 "purchase catalog identity differs")
        cid = _uint(catalog.get("city_id"), 0xffffffff, "catalog city")
        _require(cid in cities and cid not in catalog_ids,
                 "purchase catalog city is foreign or duplicated")
        catalog_ids.append(cid)
        offers = catalog.get("offers")
        _require(type(offers) is list and len(offers) <= _MAX_OFFERS,
                 "purchase offers missing or unbounded")
        seen = set()
        for offer in offers:
            _require(type(offer) is dict and set(offer) == _OFFER_FIELDS
                     and type(offer.get("command_legal")) is bool
                     and type(offer.get("can_afford")) is bool,
                     "purchase offer flags or fields differ")
            action = offer.get("action")
            identity = _validate_action(action, cid, width, height)
            _require(identity not in seen, "duplicate public purchase offer")
            seen.add(identity)
            total += 1
            _require(total <= _MAX_OFFERS, "purchase catalog exceeds global offer bound")
            key = (cid, _identity(action))
            expected[key] = offer
            locations.append({"city": cities[cid], "offer": offer})
    _require(catalog_ids == sorted(cities),
             "purchase catalogs must retain the ordered complete own-city roster")

    candidates = decisions.get("candidates")
    _require(type(candidates) is list, "purchase candidate menu missing")
    found = set()
    candidate_ids = set()
    for candidate in candidates:
        _require(type(candidate) is dict and type(candidate.get("action")) is dict,
                 "purchase candidate envelope malformed")
        action = candidate["action"]
        if action.get("kind") != "purchase":
            continue
        _validate_action(action, action.get("city_id"), width, height)
        key = (action["city_id"], _identity(action))
        offer = expected.get(key)
        candidate_id = candidate.get("candidate_id")
        _require(offer is not None and action == offer["action"]
                 and offer["command_legal"] is True and offer["can_afford"] is True
                 and key not in found and type(candidate_id) is str and bool(candidate_id)
                 and candidate_id not in candidate_ids,
                 "purchase candidate differs from an eligible public quote")
        found.add(key)
        candidate_ids.add(candidate_id)
    eligible = {key for key, offer in expected.items()
                if offer["command_legal"] and offer["can_afford"]}
    _require(not found if ended else found == eligible,
             "purchase candidates must exactly match eligible public offers")

    locations.sort(key=lambda item: (
        item["city"]["y"], item["city"]["x"], item["city"]["id"],
        *_identity(item["offer"]["action"]),
    ))
    return {"locations": locations}


def entity_features(city, offer, width, height):
    """Return the fixed 14-float projection for one validated public quote."""
    _require(type(city) is dict and type(offer) is dict and type(width) is int and type(height) is int
             and width > 0 and height > 0 and width * height <= _MAX_MAP_PLOTS,
             "invalid purchase entity inputs")
    action = offer.get("action")
    category, target, currency = _validate_action(action, action.get("city_id") if type(action) is dict else None,
                                                   width, height)
    _require(type(offer.get("command_legal")) is bool and type(offer.get("can_afford")) is bool,
             "invalid purchase entity offer flags")
    x = _uint(city.get("x"), width - 1, "city x")
    y = _uint(city.get("y"), height - 1, "city y")
    _require(type(city.get("id")) is int and city["id"] == action["city_id"],
             "purchase quote and entity city differ")
    cost = _price(action["price_token"])
    plot = category == "plot"
    return [
        1.0, float(category == "unit"), float(category == "building"), float(plot),
        float(currency == "YIELD_GOLD"), float(currency == "YIELD_FAITH"), cost,
        float(offer["command_legal"]), float(offer["can_afford"]),
        x / (width - 1) if width > 1 else 0.0,
        y / (height - 1) if height > 1 else 0.0,
        target[0] / (width - 1) if plot and width > 1 else 0.0,
        target[1] / (height - 1) if plot and height > 1 else 0.0,
        float(plot),
    ]


def entity_categories(offer, vocabulary):
    """Return frozen-vocabulary IDs; plots have no fabricated GameInfo type."""
    _require(type(offer) is dict and type(offer.get("action")) is dict,
             "purchase categorical offer malformed")
    action = offer["action"]
    category, _, currency = _validate_action(action, action.get("city_id"))
    _require(hasattr(vocabulary, "lookup"), "purchase vocabulary lookup unavailable")
    if category == "plot":
        return [0, 0, vocabulary.lookup("action.currency", currency)]
    return [vocabulary.lookup("action.category", category),
            vocabulary.lookup("action.item_type", action["item_type"]),
            vocabulary.lookup("action.currency", currency)]
