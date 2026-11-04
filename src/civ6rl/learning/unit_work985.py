"""Strict, host-independent validation for the native UnitPanel work menu.

This module validates the published candidate/catalogue join. It does not
reimplement improvement or terrain legality and does not expose engine
selectors or movement tokens as policy features.
"""
from __future__ import annotations

import math
import re


_OPS = {"HARVEST_RESOURCE", "REMOVE_FEATURE", "REMOVE_IMPROVEMENT", "REPAIR", "REPAIR_ROUTE"}
_COMMON = {"kind", "unit_id", "x", "y", "plot_id", "charges", "moves_token"}
_BUILD = _COMMON | {"index", "hash", "item_type", "selector"}
_TERRAIN = _COMMON | {"operation", "hash"}
_MOVE = re.compile(r"[0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?\Z")
ACTION_WORK_FEATURES = ("work_charges", "work_charges_known")


def _uint(value, field, maximum=0xFFFFFFFF):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid unit-work {field}")
    return value


def _moves(token):
    if type(token) is not str or len(token) > 32 or _MOVE.fullmatch(token) is None:
        raise ValueError("malformed unit-work moves_token")
    amount = float(token)
    if not math.isfinite(amount) or not 0 < amount <= 2**31:
        raise ValueError("unbounded unit-work moves_token")
    return amount


def validate_action(action):
    if type(action) is not dict or action.get("kind") not in ("build_improvement", "terrain_operation"):
        raise ValueError("invalid unit-work action")
    build = action["kind"] == "build_improvement"
    if set(action) != (_BUILD if build else _TERRAIN):
        raise ValueError("invalid unit-work action fields")
    for key in ("unit_id", "x", "y", "plot_id", "charges", "hash"):
        _uint(action[key], key)
    _moves(action["moves_token"])
    if build:
        _uint(action["index"], "index")
        _uint(action["selector"], "selector")
        if (type(action["item_type"]) is not str or
                re.fullmatch(r"IMPROVEMENT_[A-Z0-9_]{1,100}", action["item_type"]) is None):
            raise ValueError("invalid unit-work item_type")
    elif action["operation"] not in _OPS:
        raise ValueError("unsupported unit-work operation")
    return action


def _wire(action):
    validate_action(action)
    fields = [action[key] for key in ("unit_id", "x", "y", "plot_id", "charges", "moves_token")]
    if action["kind"] == "build_improvement":
        fields = ["I", *fields, action["index"], action["hash"], action["item_type"], action["selector"]]
    else:
        fields = ["O", *fields, action["operation"], action["hash"]]
    return "|".join(map(str, fields))


def validate_and_index(decisions, observation, *, ended=False):
    """Validate the menu and join every active work candidate to its offer.

    Returns candidate_id -> source/target coordinates and the two appended
    numeric features (charge count and its known bit). On a closed boundary,
    the menu remains validated but candidates must be empty and the result is
    empty. Candidate IDs are used only as dictionary keys, never as features.
    """
    if type(ended) is not bool:
        raise ValueError("ended must be boolean")
    catalog = decisions.get("unit_work") if type(decisions) is dict else None
    if catalog is None:
        candidates = decisions.get("candidates") if type(decisions) is dict else None
        has_work = type(candidates) is list and any(
            type(row) is dict and type(row.get("action")) is dict and
            row["action"].get("kind") in ("build_improvement", "terrain_operation")
            for row in candidates
        )
        if has_work:
            raise ValueError("unit-work candidates require their native catalogue")
        return {}
    if type(observation) is not dict:
        raise ValueError("unit-work validation requires player observation")
    pid, turn = observation.get("player_id"), observation.get("turn")
    if (type(catalog) is not dict or set(catalog) != {"player_id", "turn", "enabled", "offers"}
            or type(catalog["player_id"]) is not int or catalog["player_id"] != pid
            or type(catalog["turn"]) is not int or catalog["turn"] != turn
            or type(catalog["enabled"]) is not bool
            or type(catalog["offers"]) is not list or len(catalog["offers"]) > 4096
            or (not catalog["enabled"] and catalog["offers"])):
        raise ValueError("foreign or malformed unit-work catalogue")
    units = observation.get("units")
    plots = observation.get("plots")
    width, height = observation.get("width"), observation.get("height")
    if (type(units) is not list or type(plots) is not list or type(width) is not int
            or type(height) is not int or width <= 0 or height <= 0):
        raise ValueError("malformed observation for unit-work join")
    own = {}
    observed_moves = {}
    for unit in units:
        if type(unit) is not dict:
            raise ValueError("malformed owned unit")
        uid = unit.get("id")
        if type(uid) is not int or uid in own:
            raise ValueError("duplicate or invalid owned unit id")
        if unit.get("owner") != pid:
            raise ValueError("foreign unit in own observation roster")
        x, y = unit.get("x"), unit.get("y")
        if type(x) is not int or type(y) is not int or not (0 <= x < width and 0 <= y < height):
            raise ValueError("invalid owned unit coordinates")
        own[uid] = (x, y)
        moves = unit.get("moves")
        if type(moves) not in (int, float) or not math.isfinite(moves) or moves < 0:
            raise ValueError("invalid public own-unit moves")
        observed_moves[uid] = float(moves)
    visible = {}
    for plot in plots:
        if type(plot) is not dict:
            raise ValueError("malformed visible plot")
        if plot.get("visible") is not True or plot.get("revealed") is not True:
            raise ValueError("unit-work observation contains non-visible plot")
        index, x, y = plot.get("index"), plot.get("x"), plot.get("y")
        if (type(index) is not int or type(x) is not int or type(y) is not int
                or not (0 <= x < width and 0 <= y < height) or index != y * width + x
                or index in visible):
            raise ValueError("invalid or duplicate visible plot")
        visible[index] = (x, y)

    offers, seen, unit_charges = {}, set(), {}
    for row in catalog["offers"]:
        if type(row) is not dict or set(row) != {"action", "legal"} or type(row["legal"]) is not bool:
            raise ValueError("malformed unit-work offer")
        action = validate_action(row["action"])
        key = _wire(action)
        if key in seen:
            raise ValueError("duplicate unit-work offer")
        seen.add(key)
        ux, uy = own.get(action["unit_id"], (None, None))
        if (ux, uy) != (action["x"], action["y"]):
            raise ValueError("unit-work offer does not match own unit coordinates")
        if _moves(action["moves_token"]) != observed_moves[action["unit_id"]]:
            raise ValueError("unit-work moves_token differs from public unit observation")
        previous_charges = unit_charges.setdefault(action["unit_id"], action["charges"])
        if previous_charges != action["charges"]:
            raise ValueError("unit-work offers disagree on the unit's charge count")
        if action["plot_id"] != action["y"] * width + action["x"]:
            raise ValueError("unit-work plot identity differs from coordinates")
        if visible.get(action["plot_id"]) != (action["x"], action["y"]):
            raise ValueError("unit-work target is not officially visible and revealed")
        offers[key] = (action, row["legal"])

    candidates = decisions.get("candidates")
    if type(candidates) is not list or len(candidates) > 100_000:
        raise ValueError("malformed candidate catalogue")
    candidate_actions = []
    for candidate in candidates:
        if type(candidate) is not dict or set(candidate) != {"candidate_id", "action"}:
            raise ValueError("malformed candidate")
        action = candidate["action"]
        if type(action) is dict and action.get("kind") in ("build_improvement", "terrain_operation"):
            candidate_actions.append((candidate["candidate_id"], action))
    if ended:
        if candidates:
            raise ValueError("closed response must not carry candidates")
        return {}
    legal = {_wire(action) for action, is_legal in offers.values() if is_legal}
    joined, candidate_wires = {}, set()
    for cid, action in candidate_actions:
        if type(cid) is not str or not cid:
            raise ValueError("unit-work candidate ID missing")
        key = _wire(action)
        if key not in legal:
            raise ValueError("work candidate is not an exact legal native offer")
        if key in candidate_wires:
            raise ValueError("duplicate unit-work candidate action")
        candidate_wires.add(key)
        offered, _ = offers[key]
        if action != offered:
            raise ValueError("work candidate differs from native offer")
        if cid in joined:
            raise ValueError("duplicate work candidate ID")
        x, y = own[action["unit_id"]]
        joined[cid] = {"source": (x, y), "target": (action["x"], action["y"]),
                       "features": [float(action["charges"]), 1.0]}
    if candidate_wires != legal:
        raise ValueError("legal unit-work offers and candidates differ")
    return joined
