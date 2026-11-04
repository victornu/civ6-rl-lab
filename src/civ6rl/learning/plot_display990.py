"""Strict public PlotTooltip projection; no native or privileged dependency.

Missing yield lines are unknown numeric yields, not known zero: the official
tooltip omits nonpositive values and entire branches. Display presence itself
is public and is represented separately from the underlying numerical value.
"""
from __future__ import annotations

import math
import re

SCHEMA = "civ6rl.plot_display990.v1"
SOURCE_SHA256 = "fd73dda3ef4e98c69ab75bffe57e17610c226814c97fa98d103cac27a392f1c4"
CAPABILITY_SOURCE_SHA256 = "0e81de39aa32033e4b3ed0660e72f6ca63f3f3bec28ec35d3b61ef780f1ceb0c"
MAX_PLOTS = 16384
YIELDS = ("food", "production", "gold", "science", "culture", "faith")
YIELD_TYPES = tuple("YIELD_" + name.upper() for name in YIELDS)
YIELD_GROUPS = ("plot_yields", "specialist_yields", "district_yields")
DISTRICT_STATUSES = ("city", "complete", "constructing", "pillaged")
MAP_CATEGORICAL_FIELDS = ("improvement_type", "district_type")
MAP_DISPLAY_CHANNELS = (
    "plot_display_known", "improvement_displayed", "improvement_pillaged",
    "improvement_pillaged_known", "district_displayed",
    "district_status_city", "district_status_complete", "district_status_constructing",
    "district_status_pillaged", "district_status_known", "plot_yields_enabled",
    "worked_count", "worked_count_known",
) + tuple(feature for group in YIELD_GROUPS for name in YIELDS
          for feature in (group + ":" + name, group + ":" + name + "_known"))
_TYPE = re.compile(r"[A-Z][A-Z0-9_]{0,127}\Z")
_NUMBER = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")
_TOP = {"schema", "player_id", "turn", "width", "height", "yields_enabled",
        "source_sha256", "capability_source_sha256", "plots"}
_ROW = {"plot_id", "improvement", "district", "worked", *YIELD_GROUPS}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, low, high, name):
    _require(type(value) is int and low <= value <= high, "invalid plot display " + name)
    return value


def public_number(value):
    _require(type(value) is str and 1 <= len(value) <= 64 and _NUMBER.fullmatch(value) is not None,
             "plot yield must be its bounded public numeric text")
    number = float(value)
    _require(math.isfinite(number) and 0 < number <= 1e12, "plot yield must be finite and positive")
    return number


def _content_type(value, prefix):
    _require(type(value) is str and _TYPE.fullmatch(value) is not None and value.startswith(prefix),
             "invalid public plot content type")


def validate_and_index(value, player_id, turn, player_map):
    """Return rows by plot ID only after an exact revealed-map boundary join."""
    _integer(player_id, 0, 63, "player")
    _integer(turn, 0, 1_000_000, "turn")
    _require(type(value) is dict and set(value) == _TOP and value["schema"] == SCHEMA,
             "plot display schema or fields differ")
    _require(type(value["player_id"]) is int and value["player_id"] == player_id
             and type(value["turn"]) is int and value["turn"] == turn,
             "plot display player/turn differs")
    width = _integer(value["width"], 1, MAX_PLOTS, "width")
    height = _integer(value["height"], 1, MAX_PLOTS, "height")
    _require(width * height <= MAX_PLOTS, "plot display map exceeds bound")
    _require(value["source_sha256"] == SOURCE_SHA256
             and value["capability_source_sha256"] == CAPABILITY_SOURCE_SHA256,
             "plot display source pin differs")
    _require(type(value["yields_enabled"]) is bool, "plot yield capability must be Boolean")
    _require(type(player_map) is dict and player_map.get("schema") == "civ6rl.saved_player_map933.v1"
             and type(player_map.get("player_id")) is int and player_map["player_id"] == player_id
             and type(player_map.get("turn")) is int and player_map["turn"] == turn
             and type(player_map.get("width")) is int and player_map["width"] == width
             and type(player_map.get("height")) is int and player_map["height"] == height,
             "plot display and revealed map boundaries differ")
    map_rows = player_map.get("plots")
    _require(type(map_rows) is list and len(map_rows) <= width * height,
             "invalid revealed map rows")
    revealed = {}
    for row in map_rows:
        _require(type(row) is dict and row.get("revealed") is True,
                 "plot display requires revealed map rows")
        index = _integer(row.get("index"), 0, width * height - 1, "revealed index")
        x = _integer(row.get("x"), 0, width - 1, "revealed x")
        y = _integer(row.get("y"), 0, height - 1, "revealed y")
        _require(index == y * width + x and index not in revealed
                 and type(row.get("visible")) is bool and type(row.get("impassable")) is bool,
                 "revealed map geometry/state differs")
        _integer(row.get("owner"), -1, 63, "revealed owner")
        revealed[index] = row
    rows = value["plots"]
    _require(type(rows) is list and len(rows) <= width * height, "invalid plot display rows")
    result = {}
    for row in rows:
        _require(type(row) is dict and set(row) == _ROW, "plot display row fields differ")
        index = _integer(row["plot_id"], 0, width * height - 1, "plot id")
        _require(index in revealed and index not in result, "extra/duplicate/unrevealed plot display")
        if row["worked"] is not None:
            _integer(row["worked"], 1, 2147483647, "worked count")
            _require(revealed[index]["owner"] == player_id, "foreign worked count is not public")
        improvement, district = row["improvement"], row["district"]
        _require(improvement is None or district is None, "plot display branches overlap")
        if improvement is not None:
            _require(type(improvement) is dict and set(improvement) == {"type", "pillaged"}
                     and type(improvement["pillaged"]) is bool, "invalid improvement display")
            _content_type(improvement["type"], "IMPROVEMENT_")
        if district is not None:
            _require(type(district) is dict and set(district) == {"type", "status"}
                     and type(district["status"]) is str and district["status"] in DISTRICT_STATUSES,
                     "invalid district display")
            _content_type(district["type"], "DISTRICT_")
        for group in YIELD_GROUPS:
            yields = row[group]
            _require(type(yields) is list and len(yields) <= len(YIELD_TYPES), "unbounded plot yield display")
            types = []
            for item in yields:
                _require(type(item) is dict and set(item) == {"type", "value"}, "plot yield fields differ")
                _require(type(item["type"]) is str and item["type"] in YIELD_TYPES,
                         "unsupported public plot yield type; refusing omission")
                public_number(item["value"])
                types.append(item["type"])
            _require(types == sorted(set(types)), "plot yields must be unique and sorted")
            _require(value["yields_enabled"] or not yields, "disabled plot yields were supplied")
        if district is None or district["status"] == "city":
            _require(not row["specialist_yields"] and not row["district_yields"], "non-district yield branch differs")
        else:
            _require(not row["plot_yields"], "district branch contains plot yields")
            _require(not row["specialist_yields"] or revealed[index]["owner"] == player_id,
                     "foreign specialist yields are not public in this branch")
        # City/district tooltip branches run before the impassable branch.
        if revealed[index]["impassable"] and district is None:
            _require(improvement is None and district is None and not any(row[g] for g in YIELD_GROUPS),
                     "impassable plot exposes tooltip values")
        result[index] = row
    _require(list(result) == sorted(revealed), "plot display must exactly cover the ordered revealed map")
    return result


def numeric_features(row, yields_enabled):
    """Project validated public lines only; omitted yields keep known=0."""
    values = {name: 0.0 for name in MAP_DISPLAY_CHANNELS}
    values["plot_display_known"] = 1.0
    values["plot_yields_enabled"] = float(yields_enabled)
    if row["worked"] is not None:
        values["worked_count"] = float(row["worked"])
        values["worked_count_known"] = 1.0
    if row["improvement"] is not None:
        values["improvement_displayed"] = 1.0
        values["improvement_pillaged"] = float(row["improvement"]["pillaged"])
        values["improvement_pillaged_known"] = 1.0
    if row["district"] is not None:
        values["district_displayed"] = values["district_status_known"] = 1.0
        values["district_status_" + row["district"]["status"]] = 1.0
    for group in YIELD_GROUPS:
        for item in row[group]:
            name = group + ":" + item["type"][6:].lower()
            values[name] = public_number(item["value"])
            values[name + "_known"] = 1.0
    return [values[name] for name in MAP_DISPLAY_CHANNELS]
