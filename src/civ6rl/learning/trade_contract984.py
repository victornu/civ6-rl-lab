"""Pure validation and observation binding for official trade catalogues.

This module deliberately contains no native runtime, cache, or Windows imports.
"""

from __future__ import annotations

import math
from typing import Any


def uint(value: Any, label: str, maximum: int = 0xFFFFFFFF) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid {label}")
    return value


def _shape(value: Any, fields: str) -> dict:
    if type(value) is not dict or set(value) != set(fields.split()):
        raise ValueError("Invalid trade fields")
    return value


def _list(value: Any, maximum: int = 4096) -> list:
    if type(value) is not list or len(value) > maximum:
        raise ValueError("Unbounded trade list")
    return value


def _number(value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or abs(value) >= 1e12:
        raise ValueError("Invalid trade number")


def _yields(value: Any) -> None:
    _shape(value, "values majority_religion religion_pressure path_bonus tooltip")
    for number in _list(value["values"], 64):
        _number(number)
    # Religion id -1 is the native no-majority sentinel.
    if type(value["majority_religion"]) is not int or not -1 <= value["majority_religion"] <= 0x7FFFFFFF:
        raise ValueError("Invalid trade religion")
    _number(value["religion_pressure"])
    if type(value["path_bonus"]) is not bool or type(value["tooltip"]) is not str or len(value["tooltip"].encode("utf-8")) > 32768:
        raise ValueError("Invalid trade yield metadata")


def _validate_obs(obs: Any) -> tuple[int, int, int, int, dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    if type(obs) is not dict:
        raise ValueError("Invalid observation")
    pid = uint(obs.get("player_id"), "observation player", 63)
    turn = uint(obs.get("turn"), "observation turn", 0x7FFFFFFF)
    width = uint(obs.get("width"), "observation width", 16384)
    height = uint(obs.get("height"), "observation height", 16384)
    if not width or not height or width * height > 16384:
        raise ValueError("Invalid observation map")
    units: dict[int, tuple[int, int]] = {}
    for unit in _list(obs.get("units")):
        if type(unit) is not dict:
            raise ValueError("Invalid observed unit")
        owner = uint(unit.get("owner"), "unit owner", 63)
        # Read and bind only the observer's own units.
        if owner != pid:
            continue
        unit_id = uint(unit.get("id"), "unit id")
        x, y = uint(unit.get("x"), "unit x", width - 1), uint(unit.get("y"), "unit y", height - 1)
        if unit_id in units:
            raise ValueError("Duplicate own observed unit")
        units[unit_id] = (x, y)
    cities: dict[int, tuple[int, int]] = {}
    for city in _list(obs.get("cities")):
        if type(city) is not dict:
            raise ValueError("Invalid observed city")
        owner = uint(city.get("owner"), "city owner", 63)
        if owner != pid:
            continue
        city_id = uint(city.get("id"), "city id")
        x, y = uint(city.get("x"), "city x", width - 1), uint(city.get("y"), "city y", height - 1)
        if city_id in cities:
            raise ValueError("Duplicate own observed city")
        cities[city_id] = (x, y)
    return pid, turn, width, height, units, cities


def _catalog_header(value: Any, obs: Any, expected: str) -> tuple[int, int, int, int, dict, dict]:
    pid, turn, obs_width, obs_height, units, cities = _validate_obs(obs)
    _shape(value, expected)
    if type(value["player_id"]) is not int or type(value["turn"]) is not int or value["player_id"] != pid or value["turn"] != turn:
        raise ValueError("Stale or foreign trade catalogue")
    width = uint(value["width"], "trade width", 16384)
    height = uint(value["height"], "trade height", 16384)
    if not width or not height or width * height > 16384 or (width, height) != (obs_width, obs_height):
        raise ValueError("Trade grid differs from observation")
    return pid, turn, width, height, units, cities


def _route_action(trader: dict, destination: dict) -> dict:
    return {
        "kind": "make_trade_route",
        "unit_id": trader["unit_id"],
        "origin_owner": trader["origin_owner"],
        "origin_city": trader["origin_city"],
        "destination_owner": destination["owner"],
        "destination_city": destination["city_id"],
        "x0": destination["x"],
        "y0": destination["y"],
        "x1": trader["x"],
        "y1": trader["y"],
    }


def validate_routes(value: Any, obs: Any) -> list[tuple[dict, dict, dict]]:
    """Validate a route catalogue and bind its own-player claims to ``obs``.

    Foreign destination rows remain valid when offered by the native catalogue;
    this function never looks them up in observer city data.
    """
    pid, _, width, height, units, cities = _catalog_header(
        value, obs, "player_id turn width height traders active_routes"
    )
    active_ids: list[int] = []
    active_origins: list[int] = []
    for route in _list(value["active_routes"]):
        _shape(route, "unit_id origin_owner origin_city destination_owner destination_city")
        uint(route["unit_id"], "active trader")
        uint(route["origin_owner"], "active origin owner", 63)
        uint(route["origin_city"], "active origin city")
        uint(route["destination_owner"], "active destination owner", 63)
        uint(route["destination_city"], "active destination city")
        if route["origin_owner"] != pid:
            raise ValueError("Foreign outgoing trade route")
        active_ids.append(route["unit_id"])
        active_origins.append(route["origin_city"])
    if active_ids != sorted(set(active_ids)):
        raise ValueError("Duplicate active traders")
    if any(city_id not in cities for city_id in active_origins):
        raise ValueError("Outgoing route from absent own city")

    trader_ids: list[int] = []
    total_destinations = 0
    confirmed: list[tuple[dict, dict, dict]] = []
    for trader in _list(value["traders"]):
        _shape(trader, "unit_id x y origin_owner origin_city destinations")
        unit_id = uint(trader["unit_id"], "trader unit")
        x, y = uint(trader["x"], "trader x", width - 1), uint(trader["y"], "trader y", height - 1)
        origin_owner = uint(trader["origin_owner"], "origin owner", 63)
        origin_city = uint(trader["origin_city"], "origin city")
        if origin_owner != pid or units.get(unit_id) != (x, y) or cities.get(origin_city) != (x, y):
            raise ValueError("Trade roster differs from own observation")
        trader_ids.append(unit_id)
        destinations: list[tuple[int, int]] = []
        for destination in _list(trader["destinations"]):
            _shape(destination, "owner city_id x y name path portal_entrances portal_exits repeat_route can_confirm has_post origin_yields destination_yields")
            owner = uint(destination["owner"], "destination owner", 63)
            city_id = uint(destination["city_id"], "destination city")
            dx = uint(destination["x"], "destination x", width - 1)
            dy = uint(destination["y"], "destination y", height - 1)
            if type(destination["name"]) is not str or len(destination["name"].encode("utf-8")) > 32768:
                raise ValueError("Invalid trade city name")
            for key in ("repeat_route", "can_confirm", "has_post"):
                if type(destination[key]) is not bool:
                    raise ValueError("Invalid trade flags")
            if not destination["repeat_route"] and not destination["can_confirm"]:
                raise ValueError("Nonrepeat destination blocked")
            path = _list(destination["path"], 16384)
            if not path or any(type(plot) is not int or not 0 <= plot < width * height for plot in path):
                raise ValueError("Invalid native trade path")
            if path[0] != y * width + x or path[-1] != dy * width + dx:
                raise ValueError("Trade path endpoints differ from source/destination")
            for key in ("portal_entrances", "portal_exits"):
                steps: list[int] = []
                for portal in _list(destination[key], len(path)):
                    _shape(portal, "step plot")
                    step = uint(portal["step"], "portal step", len(path))
                    plot = portal["plot"]
                    if step == 0 or type(plot) is not int or not -1 <= plot < width * height:
                        raise ValueError("Invalid portal")
                    steps.append(step)
                if steps != sorted(set(steps)):
                    raise ValueError("Duplicate portal step")
            _yields(destination["origin_yields"])
            _yields(destination["destination_yields"])
            destinations.append((owner, city_id))
            total_destinations += 1
            # Own destinations can be cross-checked. Foreign ones are intentionally
            # accepted from the official trade menu, which can list beyond vision.
            if owner == pid and cities.get(city_id) != (dx, dy):
                raise ValueError("Own trade destination differs from observation")
            if destination["can_confirm"]:
                confirmed.append((_route_action(trader, destination), trader, destination))
        if destinations != sorted(set(destinations)):
            raise ValueError("Duplicate trade destination")
    if trader_ids != sorted(set(trader_ids)) or total_destinations > 4096:
        raise ValueError("Duplicate or unbounded trade roster")
    return confirmed


def _origin_action(unit: dict, target: dict, width: int) -> dict:
    return {
        "kind": "change_trade_origin",
        "unit_id": unit["unit_id"],
        "origin_city": unit["origin_city"],
        "source_x": unit["x"],
        "source_y": unit["y"],
        "destination_city": target["city_id"],
        "x": target["x"],
        "y": target["y"],
        "plot_id": target["y"] * width + target["x"],
    }


def validate_origins(value: Any, obs: Any) -> list[tuple[dict, dict, dict]]:
    """Validate origin-change offers and bind own units/cities to ``obs``."""
    pid, _, width, height, units, cities = _catalog_header(value, obs, "player_id turn width height units")
    ids: list[int] = []
    total = 0
    result: list[tuple[dict, dict, dict]] = []
    for unit in _list(value["units"]):
        _shape(unit, "unit_id origin_city x y available targets map_targets")
        unit_id = uint(unit["unit_id"], "origin unit")
        origin_city = uint(unit["origin_city"], "origin city")
        x, y = uint(unit["x"], "origin x", width - 1), uint(unit["y"], "origin y", height - 1)
        if type(unit["available"]) is not bool:
            raise ValueError("Invalid origin availability")
        targets = _list(unit["targets"])
        map_targets = _list(unit["map_targets"], width * height)
        if any(type(plot) is not int or not 0 <= plot < width * height for plot in map_targets) or map_targets != sorted(set(map_targets)):
            raise ValueError("Invalid origin highlights")
        if not unit["available"] and (targets or map_targets):
            raise ValueError("Unavailable origin exposes choices")
        if units.get(unit_id) != (x, y) or cities.get(origin_city) != (x, y):
            raise ValueError("Origin unit/city absent from own observation")
        ids.append(unit_id)
        target_ids: list[int] = []
        for target in targets:
            _shape(target, "city_id x y plot_id name")
            city_id = uint(target["city_id"], "destination city")
            tx, ty = uint(target["x"], "destination x", width - 1), uint(target["y"], "destination y", height - 1)
            plot_id = uint(target["plot_id"], "destination plot", width * height - 1)
            if type(target["name"]) is not str or len(target["name"].encode("utf-8")) > 32768:
                raise ValueError("Invalid origin city name")
            if plot_id != ty * width + tx or city_id == origin_city or (tx, ty) == (x, y):
                raise ValueError("Invalid origin destination")
            if cities.get(city_id) != (tx, ty):
                raise ValueError("Origin destination is not an observed own city")
            target_ids.append(city_id)
            total += 1
            result.append((_origin_action(unit, target, width), unit, target))
        if target_ids != sorted(set(target_ids)):
            raise ValueError("Duplicate origin destination")
    if ids != sorted(set(ids)) or total > 4096:
        raise ValueError("Duplicate or unbounded origin roster")
    return result
