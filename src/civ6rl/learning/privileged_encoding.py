"""Strict, value-only encoding for the separate Phase977 privileged sidecar.

This is deliberately not part of the player observation/action encoder. It
retains only an alive bit, signed-log-scaled gold, and unit count per native
player slot. It neither represents complete game state nor identifies private
entities. Scaling is ``sign(x) * log1p(abs(x))`` without clipping.
"""
from __future__ import annotations

import math
import re


SNAPSHOT_SCHEMA = "civ6rl.native_privileged_critic_snapshot977.v1"
PAYLOAD_SCHEMA = "civ6rl-native-privileged-payload977-v1"
PRIVILEGED_SCHEMA = "civ6rl.privileged_critic_features978.v1"
PLAYER_SLOTS = 64
MAX_UNITS = 4096
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_CONTEXT_FIELDS = (
    "core_sha256", "host_sha256", "closure_sha256", "collector_source_sha256",
)
_SNAPSHOT_FIELDS = {
    "stage", "schema", "episode_id", "game_generation", "player_id", "sequence", "turn",
    "core_sha256", "host_sha256", "closure_sha256", "collector_source_sha256", "snapshot",
}
_PAYLOAD_FIELDS = {
    "schema", "local_player_before", "observer_before", "turn_before",
    "local_player_after", "observer_after", "turn_after", "players", "units",
}
_PLAYER_FIELDS = {"player_id", "gold"}
_UNIT_FIELDS = {"id", "owner", "type", "x", "y"}

PRIVILEGED_FEATURES = tuple(
    feature
    for player_id in range(PLAYER_SLOTS)
    for feature in (
        f"player_{player_id:02d}.alive",
        f"player_{player_id:02d}.gold_signed_log1p",
        f"player_{player_id:02d}.unit_count_log1p",
    )
)


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"invalid privileged {name}")
    return value


def _finite_number(value, name):
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"invalid or nonfinite privileged {name}")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError(f"invalid or nonfinite privileged {name}") from None
    if not math.isfinite(result):
        raise ValueError(f"invalid or nonfinite privileged {name}")
    return result


def _signed_log1p(value):
    return math.copysign(math.log1p(abs(value)), value) if value else 0.0


def _validate_context(expected_context):
    if type(expected_context) is not dict or set(expected_context) != set(_EXPECTED_CONTEXT_FIELDS):
        raise ValueError("expected_context must declare exactly the four source SHA-256 pins")
    for name in _EXPECTED_CONTEXT_FIELDS:
        value = expected_context[name]
        if type(value) is not str or not _SHA256.fullmatch(value):
            raise ValueError(f"invalid expected source pin: {name}")


def _validate_response(response):
    if type(response) is not dict or not {
        "stage", "episode_id", "game_generation", "sequence", "observation", "decisions",
    }.issubset(response):
        raise ValueError("public response envelope is incomplete")
    if response["stage"] != "official_session_response":
        raise ValueError("public response is not a saved-session response")
    if response.get("schema") not in ("civ6rl.action_session920.v1", "civ6rl.hotseat_action_session923.v1"):
        raise ValueError("unsupported saved-session response schema")
    episode = response["episode_id"]
    generation = response["game_generation"]
    if (type(episode) is not str or not episode or len(episode) > 128
            or type(generation) is not str or not generation or len(generation) > 128):
        raise ValueError("invalid public episode identity")
    sequence = _integer(response["sequence"], 0, 1_000_000, "response sequence")
    observation = response["observation"]
    decisions = response["decisions"]
    if type(observation) is not dict or type(decisions) is not dict:
        raise ValueError("public response needs observation and decisions objects")
    if decisions.get("schema") not in ("civ6rl.saved_decisions972.v1", "civ6rl.saved_decisions980.v1",
                                        "civ6rl.saved_decisions982.v1", "civ6rl.saved_decisions983.v1",
                                        "civ6rl.diplomacy_decisions987.v1", "civ6rl.saved_decisions988.v1", "civ6rl.saved_decisions989.v1", "civ6rl.saved_decisions990.v1", "civ6rl.saved_decisions999.v1", "civ6rl.saved_decisions1000.v1", "civ6rl.saved_decisions.city_management.v1"):
        raise ValueError("unsupported public decision schema")
    if (type(observation.get("schema")) is not str
            or observation["schema"] != "civ6rl-player-observation-v1"):
        raise ValueError("unsupported public player observation schema")
    for name in ("episode_id", "game_generation"):
        if decisions.get(name) != response[name]:
            raise ValueError("public decision identity differs from response")
    if _integer(decisions.get("sequence"), 0, 1_000_000, "decision sequence") != sequence:
        raise ValueError("public decision sequence differs from response")
    player_id = _integer(decisions.get("player_id"), 0, PLAYER_SLOTS - 1, "decision player")
    turn = _integer(decisions.get("turn"), 0, 1_000_000, "decision turn")
    if (_integer(observation.get("player_id"), 0, PLAYER_SLOTS - 1, "observation player") != player_id
            or _integer(observation.get("turn"), 0, 1_000_000, "observation turn") != turn):
        raise ValueError("public observation and decision actor/turn differ")
    width = _integer(observation.get("width"), 1, 16_384, "map width")
    height = _integer(observation.get("height"), 1, 16_384, "map height")
    if width * height > 16_384:
        raise ValueError("public map exceeds diagnostic bound")
    economy = observation.get("economy")
    if type(economy) is not dict or "gold" not in economy:
        raise ValueError("public own gold is required for privileged cross-check")
    own_gold = _finite_number(economy["gold"], "public own gold")
    own_units = observation.get("units")
    if type(own_units) is not list or len(own_units) > 4096:
        raise ValueError("invalid public own unit list")
    expected_units = set()
    own_unit_ids = set()
    for row in own_units:
        if type(row) is not dict or not {"id", "type", "x", "y", "owner"}.issubset(row):
            raise ValueError("malformed public own unit")
        unit_id = _integer(row["id"], 0, 0xFFFFFFFF, "public unit id")
        unit_type = _integer(row["type"], 0, 0xFFFFFFFF, "public unit type")
        x = _integer(row["x"], 0, width - 1, "public own unit x")
        y = _integer(row["y"], 0, height - 1, "public own unit y")
        owner = _integer(row["owner"], 0, PLAYER_SLOTS - 1, "public own unit owner")
        if owner != player_id:
            raise ValueError("public own-unit list contains a foreign owner")
        identity = (unit_id, unit_type, x, y)
        if unit_id in own_unit_ids:
            raise ValueError("duplicate public own unit")
        own_unit_ids.add(unit_id)
        expected_units.add(identity)
    return {
        "episode_id": episode,
        "game_generation": generation,
        "sequence": sequence,
        "player_id": player_id,
        "turn": turn,
        "own_gold": own_gold,
        "own_units": expected_units,
        "width": width,
        "height": height,
    }


def _validate_snapshot(snapshot, response_context, expected_context):
    if type(snapshot) is not dict or set(snapshot) != _SNAPSHOT_FIELDS:
        raise ValueError("privileged snapshot envelope shape differs")
    if snapshot["stage"] != "privileged_critic_snapshot977" or snapshot["schema"] != SNAPSHOT_SCHEMA:
        raise ValueError("unsupported privileged snapshot schema")
    for name in ("episode_id", "game_generation", "sequence", "player_id", "turn"):
        value = snapshot[name]
        if value != response_context[name] or type(value) is not type(response_context[name]):
            raise ValueError(f"privileged snapshot {name} does not join exactly to public response")
    for name in _EXPECTED_CONTEXT_FIELDS:
        if snapshot[name] != expected_context[name]:
            raise ValueError(f"privileged snapshot source pin differs: {name}")
    payload = snapshot["snapshot"]
    if type(payload) is not dict or set(payload) != _PAYLOAD_FIELDS:
        raise ValueError("privileged payload shape differs")
    if payload["schema"] != PAYLOAD_SCHEMA:
        raise ValueError("unsupported privileged payload schema")
    for field in ("local_player_before", "observer_before", "local_player_after", "observer_after"):
        if _integer(payload[field], 0, PLAYER_SLOTS - 1, field) != response_context["player_id"]:
            raise ValueError("privileged snapshot changed local player/observer context")
    for field in ("turn_before", "turn_after"):
        if _integer(payload[field], 0, 1_000_000, field) != response_context["turn"]:
            raise ValueError("privileged snapshot crossed the public turn boundary")

    players = payload["players"]
    units = payload["units"]
    if type(players) is not list or len(players) > PLAYER_SLOTS:
        raise ValueError("invalid or unbounded privileged player enumeration")
    if type(units) is not list or len(units) > MAX_UNITS:
        raise ValueError("invalid or unbounded privileged unit enumeration")
    player_rows = {}
    for row in players:
        if type(row) is not dict or not set(row).issubset(_PLAYER_FIELDS) or "player_id" not in row:
            raise ValueError("malformed privileged player row")
        player_id = _integer(row["player_id"], 0, PLAYER_SLOTS - 1, "player id")
        if player_id in player_rows:
            raise ValueError("duplicate privileged player id")
        copied = {"player_id": player_id}
        if "gold" in row:
            copied["gold"] = _finite_number(row["gold"], "gold")
        player_rows[player_id] = copied

    unit_counts = [0] * PLAYER_SLOTS
    seen_units = set()
    own_units = set()
    for row in units:
        if type(row) is not dict or set(row) != _UNIT_FIELDS:
            raise ValueError("malformed privileged unit row")
        unit_id = _integer(row["id"], 0, 0xFFFFFFFF, "unit id")
        owner = _integer(row["owner"], 0, PLAYER_SLOTS - 1, "unit owner")
        unit_type = _integer(row["type"], 0, 0xFFFFFFFF, "unit type")
        x = _integer(row["x"], 0, response_context["width"] - 1, "unit x")
        y = _integer(row["y"], 0, response_context["height"] - 1, "unit y")
        if owner not in player_rows:
            raise ValueError("privileged unit belongs to an absent/dead player")
        identity = (owner, unit_id)
        if identity in seen_units:
            raise ValueError("duplicate privileged unit identity")
        seen_units.add(identity)
        unit_counts[owner] += 1
        if owner == response_context["player_id"]:
            own_units.add((unit_id, unit_type, x, y))
    if own_units != response_context["own_units"]:
        raise ValueError("privileged own units differ from the public observation")
    own_player = player_rows.get(response_context["player_id"])
    if own_player is None or "gold" not in own_player:
        raise ValueError("privileged own gold is missing")
    if not math.isclose(own_player["gold"], response_context["own_gold"], rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("privileged own gold differs from the public observation")
    return player_rows, unit_counts


def encode_privileged_snapshot(snapshot, response, *, expected_context):
    """Validate/join one privileged sidecar entry and emit fixed `[64,3]` data.

    Output is flat slot-major values/known masks. Alive is known for every
    slot because the collector payload is an exhaustive living-player
    enumeration. Gold is known only where a finite gold field is present.
    Unit count is known for enumerated living players (the unit list is the
    complete payload collection); absent player rows keep gold/count unknown.
    """
    _validate_context(expected_context)
    response_context = _validate_response(response)
    players, unit_counts = _validate_snapshot(snapshot, response_context, expected_context)
    values = []
    known = []
    for player_id in range(PLAYER_SLOTS):
        row = players.get(player_id)
        values.append(1.0 if row is not None else 0.0)
        known.append(True)
        if row is not None:
            if "gold" in row:
                values.append(_signed_log1p(row["gold"]))
                known.append(True)
            else:
                values.append(0.0)
                known.append(False)
            values.append(math.log1p(unit_counts[player_id]))
            known.append(True)
        else:
            values.extend((0.0, 0.0))
            known.extend((False, False))
    if len(values) != len(PRIVILEGED_FEATURES) or len(known) != len(PRIVILEGED_FEATURES):
        raise RuntimeError("privileged feature layout implementation differs")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("privileged feature scaling produced nonfinite value")
    return {
        "values": values,
        "known": known,
        "schema": PRIVILEGED_SCHEMA,
        "features": PRIVILEGED_FEATURES,
    }
