"""Join the opt-in LAN critic trace to a public decision, outside actor inputs."""

import math

from . import lan_inputs
from .privileged_encoding import (
    PLAYER_SLOTS, PRIVILEGED_FEATURES, PRIVILEGED_SCHEMA, _signed_log1p,
    _validate_context, _validate_snapshot,
)


def encode_lan_privileged_snapshot(snapshot, view, *, expected_context):
    """Return the existing critic feature layout for one exact ready LAN frame."""
    _validate_context(expected_context)
    ended = lan_inputs.validate(view)
    controls = view["decisions"]["turn_controls"]
    if ended or not (controls["active"] and controls["active_complete"]):
        raise ValueError("LAN critic snapshot requires a ready public decision")
    observation = view["observation"]
    boundary = {key: view[key] for key in ("episode_id", "game_generation", "sequence")}
    boundary.update({key: observation[key] for key in ("player_id", "turn", "width", "height")})
    boundary["own_units"] = {(u["id"], u["type"], u["x"], u["y"]) for u in observation["units"]}
    boundary["own_gold"] = observation["economy"]["gold"]
    players, unit_counts = _validate_snapshot(snapshot, boundary, expected_context,
        own_gold_projection="floor" if lan_inputs.is_display(view) else None)

    values, known = [], []
    for player_id in range(PLAYER_SLOTS):
        row = players.get(player_id)
        alive = row is not None
        gold_known = alive and "gold" in row
        values.extend((float(alive), _signed_log1p(row["gold"]) if gold_known else 0.0,
                       math.log1p(unit_counts[player_id]) if alive else 0.0))
        known.extend((True, gold_known, alive))
    return dict(schema=PRIVILEGED_SCHEMA, features=PRIVILEGED_FEATURES, values=values, known=known)
