"""Public LAN v24-v31 input joins for the existing recurrent actor features."""
from copy import deepcopy
import json
from types import SimpleNamespace

from .production_queue import ACTION_KINDS, action_features

_VIEW_FIELDS = {
    "episode_id", "game_generation", "sequence", "turn_mode", "observation",
    "decisions", "applied_action", "outcome",
}


def is_display(view):
    """Identify the host's explicit public projection after validate()."""
    from civ6_host.contracts.player_display import PUBLIC_VIEW_SCHEMA
    return view.get("schema") == PUBLIC_VIEW_SCHEMA


def policy_reconfirmation_allowed(view):
    """Grant the new policy semantics only after a complete LAN validation.

    Legacy Base views retain their changed-only contract. A LAN marker, even
    on a malformed view, must not bypass the LAN validator.
    """
    decisions = view.get("decisions", {})
    schema = decisions.get("schema")
    if "turn_mode" in view or (type(schema) is str and schema.startswith("civ6-host.network-")):
        validate(view)
        from civ6_host.contracts.network_turn import POLICY_RECONFIRM_CAPABILITY_SCHEMAS
        return schema in POLICY_RECONFIRM_CAPABILITY_SCHEMAS
    return False


def validate(view):
    from civ6_host.contracts.network_turn import (
        CAPABILITY_XP2_STRATEGIC_RESOURCES_SCHEMA, CAPABILITY_XP2_POLICIES_SCHEMA,
        CAPABILITY_XP2_POLICY_UNLOCK_SCHEMA, validate_capability_catalogue,
        CAPABILITY_XP2_GOVERNMENT_SCHEMA, CAPABILITY_XP2_POLICY_RELOCATION_SCHEMA,
        CAPABILITY_XP2_POLICY_RECONFIRM_SCHEMA,
        CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA,
        CAPABILITY_XP2_ENVOYS_SCHEMA,
        CAPABILITY_XP2_GOVERNORS_SCHEMA,
    )
    from ..native_player_observation import parse_player_observation

    if type(view) is not dict:
        raise ValueError("Expected a public SessionClient LAN view")
    display = is_display(view)
    if display:
        from civ6_host.contracts.player_display import validate_public
        validate_public(view)
        if view["locale"] != "fr_FR":
            raise ValueError("Only the source-qualified French display projection is supported")
    if ((not display and set(view) != _VIEW_FIELDS) or view["turn_mode"] != "simultaneous"
            or any(type(view[key]) is not str or not view[key] for key in
                   ("episode_id", "game_generation"))
            or type(view["sequence"]) is not int or view["sequence"] < 0):
        raise ValueError("Expected a public SessionClient LAN view")
    outcome = view["outcome"]
    if (type(outcome) is not dict
            or any(type(outcome.get(flag)) is not bool for flag in ("terminated", "truncated"))
            or outcome["terminated"] and outcome["truncated"]
            or set(outcome) != ({"terminated", "truncated", "reason", "result"}
                               if outcome["terminated"] else {"terminated", "truncated", "reason"})):
        raise ValueError("Invalid public LAN outcome")
    observation, decisions = view["observation"], view["decisions"]
    if type(observation) is not dict or type(decisions) is not dict:
        raise ValueError("LAN outcome has no observed actor boundary")
    if decisions.get("schema") not in (CAPABILITY_XP2_STRATEGIC_RESOURCES_SCHEMA,
            CAPABILITY_XP2_POLICIES_SCHEMA, CAPABILITY_XP2_POLICY_UNLOCK_SCHEMA,
            CAPABILITY_XP2_GOVERNMENT_SCHEMA, CAPABILITY_XP2_POLICY_RELOCATION_SCHEMA,
            CAPABILITY_XP2_POLICY_RECONFIRM_SCHEMA, CAPABILITY_XP2_FAITH_PURCHASE_SCHEMA,
            CAPABILITY_XP2_ENVOYS_SCHEMA, CAPABILITY_XP2_GOVERNORS_SCHEMA):
        raise ValueError("Only the public LAN v24-v32 catalogues are supported")
    player_id = observation.get("player_id")
    # Reuse raw ownership/shape and catalogue validators on a detached copy.
    # This copy is never emitted, saved or treated as native-precision data.
    shape = deepcopy(observation) if display else observation
    if display:
        from civ6_host.contracts.player_display import RAW_OBSERVATION_SCHEMA
        shape["schema"] = RAW_OBSERVATION_SCHEMA
    parse_player_observation(json.dumps(shape, allow_nan=False), player_id)
    identity = SimpleNamespace(episode_id=view["episode_id"], game_generation=view["game_generation"])
    ended = outcome["terminated"] or outcome["truncated"]
    validate_capability_catalogue(decisions, identity, view["sequence"], observation["turn"],
                                 player_id=player_id, observation=shape, actions_closed=ended)
    if decisions["schema"] == CAPABILITY_XP2_ENVOYS_SCHEMA:
        from civ6_host.contracts import envoys
        catalogue = decisions["envoys"]
        if catalogue is not None:
            envoys.validate_catalogue(catalogue, player_id, observation["turn"])
        expected = envoys.actions(catalogue)
        published = []
        for row in decisions["candidates"]:
            action = row["action"]
            if action.get("kind") in ("send_envoy", "retain_envoys"):
                envoys.validate_action(action)
                published.append(action)
        # Preserve host candidate order and IDs; validate the complete menu
        # against the exact public envoy query without pruning any candidate.
        canonical = lambda action: json.dumps(action, sort_keys=True, separators=(",", ":"))
        if sorted(map(canonical, published)) != sorted(map(canonical, expected)):
            raise ValueError("Public envoy candidates disagree with the validated catalogue")
    if decisions['schema'] == CAPABILITY_XP2_GOVERNORS_SCHEMA:
        from civ6_host.contracts import governors
        catalogue = decisions['governors']
        if catalogue is not None:
            governors.validate_catalogue(catalogue, player_id, observation['turn'])
        expected = governors.actions(catalogue)
        published = [row['action'] for row in decisions['candidates']
                     if row['action'].get('kind') in governors.ACTION_FIELDS]
        canonical = lambda action: json.dumps(action, sort_keys=True, separators=(',', ':'))
        if sorted(map(canonical, published)) != sorted(map(canonical, expected)):
            raise ValueError('Public governor candidates disagree with the validated catalogue')
    return ended


def queues(decisions, cities):
    """Project already validated entries and gestures in native order."""
    catalogs = decisions["production_queues"] or ()
    city_by_id = {row["id"]: row for row in cities}
    sizes = {row["city_id"]: len(row["entries"]) for row in catalogs}
    return {
        "locations": [dict(city=city_by_id[row["city_id"]], entry=entry)
                      for row in catalogs for entry in row["entries"]],
        "candidates": {
            row["candidate_id"]: action_features(row["action"], sizes[row["action"]["city_id"]])
            for row in decisions["candidates"] if row["action"]["kind"] in ACTION_KINDS
        },
    }


def current_production(decisions):
    """Use the observed queue head and displayed values, with absent fields unknown."""
    choice = decisions["choice_state"]
    if choice is None:
        return {}
    displayed = {row["city_id"]: row for row in decisions["production_current"]["cities"]}
    result = {}
    for row in choice["production"]:
        city_id = row["city_id"]
        item = displayed[city_id]["item"] if displayed[city_id]["state"] == "current" else None
        production = dict(kind=row["kind"], current_production_hash=row["hash"],
                          item_index=row["item_index"], queue_size=row["queue_size"],
                          has_production_progress=None)
        if item is not None:
            production.update(cost=item["cost"], progress=item["progress"], turns_left=item["turns_left"])
        result[city_id] = production
    return result


def purchase_locations(decisions, cities):
    city_by_id = {city["id"]: city for city in cities}
    return [dict(city=city_by_id[row["city_id"]], offer=offer)
            for field in ("plot_purchases", "city_purchases")
            for row in decisions[field] or () for offer in row["offers"]]


def city_controls_for_actor(decisions):
    """Build a narrow validator view for already validated LAN city controls.

    The public LAN frame keeps its actual versioned network schema. The frozen
    city indexers predate LAN and require their saved-decision schema names, so
    this explicit, transient projection supplies only the fields those indexers
    consume. It does not mutate, serialize, or relabel the live LAN frame.
    Call only after :func:`validate` has checked the complete host catalogue.
    """
    from civ6_host.contracts.network_turn import is_capability_catalogue
    from .city_focus1000 import DECISIONS_SCHEMA as FOCUS_SCHEMA
    from .city_management import DECISIONS_SCHEMA as CITY_SCHEMA

    if not is_capability_catalogue(decisions):
        raise ValueError("City actor projection requires a validated LAN capability catalogue")
    focus = decisions.get("city_focus_catalogs")
    citizens = decisions.get("citizen_catalogs")
    if focus is None and citizens is None:
        return None
    if type(focus) is not list:
        raise ValueError("LAN citizen controls require the complete public focus catalogue")
    if citizens is None:
        return {
            "schema": FOCUS_SCHEMA,
            "player_id": decisions["player_id"],
            "turn": decisions["turn"],
            "city_focus_catalogs": focus,
            "candidates": decisions["candidates"],
        }
    if type(citizens) is not list:
        raise ValueError("LAN citizen controls must carry a public citizen catalogue")
    return {
        "schema": CITY_SCHEMA,
        "player_id": decisions["player_id"],
        "turn": decisions["turn"],
        "city_focus_catalogs": focus,
        "citizen_catalogs": citizens,
        "candidates": decisions["candidates"],
    }
