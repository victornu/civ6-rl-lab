"""Bounded public diplomatic text and exact native menu joins, without host imports.

Native identifiers and internal statement causes are checked only to bind the
published menu to its actions. The returned text contains only the separately
qualified official display; opaque engine metadata is never a policy input.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib


SCHEMA = "civ6rl.diplomacy_decisions987.v1"
DISPLAY_SCHEMA = "civ6rl.diplomacy_public_display986.v2"
DISPLAY_SOURCE_SHA256 = "fb80a444174b1bf54a05c5be7329d1cd889c6eae15a24ed6f689fa39a493ea4e"
STATEMENT_SOURCE_SHA256 = "b78799f583992bcddcb16f76d3714869bf547dda7f96b79a0e9fc3abe32f13f8"
SUPPORT_SHA256 = "b9d573f6a93a8dc43bc8804b4faf53510dcbe795862f64dece76e1434bc5a0fb"
MAX_TEXT_BYTES = 4096
MAX_DISPLAY_BYTES = 131072
_FIELDS = {"StatementType", "StatementSubType", "ResponseType", "SessionID",
           "FromPlayer", "ToPlayer", "Initiator", "FromPlayerMood", "ToPlayerMood",
           "AiReason", "AiModifier", "IsFirstStatementOfType"}
_DECISION_KEYS = {"schema", "episode_id", "game_generation", "sequence", "player_id", "turn",
                  "origin_turn", "player_map", "statement", "data", "source_sha256",
                  "ui_support_sha256", "candidates", "unsupported", "complete_action_space",
                  "decision_id", "public_display", "public_display_source_sha256", "yield_display"}
_CHOICES = {"CHOICE_POSITIVE", "CHOICE_EXIT"}


def _require(condition, message):
    if not condition:
        raise ValueError("diplomacy: " + message)


def _exact(value, keys, label):
    _require(type(value) is dict and set(value) == keys, "malformed " + label + " fields")
    return value


def _uint(value, label, maximum=0xFFFFFFFF):
    _require(type(value) is int and 0 <= value <= maximum, "invalid " + label)
    return value


def _text(value, label, *, nullable=False):
    if nullable and value is None:
        return value
    _require(type(value) is str and "\0" not in value, "invalid " + label + " text")
    try:
        size = len(value.encode("utf8"))
    except UnicodeEncodeError as exc:
        raise ValueError("diplomacy: invalid UTF-8 text") from exc
    _require(size <= MAX_TEXT_BYTES, "unbounded " + label + " text")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_rosters(observation, pid):
    # The ordinary path cross-checks entities against combat/production
    # sidecars. Those readers are absent during a conversation, so validate
    # the real observation roster contract directly rather than invent them.
    kinds = (
        ("units", {"id", "type", "x", "y", "damage", "moves", "max_moves", "owner"}),
        ("cities", {"id", "x", "y", "population", "owner", "yields"}),
        ("visible_enemy_units", {"owner", "id", "type", "x", "y", "damage", "visible"}),
    )
    for kind, keys in kinds:
        rows = observation.get(kind)
        maximum = 4096 if kind != "visible_enemy_units" else observation["width"] * observation["height"] * 8
        _require(type(rows) is list and len(rows) <= maximum, "unbounded observation " + kind)
        seen = set()
        for row in rows:
            _exact(row, keys, "observation " + kind)
            owner, uid = _uint(row["owner"], "entity owner", 63), _uint(row["id"], "entity id")
            _require((owner, uid) not in seen, "duplicate observation entity")
            seen.add((owner, uid))
            if kind == "visible_enemy_units":
                _require(owner != pid and row["visible"] is True, "foreign unit is not admitted as visible")
            else:
                _require(owner == pid, "foreign entity in owned observation roster")


def validate_and_project(decisions, observation, *, ended=False,
                         allow_world_input_continuation=False,
                         allow_unnamed_response=False):
    """Return text/menu inputs after validating the exact full native choice join.

    Missing ordinary research, production, combat, and treasury readers are
    intentional for this schema. The caller represents those values as unknown,
    without constructing an ordinary catalogue or filling native values.
    """
    _require(type(ended) is bool and type(allow_world_input_continuation) is bool
             and type(allow_unnamed_response) is bool,
             "validation options must be boolean")
    keys = _DECISION_KEYS
    if allow_world_input_continuation and type(decisions) is dict and 'continuation_origin' in decisions:
        keys = _DECISION_KEYS | {'continuation_origin'}
    value = _exact(decisions, keys, "catalogue")
    _require(value["schema"] == SCHEMA, "unsupported schema")
    _require(type(observation) is dict, "missing player observation")
    pid, turn = observation.get("player_id"), observation.get("turn")
    _uint(pid, "observer", 63)
    _uint(turn, "turn")
    _validate_rosters(observation, pid)
    _require(type(value["player_id"]) is int and value["player_id"] == pid
             and type(value["turn"]) is int and value["turn"] == turn, "foreign player boundary")
    _uint(value["origin_turn"], "origin turn")
    _require(turn in (value["origin_turn"], value["origin_turn"] + 1), "origin turn differs")
    if 'continuation_origin' in value:
        _require(allow_world_input_continuation
                 and value['continuation_origin'] == 'world_input'
                 and value['origin_turn'] == turn,
                 'unsupported or stale continuation origin')
    _require(value["source_sha256"] == STATEMENT_SOURCE_SHA256
             and value["ui_support_sha256"] == SUPPORT_SHA256
             and value["public_display_source_sha256"] == DISPLAY_SOURCE_SHA256, "source pin differs")
    _require(value["complete_action_space"] is False, "unqualified completeness claim")
    _require(type(value["unsupported"]) is list and len(value["unsupported"]) <= 256
             and all(type(item) is str and len(item) <= 4096 for item in value["unsupported"]),
             "invalid unsupported metadata")
    _require(value["decision_id"] == _digest({k: v for k, v in value.items() if k != "decision_id"}),
             "catalogue digest differs")
    statement = _exact(value["statement"], {"event_sequence", "fields", "opaque_9ec5f4e8",
        "opaque_a005ae08_hex", "source_proof_sha256", "learner_exposure_validated", "automatic_response"},
        "statement")
    _require(_uint(statement["event_sequence"], "event sequence") > 0, "zero event sequence")
    fields = _exact(statement["fields"], _FIELDS, "statement scalar")
    for name, item in fields.items():
        if name == "IsFirstStatementOfType":
            _require(type(item) is bool, "invalid first-statement flag")
        else:
            _uint(item, name)
    _uint(fields["FromPlayer"], "speaker player", 63)
    _uint(fields["ToPlayer"], "recipient player", 63)
    _require(pid in (fields["FromPlayer"], fields["ToPlayer"]), "observer is not a participant")
    _uint(statement["opaque_9ec5f4e8"], "opaque scalar")
    blob = statement["opaque_a005ae08_hex"]
    _require(blob is None or (type(blob) is str and re.fullmatch(r"[0-9a-f]{16}", blob) is not None),
             "invalid opaque blob")
    _require(statement["source_proof_sha256"] == STATEMENT_SOURCE_SHA256
             and statement["learner_exposure_validated"] is False
             and statement["automatic_response"] is False, "unqualified raw statement")
    data = _exact(value["data"], {"statement_text", "mood", "selections", "session"}, "native data")
    _text(data["statement_text"], "raw statement", nullable=True)
    _require(type(data["mood"]) is int and -(1 << 31) <= data["mood"] <= 0xFFFFFFFF, "invalid raw mood")
    session = _exact(data["session"], {"player_id", "session_id", "from_player", "to_player", "is_open",
        "statement_type", "statement_subtype", "response_type"}, "session")
    for key, expected in (("player_id", pid), ("session_id", fields["SessionID"]),
                           ("from_player", fields["FromPlayer"]), ("to_player", fields["ToPlayer"])):
        _require(type(session[key]) is int and session[key] == expected, "session identity differs")
    _require(session["is_open"] is True, "native session is not open")
    for key, field in (("statement_type", "StatementType"), ("statement_subtype", "StatementSubType"),
                        ("response_type", "ResponseType")):
        text = session[key]
        if key == 'response_type' and text is None:
            _require(allow_unnamed_response,
                     'unnamed response type is unsupported by this actor contract')
            continue
        _require(type(text) is str and re.fullmatch(r"[A-Z0-9_]{1,128}", text) is not None,
                 "malformed session enum")
        _require(zlib.crc32(text.encode("ascii")) ^ 0xFFFFFFFF == fields[field], "session enum hash differs")
    _require(session["statement_type"] not in ("MAKE_DEAL", "MAKE_DEMAND"), "unsupported special handler")
    native_menu = data["selections"]
    _require(type(native_menu) is list and 1 <= len(native_menu) <= 64, "unbounded native menu")
    seen, ordering = set(), []
    for row in native_menu:
        _exact(row, {"key", "sort", "text", "tooltip", "diplomatic_action_type", "disabled"}, "native selection")
        _require(type(row["key"]) is str and row["key"] in _CHOICES and row["key"] not in seen,
                 "unknown or duplicate selection key")
        _require(type(row["disabled"]) is bool and type(row["sort"]) is int, "invalid selection metadata")
        seen.add(row["key"])
        ordering.append(row["sort"])
        for key in ("text", "tooltip", "diplomatic_action_type"):
            _text(row[key], "native selection " + key, nullable=True)
    _require(ordering == sorted(ordering), "native selection order differs")
    display = _exact(value["public_display"], {"schema", "speaker_leader", "statement_text", "reason_text", "selections"},
                     "public display")
    _require(display["schema"] == DISPLAY_SCHEMA, "unsupported display schema")
    _text(display["speaker_leader"], "speaker", nullable=True)
    _text(display["statement_text"], "statement")
    _text(display["reason_text"], "reason")
    menu = display["selections"]
    _require(type(menu) is list and len(menu) == len(native_menu), "display menu differs")
    public_rows, enabled_rows, expected_actions = [], [], []
    for i, (row, native) in enumerate(zip(menu, native_menu)):
        _exact(row, {"key", "text", "disabled", "tooltip"}, "public selection")
        _require(row["key"] == native["key"] and type(row["disabled"]) is bool
                 and row["disabled"] == native["disabled"], "display selection join differs")
        _text(row["text"], "choice")
        _text(row["tooltip"], "tooltip", nullable=True)
        _require(row["disabled"] or row["tooltip"] is None, "enabled choice has an undisplayed tooltip")
        public_rows.append({key: row[key] for key in ("text", "tooltip", "disabled")})
        if not row["disabled"]:
            enabled_rows.append(i)
            expected_actions.append({"kind": "diplomacy_response", "session_id": fields["SessionID"],
                "statement_sequence": statement["event_sequence"], "choice": row["key"]})
    _require(len(json.dumps(display, ensure_ascii=False, separators=(",", ":")).encode("utf8")) <= MAX_DISPLAY_BYTES,
             "unbounded total public display")
    candidates = value["candidates"]
    _require(type(candidates) is list, "candidate menu must be a list")
    expected_candidates = [{"candidate_id": _digest(action), "action": action} for action in expected_actions]
    if ended:
        _require(not candidates, "closed frame has action candidates")
        indices = []
    else:
        _require(bool(candidates) and candidates == expected_candidates,
                 "candidates differ from the full ordered enabled native menu")
        # Equality alone would admit integer/Boolean substitutions in control
        # tokens. Check the wire fields separately before excluding them.
        for row in candidates:
            action = _exact(row["action"], {"kind", "session_id", "statement_sequence", "choice"}, "action")
            _uint(action["session_id"], "action session")
            _uint(action["statement_sequence"], "action statement")
        indices = enabled_rows
    return {"context": [display["speaker_leader"] or "", display["statement_text"], display["reason_text"]],
            "menu": public_rows, "candidate_menu_indices": indices}
