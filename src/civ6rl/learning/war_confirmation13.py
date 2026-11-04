"""V13 learner projection for the official WorldInput declare-war popup.

Native IDs and source hashes are checked here for identity only. The returned
dialogue contains rendered public strings only; existing V12 WorldInput and
diplomacy schemas keep their strict historical meaning.
"""
from __future__ import annotations

import json
import re

SCHEMA = 'civ6rl.war_confirmation_actor.v1'
WORLD_INPUT_SCHEMA = 'civ6rl.world_input.v1'
WORLD_INPUT_DISPLAY_SCHEMA = 'civ6rl.world_input.v2'
DISPLAY_SCHEMA = 'civ6rl.native_world_war_display.v2'
DISPLAY_SOURCE_SHA256 = '5d3baf689aa250092e7490dafce23ae9d6573c3ff8e8d3c13725f976f5258c51'
CONTRACT = {
    'schema': SCHEMA,
    'source_schema': DISPLAY_SCHEMA,
    'context_fields': ['targets.display_name_and_icon_name_json', 'message', 'warmonger_text'],
    'menu_fields': ['official_decline_button_text', 'official_accept_button_text'],
    'button_order': ['decline', 'accept'],
    'action_mapping': {'world_war_response.accept_false': 0, 'world_war_response.accept_true': 1},
    'war_type_validation': 'signed int32 or uint32; checked against pinned public display identity',
    'continuation_origin_policy': 'optional exact world_input origin with origin_turn equal to current turn; validation only',
    'unnamed_response_policy': 'session response_type may be null only under V13; raw enum remains validation-only',
    'masked_candidate_menu_index': -1,
    'native_ids_and_source_sha_are_features': False,
    'continuation_origin_is_a_feature': False,
}
MAX_TEXT_BYTES = 4096
MAX_TOTAL_BYTES = 131072
MAX_DISPLAY_BYTES = 65536


def _require(condition, message):
    if not condition:
        raise ValueError('war confirmation: ' + message)


def _text(value, name):
    _require(type(value) is str and '\0' not in value, 'invalid ' + name)
    try:
        size = len(value.encode('utf-8'))
    except UnicodeEncodeError as exc:
        raise ValueError('war confirmation: invalid UTF-8 ' + name) from exc
    _require(size <= MAX_TEXT_BYTES, 'unbounded ' + name)
    return value


def _display_text(display, pid, turn, prompt):
    required = {'schema', 'player_id', 'turn', 'defender_player_ids', 'war_type',
                'targets', 'message', 'warmonger_text', 'source_sha256', 'buttons', 'static_text'}
    _require(type(display) is dict and set(display) == required,
             'display fields differ from pinned v2')
    defender = prompt['defending_player']
    _require(display['schema'] == DISPLAY_SCHEMA and type(display['player_id']) is int
             and display['player_id'] == pid and type(display['turn']) is int
             and display['turn'] == turn and display['defender_player_ids'] == [defender]
             and type(display['war_type']) is int and display['war_type'] == prompt['war_type'],
             'display identity differs from the public prompt')
    _require(type(display['source_sha256']) is str
             and re.fullmatch(r'[0-9a-f]{64}', display['source_sha256']) is not None
             and display['source_sha256'] == DISPLAY_SOURCE_SHA256,
             'display source digest differs from its pinned projection')
    targets = display['targets']
    _require(type(targets) is list and len(targets) == 1
             and type(targets[0]) is dict
             and set(targets[0]) == {'player_id', 'display_name', 'icon_name'}
             and type(targets[0]['player_id']) is int
             and targets[0]['player_id'] == defender,
             'target display does not match prompt defender')
    display_name = _text(targets[0]['display_name'], 'target name')
    icon_name = _text(targets[0]['icon_name'], 'target icon')
    message = _text(display['message'], 'message')
    warmonger = _text(display['warmonger_text'], 'warmonger text')
    buttons = display['buttons']
    _require(type(buttons) is list and len(buttons) == 2,
             'exact official decline and accept buttons are required')
    button_texts = []
    for button, accept in zip(buttons, (False, True)):
        _require(type(button) is dict and set(button) == {'accept', 'text'}
                 and type(button['accept']) is bool and button['accept'] is accept,
                 'official button order or action mapping differs')
        button_texts.append(_text(button['text'], 'official button text'))
    static = display['static_text']
    _require(type(static) is dict and set(static) == {'title', 'warmonger_heading'},
             'official static text fields differ')
    title = _text(static['title'], 'popup title')
    heading = _text(static['warmonger_heading'], 'warmonger heading')
    context0 = json.dumps({'targets': [{'display_name': display_name, 'icon_name': icon_name}],
                           'title': title, 'warmonger_heading': heading},
                          ensure_ascii=False, separators=(',', ':'))
    context = [_text(context0, 'target context'), message, warmonger]
    total = sum(len(value.encode('utf-8')) for value in context + button_texts)
    _require(total <= MAX_TOTAL_BYTES and
             len(json.dumps(display, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) <= MAX_DISPLAY_BYTES,
             'display payload exceeds public text bounds')
    return context, button_texts


def _modal_dialogue(display, pid, turn, prompt, candidates, candidate_mask, *, ended):
    context, button_texts = _display_text(display, pid, turn, prompt)
    _require(type(candidates) is list and type(candidate_mask) is list
             and len(candidates) == len(candidate_mask),
             'candidate mask does not match the retained candidate roster')
    menu = [{'text': button_texts[0], 'tooltip': None, 'disabled': False},
            {'text': button_texts[1], 'tooltip': None, 'disabled': False}]
    indices = []
    seen = set()
    active = 0
    for row, enabled in zip(candidates, candidate_mask):
        _require(type(enabled) is bool and type(row) is dict
                 and set(row) == {'candidate_id', 'action'} and type(row['action']) is dict,
                 'malformed retained candidate')
        _require(type(row['candidate_id']) is str and bool(row['candidate_id']),
                 'missing candidate identity')
        action = row['action']
        if action.get('kind') == 'world_war_response':
            _require(set(action) == {'kind', 'accept'} and type(action['accept']) is bool,
                     'malformed world war response')
            index = 1 if action['accept'] else 0
            _require(enabled and index not in seen,
                     'war response is masked or duplicated')
            seen.add(index)
            indices.append(index)
            active += 1
        else:
            _require(not enabled, 'ordinary candidate remains active during modal popup')
            indices.append(-1)
    if ended:
        _require(active == 0 and not candidates,
                 'ended confirmation frame cannot retain an active response')
    else:
        _require(active == 2 and seen == {0, 1},
                 'both exact accept/decline responses must remain available')
    return {'context': context, 'menu': menu, 'candidate_menu_indices': indices}


def validate_and_project(decisions, observation, candidates, *, ended=False, dialogue=None):
    """Return V13 selection, exact policy mask, and optional public dialogue."""
    from .world_input import validate_and_index as validate_world_input
    _require(type(decisions) is dict and type(observation) is dict
             and type(candidates) is list and type(ended) is bool,
             'invalid learner boundary')
    pid, turn = observation.get('player_id'), observation.get('turn')
    _require(type(pid) is int and 0 <= pid <= 63 and type(turn) is int and turn >= 0,
             'invalid observation identity')
    if 'world_input' not in decisions:
        selected, mask, known = validate_world_input(
            decisions, observation, candidates, ended=ended, dialogue=dialogue)
        return selected, mask, known, dialogue, None

    state = decisions['world_input']
    _require(type(state) is dict, 'missing WorldInput state')
    if state.get('schema') == WORLD_INPUT_SCHEMA:
        _require(set(state) == {'schema', 'player_id', 'turn', 'width', 'height', 'units',
                                'selected_unit_id', 'war_prompt'},
                 'ordinary V1 state gained fields')
        _require(state['war_prompt'] is None,
                 'pending war prompt requires the explicit display contract')
        selected, mask, known = validate_world_input(
            decisions, observation, candidates, ended=ended, dialogue=dialogue)
        return selected, mask, known, dialogue, None

    _require(state.get('schema') == WORLD_INPUT_DISPLAY_SCHEMA and dialogue is None,
             'unsupported WorldInput or competing dialogue schema')
    _require(set(state) == {'schema', 'player_id', 'turn', 'width', 'height', 'units',
                            'selected_unit_id', 'war_prompt', 'war_display'},
             'modal WorldInput fields differ')
    prompt = state['war_prompt']
    _require(type(prompt) is dict and set(prompt) == {
        'attacking_player', 'defending_player', 'war_type'
    } and type(prompt['attacking_player']) is int and prompt['attacking_player'] == pid
        and type(prompt['defending_player']) is int and 0 <= prompt['defending_player'] <= 63
        and prompt['defending_player'] != pid and type(prompt['war_type']) is int
        and -0x80000000 <= prompt['war_type'] <= 0xFFFFFFFF,
        'invalid or stale public war prompt')
    # Reuse the V12 roster, selection and exact map-mask checks by stripping
    # only the newer display envelope after validating its own exact contract.
    legacy_state = dict(state)
    legacy_state.pop('war_display')
    legacy_state['schema'] = WORLD_INPUT_SCHEMA
    selected, mask, known = validate_world_input(
        {'world_input': legacy_state}, observation, candidates, ended=ended,
        allow_war_type_hash=True)
    projected = _modal_dialogue(state['war_display'], pid, turn, prompt, candidates, mask,
                                ended=ended)
    return selected, mask, known, projected, list(mask)
