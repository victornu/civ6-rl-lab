"""Strict stdlib validation for published research and civic UI information."""
from __future__ import annotations

from collections.abc import Mapping


SCHEMA = 'civ6rl.native_research_info983.v1'
MAX_CHOICES = 4096
UINT32_MAX = (1 << 32) - 1
INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1
_TOPICS = ('technology', 'civic')
_ACTION_KIND = {'technology': 'research', 'civic': 'civic'}
_ROW_FIELDS = {'index', 'hash', 'cost', 'raw_turns', 'turns_display', 'boosted'}


def _mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be an object')
    return value


def _uint32(value, label: str) -> int:
    if type(value) is not int or not 0 <= value <= UINT32_MAX:
        raise ValueError(f'{label} must be an unsigned 32-bit integer')
    return value


def _int32(value, label: str) -> int:
    if type(value) is not int or not INT32_MIN <= value <= INT32_MAX:
        raise ValueError(f'{label} must be a signed 32-bit integer')
    return value


def _row(value, label: str) -> dict:
    source = _mapping(value, label)
    if set(source) != _ROW_FIELDS:
        raise ValueError(f'{label} has unknown or missing fields')
    index = _uint32(source['index'], f'{label}.index')
    native_hash = _uint32(source['hash'], f'{label}.hash')
    cost = _int32(source['cost'], f'{label}.cost')
    raw_turns = _int32(source['raw_turns'], f'{label}.raw_turns')
    expected_display = None if raw_turns == -1 else min(raw_turns, 999)
    turns_display = source['turns_display']
    if turns_display is not None and type(turns_display) is not int:
        raise ValueError(f'{label}.turns_display must be an integer or null')
    if turns_display != expected_display:
        raise ValueError(f'{label}.turns_display differs from native UI projection')
    boosted = source['boosted']
    if type(boosted) is not bool:
        raise ValueError(f'{label}.boosted must be boolean')
    return {'index': index, 'hash': native_hash, 'cost': cost,
            'raw_turns': raw_turns, 'turns_display': turns_display,
            'boosted': boosted}


def _topic(value, label: str) -> dict:
    source = _mapping(value, label)
    if set(source) != {'selected', 'selected_info', 'choices'}:
        raise ValueError(f'{label} has unknown or missing fields')
    selected_value = source['selected']
    selected = None if selected_value is None else _uint32(selected_value, f'{label}.selected')
    selected_info_value = source['selected_info']
    selected_info = (None if selected_info_value is None else
                     _row(selected_info_value, f'{label}.selected_info'))
    if ((selected is None) != (selected_info is None)
            or (selected_info is not None and selected_info['index'] != selected)):
        raise ValueError(f'{label}.selected_info does not identify the selected item')
    choices_value = source['choices']
    if type(choices_value) is not list or len(choices_value) > MAX_CHOICES:
        raise ValueError(f'{label}.choices is malformed or oversized')
    choices = []
    indexes = set()
    hashes = set()
    previous_index = -1
    for position, choice in enumerate(choices_value):
        parsed = _row(choice, f'{label}.choices[{position}]')
        if parsed['index'] <= previous_index:
            raise ValueError(f'{label}.choices must be sorted by unique index')
        if parsed['index'] in indexes or parsed['hash'] in hashes:
            raise ValueError(f'{label}.choices contain duplicate index or hash')
        previous_index = parsed['index']
        indexes.add(parsed['index'])
        hashes.add(parsed['hash'])
        choices.append(parsed)
    if selected_info is not None:
        matching = [choice for choice in choices if choice['index'] == selected]
        if matching and matching[0] != selected_info:
            raise ValueError(f'{label}.selected_info differs from its offered choice')
    return {'selected': selected, 'selected_info': selected_info, 'choices': choices}


def _validate_research(research, value: dict) -> None:
    row = _mapping(research, 'research')
    if set(row) != {'research', 'civic'}:
        raise ValueError('research mapping has unknown or missing fields')
    for name in _TOPICS:
        field = _ACTION_KIND[name]
        selected = row[field]
        if selected is not None:
            selected = _uint32(selected, f'research.{field}')
        if selected != value[name]['selected']:
            raise ValueError(f'{name} selection differs from current research observation')


def _validate_candidates(candidates, value: dict) -> None:
    if type(candidates) is not list or len(candidates) > 131072:
        raise ValueError('candidate list is malformed or oversized')
    offered = {'research': set(), 'civic': set()}
    seen_candidate_ids = set()
    for position, candidate_value in enumerate(candidates):
        if not isinstance(candidate_value, Mapping):
            continue
        action = candidate_value.get('action')
        if not isinstance(action, Mapping):
            continue
        kind = action.get('kind')
        if type(kind) is not str or kind not in offered:
            # Other native action classes are outside this parser's contract.
            continue
        candidate = _mapping(candidate_value, f'candidates[{position}]')
        if set(candidate) != {'candidate_id', 'action'} or set(action) != {'kind', 'index', 'hash'}:
            raise ValueError(f'{kind} candidate has unknown or missing fields')
        candidate_id = candidate['candidate_id']
        if type(candidate_id) is not str or not candidate_id:
            raise ValueError(f'{kind} candidate_id must be a non-empty string')
        if candidate_id in seen_candidate_ids:
            raise ValueError('duplicate candidate_id for research/civic candidate')
        seen_candidate_ids.add(candidate_id)
        pair = (_uint32(action['index'], f'{kind} candidate index'),
                _uint32(action['hash'], f'{kind} candidate hash'))
        if pair in offered[kind]:
            raise ValueError(f'duplicate {kind} candidate choice')
        offered[kind].add(pair)
    for topic in _TOPICS:
        kind = _ACTION_KIND[topic]
        expected = {(choice['index'], choice['hash']) for choice in value[topic]['choices']}
        if offered[kind] != expected:
            raise ValueError(f'{kind} candidates do not exactly cover the published choices')


def validate(value, player_id: int, turn: int, research=None, candidates=None) -> dict:
    """Validate one current player/turn research and civic observation.

    ``selected`` may identify an item no longer present in ``choices``. If a
    selected item is also offered, however, its detailed row must match exactly.
    """
    if type(player_id) is not int or not 0 <= player_id <= 63:
        raise ValueError('player_id must be an integer in [0, 63]')
    if type(turn) is not int or turn < 0:
        raise ValueError('turn must be a non-negative integer')
    source = _mapping(value, 'research information')
    if set(source) != {'schema', 'player_id', 'turn', 'technology', 'civic'}:
        raise ValueError('research information has unknown or missing fields')
    if (source['schema'] != SCHEMA or type(source['player_id']) is not int
            or source['player_id'] != player_id or type(source['turn']) is not int
            or source['turn'] != turn):
        raise ValueError('research information identity or schema mismatch')
    result = {'schema': SCHEMA, 'player_id': player_id, 'turn': turn}
    for name in _TOPICS:
        result[name] = _topic(source[name], name)
    if research is not None:
        _validate_research(research, result)
    if candidates is not None:
        _validate_candidates(candidates, result)
    return result


__all__ = ['SCHEMA', 'MAX_CHOICES', 'validate']
