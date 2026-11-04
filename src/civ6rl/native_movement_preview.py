"""Validation for public native MOVE_TO path previews.

This module operates only on decoded public observations and candidate rows. It
has no native runtime, ctypes, or pathfinding dependency.
"""
from __future__ import annotations

import hashlib
import json


SCHEMA = 'civ6rl.public_movement_preview.v1'
MAX_PLOTS = 16384
MAX_ROUTES = 4096


def _integer(value, label, maximum):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f'Invalid movement-preview {label}')
    return value


def _candidate_digest(action):
    return hashlib.sha256(json.dumps(action, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_and_index(preview, observation, candidates):
    """Validate preview routes against the exact visible state and move menu.

    Returns the complete public routes keyed by ``(unit_id, destination_plot_id)``.
    No routes are inferred, shortened, or reconstructed here.
    """
    preview_keys = {'schema', 'player_id', 'turn', 'width', 'height', 'routes'}
    if type(preview) is not dict or set(preview) != preview_keys or preview.get('schema') != SCHEMA:
        raise ValueError('Malformed public movement-preview catalogue')
    observation_keys = {'player_id', 'turn', 'width', 'height', 'plots', 'units'}
    if type(observation) is not dict or not observation_keys <= observation.keys():
        raise ValueError('Movement preview needs a public player observation')
    player_id = _integer(preview['player_id'], 'player ID', 63)
    turn = _integer(preview['turn'], 'turn', 0xFFFFFFFF)
    width = _integer(preview['width'], 'width', MAX_PLOTS)
    height = _integer(preview['height'], 'height', MAX_PLOTS)
    observed_identity = (
        _integer(observation.get('player_id'), 'observed player ID', 63),
        _integer(observation.get('turn'), 'observed turn', 0xFFFFFFFF),
        _integer(observation.get('width'), 'observed width', MAX_PLOTS),
        _integer(observation.get('height'), 'observed height', MAX_PLOTS),
    )
    if (width == 0 or height == 0 or width * height > MAX_PLOTS or
            (player_id, turn, width, height) != observed_identity):
        raise ValueError('Movement preview identity or dimensions differ from observation')
    if type(observation['plots']) is not list or len(observation['plots']) > MAX_PLOTS:
        raise ValueError('Invalid public visible-plot list')
    visible = {}
    for row in observation['plots']:
        if type(row) is not dict:
            raise ValueError('Malformed public plot row')
        index = _integer(row.get('index'), 'visible plot index', width * height - 1)
        x = _integer(row.get('x'), 'visible plot x', width - 1)
        y = _integer(row.get('y'), 'visible plot y', height - 1)
        if (row.get('visible') is not True or row.get('revealed') is not True or
                index != y * width + x or index in visible):
            raise ValueError('Movement route references hidden, duplicate, or inconsistent plot data')
        visible[index] = (x, y)
    if type(observation['units']) is not list or len(observation['units']) > MAX_ROUTES:
        raise ValueError('Invalid public own-unit list')
    units = {}
    for row in observation['units']:
        if type(row) is not dict:
            raise ValueError('Malformed public own-unit row')
        unit_id = _integer(row.get('id'), 'unit ID', 0xFFFFFFFF)
        owner = _integer(row.get('owner'), 'unit owner', 63)
        x = _integer(row.get('x'), 'unit x', width - 1)
        y = _integer(row.get('y'), 'unit y', height - 1)
        if owner != player_id or unit_id in units:
            raise ValueError('Movement route unit is foreign or duplicated')
        units[unit_id] = (x, y)
    if type(candidates) is not list or len(candidates) > 4096:
        raise ValueError('Invalid movement candidate list')
    moves = {}
    candidate_ids = set()
    action_fields = {'kind', 'unit_id', 'source_plot_id', 'source_x', 'source_y',
                     'destination_plot_id', 'destination_x', 'destination_y', 'moves_token'}
    for row in candidates:
        if type(row) is not dict or set(row) != {'candidate_id', 'action'}:
            raise ValueError('Malformed public action candidate')
        action = row['action']
        if type(action) is not dict:
            raise ValueError('Malformed public action')
        candidate_id = row['candidate_id']
        if type(candidate_id) is not str or candidate_id in candidate_ids:
            raise ValueError('Duplicate or invalid candidate identity')
        candidate_ids.add(candidate_id)
        if candidate_id != _candidate_digest(action):
            raise ValueError('Candidate identity differs from its public action')
        if action.get('kind') != 'move':
            continue
        if set(action) != action_fields:
            raise ValueError('Move candidate identity differs from its public action')
        unit_id = _integer(action['unit_id'], 'candidate unit ID', 0xFFFFFFFF)
        source = _integer(action['source_plot_id'], 'candidate source plot', width * height - 1)
        source_x = _integer(action['source_x'], 'candidate source x', width - 1)
        source_y = _integer(action['source_y'], 'candidate source y', height - 1)
        destination = _integer(action['destination_plot_id'], 'candidate destination plot', width * height - 1)
        destination_x = _integer(action['destination_x'], 'candidate destination x', width - 1)
        destination_y = _integer(action['destination_y'], 'candidate destination y', height - 1)
        if (unit_id not in units or units[unit_id] != (source_x, source_y) or
                source != source_y * width + source_x or
                destination != destination_y * width + destination_x or
                source not in visible or destination not in visible or
                visible[source] != (source_x, source_y) or
                visible[destination] != (destination_x, destination_y)):
            raise ValueError('Move candidate source or destination differs from public state')
        key = (unit_id, destination)
        if key in moves:
            raise ValueError('Duplicate movement candidate route key')
        moves[key] = source
    routes = preview['routes']
    if type(routes) is not list or len(routes) > MAX_ROUTES or len(routes) != len(moves):
        raise ValueError('Movement preview route count differs from available moves')
    indexed = {}
    previous_key = None
    route_fields = {'unit_id', 'source_plot_id', 'destination_plot_id', 'plot_ids', 'turns'}
    for route in routes:
        if type(route) is not dict or set(route) != route_fields:
            raise ValueError('Malformed public native movement route')
        unit_id = _integer(route['unit_id'], 'route unit ID', 0xFFFFFFFF)
        source = _integer(route['source_plot_id'], 'route source plot', width * height - 1)
        destination = _integer(route['destination_plot_id'], 'route destination plot', width * height - 1)
        key = (unit_id, destination)
        if (key not in moves or moves[key] != source or
                (previous_key is not None and key <= previous_key)):
            raise ValueError('Movement preview includes a missing, extra, or unordered route')
        previous_key = key
        plot_ids = route['plot_ids']
        turns = route['turns']
        if (type(plot_ids) is not list or type(turns) is not list or
                len(plot_ids) < 2 or len(plot_ids) > width * height or len(plot_ids) != len(turns)):
            raise ValueError('Movement route is empty, truncated, or unbounded')
        parsed_plots = []
        seen_plots = set()
        parsed_turns = []
        for plot_id, native_turn in zip(plot_ids, turns):
            plot_id = _integer(plot_id, 'route plot', width * height - 1)
            native_turn = _integer(native_turn, 'route turn', 0xFFFFFFFF)
            if native_turn < 1 or (parsed_turns and native_turn < parsed_turns[-1]):
                raise ValueError('Movement route turns are invalid or decreasing')
            if plot_id in seen_plots or plot_id not in visible:
                raise ValueError('Movement route repeats a plot or includes hidden terrain')
            parsed_plots.append(plot_id)
            seen_plots.add(plot_id)
            parsed_turns.append(native_turn)
        if parsed_plots[0] != source or parsed_plots[-1] != destination:
            raise ValueError('Movement route endpoints differ from its candidate')
        indexed[key] = dict(unit_id=unit_id, source_plot_id=source,
                            destination_plot_id=destination,
                            plot_ids=parsed_plots, turns=parsed_turns)
    if set(indexed) != set(moves):
        raise ValueError('Movement preview omits an available native move')
    return indexed
