"""Public own-unit queued goals and complete movement previews for actor v11."""
from __future__ import annotations

SCHEMA = 'civ6rl.learning.public_movement_information.v1'
PREVIEW_SCHEMA = 'civ6rl.public_movement_preview.v1'
GOAL_ENTITY_FEATURES = (
    'queued_goal_known', 'queued_goal_has_goal',
    'queued_goal_x_norm', 'queued_goal_y_norm',
)
POINT_ENTITY_FEATURES = (
    'movement_point_x_norm', 'movement_point_y_norm',
    'movement_point_turn', 'movement_point_order',
)
ACTION_FEATURES = (
    'movement_preview_known', 'movement_path_length', 'movement_arrival_turn',
)
ENTITY_FEATURES = GOAL_ENTITY_FEATURES + POINT_ENTITY_FEATURES
CONTRACT = {
    'schema': SCHEMA,
    'queued_orders_field': 'decisions.queued_orders',
    'preview_schema': PREVIEW_SCHEMA,
    'preview_features': list(POINT_ENTITY_FEATURES),
    'queued_goal_features': list(GOAL_ENTITY_FEATURES),
    'candidate_features': list(ACTION_FEATURES),
    'max_plot_count': 16384,
    'identity_values_are_neural_inputs': False,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_and_index(decisions, observation, candidates, *, ended=False):
    """Return own queued-goal facts and one complete route per move candidate.

    Native plot/unit identifiers are temporary join keys only. The returned
    candidate routes follow public candidate order and contain no identifiers.
    """
    _require(type(decisions) is dict and type(observation) is dict
             and type(candidates) is list and type(ended) is bool,
             'Public movement-information envelope required')
    pid, turn = observation.get('player_id'), observation.get('turn')
    width, height = observation.get('width'), observation.get('height')
    _require(type(pid) is int and 0 <= pid <= 63 and type(turn) is int and turn >= 0
             and type(width) is int and type(height) is int and 0 < width * height <= 16384,
             'Invalid public movement perspective or map')
    _require(decisions.get('player_id') == pid and decisions.get('turn') == turn,
             'Movement decisions perspective differs from observation')
    is_diplomacy = decisions.get('schema') == 'civ6rl.diplomacy_decisions987.v1'
    own_units = observation.get('units')
    _require(type(own_units) is list and len(own_units) <= 4096,
             'Missing or excessive own unit roster')
    ids = [row.get('id') for row in own_units if type(row) is dict]
    _require(len(ids) == len(own_units) and all(type(value) is int and 0 <= value <= 0xffffffff for value in ids)
             and len(ids) == len(set(ids)), 'Malformed own unit identities')
    goals = {}
    if is_diplomacy:
        _require('queued_orders' not in decisions,
                 'Pending diplomacy cannot claim an unavailable queued-order observation')
    else:
        queued = decisions.get('queued_orders')
        _require(type(queued) is dict and set(queued) == {'player_id', 'turn', 'width', 'height', 'units'}
                 and (queued['player_id'], queued['turn'], queued['width'], queued['height']) == (pid, turn, width, height)
                 and all(type(queued[k]) is int for k in ('player_id', 'turn', 'width', 'height'))
                 and type(queued['units']) is list and len(queued['units']) == len(ids),
                 'Actor v11 requires the exact own queued-order roster')
        rows = queued['units']
        row_ids = []
        for row in rows:
            _require(type(row) is dict and set(row) == {'unit_id', 'plot_id'}
                     and type(row['unit_id']) is int and 0 <= row['unit_id'] <= 0xffffffff,
                     'Malformed own queued-order row')
            plot = row['plot_id']
            _require(plot is None or (type(plot) is int and 0 <= plot < width * height),
                     'Queued movement goal lies outside the public map')
            row_ids.append(row['unit_id'])
            goals[row['unit_id']] = None if plot is None else (plot % width, plot // width)
        _require(row_ids == sorted(ids) and row_ids == sorted(set(row_ids)),
                 'Queued-order identities differ from the own unit roster')

    preview = decisions.get('movement_preview')
    if is_diplomacy:
        _require(preview is None, 'Pending diplomacy must not invent a movement preview')
        return goals, [[] for _ in candidates]
    _require(type(preview) is dict and set(preview) ==
             {'schema', 'player_id', 'turn', 'width', 'height', 'routes'}
             and preview.get('schema') == PREVIEW_SCHEMA
             and (preview.get('player_id'), preview.get('turn'), preview.get('width'), preview.get('height'))
             == (pid, turn, width, height),
             'Actor v11 requires the bound public movement-preview contract')
    from civ6rl.native_movement_preview import validate_and_index as validate_preview
    indexed = validate_preview(preview, observation, candidates)
    _require(type(indexed) is dict, 'Movement-preview validator returned an invalid index')
    routes = []
    for row in candidates:
        action = row.get('action') if type(row) is dict else None
        _require(type(action) is dict, 'Malformed movement candidate')
        if action.get('kind') != 'move':
            routes.append([])
            continue
        key = (action.get('unit_id'), action.get('destination_plot_id'))
        route = indexed.get(key)
        _require(type(route) is dict and type(route.get('plot_ids')) in (list, tuple)
                 and type(route.get('turns')) in (list, tuple)
                 and len(route['plot_ids']) == len(route['turns'])
                 and 2 <= len(route['plot_ids']) <= width * height,
                 'Move candidate lacks its complete native path preview')
        points = []
        for index, (plot, step_turn) in enumerate(zip(route['plot_ids'], route['turns'])):
            _require(type(plot) is int and 0 <= plot < width * height
                     and type(step_turn) is int and 1 <= step_turn <= 0xFFFFFFFF
                     and (index == 0 or step_turn >= route['turns'][index - 1]),
                     'Movement path point is malformed or turns decrease')
            x, y = plot % width, plot // width
            points.append([x / max(1, width - 1), y / max(1, height - 1), step_turn, index])
        routes.append(points)
    _require(len(routes) == len(candidates), 'Movement route/candidate alignment differs')
    if ended:
        _require(not any(routes), 'Ended actor menu cannot retain movement candidates')
    return goals, routes
