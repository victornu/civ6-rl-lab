"""Validation and actor projection for public WorldInput gestures (V12)."""
from __future__ import annotations

SCHEMA = 'civ6rl.world_input.v1'
CONTRACT = {
    'schema': SCHEMA,
    'decisions_field': 'decisions.world_input',
    'candidate_kinds': ['world_select_unit', 'world_deselect_unit', 'world_plot_click', 'world_war_response'],
    'unit_ids_are_neural_inputs': False,
    'hidden_plot_candidates_are_not_pruned': True,
    'pending_diplomacy_schema': 'civ6rl.diplomacy_decisions987.v1',
    'pending_diplomacy_selection_semantics': 'unknown',
}
ENTITY_FEATURES = ('world_input_selected_known', 'world_input_selected_unit')
ACTION_KINDS = ('world_select_unit', 'world_deselect_unit', 'world_plot_click', 'world_war_response')
ACTION_FEATURES = ('world_war_accept', 'world_war_decline')


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_and_index(decisions, observation, candidates, *, ended=False, dialogue=None,
                       allow_war_type_hash=False):
    """Validate an exact public unit-selection/click roster and return selected ID.

    IDs are checked as temporary joins only. This function never derives move
    legality from terrain and deliberately preserves every map coordinate.
    """
    _require(type(decisions) is dict and type(observation) is dict
             and type(candidates) is list and type(ended) is bool
             and type(allow_war_type_hash) is bool,
             'WorldInput public boundary required')
    value = decisions.get('world_input')
    if 'world_input' not in decisions:
        # The native diplomacy adapter publishes a separate, pinned schema and
        # intentionally does not read WorldInput under its modal interruption.
        # Its menu was already checked by diplomacy987.validate_and_project;
        # preserve that exact menu while representing unit selection unknown.
        _require(decisions.get('schema') == CONTRACT['pending_diplomacy_schema']
                 and type(dialogue) is dict
                 and set(dialogue) == {'context', 'menu', 'candidate_menu_indices'}
                 and type(dialogue['candidate_menu_indices']) is list
                 and len(dialogue['candidate_menu_indices']) == len(candidates)
                 and all(type(row) is dict and type(row.get('action')) is dict
                         and row['action'].get('kind') == 'diplomacy_response' for row in candidates),
                 'Missing WorldInput is allowed only for the validated pending-diplomacy schema')
        return None, [True] * len(candidates), False
    _require(type(value) is dict and set(value) == {
        'schema', 'player_id', 'turn', 'width', 'height', 'units', 'selected_unit_id', 'war_prompt'
    } and value.get('schema') == SCHEMA,
        'Actor v12 requires the exact WorldInput contract')
    pid, turn = observation.get('player_id'), observation.get('turn')
    width, height = observation.get('width'), observation.get('height')
    _require(type(pid) is int and 0 <= pid <= 63 and type(turn) is int and turn >= 0
             and type(width) is int and type(height) is int and width > 0 and height > 0
             and width * height <= 16384,
             'Invalid WorldInput public boundary values')
    _require(all(type(value.get(key)) is int for key in ('player_id', 'turn', 'width', 'height')),
             'WorldInput boundary fields must be exact integers')
    _require((value['player_id'], value['turn'], value['width'], value['height']) ==
             (pid, turn, width, height), 'WorldInput perspective differs from observation')
    own = observation.get('units')
    _require(type(own) is list and len(own) <= 4096, 'Invalid own WorldInput unit roster')
    own_ids = [row.get('id') for row in own if type(row) is dict]
    _require(len(own_ids) == len(own) and all(type(i) is int and 0 <= i <= 0xffffffff for i in own_ids)
             and len(own_ids) == len(set(own_ids)), 'Malformed own unit roster')
    units = value['units']
    _require(type(units) is list and len(units) == len(own_ids), 'WorldInput roster differs from own units')
    _require(all(type(row) is dict and set(row) == {'unit_id', 'ignore_moves'}
                 and type(row['unit_id']) is int and 0 <= row['unit_id'] <= 0xffffffff
                 and type(row['ignore_moves']) is bool for row in units)
             and [row['unit_id'] for row in units] == sorted(own_ids)
             and [row['unit_id'] for row in units] == sorted(set(row['unit_id'] for row in units)),
             'WorldInput unit roster is malformed or unordered')
    selected = value['selected_unit_id']
    _require(selected is None or (type(selected) is int and selected in set(own_ids)),
             'Selected WorldInput unit is not owned and observed')
    movable_ids = {row['unit_id'] for row in units if not row['ignore_moves']}
    _require(selected is None or selected in movable_ids,
             'Selected WorldInput unit is not movable in the public menu')
    prompt = value['war_prompt']
    war_type_valid = (lambda value: type(value) is int and
                      (-0x80000000 <= value <= 0xFFFFFFFF if allow_war_type_hash
                       else 0 <= value <= 65535))
    _require(prompt is None or (type(prompt) is dict and set(prompt) == {
        'attacking_player', 'defending_player', 'war_type'
    } and type(prompt['attacking_player']) is int and 0 <= prompt['attacking_player'] <= 63
        and type(prompt['defending_player']) is int and 0 <= prompt['defending_player'] <= 63
        and prompt['attacking_player'] == pid
        and prompt['defending_player'] != prompt['attacking_player']
        and war_type_valid(prompt['war_type'])),
        'Malformed public WorldInput war prompt')
    if ended:
        _require(not candidates, 'Ended WorldInput frame cannot retain candidates')
        return selected, [], True
    _require(width > 0 and height > 0 and width * height <= 16384,
             'WorldInput map exceeds bounded coordinate space')
    actions = [row.get('action') for row in candidates if type(row) is dict]
    _require(len(actions) == len(candidates) and all(type(a) is dict for a in actions),
             'Malformed WorldInput candidate')
    world = [a for a in actions if a.get('kind') in ACTION_KINDS]
    if prompt is not None:
        _require(len(world) == 2 and
                 {a.get('kind') for a in world} == {'world_war_response'} and
                 {a.get('accept') for a in world} == {True, False} and
                 all(set(a) == {'kind', 'accept'} for a in world),
                 'Modal WorldInput prompt requires only accept and decline')
    else:
        select_rows = [a for a in world if a.get('kind') == 'world_select_unit']
        _require(all(a.get('kind') != 'world_war_response' for a in world),
                 'War response cannot appear without a modal prompt')
        eligible = {row['unit_id'] for row in units if not row['ignore_moves']}
        _require(len(select_rows) == len(eligible) and
                 {a.get('unit_id') for a in select_rows} == eligible and
                 all(set(a) == {'kind', 'unit_id'} for a in select_rows),
                 'WorldInput unit selection menu differs from movable own units')
        clicks = [a for a in world if a.get('kind') == 'world_plot_click']
        deselects = [a for a in world if a.get('kind') == 'world_deselect_unit']
        if selected is None:
            _require(not clicks and not deselects, 'WorldInput click/deselect requires selected unit')
        else:
            _require(len(clicks) == width * height and len(deselects) == 1 and
                     deselects[0] == {'kind': 'world_deselect_unit'},
                     'Selected WorldInput state requires all map clicks and one deselect')
            expected = {(x, y) for y in range(height) for x in range(width)}
            received = set()
            for action in clicks:
                _require(set(action) == {'kind', 'unit_id', 'plot_id', 'x', 'y'}
                         and action['unit_id'] == selected
                         and type(action['x']) is int and type(action['y']) is int
                         and type(action['plot_id']) is int
                         and (action['x'], action['y']) in expected
                         and action['plot_id'] == action['y'] * width + action['x'],
                         'Malformed or mismatched public WorldInput map click')
                received.add((action['x'], action['y']))
            _require(received == expected, 'WorldInput click menu pruned or duplicated map coordinates')
    return selected, [prompt is None or action.get('kind') == 'world_war_response' for action in actions], True


def action_flags(action):
    """Named V12 action columns; non-WorldInput actions get neutral flags."""
    _require(type(action) is dict, 'Action dictionary required')
    kind = action.get('kind')
    if kind == 'world_war_response':
        _require(set(action) == {'kind', 'accept'} and type(action['accept']) is bool,
                 'Malformed WorldInput war-response action')
        return [float(action['accept']), float(not action['accept'])]
    return [0.0, 0.0]
