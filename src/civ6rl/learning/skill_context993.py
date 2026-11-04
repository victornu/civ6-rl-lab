"""Diagnostic skill scenarios and strictly public actor instructions.

Hashes bind declared inputs; their syntax and equality do not prove provenance,
native reachability, held-out independence or readiness for strategic training.
The full scenario is audit metadata. Only the projected task vector and public
content-type tokens reach the actor. Collection caps never become instructions.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re

from . import skill_task991 as skill

SCHEMA = 'civ6rl.skill_scenario993.v1'
TASK_SCHEMA = 'civ6rl.public_task_context.v2'
TASK_FEATURES = (
    'economy_gold_threshold','target_gold',
    'development_capacity_and_reserve','settlement_and_survival','short_tactical_defense',
    'target_unit_count','target_city_count','minimum_force','deadline_remaining',
    'protected_city_x','protected_city_y','target_unit_type_count',
)
TASK_TYPE_ENCODER = 'public_unit_type_set_mean_embedding_v1'
_IDENTITY = {'save_sha256','gamecore_sha256','application_config_sha256',
             'final_database_sha256','gameplay_scripts_sha256','lineage_sha256'}
_CATALOGUE = {'unit_types_sha256','gameplay_database_sha256','index_order_source_sha256'}
_HASH = re.compile(r'[0-9a-f]{64}\Z')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,
        allow_nan=False,separators=(',',':')).encode('ascii')).hexdigest()


def validate_skill_scenario(value):
    fields = {'schema','scenario_id','skill_task','horizon','identity','catalogue_source','training_eligible'}
    _require(type(value) is dict and set(value) == fields and value['schema'] == SCHEMA,
             'skill scenario fields/schema differ')
    _require(type(value['scenario_id']) is str and 0 < len(value['scenario_id']) <= 128
             and '\x00' not in value['scenario_id'], 'invalid skill scenario id')
    task = skill.validate(value['skill_task'])
    horizon = value['horizon']
    _require(type(horizon) is dict and set(horizon) == {'max_decisions','max_turns'}, 'skill horizon fields differ')
    for name in horizon:
        _require(type(horizon[name]) is int and 1 <= horizon[name] <= 1000000, 'invalid collection ' + name)
    for field, expected in (('identity',_IDENTITY),('catalogue_source',_CATALOGUE)):
        row = value[field]
        _require(type(row) is dict and set(row) == expected, 'skill ' + field + ' fields differ')
        _require(all((field == 'identity' and v is None) or
                     (type(v) is str and _HASH.fullmatch(v) is not None) for v in row.values()),
                 'invalid declared source digest')
    _require(value['catalogue_source']['unit_types_sha256'] == _digest(task['unit_types']),
             'public unit catalogue digest differs')
    _require(value['identity']['final_database_sha256'] is None or
             value['catalogue_source']['gameplay_database_sha256'] == value['identity']['final_database_sha256'],
             'public catalogue database differs from declared scenario database')
    _require(value['training_eligible'] is False, 'declared source hashes cannot certify training eligibility')
    return value


def build_skill_scenario(*, scenario_id, skill_task, max_decisions, max_turns, identity, catalogue_source):
    value = dict(schema=SCHEMA,scenario_id=scenario_id,skill_task=copy.deepcopy(skill_task),
        horizon=dict(max_decisions=max_decisions,max_turns=max_turns),identity=copy.deepcopy(identity),
        catalogue_source=copy.deepcopy(catalogue_source),training_eligible=False)
    return validate_skill_scenario(value)


def skill_digest(scenario):
    return _digest(validate_skill_scenario(scenario))


def _count(scenario, completed_decisions):
    _require(type(completed_decisions) is int and 0 <= completed_decisions <= scenario['horizon']['max_decisions'],
             'invalid completed skill decision count')


def encode_skill_task(frame, response, scenario, completed_decisions, allow_horizon=False, *, vocabulary):
    """Attach only public instructions to an already validated player frame."""
    validate_skill_scenario(scenario); _count(scenario,completed_decisions)
    _require(type(allow_horizon) is bool, 'allow_horizon must be Boolean')
    task = scenario['skill_task']; obs = response['observation']
    _require(frame.get('player_id') == task['player_id'] == obs.get('player_id')
             and frame.get('episode_id') == response.get('episode_id')
             and frame.get('sequence') == response.get('sequence'), 'skill frame identity differs')
    _require(frame.get('vocabulary_fingerprint') == vocabulary.fingerprint, 'skill vocabulary differs from frame')
    turn,metrics,_ = skill._observation(obs,task['player_id'],skill._catalog(task['unit_types']),task['objective'])
    _require(task['start_turn'] <= turn < task['deadline_turn'], 'skill deadline reached or turn outside task')
    _require(metrics.get('protected_city_present',True), 'protected city lost; do not sample or bootstrap')
    _require(not skill._success(task,metrics,turn), 'skill already successful; do not sample or bootstrap')
    if completed_decisions == 0:
        skill.validate_start(task,obs)
    cap = task['start_turn'] + scenario['horizon']['max_turns']
    _require(turn <= cap, 'skill observation crossed collection horizon')
    _require(allow_horizon or (completed_decisions < scenario['horizon']['max_decisions'] and turn < cap),
             'skill collection horizon reached')
    values = [0.]*len(TASK_FEATURES); masks = [False]*len(TASK_FEATURES)
    def put(name,value):
        index = TASK_FEATURES.index(name); values[index] = float(value); masks[index] = True
    for name in ('economy_gold_threshold',skill.DEVELOPMENT,skill.SETTLEMENT,skill.DEFENSE):
        put(name,0)
    instruction = skill.public_instruction(task); objective = instruction['objective']; kind = objective['kind']
    put(kind,1); put('deadline_remaining',task['deadline_turn']-turn)
    types = []
    if kind == skill.DEVELOPMENT:
        put('target_gold',objective['gold_reserve']); put('target_unit_count',objective['target_count'])
        types = [objective['unit_type']]
    elif kind == skill.SETTLEMENT:
        put('target_city_count',objective['target_cities'])
    else:
        put('minimum_force',objective['minimum_force']); types = objective['force_types']
    for field in ('origin_city','defended_city'):
        if field in objective:
            for axis in ('x','y'): put('protected_city_'+axis,objective[field][axis])
    put('target_unit_type_count',len(types))
    tokens = [vocabulary.lookup('action.item_type',name) for name in types]
    result = dict(frame)
    result.update(task={'values':values,'mask':masks},task_unit_types=tokens,scenario_sha256=skill_digest(scenario))
    return result


def evaluate_skill_transition(scenario, before_observation, after_observation, *, decision_count,
                              official_game_ended=False, official_game_outcome=None, host_failure=None):
    validate_skill_scenario(scenario); _count(scenario,decision_count)
    task = scenario['skill_task']
    if host_failure is not None:
        result = skill.evaluate_step(task,before_observation,after_observation,host_failure=host_failure)
    else:
        _require(decision_count >= 1, 'skill transition needs a completed decision')
        before_turn = before_observation.get('turn'); after_turn = after_observation.get('turn')
        cap = task['start_turn'] + scenario['horizon']['max_turns']
        _require(type(before_turn) is int and type(after_turn) is int and before_turn < cap and after_turn <= cap,
                 'skill transition crosses collection horizon without boundary')
        truncated = decision_count == scenario['horizon']['max_decisions'] or after_turn == cap
        result = skill.evaluate_step(task,before_observation,after_observation,collection_truncated=truncated,
            official_game_ended=official_game_ended,official_game_outcome=official_game_outcome)
    return dict(result,scenario_id=scenario['scenario_id'],scenario_sha256=skill_digest(scenario))


def monitor_skill_context(scenario):
    value = validate_skill_scenario(scenario)
    return dict(objective=value['skill_task']['objective']['kind'],
        instruction=skill.public_instruction(value['skill_task']),**value['horizon'],
        scenario_sha256=skill_digest(value),scope='diagnostic',training_eligible=False,
        provenance_status='declared_hash_bindings_only_not_verified_admission')


def evaluate_public_skill_interval(scenario, before_response, witnesses, *, decision_count,
                                   sources, next_response=None, action_receipt=None):
    """Evaluate every supplied global action boundary in the player's own view.

    This is continuity at observed receipt boundaries, not evidence that no
    transient event occurred inside a native command. The first task terminal
    is irreversible, even if a later view contains a recaptured city. Views
    never become policy inputs or fabricated decision responses.
    """
    validate_skill_scenario(scenario); _count(scenario,decision_count)
    _require(decision_count >= 1, 'skill interval needs a completed decision')
    skill.validate_public_sources(sources)
    _require(scenario['identity']['gamecore_sha256'] is None or
             scenario['identity']['gamecore_sha256'] == sources['gamecore_sha256'],
             'public continuity GameCore differs from scenario')
    task = scenario['skill_task']; pid = task['player_id']
    _require(type(before_response) is dict and type(before_response.get('sequence')) is int,
             'public interval start response missing')
    _require(type(witnesses) is list and 0 < len(witnesses) <= 4096, 'public continuity witnesses missing or unbounded')
    if action_receipt is not None:
        _require(type(action_receipt) is dict and action_receipt.get('episode_id') == before_response['episode_id']
                 and action_receipt.get('game_generation') == before_response['game_generation']
                 and type(action_receipt.get('sequence')) is int
                 and action_receipt['sequence'] == before_response['sequence']+1
                 and action_receipt.get('operation') == 'act'
                 and action_receipt.get('automatic_choices_applied') is False,
                 'public continuity immediate action receipt differs')
    catalogue = skill._catalog(task['unit_types']); objective = task['objective']
    previous = before_response['observation']
    old_turn,_,shape = skill._observation(previous,pid,catalogue,objective)
    cap = task['start_turn'] + scenario['horizon']['max_turns']
    _require(old_turn < cap, 'skill interval begins after collection horizon')
    terminal = None; result = None; terminal_sequence = None
    for index,witness in enumerate(witnesses):
        expected_sequence = before_response['sequence'] + index + 1
        skill.validate_public_witness(witness,sources=sources,episode_id=before_response['episode_id'],
            game_generation=before_response['game_generation'],player_id=pid,sequence=expected_sequence)
        observation = witness['observation']
        turn,_,geometry = skill._observation(observation,pid,catalogue,objective)
        _require(geometry == shape and old_turn <= turn, 'public continuity geometry or turn differs')
        if terminal is None:
            _require(turn <= cap, 'public continuity crosses collection horizon without boundary')
            result = skill.evaluate_step(task,previous,observation)
            if result['terminated']:
                terminal,terminal_sequence = result,expected_sequence
            elif turn == cap and index != len(witnesses)-1:
                raise ValueError('public continuity extends after collection horizon')
        previous,old_turn = observation,turn
    if next_response is not None:
        _require(type(next_response) is dict and next_response.get('episode_id') == before_response['episode_id']
                 and next_response.get('game_generation') == before_response['game_generation']
                 and type(next_response.get('sequence')) is int
                 and next_response['sequence'] == witnesses[-1]['sequence']
                 and skill.project_public_observation(next_response.get('observation')) == witnesses[-1]['observation'],
                 'public continuity endpoint differs from actual same-player successor')
    if terminal is not None:
        result = terminal
    elif decision_count == scenario['horizon']['max_decisions'] or old_turn == cap:
        result = dict(result,status='truncated',reason='collection_limit',truncated=True)
    return dict(result,scenario_id=scenario['scenario_id'],scenario_sha256=skill_digest(scenario),
        public_continuity=dict(sequences=[w['sequence'] for w in witnesses],
            terminal_sequence=terminal_sequence,terminal_turn=(witnesses[terminal_sequence-before_response['sequence']-1]
                ['observation']['turn'] if terminal_sequence is not None else None),
            sources=copy.deepcopy(sources),scope='observed_global_action_boundaries_only'))
