"""Public sparse skill objectives; pure classification, never scenario admission.

The caller binds observations to an actual episode/action receipt and resolves
the public GameInfo.Units catalogue. Native GetType() is an integer index, not
a content hash. This module never runs an action or constructs a native result.

Outcome order: host fault invalidates everything; malformed/stale boundaries
raise; loss of a protected city fails irreversibly; a reached task predicate
succeeds; otherwise an official game end fails the unfinished task; the public
deadline fails an unmet objective; a collection cap truncates with bootstrap;
otherwise continue. Thus task success can coincide with an official defeat
and is reported separately. Protected-city loss always overrides success.
Defense succeeds only at its announced deadline. Losing replaceable units
earlier is not failure. No terminal may be crossed without an observation.

All outputs remain diagnostic. A flag cannot qualify a task for training.
"""
from __future__ import annotations

import copy
import json
import math
import re

SCHEMA = 'civ6rl.public_skill_task991.v1'
STEP_SCHEMA = 'civ6rl.public_skill_step991.v1'
INSTRUCTION_SCHEMA = 'civ6rl.public_skill_instruction991.v1'
DEVELOPMENT = 'development_capacity_and_reserve'
SETTLEMENT = 'settlement_and_survival'
DEFENSE = 'short_tactical_defense'
_TYPE = re.compile(r'UNIT_[A-Z0-9_]{1,123}\Z')
_TOP = {'schema','task_id','player_id','start_turn','deadline_turn','objective',
        'unit_types','initial_metrics','training_eligible'}
PUBLIC_WITNESS_SCHEMA = 'civ6rl.public_skill_witness.v1'
PUBLIC_OBSERVATION_SCHEMA = 'civ6rl-public-skill-observation-v1'
_PUBLIC_SOURCES = {'gamecore_sha256','host_sha256','collector_source_sha256'}


def project_public_observation(value):
    """Minimal owned-state reward view; never a replacement for actor observations."""
    _require(type(value) is dict and value.get('schema') in
             ('civ6rl-player-observation-v1',PUBLIC_OBSERVATION_SCHEMA), 'unsupported public objective observation')
    if value['schema'] == 'civ6rl-player-observation-v1':
        from ..native_player_observation import parse_player_observation
        parse_player_observation(json.dumps(value,allow_nan=False),value.get('player_id'))
        result = {key:copy.deepcopy(value[key]) for key in ('player_id','turn','width','height')}
        result.update(schema=PUBLIC_OBSERVATION_SCHEMA,
            cities=[{key:row[key] for key in ('id','owner','x','y')} for row in value['cities']],
            units=[{key:row[key] for key in ('id','owner','x','y','type')} for row in value['units']],
            economy={'gold':value['economy']['gold']})
    else:
        result = copy.deepcopy(value)
    _require(set(result) == {'schema','player_id','turn','width','height','cities','units','economy'},
             'public objective observation fields differ')
    pid = _int(result['player_id'],0,63,'public objective player')
    _int(result['turn'],0,1000000,'public objective turn')
    width = _int(result['width'],1,16384,'public objective width')
    height = _int(result['height'],1,16384,'public objective height')
    _require(width*height <= 16384, 'public objective map exceeds bound')
    for name in ('cities','units'):
        rows = result[name]; _require(type(rows) is list and len(rows) <= 4096,'invalid public owned entities')
        ids = set(); coordinates = set()
        fields = {'id','owner','x','y'} | ({'type'} if name == 'units' else set())
        for row in rows:
            _require(type(row) is dict and set(row) == fields, 'public owned entity fields differ')
            _require(type(row['owner']) is int and row['owner'] == pid, 'foreign entity in public objective observation')
            ident = _int(row['id'],0,0xffffffff,'public entity id')
            _require(ident not in ids, 'duplicate public owned entity'); ids.add(ident)
            point = (_int(row['x'],0,width-1,'public entity x'),_int(row['y'],0,height-1,'public entity y'))
            if name == 'cities':
                _require(point not in coordinates,'duplicate public city coordinates'); coordinates.add(point)
            else:
                _int(row['type'],0,2147483647,'public unit type index')
    economy = result['economy']
    _require(type(economy) is dict and set(economy) == {'gold'}, 'public objective economy fields differ')
    gold = economy['gold']
    _require(type(gold) in (int,float) and math.isfinite(gold) and abs(gold) <= 1e12, 'invalid public objective gold')
    return result


def validate_public_sources(sources):
    """Check declared producer pins; this does not authenticate native execution."""
    _require(type(sources) is dict and set(sources) == _PUBLIC_SOURCES,
             'public continuity source fields differ')
    _require(all(type(v) is str and re.fullmatch(r'[0-9a-f]{64}',v) for v in sources.values()),
             'invalid public continuity source digest')
    return copy.deepcopy(sources)


def validate_public_witness(value, *, sources, episode_id, game_generation, player_id, sequence):
    """Bind one public view to an actual global receipt boundary, outside actor inputs."""
    validate_public_sources(sources)
    fields = {'schema','episode_id','game_generation','player_id','sequence','sources','observation'}
    _require(type(value) is dict and set(value) == fields and value['schema'] == PUBLIC_WITNESS_SCHEMA,
             'public continuity witness fields/schema differ')
    for key,expected in (('episode_id',episode_id),('game_generation',game_generation),
                         ('player_id',player_id),('sequence',sequence)):
        _require(type(value[key]) is type(expected) and value[key] == expected,
                 'public continuity witness ' + key + ' differs')
    _require(type(episode_id) is str and bool(episode_id) and type(game_generation) is str
             and bool(game_generation), 'public continuity episode identity missing')
    _int(player_id,0,63,'public witness player'); _int(sequence,0,2147483647,'public witness sequence')
    _require(value['sources'] == sources, 'public continuity witness sources differ')
    _require(type(value['observation']) is dict and value['observation'].get('schema') == PUBLIC_OBSERVATION_SCHEMA,
             'public continuity requires its narrow observation schema')
    observation = project_public_observation(value['observation'])
    _require(observation['player_id'] == player_id, 'public continuity observation is foreign')
    return copy.deepcopy(value)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _int(value, low, high, label):
    _require(type(value) is int and low <= value <= high, 'invalid ' + label)
    return value


def _type(value):
    _require(type(value) is str and _TYPE.fullmatch(value) is not None, 'invalid public unit type')
    return value


def _catalog(rows):
    _require(type(rows) is list and 0 < len(rows) <= 100000, 'invalid public unit catalogue')
    result = {}; names = set()
    for row in rows:
        _require(type(row) is dict and set(row) == {'index','type'}, 'unit catalogue fields differ')
        index = _int(row['index'], 0, 2147483647, 'public unit index'); name = _type(row['type'])
        _require(index not in result and name not in names, 'ambiguous public unit catalogue')
        result[index] = name; names.add(name)
    _require(list(result) == sorted(result), 'unit catalogue must be ordered by index')
    return result


def _city(value):
    _require(type(value) is dict and set(value) == {'id','x','y'}, 'protected city fields differ')
    _int(value['id'], 0, 0xffffffff, 'city id')
    _int(value['x'], 0, 16383, 'city x'); _int(value['y'], 0, 16383, 'city y')


def _objective(value, catalogue):
    _require(type(value) is dict, 'objective must be an object')
    kind = value.get('kind'); types = set(catalogue.values())
    if kind == DEVELOPMENT:
        _require(set(value) == {'kind','unit_type','target_count','gold_reserve'}, 'development fields differ')
        _require(value['unit_type'] in ('UNIT_BUILDER','UNIT_SETTLER') and value['unit_type'] in types,
                 'development needs a catalogued builder or settler')
        _int(value['target_count'], 1, 4096, 'unit target')
        _int(value['gold_reserve'], 0, 2147483647, 'public integer reserve')
    elif kind == SETTLEMENT:
        _require(set(value) == {'kind','target_cities','origin_city'}, 'settlement fields differ')
        _int(value['target_cities'], 1, 4096, 'city target'); _city(value['origin_city'])
    elif kind == DEFENSE:
        _require(set(value) == {'kind','defended_city','force_types','minimum_force'}, 'defense fields differ')
        _city(value['defended_city'])
        force = value['force_types']
        _require(type(force) is list and 0 < len(force) <= len(catalogue), 'defense force types missing')
        _require(all(type(name) is str for name in force) and force == sorted(set(force)), 'force types must be unique and ordered')
        for name in force:
            _type(name); _require(name in types, 'defense type absent from public catalogue')
        _int(value['minimum_force'], 1, 4096, 'force target')
    else:
        raise ValueError('unsupported skill objective')
    return kind


def _observation(value, player_id, catalogue, objective):
    _require(type(value) is dict and value.get('schema') in ('civ6rl-player-observation-v1',PUBLIC_OBSERVATION_SCHEMA),
             'unsupported player observation')
    if value['schema'] == PUBLIC_OBSERVATION_SCHEMA:
        project_public_observation(value)
    _require(type(value.get('player_id')) is int and value['player_id'] == player_id, 'observation player differs')
    turn = _int(value.get('turn'), 0, 1000000, 'observation turn')
    width = _int(value.get('width'), 1, 16384, 'map width')
    height = _int(value.get('height'), 1, 16384, 'map height')
    _require(width*height <= 16384, 'map exceeds bound')
    cities = value.get('cities'); units = value.get('units')
    _require(type(cities) is list and len(cities) <= 4096 and type(units) is list and len(units) <= 4096,
             'owned entity list missing or unbounded')
    city_ids = set(); city_positions = set(); owned = []; unit_ids = set(); counts = {}
    for rows, seen in ((cities, city_ids), (units, unit_ids)):
        for row in rows:
            _require(type(row) is dict and type(row.get('owner')) is int and row['owner'] == player_id,
                     'foreign entity in owned observation')
            index = _int(row.get('id'), 0, 0xffffffff, 'entity id')
            _require(index not in seen, 'duplicate owned entity'); seen.add(index)
            x = _int(row.get('x'), 0, width-1, 'entity x'); y = _int(row.get('y'), 0, height-1, 'entity y')
            if rows is cities:
                _require((x,y) not in city_positions, 'duplicate city coordinates'); city_positions.add((x,y))
                owned.append({'id':index,'x':x,'y':y})
            else:
                native_type = _int(row.get('type'), 0, 2147483647, 'native unit type index')
                _require(native_type in catalogue, 'native unit type missing from public catalogue')
                name = catalogue[native_type]; counts[name] = counts.get(name, 0) + 1
    kind = objective['kind']
    if kind == DEVELOPMENT:
        economy = value.get('economy')
        _require(type(economy) is dict, 'public economy missing')
        gold = economy.get('gold')
        _require(type(gold) in (int,float) and abs(gold) <= 1e12 and math.isfinite(gold), 'invalid gold observation')
        metrics = {'unit_count':counts.get(objective['unit_type'],0),'public_gold':math.floor(gold)}
    elif kind == SETTLEMENT:
        metrics = {'city_count':len(owned),'protected_city_present':objective['origin_city'] in owned}
    else:
        metrics = {'force_count':sum(counts.get(name,0) for name in objective['force_types']),
                   'protected_city_present':objective['defended_city'] in owned}
    return turn, metrics, (width,height)


def validate(task):
    _require(type(task) is dict and set(task) == _TOP and task['schema'] == SCHEMA, 'skill task fields/schema differ')
    _require(type(task['task_id']) is str and 0 < len(task['task_id']) <= 128 and '\x00' not in task['task_id'], 'invalid task id')
    _int(task['player_id'],0,63,'task player'); _int(task['start_turn'],0,999999,'start turn')
    _int(task['deadline_turn'],task['start_turn']+1,1000000,'public deadline')
    catalogue = _catalog(task['unit_types']); kind = _objective(task['objective'],catalogue)
    initial = task['initial_metrics']; _require(type(initial) is dict, 'initial metrics missing')
    if kind == DEVELOPMENT:
        _require(set(initial) == {'unit_count','public_gold'}, 'initial development metrics differ')
        _int(initial['unit_count'],0,4096,'initial unit count'); _int(initial['public_gold'],-10**12,10**12,'initial public gold')
        _require(task['objective']['target_count'] > initial['unit_count'], 'development must add an owned civilian unit')
    else:
        count = 'city_count' if kind == SETTLEMENT else 'force_count'
        _require(set(initial) == {count,'protected_city_present'} and initial['protected_city_present'] is True,
                 'protected city must exist initially')
        _int(initial[count],1,4096,'initial owned count')
        if kind == SETTLEMENT:
            _require(task['objective']['target_cities'] > initial[count], 'settlement target must add a city')
        else:
            _require(initial[count] >= task['objective']['minimum_force'], 'defense starts below its retained-force target')
    _require(task['training_eligible'] is False, 'skill contract cannot certify training eligibility')
    return task


def build_task(*, task_id, player_id, start_observation, objective, deadline_turn, unit_types):
    """Bind a public objective to initial metrics, without qualifying its bundle."""
    _int(player_id,0,63,'task player'); catalogue = _catalog(unit_types); _objective(objective,catalogue)
    start, metrics, _ = _observation(start_observation,player_id,catalogue,objective)
    task = dict(schema=SCHEMA,task_id=task_id,player_id=player_id,start_turn=start,
        deadline_turn=deadline_turn,objective=copy.deepcopy(objective),unit_types=copy.deepcopy(unit_types),
        initial_metrics=metrics,training_eligible=False)
    return validate(task)


def public_instruction(task):
    """Typed public instruction, not a neural tensor or an admission certificate."""
    validate(task); objective = copy.deepcopy(task['objective'])
    for field in ('origin_city','defended_city'):
        if field in objective:
            objective[field] = {key:objective[field][key] for key in ('x','y')}
    return {'schema':INSTRUCTION_SCHEMA,'objective':objective,'deadline_turn':task['deadline_turn']}


def validate_start(task, observation):
    validate(task)
    turn, metrics, _ = _observation(observation,task['player_id'],_catalog(task['unit_types']),task['objective'])
    _require(turn == task['start_turn'] and metrics == task['initial_metrics'], 'initial task metrics or turn differ')
    return metrics


def _success(task, metrics, turn):
    objective = task['objective']; kind = objective['kind']
    if kind == DEVELOPMENT:
        return metrics['unit_count'] >= objective['target_count'] and metrics['public_gold'] >= objective['gold_reserve']
    if kind == SETTLEMENT:
        return metrics['protected_city_present'] and metrics['city_count'] >= objective['target_cities']
    return (turn == task['deadline_turn'] and metrics['protected_city_present']
            and metrics['force_count'] >= objective['minimum_force'])


def evaluate_step(task, before_observation, after_observation, *, collection_truncated=False,
                  official_game_ended=False, official_game_outcome=None, host_failure=None):
    """Classify one post-action boundary. Collection truncation bootstraps.

    A protected city cannot be lost and later recaptured within this task:
    the first observed loss terminates it. Missing intermediate observations
    cannot certify continuous survival and remain the collector's obligation.
    """
    validate(task)
    base = {'schema':STEP_SCHEMA,'task_id':task['task_id'],'player_id':task['player_id'],
            'training_eligible':False}
    if host_failure is not None:
        _require(type(host_failure) is str and 0 < len(host_failure.strip()) <= 4096, 'invalid host failure')
        return dict(base,status='invalid',reason='host_failure',reward=None,terminated=False,truncated=False,
                    bootstrap=False,episode_valid=False,task_success=False,native_game_ended=None,
                    official_game_outcome=None,observed_metrics=None)
    _require(type(collection_truncated) is bool and type(official_game_ended) is bool, 'terminal controls must be Boolean')
    _require(official_game_outcome in (None,'victory','defeat','draw') and
             (official_game_outcome is None or official_game_ended), 'official outcome requires verified terminal')
    catalogue = _catalog(task['unit_types']); objective = task['objective']
    before, old, shape = _observation(before_observation,task['player_id'],catalogue,objective)
    after, metrics, next_shape = _observation(after_observation,task['player_id'],catalogue,objective)
    _require(shape == next_shape and task['start_turn'] <= before <= after <= task['deadline_turn'],
             'observation geometry/turn differs or crosses deadline without a boundary')
    _require(before < task['deadline_turn'] and not _success(task,old,before), 'transition begins after task termination')
    _require(old.get('protected_city_present',True), 'transition begins after irreversible protected-city loss')
    if not metrics.get('protected_city_present',True):
        status,reason,reward = 'failure','protected_city_lost',-1.
    elif _success(task,metrics,after):
        status,reason,reward = 'success','public_objective_reached',1.
    elif official_game_ended:
        status,reason,reward = 'failure','official_game_ended_before_task_success',-1.
    elif after == task['deadline_turn']:
        status,reason,reward = 'failure','public_deadline_unmet',-1.
    elif collection_truncated:
        status,reason,reward = 'truncated','collection_limit',0.
    else:
        status,reason,reward = 'running',None,0.
    terminal = status in ('success','failure')
    return dict(base,status=status,reason=reason,reward=reward,terminated=terminal,
                truncated=status=='truncated',bootstrap=not terminal,episode_valid=True,
                task_success=status=='success',native_game_ended=official_game_ended,
                official_game_outcome=official_game_outcome,observed_metrics=metrics)
