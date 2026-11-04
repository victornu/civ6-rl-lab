"""Explicit, public exercise instructions, separate from native observations.

Scenario identity binds the rollout but is never a neural feature. Collection
caps are truncations, not deadlines in the game, and are not policy inputs.
Only the supported announced objective and its target condition the policy.
"""
from __future__ import annotations
import hashlib
import json
import math

from .encoding import encode_response
from .scenario import _validate_contract, _player_observation
from .skill_context993 import (SCHEMA as SKILL_SCENARIO_SCHEMA, TASK_FEATURES, TASK_SCHEMA,
    skill_digest, encode_skill_task, monitor_skill_context)


def scenario_digest(scenario):
    if type(scenario) is dict and scenario.get('schema') == SKILL_SCENARIO_SCHEMA:
        return skill_digest(scenario)
    value = _validate_contract(scenario)
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def encode_task_response(response, *, vocabulary, scenario=None, completed_decisions=None,
                         allow_horizon=False, actor_schema=None):
    # An existing actor determines its tensor layout, including on historical
    # and diplomatic views where newer public features are explicitly unknown.
    # None preserves the encoder's schema-based default for offline callers.
    if scenario is None:
        frame = encode_response(response, vocabulary=vocabulary, actor_schema=actor_schema)
        if completed_decisions is not None:
            raise ValueError('decision count requires an explicit scenario')
        frame['task'] = {'values': [0.0]*len(TASK_FEATURES), 'mask': [False]*len(TASK_FEATURES)}
        frame['task_unit_types'] = []
        frame['scenario_sha256'] = None
        return frame
    if type(scenario) is dict and scenario.get('schema') == SKILL_SCENARIO_SCHEMA:
        frame = encode_response(response,vocabulary=vocabulary,actor_schema=actor_schema)
        return encode_skill_task(frame,response,scenario,completed_decisions,allow_horizon,
                                 vocabulary=vocabulary)
    task = _validate_contract(scenario)
    # The current actor conditioning supports only a public integer gold
    # threshold. Older fractional-target contracts remain valid for historical
    # analysis but cannot be sent to the policy: raw treasury decimals may be
    # hidden by the observation and must not re-enter through the target.
    if task['objective']['target'] != math.floor(task['objective']['target']):
        raise ValueError('fractional task targets are unsupported by the public policy context')
    frame = encode_response(response, vocabulary=vocabulary, actor_schema=actor_schema)
    if type(completed_decisions) is not int or not 0 <= completed_decisions <= task['horizon']['max_decisions']:
        raise ValueError('invalid task completed decision count')
    turn, gold = _player_observation(response['observation'], task['player_id'], 'task observation')
    if not task['start_turn'] <= turn <= task['start_turn'] + task['horizon']['max_turns']:
        raise ValueError('task observation outside collection horizon')
    if completed_decisions == 0 and (turn != task['start_turn'] or gold != task['objective']['starting_value']):
        raise ValueError('task initial observation differs from declared start')
    if gold >= task['objective']['target']:
        raise ValueError('task is already successful; do not sample or bootstrap')
    if not allow_horizon and (completed_decisions == task['horizon']['max_decisions']
                             or turn == task['start_turn'] + task['horizon']['max_turns']):
        raise ValueError('task collection horizon reached')
    frame['task'] = {'values': [1.0, task['objective']['target']] + [0.0]*(len(TASK_FEATURES)-2),
                     'mask': [True, True] + [False]*(len(TASK_FEATURES)-2)}
    frame['task_unit_types'] = []
    frame['scenario_sha256'] = scenario_digest(task)
    return frame


def validate_collection_binding(model, response, frame, completed_decisions):
    """Reject objective swaps, omitted context, replayed counts and late starts.

    Runs before sampling or recurrent-state mutation. A native episode/player
    is bound by the first sampled instruction, including explicitly no task.
    """
    identity = response['episode_id'], response['game_generation'], frame['player_id']
    previous = [row['choice'] for key, row in model._sample_audit.items() if key[:3] == identity]
    digest = frame['scenario_sha256']
    if any(row.get('scenario_sha256') != digest for row in previous):
        raise ValueError('scenario changed during recurrent episode')
    if digest is not None and completed_decisions != len(previous):
        raise ValueError('task count differs from actually sampled episode history')


def monitor_task_context(scenario):
    if type(scenario) is dict and scenario.get('schema') == SKILL_SCENARIO_SCHEMA:
        return monitor_skill_context(scenario)
    value = _validate_contract(scenario)
    return dict(objective=value['objective']['kind'], target=value['objective']['target'],
                **value['horizon'], scenario_sha256=scenario_digest(value), scope='diagnostic')
