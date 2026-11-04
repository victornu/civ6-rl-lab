"""Versioned task contracts for short, player-scoped native scenarios.

This module classifies observed task outcomes only. It never scores actions,
infers research completion from a changed selection, or turns host failures
into player losses. A contract may be useful for diagnostics while remaining
ineligible for training until every independent qualification gate is met.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping


SCHEMA = 'civ6rl.learning_scenario.v1'
OBSERVATION_SCHEMA = 'civ6rl-player-observation-v1'
OBJECTIVE = 'economy_gold_threshold'
_HASH_RE = re.compile(r'^[0-9a-fA-F]{64}$')
_IDENTITY_FIELDS = {
    'game_version', 'seed', 'map_seed', 'save_sha256', 'checkpoint_manifest_sha256', 'gamecore_sha256',
    'application_config_sha256', 'final_database_sha256', 'gameplay_script_sha256',
}
_RESET_FIELDS = {'strict_reload_verified', 'checkpoint_sha256', 'evidence_sha256', 'evidence_ref'}
_OBSERVATION_PROOF_FIELDS = {
    'schema', 'scope', 'source_sha256', 'visibility_qualified', 'available_fields',
}


def _require_mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be an object')
    return value


def _sha256(value, label: str, *, optional: bool = True) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or not _HASH_RE.fullmatch(value):
        raise ValueError(f'{label} must be a 64-character SHA-256 hex digest')
    return value.lower()


def _finite_number(value, label: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f'{label} must be a finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def _player_observation(observation, player_id: int, label: str) -> tuple[int, float]:
    row = _require_mapping(observation, label)
    if row.get('schema') != OBSERVATION_SCHEMA:
        raise ValueError(f'{label} has an unknown player-observation schema')
    if type(row.get('player_id')) is not int or row['player_id'] != player_id:
        raise ValueError(f'{label} belongs to another player')
    turn = row.get('turn')
    if type(turn) is not int or turn < 0:
        raise ValueError(f'{label} has an invalid turn')
    economy = _require_mapping(row.get('economy'), f'{label}.economy')
    if 'gold' not in economy:
        raise ValueError(f'{label} does not publish economy.gold')
    return turn, _finite_number(economy['gold'], f'{label}.economy.gold')


def _identity(value) -> dict:
    source = {} if value is None else _require_mapping(value, 'identity')
    unknown = set(source) - _IDENTITY_FIELDS
    if unknown:
        raise ValueError(f'Unknown identity fields: {sorted(unknown)}')
    result = {key: source.get(key) for key in sorted(_IDENTITY_FIELDS)}
    for key in ('save_sha256', 'checkpoint_manifest_sha256', 'gamecore_sha256', 'application_config_sha256'):
        result[key] = _sha256(result[key], f'identity.{key}')
    for key in ('final_database_sha256', 'gameplay_script_sha256'):
        files = result[key]
        if files is None:
            result[key] = {}
            continue
        files = _require_mapping(files, f'identity.{key}')
        if any(type(name) is not str or not name for name in files):
            raise ValueError(f'identity.{key} needs non-empty source names')
        result[key] = {name: _sha256(digest, f'identity.{key}[{name!r}]', optional=False)
                       for name, digest in sorted(files.items())}
    if result['game_version'] is not None and (
            type(result['game_version']) is not str or not result['game_version'].strip()):
        raise ValueError('identity.game_version must be a non-empty string')
    for key in ('seed', 'map_seed'):
        seed = result[key]
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError(f'identity.{key} must be a non-negative integer')
    return result


def _reset(value) -> dict:
    source = {} if value is None else _require_mapping(value, 'reset_evidence')
    unknown = set(source) - _RESET_FIELDS
    if unknown:
        raise ValueError(f'Unknown reset evidence fields: {sorted(unknown)}')
    strict = source.get('strict_reload_verified', False)
    if type(strict) is not bool:
        raise ValueError('reset_evidence.strict_reload_verified must be boolean')
    evidence_ref = source.get('evidence_ref')
    if evidence_ref is not None and (type(evidence_ref) is not str or not evidence_ref.strip()):
        raise ValueError('reset_evidence.evidence_ref must be a non-empty string')
    return {
        'strict_reload_verified': strict,
        'checkpoint_sha256': _sha256(source.get('checkpoint_sha256'), 'reset_evidence.checkpoint_sha256'),
        'evidence_sha256': _sha256(source.get('evidence_sha256'), 'reset_evidence.evidence_sha256'),
        'evidence_ref': evidence_ref,
    }


def _observation_proof(value) -> dict:
    source = {} if value is None else _require_mapping(value, 'observation_proof')
    unknown = set(source) - _OBSERVATION_PROOF_FIELDS
    if unknown:
        raise ValueError(f'Unknown observation proof fields: {sorted(unknown)}')
    schema = source.get('schema')
    scope = source.get('scope')
    visibility = source.get('visibility_qualified', False)
    if type(visibility) is not bool:
        raise ValueError('observation_proof.visibility_qualified must be boolean')
    available = source.get('available_fields', [])
    if type(available) not in (list, tuple) or any(type(field) is not str for field in available):
        raise ValueError('observation_proof.available_fields must be a string list')
    if schema is not None and type(schema) is not str:
        raise ValueError('observation_proof.schema must be a string')
    if scope is not None and type(scope) is not str:
        raise ValueError('observation_proof.scope must be a string')
    return {
        'schema': schema,
        'scope': scope,
        'source_sha256': _sha256(source.get('source_sha256'), 'observation_proof.source_sha256'),
        'visibility_qualified': visibility,
        'available_fields': sorted(set(available)),
    }


def _blockers(identity: dict, reset: dict, observation_proof: dict,
              action_space_complete: bool, held_out_eval_qualified: bool) -> list[str]:
    blockers = []
    if not identity['game_version']:
        blockers.append('game_version_missing')
    for key in ('save_sha256', 'checkpoint_manifest_sha256', 'gamecore_sha256', 'application_config_sha256'):
        if not identity[key]:
            blockers.append(f'{key}_missing')
    for key in ('final_database_sha256', 'gameplay_script_sha256'):
        if not identity[key]:
            blockers.append(f'{key}_missing')
    for key in ('seed', 'map_seed'):
        if identity[key] is None:
            blockers.append(f'{key}_missing')
    if not reset['strict_reload_verified']:
        blockers.append('strict_reload_unverified')
    if not reset['checkpoint_sha256'] or reset['checkpoint_sha256'] != identity['checkpoint_manifest_sha256']:
        blockers.append('reset_checkpoint_manifest_mismatch_or_missing')
    if not reset['evidence_sha256'] or not reset['evidence_ref']:
        blockers.append('reset_evidence_missing')
    if observation_proof['schema'] != OBSERVATION_SCHEMA:
        blockers.append('player_observation_schema_unverified')
    if observation_proof['scope'] != 'player':
        blockers.append('player_scope_unverified')
    if not observation_proof['source_sha256']:
        blockers.append('observation_source_pin_missing')
    if not observation_proof['visibility_qualified']:
        blockers.append('visibility_qualification_missing')
    if 'economy.gold' not in observation_proof['available_fields']:
        blockers.append('economy_gold_observation_unverified')
    if not action_space_complete:
        blockers.append('complete_action_space_unverified')
    if not held_out_eval_qualified:
        blockers.append('held_out_evaluation_unverified')
    return blockers


def build_economy_gold_scenario(*, scenario_id: str, player_id: int,
                                start_observation, target_gold: int | float,
                                max_decisions: int, max_turns: int,
                                identity=None, reset_evidence=None,
                                observation_proof=None,
                                action_space_complete: bool = False,
                                held_out_eval_qualified: bool = False) -> dict:
    """Build a multi-decision gold-balance milestone contract.

    The outcome is an absolute native balance threshold, not a per-action
    reward or a claim that any single action caused the balance change.
    Missing qualification evidence leaves training_eligible false.
    """
    if type(scenario_id) is not str or not scenario_id.strip() or len(scenario_id) > 128:
        raise ValueError('scenario_id must be a non-empty string of at most 128 characters')
    if type(player_id) is not int or not 0 <= player_id <= 63:
        raise ValueError('player_id must be an integer in [0, 63]')
    if type(max_decisions) is not int or max_decisions < 2:
        raise ValueError('max_decisions must allow a multi-decision episode')
    if type(max_turns) is not int or max_turns < 1:
        raise ValueError('max_turns must be a positive integer')
    if type(action_space_complete) is not bool or type(held_out_eval_qualified) is not bool:
        raise ValueError('scenario qualification flags must be boolean')
    start_turn, start_gold = _player_observation(start_observation, player_id, 'start_observation')
    target = _finite_number(target_gold, 'target_gold')
    if target <= start_gold:
        raise ValueError('target_gold must exceed the observed starting balance')
    normalized_identity = _identity(identity)
    normalized_reset = _reset(reset_evidence)
    normalized_observation = _observation_proof(observation_proof)
    metadata_blockers = _blockers(normalized_identity, normalized_reset, normalized_observation,
                                  action_space_complete, held_out_eval_qualified)
    return {
        'schema': SCHEMA,
        'scenario_id': scenario_id,
        'player_id': player_id,
        'identity': normalized_identity,
        'reset_evidence': normalized_reset,
        'observation_proof': normalized_observation,
        'action_contract': {'complete_action_space': action_space_complete},
        'held_out_eval_qualified': held_out_eval_qualified,
        'objective': {
            'kind': OBJECTIVE,
            'metric_path': 'economy.gold',
            'comparison': 'greater_than_or_equal',
            'target': target,
            'starting_value': start_gold,
            'terminal_success_reward': 1.0,
            'terminal_failure_reward': -1.0,
            'per_decision_reward': 0.0,
        },
        'start_turn': start_turn,
        'horizon': {'max_decisions': max_decisions, 'max_turns': max_turns},
        'terminal_contract': {
            'task_success': 'economy.gold >= objective.target in a player-scoped observation',
            'task_failure': 'verified official game terminal before task success',
            'official_game_outcome_is_separate': True,
            'horizon_is_truncation': True,
            'host_failure_is_invalid_data': True,
        },
        'qualification_note': (
            'Illustrative curriculum objective only; no evidence establishes its utility or '
            'qualifies this scenario. Phase 1 is not limited to gold-balance tasks.'
        ),
        'eligibility': {
            'metadata_complete': not metadata_blockers,
            'metadata_blockers': metadata_blockers,
            'training_eligible': False,
            'blockers': [*metadata_blockers, 'qualification_evidence_not_verified'],
        },
    }


def build_public_gold_gain_scenario(*, gold_gain: int, **kwargs) -> dict:
    """Build the currently supported public-gold curriculum instruction.

    The actor receives only ``floor(start gold) + gold_gain`` as its target.
    The exact starting balance remains scenario join metadata for collection
    validation; it is not a policy feature. This constrains this particular
    curriculum to public integer gold thresholds and says nothing about the
    rest of the reward space or other possible tasks.
    """
    if type(gold_gain) is not int or gold_gain <= 0:
        raise ValueError('gold_gain must be a positive integer')
    start_observation = kwargs.get('start_observation')
    if start_observation is None:
        raise ValueError('start_observation is required')
    player_id = kwargs.get('player_id')
    if type(player_id) is not int or not 0 <= player_id <= 63:
        raise ValueError('player_id must be an integer in [0, 63]')
    _, starting_gold = _player_observation(start_observation, player_id, 'start_observation')
    target_gold = math.floor(starting_gold) + gold_gain
    return build_economy_gold_scenario(target_gold=target_gold, **kwargs)


def _validate_contract(contract) -> dict:
    value = _require_mapping(contract, 'scenario')
    required = {
        'schema', 'scenario_id', 'player_id', 'identity', 'reset_evidence',
        'observation_proof', 'action_contract', 'held_out_eval_qualified',
        'objective', 'start_turn', 'horizon', 'terminal_contract', 'qualification_note', 'eligibility',
    }
    if set(value) != required or value.get('schema') != SCHEMA:
        raise ValueError('Unknown scenario contract schema or fields')
    if type(value['scenario_id']) is not str or not value['scenario_id'].strip() or len(value['scenario_id']) > 128:
        raise ValueError('Invalid scenario id')
    if type(value['player_id']) is not int or not 0 <= value['player_id'] <= 63:
        raise ValueError('Invalid scenario player id')
    objective = _require_mapping(value['objective'], 'objective')
    objective_fields = {
        'kind', 'metric_path', 'comparison', 'target', 'starting_value',
        'terminal_success_reward', 'terminal_failure_reward', 'per_decision_reward',
    }
    if set(objective) != objective_fields or objective.get('kind') != OBJECTIVE or objective.get('metric_path') != 'economy.gold':
        raise ValueError('Unsupported scenario objective')
    if objective.get('comparison') != 'greater_than_or_equal':
        raise ValueError('Unsupported scenario objective comparison')
    target = _finite_number(objective.get('target'), 'objective.target')
    start_value = _finite_number(objective.get('starting_value'), 'objective.starting_value')
    if target <= start_value:
        raise ValueError('Scenario target must exceed its starting value')
    if (objective.get('terminal_success_reward') != 1.0
            or objective.get('terminal_failure_reward') != -1.0
            or objective.get('per_decision_reward') != 0.0):
        raise ValueError('Scenario contract must not reward individual decisions')
    horizon = _require_mapping(value['horizon'], 'horizon')
    if set(horizon) != {'max_decisions', 'max_turns'}:
        raise ValueError('Unknown scenario horizon fields')
    if (type(horizon['max_decisions']) is not int or horizon['max_decisions'] < 2
            or type(horizon['max_turns']) is not int or horizon['max_turns'] < 1):
        raise ValueError('Invalid scenario horizon')
    if type(value['start_turn']) is not int or value['start_turn'] < 0:
        raise ValueError('Invalid scenario start turn')
    action_contract = _require_mapping(value['action_contract'], 'action_contract')
    if set(action_contract) != {'complete_action_space'} or type(action_contract['complete_action_space']) is not bool:
        raise ValueError('Invalid action contract')
    held_out = value['held_out_eval_qualified']
    if type(held_out) is not bool:
        raise ValueError('Invalid held-out evaluation flag')
    identity = _identity(value['identity'])
    reset = _reset(value['reset_evidence'])
    observation_proof = _observation_proof(value['observation_proof'])
    metadata_blockers = _blockers(identity, reset, observation_proof,
                                  action_contract['complete_action_space'], held_out)
    eligibility = _require_mapping(value['eligibility'], 'eligibility')
    expected_note = (
        'Illustrative curriculum objective only; no evidence establishes its utility or '
        'qualifies this scenario. Phase 1 is not limited to gold-balance tasks.'
    )
    if value['qualification_note'] != expected_note:
        raise ValueError('Scenario qualification note is missing or unsupported')
    if (set(eligibility) != {'metadata_complete', 'metadata_blockers', 'training_eligible', 'blockers'}
            or type(eligibility['metadata_complete']) is not bool
            or eligibility['metadata_complete'] != (not metadata_blockers)
            or eligibility['metadata_blockers'] != metadata_blockers
            or eligibility['training_eligible'] is not False
            or eligibility['blockers'] != [*metadata_blockers, 'qualification_evidence_not_verified']):
        raise ValueError('Scenario eligibility does not match its evidence gates')
    terminal = _require_mapping(value['terminal_contract'], 'terminal_contract')
    expected_terminal = {
        'task_success': 'economy.gold >= objective.target in a player-scoped observation',
        'task_failure': 'verified official game terminal before task success',
        'official_game_outcome_is_separate': True,
        'horizon_is_truncation': True,
        'host_failure_is_invalid_data': True,
    }
    if dict(terminal) != expected_terminal:
        raise ValueError('Unknown scenario terminal contract')
    return dict(value)


def evaluate_step(contract, before_observation, after_observation, *, decision_count: int,
                  official_game_ended: bool = False,
                  official_game_outcome: str | None = None,
                  host_failure: str | None = None) -> dict:
    """Classify one transition without assigning reward to its action.

    ``official_game_ended`` and ``official_game_outcome`` must come from the
    native terminal reader. A host failure invalidates the transition and is
    never encoded as a terminal loss.
    """
    scenario = _validate_contract(contract)
    player_id = scenario['player_id']
    if type(decision_count) is not int or not 0 <= decision_count <= scenario['horizon']['max_decisions']:
        raise ValueError('decision_count is outside the declared horizon')
    if type(official_game_ended) is not bool:
        raise ValueError('official_game_ended must be boolean')
    if official_game_outcome not in (None, 'victory', 'defeat', 'draw'):
        raise ValueError('Unknown official game outcome')
    if official_game_outcome is not None and not official_game_ended:
        raise ValueError('An official game outcome requires a verified terminal')
    if host_failure is not None:
        if type(host_failure) is not str or not host_failure.strip() or official_game_ended:
            raise ValueError('Invalid or contradictory host failure')
        return {
            'schema': 'civ6rl.learning_scenario_step.v1',
            'scenario_id': scenario['scenario_id'], 'player_id': player_id,
            'status': 'host_failure', 'task_success': False,
            'native_game_ended': False, 'official_game_outcome': None,
            'terminated': False, 'truncated': False, 'episode_valid': False,
            'reward': None, 'reason': 'host_failure_excluded_from_player_outcome',
            'training_eligible': False,
        }
    if decision_count < 1:
        raise ValueError('A non-crash transition needs at least one completed decision')
    before_turn, before_gold = _player_observation(before_observation, player_id, 'before_observation')
    after_turn, after_gold = _player_observation(after_observation, player_id, 'after_observation')
    if before_turn < scenario['start_turn'] or after_turn < before_turn:
        raise ValueError('Transition observation turns are inconsistent with the scenario start')
    deadline = scenario['start_turn'] + scenario['horizon']['max_turns']
    if before_turn >= deadline or after_turn > deadline:
        raise ValueError('Transition crosses beyond the scenario horizon; timely outcome unknown')
    if before_gold >= scenario['objective']['target']:
        raise ValueError('Transition starts after the task success condition was already met')

    succeeded = after_gold >= scenario['objective']['target']
    if succeeded:
        status, reason, terminated, truncated = 'success', 'task_threshold_observed', True, False
        reward = scenario['objective']['terminal_success_reward']
    elif official_game_ended:
        status, reason, terminated, truncated = 'failure', 'native_game_ended_before_task_success', True, False
        reward = scenario['objective']['terminal_failure_reward']
    else:
        turn_limit = after_turn - scenario['start_turn'] >= scenario['horizon']['max_turns']
        decision_limit = decision_count >= scenario['horizon']['max_decisions']
        if turn_limit or decision_limit:
            status, reason, terminated, truncated = 'truncated', (
                'turn_horizon' if turn_limit else 'decision_horizon'), False, True
        else:
            status, reason, terminated, truncated = 'running', None, False, False
        reward = 0.0
    return {
        'schema': 'civ6rl.learning_scenario_step.v1',
        'scenario_id': scenario['scenario_id'], 'player_id': player_id,
        'status': status, 'task_success': succeeded,
        'native_game_ended': official_game_ended,
        'official_game_outcome': official_game_outcome,
        'observed_metric': after_gold, 'target_metric': scenario['objective']['target'],
        'terminated': terminated, 'truncated': truncated,
        'episode_valid': True, 'reward': reward, 'reason': reason,
        'training_eligible': False,
    }
