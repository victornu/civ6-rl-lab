"""Explicit, bounded opt-in for one multiplayer diagnostic collection."""

SCHEMA = 'civ6rl.multiplayer_diagnostic_budget998.v1'
MAX_DECISIONS = 63
MAX_UPDATES = 1
EXTENDED_SCHEMA = 'civ6rl.bounded_episode_budget.v1'
MAX_EPISODE_DECISIONS = 256
MAX_EPISODE_ROUNDS = 100
MAX_EPISODE_SECONDS = 1000
MAX_EPISODE_REQUEST_SECONDS = 60


def validate_collection(value):
    """Return a detached canonical contract; never infer or raise a budget."""
    if type(value) is dict and value.get('schema') == EXTENDED_SCHEMA:
        fields = {'schema','max_decisions','max_updates','max_rounds','wall_seconds','request_seconds'}
        if set(value) != fields:
            raise ValueError('invalid bounded episode budget fields')
        for name, low, high in (('max_decisions',2,MAX_EPISODE_DECISIONS),
                ('max_updates',1,1),('max_rounds',1,MAX_EPISODE_ROUNDS),
                ('wall_seconds',120,MAX_EPISODE_SECONDS),
                ('request_seconds',20,MAX_EPISODE_REQUEST_SECONDS)):
            if type(value[name]) is not int or not low <= value[name] <= high:
                raise ValueError('invalid bounded episode '+name)
        if value['wall_seconds'] <= value['request_seconds']+30:
            raise ValueError('bounded episode leaves no collection/finalization window')
        return {name:value[name] for name in ('schema','max_decisions','max_updates',
            'max_rounds','wall_seconds','request_seconds')}
    if type(value) is not dict or set(value) != {'schema', 'max_decisions', 'max_updates'}:
        raise ValueError('invalid diagnostic collection fields')
    if value['schema'] != SCHEMA:
        raise ValueError('unknown diagnostic collection schema')
    if type(value['max_decisions']) is not int or not 2 <= value['max_decisions'] <= MAX_DECISIONS:
        raise ValueError('diagnostic max_decisions must be an integer in [2, 63]')
    if type(value['max_updates']) is not int or value['max_updates'] != MAX_UPDATES:
        raise ValueError('diagnostic max_updates must be exactly one')
    return dict(schema=SCHEMA, max_decisions=value['max_decisions'], max_updates=MAX_UPDATES)


def is_extended(value):
    """Recognize a validated opt-in, never a training-admission certificate."""
    return validate_collection(value)['schema'] == EXTENDED_SCHEMA
