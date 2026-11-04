"""Pure-stdlib validation for public native yield display observations."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping


SCHEMA = 'civ6rl.native_yield_display982.v1'
OBSERVATION_SCHEMA = 'civ6rl-player-observation-v1'
GLOBAL_YIELDS = ('science', 'culture', 'faith', 'faith_yield')
CITY_YIELDS = ('food', 'production', 'gold', 'science', 'culture', 'faith')
_DISPLAY_NUMBER = re.compile(
    r'[+-]?(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)(?:\.[0-9])?\Z')
_FRENCH_DISPLAY_NUMBER = re.compile(
    r'[+-]?(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:\u00a0[0-9]{3})+)(?:,[0-9])?\Z')
# The native formatter does not promise nearest rounding: the recorded
# constructor/Lookup witness formats 1234.56 as 1,234.5. This is only a
# one-decimal consistency envelope; authoritative text comes from the DLL.
_GLOBAL_TOLERANCE = 0.100000001
_CITY_TOLERANCE = 1e-9
_MAX_CITIES = 4096


def _mapping(value, label: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError(f'{label} must be an object')
    return value


def _finite_number(value, label: str) -> int | float:
    if type(value) not in (int, float):
        raise ValueError(f'{label} must be a finite number')
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f'{label} must be a finite number')
    return value


def public_number(text: str) -> float:
    """Parse the qualified English/French one-decimal public formats.

    English comma groups contain exactly three digits; a French decimal has
    exactly one. The only overlap is an ungrouped integer with the same value.
    Keep the native display text untouched; this converts the actor's numeric
    feature, never a raw getter or a guessed locale/rounding convention.
    """
    if type(text) is not str or len(text) > 64:
        raise ValueError('unsupported public yield number')
    if _DISPLAY_NUMBER.fullmatch(text):
        number = text.replace(',', '')
    elif _FRENCH_DISPLAY_NUMBER.fullmatch(text):
        number = text.replace('\u00a0', '').replace(',', '.')
    else:
        raise ValueError('unsupported public yield number')
    if len(text.encode('utf-8')) > 64:
        raise ValueError('public yield number exceeds its UTF-8 byte bound')
    value = float(number)
    if not math.isfinite(value):
        raise ValueError('public yield number is non-finite')
    return value


def _exact_fields(value, fields, label: str) -> Mapping:
    row = _mapping(value, label)
    if set(row) != set(fields):
        raise ValueError(f'{label} has unknown or missing fields')
    return row


def _yield_map(value, names, label: str, *, display: bool) -> dict:
    row = _exact_fields(value, names, label)
    result = {}
    for name in names:
        item = row[name]
        if display:
            result[name] = item
            public_number(item)
        else:
            result[name] = _finite_number(item, f'{label}.{name}')
    return result


def _validate_city_id(value, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f'{label} must be a non-negative integer')
    return value


def _floor_tenth(value: int | float) -> float:
    scaled = value * 10
    if not math.isfinite(scaled):
        raise ValueError('city yield is too large to display')
    try:
        return math.floor(scaled) / 10
    except (OverflowError, ValueError) as exc:
        raise ValueError('city yield is too large to display') from exc


def _observation_binding(observation, player_id: int, turn: int,
                         raw_global: Mapping, cities: list[dict]) -> None:
    obs = _mapping(observation, 'observation')
    if (obs.get('schema') != OBSERVATION_SCHEMA or type(obs.get('player_id')) is not int
            or obs['player_id'] != player_id or type(obs.get('turn')) is not int
            or obs['turn'] != turn):
        raise ValueError('yield observation identity mismatch')
    economy = _mapping(obs.get('economy'), 'observation.economy')
    for name in ('science', 'culture', 'faith'):
        observed = _finite_number(economy.get(name), f'observation.economy.{name}')
        if raw_global[name] != observed:
            raise ValueError(f'yield raw {name} differs from player observation')

    observation_cities = obs.get('cities')
    if type(observation_cities) is not list or len(observation_cities) > _MAX_CITIES * 4:
        raise ValueError('observation city roster is malformed or oversized')
    own = {}
    all_ids = set()
    for index, city_value in enumerate(observation_cities):
        city = _mapping(city_value, f'observation.cities[{index}]')
        city_id = _validate_city_id(city.get('id'), f'observation.cities[{index}].id')
        owner = city.get('owner')
        if type(owner) is not int or owner < 0:
            raise ValueError(f'observation.cities[{index}].owner must be an integer')
        if city_id in all_ids:
            raise ValueError('duplicate city id in player observation')
        all_ids.add(city_id)
        if owner == player_id:
            own[city_id] = city
    if len(own) > _MAX_CITIES:
        raise ValueError('observation own-city roster exceeds limit')
    if [row['city_id'] for row in cities] != sorted(own):
        raise ValueError('yield city roster differs from player observation')
    for row in cities:
        city = own[row['city_id']]
        yields = _yield_map(city.get('yields'), CITY_YIELDS,
                            f'observation city {row["city_id"]}.yields', display=False)
        if row['raw'] != yields:
            raise ValueError(f'yield raw values differ from observation city {row["city_id"]}')


def validate(value, player_id: int, turn: int, observation=None) -> dict:
    """Validate strict identity, raw/public values, and optional observation binding."""
    if type(player_id) is not int or not 0 <= player_id <= 63:
        raise ValueError('player_id must be an integer in [0, 63]')
    if type(turn) is not int or turn < 0:
        raise ValueError('turn must be a non-negative integer')
    row = _exact_fields(value, {'schema', 'player_id', 'turn', 'raw', 'public_display', 'cities'},
                        'yield display')
    if (row['schema'] != SCHEMA or type(row['player_id']) is not int
            or row['player_id'] != player_id or type(row['turn']) is not int
            or row['turn'] != turn):
        raise ValueError('yield display identity or schema mismatch')

    raw_global = _yield_map(row['raw'], GLOBAL_YIELDS, 'yield display.raw', display=False)
    public_global = _yield_map(row['public_display'], GLOBAL_YIELDS,
                               'yield display.public_display', display=True)
    for name in GLOBAL_YIELDS:
        shown = public_number(public_global[name])
        if abs(shown - raw_global[name]) > _GLOBAL_TOLERANCE:
            raise ValueError(f'public global {name} differs from raw value')

    city_values = row['cities']
    if type(city_values) is not list or len(city_values) > _MAX_CITIES:
        raise ValueError('yield city roster is malformed or oversized')
    cities = []
    previous_id = -1
    for index, city_value in enumerate(city_values):
        city = _exact_fields(city_value, {'city_id', 'raw', 'public_display'},
                             f'yield cities[{index}]')
        city_id = _validate_city_id(city['city_id'], f'yield cities[{index}].city_id')
        if city_id <= previous_id:
            raise ValueError('yield cities must be sorted by unique city id')
        previous_id = city_id
        raw_city = _yield_map(city['raw'], CITY_YIELDS, f'yield cities[{index}].raw', display=False)
        public_city = _yield_map(city['public_display'], CITY_YIELDS,
                                 f'yield cities[{index}].public_display', display=True)
        for name in CITY_YIELDS:
            expected = _floor_tenth(raw_city[name])
            shown = public_number(public_city[name])
            if abs(shown - expected) > _CITY_TOLERANCE:
                raise ValueError(f'public city {name} differs from floor(raw * 10) / 10')
        cities.append({'city_id': city_id, 'raw': raw_city,
                       'public_display': public_city})

    if observation is not None:
        _observation_binding(observation, player_id, turn, raw_global, cities)
    return {'schema': SCHEMA, 'player_id': player_id, 'turn': turn,
            'raw': raw_global, 'public_display': public_global, 'cities': cities}


__all__ = ['SCHEMA', 'GLOBAL_YIELDS', 'CITY_YIELDS', 'public_number', 'validate']
