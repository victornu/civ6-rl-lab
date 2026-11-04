"""Public ordered BuildQueue projection, without production rules or legality.

The native menu is authoritative. This validator joins its offered gestures to
the observed own-city queue; it does not invent, prune or execute any choice.
Sources: work/production-queue962/source.json and
work/ui-availability/queue-public-source.json (official queue icons/formation).
"""
from __future__ import annotations
import json

ACTOR_SCHEMA = 'civ6rl.recurrent_policy.v9'
CATALOG_SCHEMA = 'civ6rl.production_queue.public.v2'
FEATURE_CONTRACT = 'public_ordered_production_queue_and_exact_positions_v1'
SOURCE_SHA256 = '51cda4a2c0a0bf7f29c58f853186cf364382ab6e882660ae04e58ae467afab6b'
NATIVE_MENU_SOURCE_SHA256 = '1ae2a3a5e331bed654ac98fe7b79b2305c5980652c6dc06a4201a53b32e0f322'
KINDS = ('unit', 'building', 'district', 'project')
ACTION_KINDS = ('queue_append', 'queue_remove', 'queue_swap')
ENTITY_FEATURES = ('kind_production_queue_entry', 'queue_position', 'queue_position_known',
                   'queue_is_corps', 'queue_is_army', 'queue_formation_known')
ACTION_FEATURES = ('queue_remove_position', 'queue_remove_position_known',
                   'queue_source_position', 'queue_source_position_known',
                   'queue_destination_position', 'queue_destination_position_known')
FIELDS = {
    'queue_append': {'kind', 'city_id', 'category', 'index', 'hash', 'item_type', 'mode'},
    'queue_remove': {'kind', 'city_id', 'position'},
    'queue_swap': {'kind', 'city_id', 'source_position', 'destination_position'},
}


def _require(ok, message):
    if not ok: raise ValueError(message)


def _uint(value, maximum, name):
    _require(type(value) is int and 0 <= value <= maximum, 'Invalid queue ' + name)
    return value


def _key(action):
    return json.dumps(action, sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def entry_features(entry):
    # These are the two actual UI markers, not a guessed enum-to-name map.
    display = entry['formation_display']
    corps, army, known = (display['is_corps'], display['is_army'], True) if display is not None else (False, False, False)
    return [1.0, float(entry['position']), 1.0, float(corps), float(army), float(known)]


def action_features(action, size):
    values = [0.0] * len(ACTION_FEATURES)
    if action['kind'] == 'queue_remove': values[0:2] = [float(action['position']), 1.0]
    elif action['kind'] == 'queue_swap':
        values[2:6] = [float(action['source_position']), 1.0, float(action['destination_position']), 1.0]
    else:
        values[4:6] = [float(7 if action['mode'] == 'replace_tail' else size), 1.0]
    return values


def validate_and_index(decisions, observation, *, ended=False):
    _require(type(ended) is bool, 'Queue closure must be Boolean')
    pid, turn = observation.get('player_id'), observation.get('turn')
    _uint(pid, 63, 'player'); _uint(turn, 0x7fffffff, 'turn')
    width, height = observation.get('width'), observation.get('height')
    _require(type(width) is int and type(height) is int and width > 0 and height > 0,
             'Invalid queue map dimensions')
    _require(type(decisions.get('player_id')) is int and decisions['player_id'] == pid
             and type(decisions.get('turn')) is int and decisions['turn'] == turn,
             'Queue decision boundary differs')
    roster, catalogs = observation.get('cities'), decisions.get('production_queues')
    _require(type(roster) is list and len(roster) <= 256, 'Queue owned roster missing')
    cities = {}
    for city in roster:
        _require(type(city) is dict and type(city.get('owner')) is int and city['owner'] == pid,
                 'Queue city is not owned by observed player')
        cid = _uint(city.get('id'), 0xffffffff, 'city identity')
        _uint(city.get('x'), width-1, 'city x'); _uint(city.get('y'), height-1, 'city y')
        _require(cid not in cities, 'Duplicate queue city')
        cities[cid] = city
    _require(type(catalogs) is list and len(catalogs) == len(cities),
             'Queue catalogs must match all observed own cities')
    production = decisions.get('production')
    _require(type(production) is dict and production.get('player_id') == pid
             and production.get('game_turn') == turn and type(production.get('cities')) is list,
             'Queue current production boundary differs')
    heads = {}
    for row in production['cities']:
        _require(type(row) is dict and type(row.get('city_id')) is int
                 and row['city_id'] in cities and row['city_id'] not in heads
                 and type(row.get('owner_id')) is int and row['owner_id'] == pid,
                 'Queue current production roster differs')
        heads[row['city_id']] = row
    _require(set(heads) == set(cities), 'Queue current production is incomplete')
    expected, locations, catalog_ids = {}, [], []
    content_identities = {}
    for catalog in catalogs:
        _require(type(catalog) is dict and set(catalog) ==
                 {'schema', 'player_id', 'turn', 'city_id', 'enabled', 'entries', 'offers'}
                 and catalog['schema'] == CATALOG_SCHEMA,
                 'Queue catalog fields differ')
        cid = catalog['city_id']
        _require(type(catalog['player_id']) is int and catalog['player_id'] == pid
                 and type(catalog['turn']) is int and catalog['turn'] == turn
                 and type(cid) is int and cid in cities and cid not in catalog_ids
                 and type(catalog['enabled']) is bool, 'Queue catalog identity differs')
        catalog_ids.append(cid)
        entries, offers = catalog['entries'], catalog['offers']
        _require(type(entries) is list and len(entries) <= 8
                 and type(offers) is list and len(offers) <= 320, 'Queue catalog exceeds bound')
        for position, entry in enumerate(entries):
            _require(type(entry) is dict and set(entry) == {'position', 'category', 'index', 'hash', 'formation', 'formation_display'}
                     and type(entry['position']) is int and entry['position'] == position
                     and entry['category'] in KINDS, 'Queue entries must retain exact native order')
            _uint(entry['index'], 0x7fffffff, 'content index'); _uint(entry['hash'], 0xffffffff, 'content hash')
            display = entry['formation_display']
            if entry['category'] == 'unit':
                _require(type(entry['formation']) is int and -0x80000000 <= entry['formation'] <= 0x7fffffff,
                         'Invalid raw queue formation audit value')
                _require(type(display) is dict and set(display) == {'is_corps', 'is_army'}
                         and type(display['is_corps']) is bool and type(display['is_army']) is bool
                         and not (display['is_corps'] and display['is_army']),
                         'Invalid public queue formation markers')
            else:
                _require(entry['formation'] is None and display is None, 'Nonunit queue formation must be absent')
            identity = (entry['category'], entry['index'])
            _require(identity not in content_identities or content_identities[identity] == entry['hash'],
                     'Queue content index/hash identity disagrees')
            content_identities[identity] = entry['hash']
            locations.append(dict(city=cities[cid], entry=entry))
        head = heads[cid]
        _require(type(head.get('queue_size')) is int and head['queue_size'] == len(entries),
                 'Queue size differs from current production')
        if entries:
            first = entries[0]
            _require(type(head.get('current_production_hash')) is int
                     and (first['category'], first['index'], first['hash']) ==
                     ({'U':'unit', 'B':'building', 'D':'district', 'P':'project'}.get(head.get('kind')),
                      head.get('item_index'), head['current_production_hash'] & 0xffffffff),
                     'Queue head differs from current production')
        else:
            _require(head.get('kind') == 'N' and head.get('current_production_hash') == 0
                     and head.get('item_index') is None, 'Empty queue has current production')
        for action in offers:
            _require(type(action) is dict and action.get('kind') in FIELDS
                     and set(action) == FIELDS[action['kind']] and type(action['city_id']) is int
                     and action['city_id'] == cid and catalog['enabled'],
                     'Queue offer is malformed, foreign or disabled')
            if action['kind'] == 'queue_append':
                _require(action['category'] in KINDS and type(action['item_type']) is str
                         and action['item_type'].startswith(action['category'].upper()+'_')
                         and action['mode'] == ('replace_tail' if len(entries) == 8 else 'append'),
                         'Queue append category or destination differs')
                _uint(action['index'], 0x7fffffff, 'append index'); _uint(action['hash'], 0xffffffff, 'append hash')
                identity = (action['category'], action['index'])
                _require(identity not in content_identities or content_identities[identity] == action['hash'],
                         'Queue append content identity differs')
                content_identities[identity] = action['hash']
            elif action['kind'] == 'queue_remove':
                _uint(action['position'], len(entries)-1, 'remove position')
            else:
                _uint(action['source_position'], len(entries)-1, 'source position')
                _uint(action['destination_position'], len(entries)-1, 'destination position')
                _require(action['source_position'] != action['destination_position'], 'Queue swap must change positions')
            key = _key(action)
            _require(key not in expected, 'Duplicate queue offer')
            expected[key] = action_features(action, len(entries))
    _require(catalog_ids == sorted(cities), 'Queue catalogs must retain ordered complete owned city roster')
    candidates, found, indexed = decisions.get('candidates'), set(), {}
    _require(type(candidates) is list, 'Queue candidate menu missing')
    for row in candidates:
        _require(type(row) is dict and type(row.get('action')) is dict, 'Malformed queue candidate')
        action = row['action']
        if action.get('kind') not in ACTION_KINDS: continue
        key = _key(action); candidate_id = row.get('candidate_id')
        _require(key in expected and key not in found and type(candidate_id) is str
                 and bool(candidate_id) and candidate_id not in indexed,
                 'Queue candidate differs from exact native offers')
        found.add(key); indexed[candidate_id] = expected[key]
    _require(not found if ended else found == set(expected),
             'Queue menu must contain every offered gesture exactly once')
    locations.sort(key=lambda item: (item['city']['y'], item['city']['x'], item['entry']['position']))
    return dict(locations=locations, candidates=indexed)
