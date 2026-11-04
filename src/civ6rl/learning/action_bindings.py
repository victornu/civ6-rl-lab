"""Public candidate-to-entity joins, independent of opaque instance ID values.

This is preparation for the next grouped actor contract, not a replacement for
the v9 encoder. IDs are transient join keys; only row indices reach a model.
The native menu determines availability. This module never removes candidates,
queries hidden state, executes actions or calculates combat legality.
"""
from __future__ import annotations

SCHEMA = 'civ6rl.public_action_source_bindings.v1'
DISTRICT_ENTITY_FEATURES = (
    'kind_own_district', 'district_complete', 'district_banner_eligible',
    'district_attack_available',
)
DISTRICT_CATALOG_SOURCE = {
    'path': 'civ6rl/native_saved_district_shot.py',
    'sha256': '2b271f8c8754b4241a177c44c3bced7e94db511e8801da96ea4bcc5a29c7fb57',
}
_DISTRICT_FIELDS = {'district_id', 'city_id', 'type_index', 'x', 'y',
                    'complete', 'banner_eligible', 'available', 'targets'}
_DISTRICT_ACTION_FIELDS = {'kind', 'district_id', 'plot_id', 'x', 'y'}


def _require(condition, message):
    if not condition: raise ValueError(message)


def _uint(value, name, maximum=0xffffffff):
    _require(type(value) is int and 0 <= value <= maximum, 'Invalid public source ' + name)
    return value


def district_features(row):
    """Only flags already in the native public district catalogue."""
    _require(type(row) is dict and all(type(row.get(k)) is bool
             for k in ('complete', 'banner_eligible', 'available')), 'Invalid public district flags')
    return [1.0, float(row['complete']), float(row['banner_eligible']), float(row['available'])]


def build_source_bindings(decisions, observation, *, ended=False):
    """Validate joins for every offered action, retaining the native candidate order.

District rows include IDs solely for the encoder to create its entity-key map.
Do not serialize those keys into numeric/categorical actor inputs.
"""
    _require(type(decisions) is dict and type(observation) is dict and type(ended) is bool,
             'Public binding envelope required')
    pid = _uint(observation.get('player_id'), 'player', 63)
    turn = _uint(observation.get('turn'), 'turn', 0x7fffffff)
    width = _uint(observation.get('width'), 'width', 16384)
    height = _uint(observation.get('height'), 'height', 16384)
    _require(0 < width * height <= 16384, 'Public binding map outside supported bound')
    _require(type(decisions.get('player_id')) is int and decisions['player_id'] == pid
             and type(decisions.get('turn')) is int and decisions['turn'] == turn,
             'Public binding decision perspective differs')
    own = {}
    for kind, field in (('unit', 'units'), ('city', 'cities')):
        rows = observation.get(field)
        _require(type(rows) is list and len(rows) <= 4096, 'Missing or unbounded own source roster')
        for row in rows:
            _require(type(row) is dict and type(row.get('owner')) is int and row['owner'] == pid,
                     'Source entity is not owned by observed player')
            identity = _uint(row.get('id'), kind + ' ID')
            x = _uint(row.get('x'), kind + ' x', width - 1)
            y = _uint(row.get('y'), kind + ' y', height - 1)
            key = (kind, identity)
            _require(key not in own, 'Duplicate own source entity')
            own[key] = (x, y)
    districts, expected_shots = [], set()
    catalog = decisions.get('district_shots')
    if catalog is not None:
        _require(type(catalog) is dict and set(catalog) == {'player_id', 'turn', 'width', 'height', 'districts'},
                 'Public district catalogue fields differ')
        for field, value in (('player_id', pid), ('turn', turn), ('width', width), ('height', height)):
            _require(type(catalog[field]) is int and catalog[field] == value,
                     'Public district catalogue boundary differs')
        _require(type(catalog['districts']) is list and len(catalog['districts']) <= 4096,
                 'Public district catalogue outside bound')
        ids, plots = [], set()
        for row in catalog['districts']:
            _require(type(row) is dict and set(row) == _DISTRICT_FIELDS, 'Public district row fields differ')
            did = _uint(row['district_id'], 'district ID')
            cid = _uint(row['city_id'], 'district city ID')
            _uint(row['type_index'], 'district type', 0x7fffffff)
            x = _uint(row['x'], 'district x', width - 1)
            y = _uint(row['y'], 'district y', height - 1)
            district_features(row)
            _require(('city', cid) in own and own['city', cid] != (x, y),
                     'Independent district lacks its own city or is the city center')
            _require((x, y) not in plots and ('district', did) not in own, 'Duplicate public district')
            _require((not row['available'] or row['banner_eligible'])
                     and (not row['banner_eligible'] or row['complete']), 'Inconsistent public district flags')
            targets = row['targets']
            _require(type(targets) is list and len(targets) <= 4096
                     and all(type(t) is int and 0 <= t < width * height for t in targets)
                     and targets == sorted(set(targets)) and (not targets or row['available']),
                     'Invalid public district target catalogue')
            ids.append(did); plots.add((x, y)); own['district', did] = (x, y)
            expected_shots.update((did, t) for t in targets)
            districts.append(dict(row, targets=list(targets)))
        _require(ids == sorted(set(ids)) and len(expected_shots) <= 4096,
                 'Unordered or unbounded public district catalogue')
    candidates = decisions.get('candidates')
    _require(type(candidates) is list, 'Public candidate menu missing')
    sources, found_shots = {}, set()
    for candidate in candidates:
        _require(type(candidate) is dict and set(candidate) == {'candidate_id', 'action'}, 'Malformed public candidate')
        cid, action = candidate['candidate_id'], candidate['action']
        _require(type(cid) is str and bool(cid) and cid not in sources and type(action) is dict,
                 'Duplicate or malformed public candidate identity')
        key = None
        if action.get('kind') == 'district_range_attack':
            _require(set(action) == _DISTRICT_ACTION_FIELDS, 'District action fields differ')
            did = _uint(action['district_id'], 'shot district ID')
            target = _uint(action['plot_id'], 'shot target', width * height - 1)
            _require(type(action['x']) is int and type(action['y']) is int
                     and (action['x'], action['y']) == (target % width, target // width),
                     'District action target differs from plot')
            pair = did, target
            _require(pair in expected_shots and pair not in found_shots,
                     'District action is not exactly one native offered target')
            found_shots.add(pair); key = ('district', did)
        else:
            _require('district_id' not in action, 'Unqualified district binding on another action')
            # A trader owns its order even when the action also identifies cities.
            # Check every public owner join; only the acting source is selected.
            for field, kind in (('unit_id', 'unit'), ('city_id', 'city'), ('origin_city_id', 'city')):
                if field in action:
                    identity = _uint(action[field], field)
                    _require((kind, identity) in own, 'Action source is absent from own public roster')
                    if key is None: key = (kind, identity)
        if key is not None:
            _require(key in own, 'Action source is absent from own public roster')
        sources[cid] = key
    _require(not candidates if ended else found_shots == expected_shots,
             'Public binding catalogue lost an offered action or contains actions after closure')
    # Geometry gives a stable district token order without embedding native IDs.
    districts.sort(key=lambda d: (d['y'], d['x']))
    return dict(schema=SCHEMA, sources=sources, districts=districts)


def resolve_source_rows(bindings, entity_keys):
    """Translate transient own-entity keys to dense token rows; -1 means no source.

Pass None for an entity token without an acting-source identity (for example
a public queue entry or citizen location). The caller gathers these token rows
in the candidate encoder; it must not embed the native key itself.
"""
    _require(type(bindings) is dict and set(bindings) == {'schema', 'sources', 'districts'}
             and bindings['schema'] == SCHEMA and type(bindings['sources']) is dict,
             'Invalid public binding contract')
    _require(type(entity_keys) is list, 'Entity key rows must be explicit')
    rows = {}
    for index, key in enumerate(entity_keys):
        if key is None: continue
        _require(type(key) is tuple and len(key) == 2 and key[0] in ('unit', 'city', 'district'),
                 'Unqualified source entity key')
        _uint(key[1], 'encoded source ID')
        _require(key not in rows, 'Duplicate encoded source entity')
        rows[key] = index
    result = []
    for key in bindings['sources'].values():
        _require(key is None or key in rows, 'Public action source has no encoded entity row')
        result.append(-1 if key is None else rows[key])
    return result
