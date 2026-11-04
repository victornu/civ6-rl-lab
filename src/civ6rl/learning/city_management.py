"""Strict public citizen/specialist click contract for the grouped v8 actor.

The official UI uses one MANAGE click for ordinary plots and specialist slots.
Its icon exposes occupation and a lock Boolean. Exact worker/capacity numbers
are displayed only by the multi-slot meter; absent numbers stay unknown.
"""
from __future__ import annotations

ACTOR_SCHEMA = 'civ6rl.recurrent_policy.v8'
DECISIONS_SCHEMA = 'civ6rl.saved_decisions.city_management.v1'
CATALOG_SCHEMA = 'civ6rl.native_city_citizens.v1'
FEATURE_CONTRACT = 'public_manage_citizen_single_click_and_focus_v1'
ACTION_KIND = 'city_citizen'
ACTION_FIELDS = {'kind', 'city_id', 'plot_id', 'x', 'y', 'occupied', 'locked', 'workers', 'capacity'}
PLOT_FIELDS = {'plot_id', 'x', 'y', 'occupied', 'locked', 'workers', 'capacity'}
ACTION_FEATURES = ('citizen_occupied', 'citizen_occupied_known', 'citizen_locked', 'citizen_locked_known',
                   'citizen_workers', 'citizen_workers_known', 'citizen_capacity', 'citizen_capacity_known')
ENTITY_FEATURES = ('kind_citizen_location', 'citizen_city_x_norm', 'citizen_city_y_norm',
                   'citizen_city_xy_known') + ACTION_FEATURES


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _uint(value, maximum, message):
    _require(type(value) is int and 0 <= value <= maximum, message)
    return value


def public_features(row):
    if row is None:
        return [0.0] * len(ACTION_FEATURES)
    workers, capacity = row['workers'], row['capacity']
    return [float(row['occupied']), 1.0, float(row['locked']), 1.0,
            float(workers) if workers is not None else 0.0, float(workers is not None),
            float(capacity) if capacity is not None else 0.0, float(capacity is not None)]


def location_features(city, row, width, height):
    norm = lambda value, span: value / max(1, span - 1)
    return [1.0, norm(city['x'], width), norm(city['y'], height), 1.0] + public_features(row)


def validate_and_index(decisions, observation, player_map, *, ended=False):
    _require(type(ended) is bool, 'Citizen closure must be Boolean')
    _require(type(decisions) is dict and decisions.get('schema') == DECISIONS_SCHEMA,
             'Unsupported grouped city-management decision schema')
    _require(type(observation) is dict, 'Citizen player observation missing')
    pid = _uint(observation.get('player_id'), 63, 'Invalid citizen player')
    turn = _uint(observation.get('turn'), 0x7fffffff, 'Invalid citizen turn')
    width, height = observation.get('width'), observation.get('height')
    _require(type(width) is int and type(height) is int and width > 0 and height > 0
             and width * height <= 16384, 'Invalid citizen map dimensions')
    _require(type(decisions.get('player_id')) is int and decisions['player_id'] == pid
             and type(decisions.get('turn')) is int and decisions['turn'] == turn,
             'Citizen decision boundary differs')
    _require(type(player_map) is dict and type(player_map.get('player_id')) is int
             and player_map['player_id'] == pid and type(player_map.get('turn')) is int
             and player_map['turn'] == turn and type(player_map.get('width')) is int
             and player_map['width'] == width and type(player_map.get('height')) is int
             and player_map['height'] == height and type(player_map.get('plots')) is list,
             'Citizen public map boundary differs')
    revealed = {}
    for plot in player_map['plots']:
        _require(type(plot) is dict and plot.get('revealed') is True
                 and type(plot.get('index')) is int and plot['index'] not in revealed,
                 'Citizen public map has unknown or duplicate plots')
        revealed[plot['index']] = plot
    roster = observation.get('cities')
    _require(type(roster) is list and len(roster) <= 4096, 'Citizen owned city roster missing')
    cities = {}
    for city in roster:
        _require(type(city) is dict and type(city.get('owner')) is int and city['owner'] == pid,
                 'Citizen city is not owned by the observed player')
        cid = _uint(city.get('id'), 0xffffffff, 'Invalid citizen city identity')
        _uint(city.get('x'), width-1, 'Invalid citizen city x')
        _uint(city.get('y'), height-1, 'Invalid citizen city y')
        _require(cid not in cities, 'Duplicate citizen city')
        cities[cid] = city
    catalogs = decisions.get('citizen_catalogs')
    _require(type(catalogs) is list and len(catalogs) == len(cities),
             'Citizen catalogs must match every observed owned city')
    seen_cities, expected, locations = set(), {}, []
    for catalog in catalogs:
        _require(type(catalog) is dict and set(catalog) ==
                 {'schema', 'player_id', 'turn', 'city_id', 'plots'}
                 and catalog['schema'] == CATALOG_SCHEMA
                 and type(catalog['player_id']) is int and catalog['player_id'] == pid
                 and type(catalog['turn']) is int and catalog['turn'] == turn
                 and type(catalog['city_id']) is int and catalog['city_id'] in cities
                 and catalog['city_id'] not in seen_cities, 'Citizen catalogue identity differs')
        cid = catalog['city_id']; seen_cities.add(cid)
        _require(type(catalog['plots']) is list and len(catalog['plots']) <= 16384,
                 'Citizen public target list exceeds bound')
        seen_plots, previous_plot = set(), -1
        for row in catalog['plots']:
            _require(type(row) is dict and set(row) == PLOT_FIELDS, 'Citizen public target fields differ')
            plot_id = _uint(row['plot_id'], width*height-1, 'Citizen plot outside map')
            x = _uint(row['x'], width-1, 'Citizen target x outside map')
            y = _uint(row['y'], height-1, 'Citizen target y outside map')
            _require(plot_id == y*width+x and plot_id not in seen_plots and plot_id > previous_plot,
                     'Citizen targets must be ordered, unique and match their coordinates')
            seen_plots.add(plot_id)
            previous_plot = plot_id
            mapped = revealed.get(plot_id)
            _require(mapped is not None and mapped.get('x') == x and mapped.get('y') == y,
                     'Citizen target is not a revealed plot with matching coordinates')
            _require(type(row['occupied']) is bool and type(row['locked']) is bool,
                     'Citizen icon states must be Boolean')
            workers, capacity = row['workers'], row['capacity']
            _require((workers is None) == (capacity is None), 'Citizen meter must publish both values or neither')
            if workers is not None:
                _uint(workers, 0x7fffffff, 'Invalid displayed citizen count')
                _uint(capacity, 0x7fffffff, 'Invalid displayed citizen capacity')
                _require(capacity > 1 and row['occupied'] is (workers >= 1),
                         'Citizen displayed meter and icon disagree')
            key = cid, plot_id
            expected[key] = dict(kind=ACTION_KIND, city_id=cid, **row)
            locations.append(dict(city=cities[cid], plot=row))
    _require(seen_cities == set(cities), 'Citizen owned roster incomplete')
    candidates = decisions.get('candidates')
    _require(type(candidates) is list, 'Citizen action menu missing')
    found, choices = set(), {}
    for candidate in candidates:
        _require(type(candidate) is dict and type(candidate.get('action')) is dict,
                 'Malformed citizen candidate envelope')
        action = candidate['action']
        if action.get('kind') != ACTION_KIND:
            continue
        _require(set(action) == ACTION_FIELDS and type(action['city_id']) is int
                 and type(action['plot_id']) is int and type(action['x']) is int
                 and type(action['y']) is int and type(action['occupied']) is bool
                 and type(action['locked']) is bool
                 and (action['workers'] is None or type(action['workers']) is int)
                 and (action['capacity'] is None or type(action['capacity']) is int),
                 'Citizen action must be an exact public UI click')
        key = action['city_id'], action['plot_id']
        _require(key in expected and key not in found and action == expected[key],
                 'Citizen action differs from its public target state')
        candidate_id = candidate.get('candidate_id')
        _require(type(candidate_id) is str and bool(candidate_id) and candidate_id not in choices,
                 'Citizen candidate identity missing or duplicated')
        found.add(key)
        city = cities[action['city_id']]
        choices[candidate_id] = dict(source=(city['x'], city['y']), target=(action['x'], action['y']),
                                     features=public_features(action))
    _require(not found if ended else found == set(expected),
             'Citizen menu must contain every native public target exactly once')
    # Geometry, not opaque city IDs, determines the observation entity order.
    locations.sort(key=lambda item: (item['city']['y'], item['city']['x'], item['plot']['y'], item['plot']['x']))
    return dict(candidates=choices, locations=locations)
