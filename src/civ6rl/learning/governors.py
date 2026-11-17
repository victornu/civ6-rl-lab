"""Public governor catalog features for the explicit actor v16 input."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

CONTRACT = 'public_native_governor_catalog_v1'
ACTION_KINDS = ('appoint_governor', 'promote_governor', 'assign_governor',
                'consider_governor_titles')

ENTITY_FEATURES = (
    'kind_governor_titles', 'governor_points_obtained', 'governor_points_obtained_known',
    'governor_points_spent', 'governor_points_spent_known', 'governor_can_appoint',
    'governor_titles_considered',
    'kind_governor', 'governor_appointed', 'governor_can_ever_appoint',
    'governor_can_promote', 'governor_assignment_open', 'governor_turns_to_establish',
    'governor_turns_to_establish_known', 'governor_status_known',
    'governor_neutralized_turns', 'governor_neutralized_turns_known',
    'governor_established', 'governor_established_known',
    'governor_status_turns_to_establish', 'governor_status_turns_to_establish_known',
    'governor_turns_on_site', 'governor_turns_on_site_known',
    'governor_assigned_city_known', 'governor_govern_city_state',
    'governor_govern_city_state_known', 'governor_can_assign_to_major_civ',
    'governor_can_assign_to_major_civ_known', 'governor_promotion_count',
    'governor_assignment_target_count', 'governor_assignable_target_count',
)

ACTION_FEATURES = (
    'governor_titles_context', 'governor_points_obtained',
    'governor_points_obtained_known', 'governor_points_spent',
    'governor_points_spent_known', 'governor_can_appoint', 'governor_titles_considered',
    'governor_selected', 'governor_appointed', 'governor_can_ever_appoint',
    'governor_can_promote', 'governor_assignment_open',
    'governor_turns_to_establish', 'governor_turns_to_establish_known',
    'governor_status_known', 'governor_neutralized_turns',
    'governor_neutralized_turns_known', 'governor_established',
    'governor_established_known', 'governor_status_turns_to_establish',
    'governor_status_turns_to_establish_known', 'governor_turns_on_site',
    'governor_turns_on_site_known', 'governor_assigned_city_known',
    'governor_govern_city_state', 'governor_govern_city_state_known',
    'governor_can_assign_to_major_civ', 'governor_can_assign_to_major_civ_known',
    'governor_promotion_count', 'governor_assignment_target_count',
    'governor_assignable_target_count', 'promotion_selected', 'promotion_owned',
    'promotion_can_earn', 'promotion_hidden', 'promotion_base_ability',
    'promotion_level', 'promotion_level_known', 'assignment_target_selected',
    'assignment_target_can_assign', 'governor_target_city_is_own',
    'governor_target_city_x_norm', 'governor_target_city_x_known',
    'governor_target_city_y_norm', 'governor_target_city_y_known',
    'governor_target_city_population', 'governor_target_city_population_known',
    'governor_target_city_food_yield', 'governor_target_city_food_yield_known',
    'governor_target_city_production_yield', 'governor_target_city_production_yield_known',
    'governor_target_city_gold_yield', 'governor_target_city_gold_yield_known',
    'governor_target_city_science_yield', 'governor_target_city_science_yield_known',
    'governor_target_city_culture_yield', 'governor_target_city_culture_yield_known',
    'governor_target_city_faith_yield', 'governor_target_city_faith_yield_known',
)

_CITY_BYTE_FIELDS = tuple(f'city_id_byte{i}' for i in range(4))
ENTITY_CATEGORICAL_FIELDS = (
    'governor_index', 'governor_hash', 'city_owner', *_CITY_BYTE_FIELDS,
)
ACTION_CATEGORICAL_FIELDS = (
    'governor_index', 'governor_hash', 'promotion_index', 'promotion_hash',
    'city_owner', *_CITY_BYTE_FIELDS,
)

def entity_city_categories(city, vocabulary):
    if city is None:
        return [0] * len(ENTITY_CATEGORICAL_FIELDS)
    return [0, 0, vocabulary.lookup('city.owner', city['owner'])] + [
        vocabulary.lookup(f'city.id_byte{i}', byte)
        for i, byte in enumerate(city_id_bytes(city['id']))]


def action_categorical_values(action, vocabulary):
    values = {field: 0 for field in ACTION_CATEGORICAL_FIELDS}
    for field, namespace in (('governor_index', 'governor.index'),
                             ('governor_hash', 'governor.hash'),
                             ('promotion_index', 'promotion.index'),
                             ('promotion_hash', 'promotion.hash')):
        if field in action:
            values[field] = vocabulary.lookup(namespace, action[field])
    if 'city_owner' in action:
        values['city_owner'] = vocabulary.lookup('city.owner', action['city_owner'])
        for index, byte in enumerate(city_id_bytes(action['city_id'])):
            values[f'city_id_byte{index}'] = vocabulary.lookup(f'city.id_byte{index}', byte)
    return [values[field] for field in ACTION_CATEGORICAL_FIELDS]


def city_id_bytes(value: int) -> tuple[int, int, int, int]:
    if type(value) is not int or not 0 <= value <= 0xffffffff:
        raise ValueError('governor city id must be uint32')
    return tuple((value >> (8 * index)) & 0xff for index in range(4))


def categorical_tokens():
    """Return the bounded governor identity and dynamic-city token set."""
    tokens = set()
    for namespace, values in (
        ('city.owner', range(64)),
        *((f'city.id_byte{i}', range(256)) for i in range(4)),
    ):
        tokens.update(_token(namespace, value) for value in values)
    return tokens


def _token(namespace, value):
    return namespace + ':' + json.dumps(value, ensure_ascii=True, separators=(',', ':'))


def extend_governor_vocabulary(parent, views):
    """Extend a frozen vocabulary from validated public v32 catalogues only."""
    from .encoding import ContentVocabulary
    from . import lan_inputs

    if type(parent) is not ContentVocabulary or type(views) not in (list, tuple) or not views:
        raise ValueError('governor vocabulary extension requires a parent and bounded public views')
    if len(views) > 100_000:
        raise ValueError('governor vocabulary view bound exceeded')
    tokens = set(parent.tokens)
    tokens.update(categorical_tokens())
    catalogue_digests = []
    for view in views:
        lan_inputs.validate(view)
        decisions, obs = view['decisions'], view['observation']
        if decisions.get('schema') != 'civ6-host.network-capability-decisions.v32':
            raise ValueError('governor vocabulary requires validated v32 views')
        catalogue = decisions['governors']
        if catalogue is None:
            continue
        from civ6_host.contracts.governors import validate_catalogue
        validate_catalogue(catalogue, obs['player_id'], obs['turn'])
        for governor in catalogue['governors']:
            for namespace, value in (('governor.index', governor['index']),
                                     ('governor.hash', governor['hash'])):
                tokens.add(_token(namespace, value))
            for promotion in governor['promotions']:
                tokens.add(_token('promotion.index', promotion['index']))
                tokens.add(_token('promotion.hash', promotion['hash']))
        canonical = json.dumps(catalogue, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
        catalogue_digests.append(hashlib.sha256(canonical.encode('utf-8')).hexdigest())
    manifest = deepcopy(parent.manifest) if parent.manifest is not None else {}
    manifest['governor_extension'] = {
        'schema': 'civ6rl.learning.governor-vocabulary-extension.v1',
        'parent_fingerprint': parent.fingerprint,
        'public_catalogue_sha256': catalogue_digests,
        'dynamic_city_id_encoding': 'uint32_little_endian_4x_uint8_v1',
    }
    return ContentVocabulary(sorted(tokens), manifest=manifest)


def _features(governor=None, catalogue=None, *, selected=False, promotion=None, target=None,
              target_city=None, target_city_yields=None, player_id=None, width=None, height=None):
    values = []
    if catalogue is not None:
        values.extend([1.0, float(catalogue['points_obtained']), 1.0,
                       float(catalogue['points_spent']), 1.0,
                       float(catalogue['can_appoint']), float(catalogue['titles_considered'])])
    else:
        values.extend([0.0] * 7)
    if governor is None:
        values.extend([0.0] * 24)
    else:
        status = governor['status']
        values.extend([
            float(selected), float(governor['appointed']), float(governor['can_ever_appoint']),
            float(governor['can_promote']), float(governor['assignment_open']),
            float(governor['turns_to_establish']), 1.0, float(status is not None),
            float(status['neutralized_turns']) if status else 0.0, float(status is not None),
            float(status['established']) if status else 0.0, float(status is not None),
            float(status['turns_to_establish']) if status else 0.0, float(status is not None),
            float(status['turns_on_site']) if status else 0.0, float(status is not None),
            float(status is not None),
            float(status['govern_city_state']) if status else 0.0, float(status is not None),
            float(status['can_assign_to_major_civ']) if status else 0.0, float(status is not None),
            float(len(governor['promotions'])), float(len(governor['assignment_targets'])),
            float(sum(target_row['can_assign'] for target_row in governor['assignment_targets'])),
        ])
    if promotion is None:
        values.extend([0.0] * 7)
    else:
        values.extend([1.0, float(promotion['owned']), float(promotion['can_earn']),
                       float(promotion['hidden']), float(promotion['base_ability']),
                       float(promotion['level']), 1.0])
    values.extend([float(target is not None), float(target['can_assign']) if target else 0.0])
    own_target = (target is not None and target_city is not None
                  and target['owner'] == player_id and target_city.get('owner') == player_id
                  and target_city.get('id') == target['city_id'])
    values.append(float(own_target))
    if own_target:
        from .encoding import _norm_coord
        x, y = target_city['x'], target_city['y']
        values.extend([_norm_coord(x, width), 1.0,
                       _norm_coord(y, height), 1.0,
                       float(target_city['population']), 1.0])
        yields = target_city_yields if target_city_yields is not None else target_city.get('yields', {})
        for name in ('food', 'production', 'gold', 'science', 'culture', 'faith'):
            known = name in yields
            values.extend([float(yields[name]) if known else 0.0, float(known)])
    else:
        values.extend([0.0] * 18)
    if len(values) != len(ACTION_FEATURES):
        raise ValueError('governor action feature layout differs')
    return values


def action_features(action, catalogue, *, observation=None, city_yields=None, width=None, height=None):
    if catalogue is None:
        raise ValueError('governor action requires a public catalogue')
    kind = action['kind']
    if kind == 'consider_governor_titles':
        return _features(catalogue=catalogue)
    governor = next((row for row in catalogue['governors']
                     if row['index'] == action['governor_index']
                     and row['hash'] == action['governor_hash']), None)
    if governor is None:
        raise ValueError('governor candidate identity is absent from its public catalogue')
    promotion = None
    target = None
    if kind == 'promote_governor':
        promotion = next((row for row in governor['promotions']
                          if row['index'] == action['promotion_index']
                          and row['hash'] == action['promotion_hash']), None)
        if promotion is None:
            raise ValueError('promotion candidate identity is absent from its public catalogue')
    if kind == 'assign_governor':
        target = next((row for row in governor['assignment_targets']
                       if row['owner'] == action['city_owner']
                       and row['city_id'] == action['city_id']), None)
        if target is None:
            raise ValueError('assignment candidate target is absent from its public catalogue')
    target_city = None
    target_city_yields = None
    if target is not None and observation is not None:
        if target['owner'] == observation['player_id']:
            target_city = next((row for row in observation['cities']
                                if row['owner'] == observation['player_id']
                                and row['id'] == target['city_id']), None)
            if city_yields is not None and target_city is not None:
                target_city_yields = city_yields.get(target['city_id'])
    return _features(governor, catalogue, selected=True, promotion=promotion, target=target,
                     target_city=target_city, target_city_yields=target_city_yields,
                     player_id=observation['player_id'] if observation is not None else None,
                     width=width, height=height)


def entity_rows(catalogue, vocabulary):
    """Return appended entity numeric/category rows and identity join keys."""
    rows, cats, keys = [], [], []
    if catalogue is None:
        return rows, cats, keys
    title = [1.0, float(catalogue['points_obtained']), 1.0,
             float(catalogue['points_spent']), 1.0, float(catalogue['can_appoint']),
             float(catalogue['titles_considered'])]
    rows.append(title + [0.0] * (len(ENTITY_FEATURES) - len(title)))
    cats.append([0] * len(ENTITY_CATEGORICAL_FIELDS))
    keys.append(None)
    for governor in catalogue['governors']:
        status = governor['status']
        extra = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, float(governor['appointed']), float(governor['can_ever_appoint']),
            float(governor['can_promote']), float(governor['assignment_open']),
            float(governor['turns_to_establish']), 1.0, float(status is not None),
            float(status['neutralized_turns']) if status else 0.0, float(status is not None),
            float(status['established']) if status else 0.0, float(status is not None),
            float(status['turns_to_establish']) if status else 0.0, float(status is not None),
            float(status['turns_on_site']) if status else 0.0, float(status is not None),
            float(status is not None),
            float(status['govern_city_state']) if status else 0.0, float(status is not None),
            float(status['can_assign_to_major_civ']) if status else 0.0, float(status is not None),
            float(len(governor['promotions'])), float(len(governor['assignment_targets'])),
            float(sum(target['can_assign'] for target in governor['assignment_targets'])),
        ]
        if len(extra) != len(ENTITY_FEATURES):
            raise ValueError('governor entity feature layout differs')
        values = [vocabulary.lookup('governor.index', governor['index']),
                  vocabulary.lookup('governor.hash', governor['hash'])]
        values.extend([0] * (len(ENTITY_CATEGORICAL_FIELDS) - len(values)))
        if status is not None and status['assigned_city'] is not None:
            city = status['assigned_city']
            values[2] = vocabulary.lookup('city.owner', city['owner'])
            values[3:] = [vocabulary.lookup(f'city.id_byte{i}', byte)
                          for i, byte in enumerate(city_id_bytes(city['city_id']))]
        rows.append(extra)
        cats.append(values)
        keys.append(('governor', (governor['index'], governor['hash'])))
    return rows, cats, keys

