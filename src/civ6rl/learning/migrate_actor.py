"""Explicit actor-only feature extensions for grouped public actor contracts.

Preserves named features and public-content tokens from a strict parent actor.
The feature, WorldInput, war-display and resource-quantity routes are explicit,
never chained implicitly.
New numeric columns and new tokens start at zero. The architecture and every
other tensor stay unchanged. No optimizer, recurrent state or critic migrates.
The v10 source-token join changes the forward semantics; copied tensors do not
imply equal action probabilities or authorize replaying an old rollout as v10.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import torch
from .artifacts import load_policy_bytes, policy_bytes
from .model import RecurrentPolicy, actor_schema_for_config
from .encoding import ContentVocabulary, ACTOR_V6, ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15, ACTOR_V16, actor_feature_names

ACTOR_V15 = 'civ6rl.recurrent_policy.v15'
ACTOR_V16 = 'civ6rl.recurrent_policy.v16'


def migrate_v15_parameter_tensor(tensor, target_shape, mapping):
    """Copy a V15 parameter or optimizer tensor into its named V16 axes.

    ``mapping`` is ``None`` for a strict shape-preserving copy, or a mapping
    with ``axis`` and ``indices`` where each source-axis index names its target
    index. New target entries remain zero.
    """
    if not isinstance(tensor, torch.Tensor) or not isinstance(target_shape, (tuple, list, torch.Size)):
        raise ValueError('V15 tensor migration requires a tensor and target shape')
    shape = tuple(target_shape)
    if any(type(size) is not int or size < 0 for size in shape) or len(shape) != tensor.ndim:
        raise ValueError('V16 tensor target shape differs in rank or dimensions')
    if mapping is None:
        if tuple(tensor.shape) != shape:
            raise ValueError('identity tensor migration requires an exact shape')
        return tensor.detach().clone()
    if type(mapping) is not dict or set(mapping) != {'axis', 'indices'}:
        raise ValueError('V16 parameter mapping must name one axis and its parent indices')
    axis, indices = mapping['axis'], mapping['indices']
    if (type(axis) is not int or axis not in (0, 1) or axis >= tensor.ndim
            or type(indices) is not list or len(indices) != tensor.shape[axis]
            or shape[axis] < tensor.shape[axis]
            or any(type(index) is not int or not 0 <= index < shape[axis] for index in indices)
            or len(set(indices)) != len(indices)
            or any(tensor.shape[i] != shape[i] for i in range(tensor.ndim) if i != axis)):
        raise ValueError('V16 tensor mapping is not a bounded injective axis extension')
    target = torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
    if indices:
        target.index_copy_(axis, torch.tensor(indices, dtype=torch.long, device=tensor.device), tensor.detach())
    return target


def migrate_v15_model(parent, original_vocabulary, vocabulary):
    """Build an explicit actor V15→V16 child and optimizer-axis migration map."""
    from dataclasses import replace
    from .model import RecurrentPolicy, actor_schema_for_config
    from .encoding import (entity_categorical_fields, action_categorical_fields,
                           ACTOR_V15 as ENCODING_ACTOR_V15,
                           ACTOR_V16 as ENCODING_ACTOR_V16)
    if (not isinstance(parent, RecurrentPolicy)
            or actor_schema_for_config(parent.config) != ENCODING_ACTOR_V15):
        raise ValueError('V15->V16 migration requires an exact V15 actor model')
    if (type(original_vocabulary) is not ContentVocabulary
            or type(vocabulary) is not ContentVocabulary
            or parent.config.vocabulary_size != original_vocabulary.vocabulary_size
            or not set(original_vocabulary.tokens) <= set(vocabulary.tokens)):
        raise ValueError('V15->V16 migration requires a compatible frozen vocabulary extension')
    old_entities, old_actions = actor_feature_names(ENCODING_ACTOR_V15)
    new_entities, new_actions = actor_feature_names(ENCODING_ACTOR_V16)
    if (len(set(old_entities)) != len(old_entities) or len(set(old_actions)) != len(old_actions)
            or len(set(new_entities)) != len(new_entities) or len(set(new_actions)) != len(new_actions)
            or not set(old_entities) <= set(new_entities) or not set(old_actions) <= set(new_actions)):
        raise ValueError('V15 feature names are not an injective subset of V16')
    from .governors import ACTION_KINDS
    added_kinds = [name[5:] for name in new_actions if name.startswith('kind:')
                   and name not in old_actions]
    if set(added_kinds) != set(ACTION_KINDS) or len(added_kinds) != len(ACTION_KINDS):
        raise ValueError('V16 must add the complete four-kind governor action family')
    if parent.config.candidate_family_prior is not None:
        from .model import FAMILY_PRIOR
        if parent.config.candidate_family_prior != FAMILY_PRIOR:
            raise ValueError('V15 parent uses an unsupported action-family prior')

    config = replace(parent.config, actor_schema=ENCODING_ACTOR_V16,
        entity_features=len(new_entities), action_features=len(new_actions),
        vocabulary_size=vocabulary.vocabulary_size)
    parent_device = next(parent.parameters()).device
    with torch.random.fork_rng(devices=[]):
        child = RecurrentPolicy(config).to(parent_device)
    source, target = parent.state_dict(), child.state_dict()
    if set(source) != set(target):
        raise ValueError('V15->V16 changed actor tensor keys')
    entity_map = [new_entities.index(name) for name in old_entities]
    action_map = [new_actions.index(name) for name in old_actions]
    mappings = {name: None for name in source}
    for name, old_width, new_width, indices in (
            ('entity.0.weight', len(old_entities), len(new_entities), entity_map),
            ('candidate.0.weight', len(old_actions), len(new_actions), action_map)):
        if name not in source or source[name].shape[1] != old_width + parent.config.category_size:
            raise ValueError('V15 actor projection layer differs: ' + name)
        if target[name].shape[1] != new_width + parent.config.category_size:
            raise ValueError('V16 actor projection layer differs: ' + name)
        categorical = [new_width + index for index in range(parent.config.category_size)]
        mappings[name] = {'axis': 1, 'indices': indices + categorical}
    if ('categories.weight' not in source
            or source['categories.weight'].shape != (original_vocabulary.vocabulary_size,
                                                      parent.config.category_size)
            or target['categories.weight'].shape != (vocabulary.vocabulary_size,
                                                       parent.config.category_size)):
        raise ValueError('V15/V16 shared categorical embedding differs from the selected vocabularies')
    vocabulary_indices = [0] + [vocabulary.tokens.index(token) + 1
                                 for token in original_vocabulary.tokens]
    mappings['categories.weight'] = {'axis': 0, 'indices': vocabulary_indices}
    for name in source:
        if mappings[name] is None and source[name].shape != target[name].shape:
            raise ValueError('Unexpected V15->V16 tensor widening: ' + name)
    migrated = {name: migrate_v15_parameter_tensor(source[name], target[name].shape,
                                                   mappings[name]) for name in source}
    child.load_state_dict(migrated, strict=True)
    child.eval()
    from .governors import CONTRACT as GOVERNOR_CONTRACT
    report = dict(schema='civ6rl.actor_feature_migration.v1',
        parent_schema=ENCODING_ACTOR_V15, child_schema=ENCODING_ACTOR_V16,
        parent_vocabulary_sha256=original_vocabulary.fingerprint,
        child_vocabulary_sha256=vocabulary.fingerprint,
        feature_mapping='exact_numeric_feature_name_plus_shared_categorical_block',
        embedding_mapping='exact_token_rows_including_padding_row_zero',
        new_parameters_initialization='zero',
        new_entity_features=[name for name in new_entities if name not in old_entities],
        new_action_features=[name for name in new_actions if name not in old_actions],
        new_tokens=[token for token in vocabulary.tokens if token not in set(original_vocabulary.tokens)],
        governor_contract=GOVERNOR_CONTRACT, parameter_mappings=mappings,
        forward_semantics=('V16 adds public governor entities and four explicit governor action kinds. '
            'New numeric projection columns and vocabulary rows start at zero; legacy shared categorical '
            'projection columns and token rows are copied by their unchanged relative positions/identities. '
            'The admitted input and action support changes, so V15 policy probabilities and PPO equivalence '
            'are not claimed.'),
        retained_rollouts_reusable_for_update=False, optimizer_state_migrated=False,
        learner_checkpoint_migrated=False, critic_migrated=False,
        recurrent_state_migrated=False, fresh_optimizer_required=True,
        training_eligible=False)
    return child, report


def migrate_policy_bytes(data, *, sha256, vocabulary=None, target_schema=ACTOR_V8):
    if target_schema not in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15, ACTOR_V16):
        raise ValueError('Explicit migrations target grouped v8 through governor-observation v16 only')
    with torch.random.fork_rng(devices=[]):
        parent, original = load_policy_bytes(data, sha256=sha256, device='cpu')
        parent_schema = actor_schema_for_config(parent.config)
        allowed_parents = ((ACTOR_V6, ACTOR_V7) if target_schema == ACTOR_V8 else
                           (ACTOR_V8,) if target_schema == ACTOR_V9 else
                           (ACTOR_V9,) if target_schema == ACTOR_V10 else
                           (ACTOR_V10,) if target_schema == ACTOR_V11 else (ACTOR_V11,))
        if target_schema == ACTOR_V13:
            allowed_parents = (ACTOR_V12,)
        elif target_schema == ACTOR_V14:
            allowed_parents = (ACTOR_V13,)
        elif target_schema == ACTOR_V15:
            allowed_parents = (ACTOR_V14,)
        elif target_schema == ACTOR_V16:
            allowed_parents = (ACTOR_V15,)
        if parent_schema not in allowed_parents:
            if target_schema == ACTOR_V13:
                raise ValueError('V12->V13 migration requires an exact v12 parent')
            if target_schema == ACTOR_V14:
                raise ValueError('V13->V14 migration requires an exact v13 parent')
            if target_schema == ACTOR_V15:
                raise ValueError('V14->V15 migration requires an exact v14 parent')
            if target_schema == ACTOR_V16:
                raise ValueError('V15->V16 migration requires an exact v15 parent')
            raise ValueError('Migration requires the exact parent actor schema for the requested target')
        vocabulary = original if vocabulary is None else vocabulary
        if type(vocabulary) is not ContentVocabulary:
            raise ValueError('Migration requires a frozen ContentVocabulary')
        if not set(original.tokens) <= set(vocabulary.tokens):
            raise ValueError('Migration cannot drop parent vocabulary tokens')
        if target_schema == ACTOR_V16:
            child, report = migrate_v15_model(parent, original, vocabulary)
            result = policy_bytes(child, vocabulary)
            report.update(parent_sha256=sha256,
                child_sha256=hashlib.sha256(result).hexdigest(),
                parent_vocabulary_sha256=original.fingerprint,
                child_vocabulary_sha256=vocabulary.fingerprint)
            return result, report
        if target_schema == ACTOR_V13:
            # V13 changes the strict public war-confirmation contract only.
            # Keep the vocabulary and every actor tensor byte-for-byte stable;
            # record the new contract identity explicitly in serialized config.
            if vocabulary.fingerprint != original.fingerprint or vocabulary.tokens != original.tokens:
                raise ValueError('V12->V13 migration cannot change the vocabulary')
            child = RecurrentPolicy(replace(parent.config, actor_schema=ACTOR_V13))
            source, target = parent.state_dict(), child.state_dict()
            if set(source) != set(target) or any(source[key].shape != target[key].shape for key in source):
                raise ValueError('V12->V13 migration requires identical actor architecture')
            for key, tensor in target.items():
                tensor.copy_(source[key])
            child.load_state_dict(target, strict=True); child.eval()
            result = policy_bytes(child, vocabulary)
            report = dict(schema='civ6rl.actor_feature_migration.v1', parent_schema=ACTOR_V12,
                child_schema=ACTOR_V13, parent_sha256=sha256,
                child_sha256=hashlib.sha256(result).hexdigest(),
                parent_vocabulary_sha256=original.fingerprint,
                child_vocabulary_sha256=vocabulary.fingerprint,
                feature_mapping='identical_v12_layout', embedding_mapping='identical_vocabulary',
                new_parameters_initialization='none', new_entity_features=[], new_action_features=[],
                new_tokens=[], contract_only=True,
                forward_semantics=('V13 accepts the pinned official declare-war confirmation display contract; '
                    'architecture, feature widths and tensors are copied unchanged, and behavioral equivalence '
                    'for newly admitted modal observations is not claimed.'),
                retained_rollouts_reusable_for_update=False, optimizer_state_migrated=False,
                learner_checkpoint_migrated=False, critic_migrated=False,
                fresh_optimizer_required=True, training_eligible=False)
            return result, report
        if target_schema == ACTOR_V15:
            if parent_schema != ACTOR_V14:
                raise ValueError('V14->V15 migration requires an exact v14 parent')
            # Purchase observation adds only fourteen named numeric entity
            # columns. Action features and the frozen token vocabulary remain
            # exact, so the V14 policy prior and every shared tensor can carry
            # over without reinitialization.
            if vocabulary.fingerprint != original.fingerprint or vocabulary.tokens != original.tokens:
                raise ValueError('V14->V15 migration cannot change the vocabulary')
            old_entities, old_actions = actor_feature_names(ACTOR_V14)
            new_entities, new_actions = actor_feature_names(ACTOR_V15)
            added = [name for name in new_entities if name not in old_entities]
            if (len(set(old_entities)) != len(old_entities)
                    or len(set(new_entities)) != len(new_entities)
                    or not set(old_entities) <= set(new_entities)
                    or len(added) != 14 or new_actions != old_actions):
                raise ValueError('V14->V15 must add exactly fourteen entity features and preserve action layout')
            child = RecurrentPolicy(replace(parent.config, actor_schema=ACTOR_V15,
                entity_features=len(new_entities), action_features=len(new_actions),
                vocabulary_size=vocabulary.vocabulary_size))
            source, target = parent.state_dict(), child.state_dict()
            if set(source) != set(target):
                raise ValueError('V14->V15 migration requires identical actor tensor keys')
            for key, tensor in target.items():
                if key == 'entity.0.weight':
                    if tensor.shape[0] != source[key].shape[0] or tensor.shape[1] != source[key].shape[1] + 14:
                        raise ValueError('V14->V15 entity projection width is not the expected extension')
                    tensor.zero_()
                    for index, name in enumerate(old_entities):
                        tensor[:, new_entities.index(name)].copy_(source[key][:, index])
                    tensor[:, len(new_entities):].copy_(source[key][:, len(old_entities):])
                elif tensor.shape != source[key].shape:
                    raise ValueError('V14->V15 changed a non-entity actor tensor: ' + key)
                else:
                    tensor.copy_(source[key])
            child.load_state_dict(target, strict=True); child.eval()
            result = policy_bytes(child, vocabulary)
            report = dict(schema='civ6rl.actor_feature_migration.v1', parent_schema=ACTOR_V14,
                child_schema=ACTOR_V15, parent_sha256=sha256,
                child_sha256=hashlib.sha256(result).hexdigest(),
                parent_vocabulary_sha256=original.fingerprint,
                child_vocabulary_sha256=vocabulary.fingerprint,
                feature_mapping='exact_name', embedding_mapping='identical_vocabulary',
                new_parameters_initialization='zero', new_entity_features=added,
                new_action_features=[], new_tokens=[], contract_only=False,
                forward_semantics=('V15 adds fourteen named public purchase quote fields to entities. '
                    'New entity input columns are zero initialized; all V14 feature weights and other actor '
                    'tensors are copied by name/shape. New entity-category slots reuse the existing mean '
                    'embedding. Input semantics changed, so behavioral equivalence is not claimed.'),
                retained_rollouts_reusable_for_update=False, optimizer_state_migrated=False,
                learner_checkpoint_migrated=False, critic_migrated=False,
                fresh_optimizer_required=True, training_eligible=False)
            return result, report
        if target_schema == ACTOR_V14:
            # V14 has the same model tensors and vocabulary as V13. Only its
            # plot-quantity projection contract differs; record that identity
            # explicitly without migrating learner or optimizer state.
            if vocabulary.fingerprint != original.fingerprint or vocabulary.tokens != original.tokens:
                raise ValueError('V13->V14 migration cannot change the vocabulary')
            child = RecurrentPolicy(replace(parent.config, actor_schema=ACTOR_V14))
            source, target = parent.state_dict(), child.state_dict()
            if set(source) != set(target) or any(source[key].shape != target[key].shape for key in source):
                raise ValueError('V13->V14 migration requires identical actor architecture')
            for key, tensor in target.items():
                tensor.copy_(source[key])
            child.load_state_dict(target, strict=True); child.eval()
            result = policy_bytes(child, vocabulary)
            report = dict(schema='civ6rl.actor_feature_migration.v1', parent_schema=ACTOR_V13,
                child_schema=ACTOR_V14, parent_sha256=sha256,
                child_sha256=hashlib.sha256(result).hexdigest(),
                parent_vocabulary_sha256=original.fingerprint,
                child_vocabulary_sha256=vocabulary.fingerprint,
                feature_mapping='identical_v13_layout', embedding_mapping='identical_vocabulary',
                new_parameters_initialization='none', new_entity_features=[], new_action_features=[],
                new_tokens=[], contract_only=True,
                forward_semantics=('V14 reserves per-plot resource quantity channels at zero while preserving '
                    'the V13 architecture, feature layout and tensors. Copied weights do not imply behavioral '
                    'equivalence because the input projection semantics changed.'),
                retained_rollouts_reusable_for_update=False, optimizer_state_migrated=False,
                learner_checkpoint_migrated=False, critic_migrated=False,
                fresh_optimizer_required=True, training_eligible=False)
            return result, report
        old_entities, old_actions = actor_feature_names(parent_schema)
        new_entities, new_actions = actor_feature_names(target_schema)
        child = RecurrentPolicy(replace(parent.config, entity_features=len(new_entities),
            action_features=len(new_actions), vocabulary_size=vocabulary.vocabulary_size))
        source, target = parent.state_dict(), child.state_dict()
        if set(source) != set(target):
            raise ValueError('Actor architecture changed beyond feature columns')
        columns = {'entity.0.weight': (old_entities, new_entities),
                   'candidate.0.weight': (old_actions, new_actions)}
        for key, tensor in target.items():
            if key in columns:
                previous, current = columns[key]
                if (len(set(previous)) != len(previous) or len(set(current)) != len(current)
                        or not set(previous) <= set(current)):
                    raise ValueError('Actor feature names are not an injective extension')
                tensor.zero_()
                for index, name in enumerate(previous):
                    tensor[:, current.index(name)].copy_(source[key][:, index])
                tensor[:, len(current):].copy_(source[key][:, len(previous):])
            elif key == 'categories.weight':
                tensor.zero_(); tensor[0].copy_(source[key][0])
                indices = {token: index+1 for index, token in enumerate(vocabulary.tokens)}
                for index, token in enumerate(original.tokens):
                    tensor[indices[token]].copy_(source[key][index+1])
            else:
                if tensor.shape != source[key].shape:
                    raise ValueError('Unexpected actor tensor shape change: ' + key)
                tensor.copy_(source[key])
        child.load_state_dict(target, strict=True); child.eval()
        result = policy_bytes(child, vocabulary)
    report = dict(schema='civ6rl.actor_feature_migration.v1', parent_schema=parent_schema,
        child_schema=target_schema, parent_sha256=sha256, child_sha256=hashlib.sha256(result).hexdigest(),
        parent_vocabulary_sha256=original.fingerprint, child_vocabulary_sha256=vocabulary.fingerprint,
        feature_mapping='exact_name', embedding_mapping='exact_token', new_parameters_initialization='zero',
        new_entity_features=[name for name in new_entities if name not in old_entities],
        new_action_features=[name for name in new_actions if name not in old_actions],
        new_tokens=[token for token in vocabulary.tokens if token not in set(original.tokens)],
        action_content_catalog_retained=((vocabulary.manifest or {}).get('action_content') if target_schema in (ACTOR_V10,ACTOR_V11,ACTOR_V12) else None),
        optimizer_state_migrated=False, learner_checkpoint_migrated=False, critic_migrated=False,
        fresh_optimizer_required=True, training_eligible=False)
    if target_schema == ACTOR_V10:
        report['forward_semantics'] = 'Public source token now conditions each candidate; behavioral equivalence is not claimed.'
        report['retained_rollouts_reusable_for_update'] = False
    if target_schema == ACTOR_V11:
        report['forward_semantics'] = ('Own queued goals and complete candidate-bound movement paths now enter the actor. '
            'Copied parameters do not imply behavioral equivalence; optimizer, critic, recurrent state and rollout state were not migrated.')
        report['retained_rollouts_reusable_for_update'] = False
    if target_schema == ACTOR_V12:
        report['forward_semantics'] = ('Public WorldInput selection, all-coordinate map clicks and modal war responses '
            'now enter the same candidate/entity encoders; game-turn advancement is represented by the outer native receipt. '
            'Parameters from the v11 architecture are copied by exact feature name, but behavioral equivalence is not claimed.')
        report['retained_rollouts_reusable_for_update'] = False
    return result, report
