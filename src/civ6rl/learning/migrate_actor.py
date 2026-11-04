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
from .encoding import ContentVocabulary, ACTOR_V6, ACTOR_V7, ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, actor_feature_names

ACTOR_V15 = 'civ6rl.recurrent_policy.v15'


def migrate_policy_bytes(data, *, sha256, vocabulary=None, target_schema=ACTOR_V8):
    if target_schema not in (ACTOR_V8, ACTOR_V9, ACTOR_V10, ACTOR_V11, ACTOR_V12, ACTOR_V13, ACTOR_V14, ACTOR_V15):
        raise ValueError('Explicit migrations target grouped v8 through purchase-observation v15 only')
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
        if parent_schema not in allowed_parents:
            if target_schema == ACTOR_V13:
                raise ValueError('V12->V13 migration requires an exact v12 parent')
            if target_schema == ACTOR_V14:
                raise ValueError('V13->V14 migration requires an exact v13 parent')
            if target_schema == ACTOR_V15:
                raise ValueError('V14->V15 migration requires an exact v14 parent')
            raise ValueError('Migration requires the exact parent actor schema for the requested target')
        vocabulary = original if vocabulary is None else vocabulary
        if type(vocabulary) is not ContentVocabulary:
            raise ValueError('Migration requires a frozen ContentVocabulary')
        if not set(original.tokens) <= set(vocabulary.tokens):
            raise ValueError('Migration cannot drop parent vocabulary tokens')
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
