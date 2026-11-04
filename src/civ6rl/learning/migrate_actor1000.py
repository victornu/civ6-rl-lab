"""Explicit actor-only v6 -> v7 transfer. Never migrate learner/optimizer state.

New feature columns, the city_focus kind column and new token embeddings start
at zero. Old columns are joined by feature NAME, embeddings by TOKEN, and all
other tensors are copied exactly. A new actor digest and external provenance
report are returned; callers must initialize a fresh optimizer separately.
"""
from __future__ import annotations
from dataclasses import replace
import hashlib
import torch
from .artifacts import load_policy_bytes, policy_bytes
from .model import RecurrentPolicy, actor_schema_for_config
from .encoding import ContentVocabulary, ACTOR_V6, ACTOR_V7, actor_feature_names


def migrate_policy_bytes(data, *, sha256, vocabulary=None):
    # Loading a model also initializes tensors before replacing their values;
    # preserve RNG across both the strict parent load and child construction.
    with torch.random.fork_rng(devices=[]):
        return _migrate(data, sha256=sha256, vocabulary=vocabulary)


def _migrate(data, *, sha256, vocabulary):
    parent, old_vocabulary = load_policy_bytes(data, sha256=sha256, device="cpu")
    if actor_schema_for_config(parent.config) != ACTOR_V6:
        raise ValueError("migration requires an exact v6 parent actor")
    target_vocabulary = old_vocabulary if vocabulary is None else vocabulary
    if type(target_vocabulary) is not ContentVocabulary:
        raise ValueError("migration requires a frozen ContentVocabulary")
    old_tokens = set(old_vocabulary.tokens)
    if not old_tokens <= set(target_vocabulary.tokens):
        raise ValueError("migration cannot drop parent vocabulary tokens")
    old_entities, old_actions = actor_feature_names(ACTOR_V6)
    new_entities, new_actions = actor_feature_names(ACTOR_V7)
    config = replace(parent.config, entity_features=len(new_entities),
                     action_features=len(new_actions), vocabulary_size=target_vocabulary.vocabulary_size)
    # Do not alter the caller's RNG stream as a side effect of actor migration.
    with torch.random.fork_rng(devices=[]):
        child = RecurrentPolicy(config)
    source = parent.state_dict(); target = child.state_dict()
    if set(source) != set(target): raise ValueError("actor tensor names differ")
    remapped = {"entity.0.weight": (old_entities, new_entities),
                "candidate.0.weight": (old_actions, new_actions)}
    for key, tensor in target.items():
        if key in remapped:
            old_names, new_names = remapped[key]
            if (len(set(old_names)) != len(old_names) or len(set(new_names)) != len(new_names)
                    or not set(old_names) <= set(new_names)):
                raise ValueError("actor feature names are not an injective extension")
            tensor.zero_()
            for old_index, name in enumerate(old_names):
                tensor[:, new_names.index(name)].copy_(source[key][:, old_index])
            # Shared categorical vector follows the numeric feature block.
            tensor[:, len(new_names):].copy_(source[key][:, len(old_names):])
        elif key == "categories.weight":
            tensor.zero_(); tensor[0].copy_(source[key][0])
            target_ids = {token: index + 1 for index, token in enumerate(target_vocabulary.tokens)}
            for index, token in enumerate(old_vocabulary.tokens):
                tensor[target_ids[token]].copy_(source[key][index + 1])
        else:
            if tensor.shape != source[key].shape: raise ValueError("unexpected actor tensor shape change: " + key)
            tensor.copy_(source[key])
    child.load_state_dict(target, strict=True); child.eval()
    result = policy_bytes(child, target_vocabulary)
    report = dict(schema="civ6rl.actor_migration1000.v1", parent_schema=ACTOR_V6, child_schema=ACTOR_V7,
        parent_sha256=sha256, child_sha256=hashlib.sha256(result).hexdigest(),
        parent_vocabulary_sha256=old_vocabulary.fingerprint,
        child_vocabulary_sha256=target_vocabulary.fingerprint,
        feature_mapping="exact_name", embedding_mapping="exact_token", new_parameters_initialization="zero",
        new_entity_features=[name for name in new_entities if name not in old_entities],
        new_action_features=[name for name in new_actions if name not in old_actions],
        new_tokens=[token for token in target_vocabulary.tokens if token not in old_tokens],
        optimizer_state_migrated=False, learner_checkpoint_migrated=False,
        fresh_optimizer_required=True, training_eligible=False)
    return result, report
