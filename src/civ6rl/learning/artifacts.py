"""Portable diagnostic policy snapshots with weights-only loading and identity."""
from __future__ import annotations
import hashlib,io
import torch
from .model import checkpoint_payload,RecurrentPolicy,PolicyConfig
from .encoding import ContentVocabulary


def policy_bytes(model, vocabulary):
    payload=checkpoint_payload(model,vocabulary=vocabulary)
    payload['state_dict']={key:value.detach().cpu().contiguous() for key,value in model.state_dict().items()}
    stream=io.BytesIO();torch.save(payload,stream)
    return stream.getvalue()


def policy_version(model,vocabulary):
    return hashlib.sha256(policy_bytes(model,vocabulary)).hexdigest()


def load_policy_bytes(data,*,sha256,device='cpu'):
    if type(data) is not bytes or not 0<len(data)<=64*1024*1024:
        raise ValueError('policy artifact exceeds bound')
    if hashlib.sha256(data).hexdigest()!=sha256:raise ValueError('policy digest differs')
    payload=torch.load(io.BytesIO(data),weights_only=True,map_location='cpu')
    if payload.get('schema') not in ('civ6rl.recurrent_policy.v6','civ6rl.recurrent_policy.v7','civ6rl.recurrent_policy.v8','civ6rl.recurrent_policy.v9','civ6rl.recurrent_policy.v10','civ6rl.recurrent_policy.v11','civ6rl.recurrent_policy.v12','civ6rl.recurrent_policy.v13','civ6rl.recurrent_policy.v14','civ6rl.recurrent_policy.v15') or payload.get('training_eligible') is not False:
        raise ValueError('unsupported diagnostic checkpoint; actor requires explicit v6-v14 without implicit migration')
    vocabulary=ContentVocabulary.from_serialized(payload['vocabulary'])
    if vocabulary.fingerprint!=payload['vocabulary_sha256']:raise ValueError('vocabulary digest differs')
    config=PolicyConfig(**payload['config'])
    from .model import actor_schema_for_config
    if actor_schema_for_config(config) != payload['schema']:
        raise ValueError('checkpoint actor schema identity differs from its config')
    if (not 4<=config.hidden_size<=1024 or config.hidden_size%4 or not 1<=config.spatial_size<=256
            or not 1<=config.category_size<=256 or config.vocabulary_size!=vocabulary.vocabulary_size
            or not 1<=config.text_embedding_size<=128 or not 1<=config.text_channels<=128):
        raise ValueError('unsupported diagnostic model dimensions')
    from .encoding import MAP_CHANNELS,GLOBAL_FEATURES,ENTITY_FEATURES,ACTION_FEATURES,GLOBAL_CATEGORICAL_FIELDS,MAP_CATEGORICAL_FIELDS
    from .encoding import actor_feature_names
    entity_names, action_names = actor_feature_names(payload['schema'])
    from .task_context import TASK_FEATURES
    if config.task_features != len(TASK_FEATURES):raise ValueError('checkpoint task dimensions differ')
    if config.map_categories != len(MAP_CATEGORICAL_FIELDS):raise ValueError('checkpoint map categorical dimensions differ')
    if (config.map_channels,config.global_features,config.entity_features,config.action_features,config.global_categories)!=(len(MAP_CHANNELS),len(GLOBAL_FEATURES),len(entity_names),len(action_names),len(GLOBAL_CATEGORICAL_FIELDS)):
        raise ValueError('checkpoint feature dimensions differ; implicit feature-layout migration is unsupported')
    model=RecurrentPolicy(config)
    # Ordered feature names are part of the checkpoint contract. Extending the
    # public global vector (for example with schema-980 treasury inputs) makes
    # older artifacts incompatible; weights are never padded or migrated here.
    if checkpoint_payload(model,vocabulary=vocabulary)['feature_layout']!=payload['feature_layout']:
        raise ValueError('checkpoint feature layout differs; implicit feature-layout migration is unsupported')
    model.load_state_dict(payload['state_dict'],strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite policy weights')
    return model.to(device).eval(),vocabulary
