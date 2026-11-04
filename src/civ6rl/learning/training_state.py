"""Versioned learner resume at a completed PPO/task boundary.

There must be no pending actor samples or player memory: native games restart
from explicitly chosen scenarios after this boundary. Mid-rollout snapshots,
curriculum scheduling, backend migration and strategic qualification are not
silently approximated. Loading restores RNG only after all validation passes.
"""
from __future__ import annotations
import hashlib, io, json, math, platform, random, zipfile
from dataclasses import replace
from pathlib import Path
import torch
from .artifacts import policy_bytes, load_policy_bytes, policy_version
from .model import PlayerMemory

SCHEMA = 'civ6rl.learner_boundary.v3'
CAMPAIGN_SCHEMA = 'civ6rl.learner_boundary.v4'
MAX_BYTES = 128 * 1024 * 1024
# This audited source-only transition enables a fresh bounded collection after
# a completed learner boundary. It does not migrate tensors or model layouts.
BOUNDED_COLLECTION_RESUME_V1 = {
    'worker.py': '94f886f6eda1cb92505285e3b40e336c54346ef6a79955baf3f0ab68c5642571',
    'training_state.py': '068830b3772a8c9b79a833d81f6730c63d88f15a9d08f6cfc3394a15dcd28500',
    'campaign.py': None,
}
BOUNDED_EPISODE_BUDGET_V1 = {
    'worker.py': '0bda2458d50864c78c4fbc7db51284d8e4a45adc79ff98656ea359223b6851a7',
    'diagnostic_budget998.py': '3eef4fb8baccd087aa2bf351c00f5fc40e3f9e2892ce3e6570cd5d106e2b7307',
    'learner.py': '61a96e44beff88d35ee111ab58af280f6d54aa8d27acbe3dd087949895f30db6',
    'ppo.py': '0a950d056d2eee289701807d830a38ab681cb5b72f7e3aba900dbd3fd7132c29',
    'training_state.py': '379f1f3eeb3442d40f18e973f5246a81add7c86a3d3c808637aef2cb62206bcc',
    'campaign.py': None,
}
BOUNDED_CAMPAIGN_V1 = {
    'worker.py': '12c3da973c7aff41624cfbe5c4377d9a8d50e067017a91d7d408938e59d5caa7',
    'training_state.py': 'd8295932176898d9ce964f573016a96f53fcf588be4f94cd13f6256f36a59234',
    'campaign.py': None,
}
FAMILY_PRIOR_RECONFIGURATION_V1 = {
    'model.py': '6967374b9501c89ee4605b227f4b386846219742926151f8b09d18e6e686d168',
    'training_state.py': '2499e02b2345044ea8e75169cabf807426b86157e22b6c569136c7fa6c699e17',
}
BOUNDED_UNRESOLVED_TAIL_V1 = {
    'learner.py': 'c0646735c854a7d04d897d5ae89235c4b823a348cb0b644467e10ebab984cdf3',
    'training_state.py': '3a74a23d03b983ab0351379bc59f4935ff1c34e8afd35adcef70b95fbdcca152',
}
# Exact native comparison: all 61 transition records retain digest
# 76cbb5de0ea43f9a9978bdf8fc2d1b3e8dbff8d7d875966be91efc17a3e9f008.
# Only internal ownership/caching changes; no actor, reward, optimizer or RNG
# change is admitted by this source transition.
DETACHED_TRANSITION_REUSE_V1 = {
    'multiplayer_trajectory.py': '44a1a2f562b2dcb3f426a40dadd30d5b5db32fb3dc06ca473a61bb50de6e10d8',
    'training_state.py': 'f251f4b0dc5c2b1d6bb936189a65360ed4ac845400a13f81d6284309b624d6d7',
}
CAMPAIGN_ORIGIN_REATTACH_V1 = {
    'training_state.py': '6c2170170d70bdf77a0efff5a1c7e5c5a159576d83303a26f81880954268d46b',
}
CAMPAIGN_ORIGIN_REATTACH_V1_ID = 'campaign_origin_reattach_v1'
COMPLETED_CAMPAIGN_SUCCESSOR_V1 = {
    'worker.py': '523ab0927cfd6361f6123b090f726698cad79789a30b73b1fa0e6a1155ecf703',
    'campaign.py': 'd0577a7b930c1ae36b1d33fafc7d43ae51871e4587dcbd38e139c1e7117efa1f',
    'training_state.py': '63ca415142a67ed873d13c2ceefb8c374595aaf763472006899c8d990d093254',
    'diagnostic_budget998.py': 'ab4f0405fe427ed92adbf145ea4311e821dc053136bbac66e0c30fb0fca25a79',
    'ppo.py': '6e86d6d580f6b2ce0e97cc1ac79e0357dce4c964a8fa167edfef54115829b5c9',
}
COMPLETED_CAMPAIGN_SUCCESSOR_V1_ID = 'completed_campaign_successor_v1'
COMPLETED_CAMPAIGN_SUPPORTING_SOURCES_V1 = {
    'civ6rl/learning/transport.py': '3b7191a12a16d9d28d9df999b2c8c1c90972106be69dabc012cc51ea093a39e0',
    'civ6rl/learning/wire.py': 'cd0bb0626a3b538f20d31d694f4c5c77caad59af25069648ea3a9c22088e83c0',
}
V14_PARENT_SOURCE_RESTORE = 'v14_parent_source_only_v1'
PUBLIC_PURCHASE_OBSERVATION_V15 = 'public_purchase_observation_v15'
V14_PARENT_JSON_SHA256 = '5cdc2efbecd2afdf2e1b997f9d47c56177768d96bb69a6caff2f2fa398e2b8d8'
V14_PARENT_ARCHIVE_SHA256 = 'dba5477f2df79577adff755073530fac09042c6bf66fcc19a2c56ebfbb3a1a1e'
V14_PARENT_LEARNER_SHA256 = '1227a0cea953c0a5c8052c19eb19f0ab47a2a7f5d0c12cc87a390d7066b608fb'
V14_PARENT_POLICY_SHA256 = '0711124d7a0c302704ce8bd5800e6f77665d04ea10a1dac6ba4c2a3974d79e1d'
SOURCE_TRANSITIONS = {'bounded_collection_resume_v1': BOUNDED_COLLECTION_RESUME_V1,
                      'bounded_episode_budget_v1': BOUNDED_EPISODE_BUDGET_V1,
                      'bounded_campaign_v1': BOUNDED_CAMPAIGN_V1,
                      'candidate_family_prior_reconfiguration_v1': FAMILY_PRIOR_RECONFIGURATION_V1,
                      'bounded_unresolved_tail_v1': BOUNDED_UNRESOLVED_TAIL_V1,
                      'detached_transition_reuse_v1': DETACHED_TRANSITION_REUSE_V1,
                      CAMPAIGN_ORIGIN_REATTACH_V1_ID: CAMPAIGN_ORIGIN_REATTACH_V1,
                      COMPLETED_CAMPAIGN_SUCCESSOR_V1_ID: COMPLETED_CAMPAIGN_SUCCESSOR_V1}
IMPLEMENTATION = ('artifacts.py','encoding.py','model.py','policy_assignments.py',
                  'action_receipt.py','scenario.py','task_context.py','ppo.py','learner.py','worker.py',
                  'privileged_critic.py','privileged_encoding.py','critic_modes.py','training_state.py',
                  'diplomacy987.py','public_text987.py','yield_display982.py','research_info983.py',
                  'trade_choices984.py','unit_work985.py','production_info988.py','plot_display990.py',
                  'skill_task991.py','skill_context993.py','multiplayer_trajectory.py','diagnostic_budget998.py',
                  'city_focus1000.py','city_management.py','production_queue.py',
                  'action_bindings.py','action_content.py','campaign.py')
IMPLEMENTATION = IMPLEMENTATION + ('movement_information.py','civ6rl/native_movement_preview.py','world_input.py',
                                  'war_confirmation13.py','resource_quantity14.py')
IMPLEMENTATION = IMPLEMENTATION + ('public_purchases.py',)


def _pinned_v14_parent_sources():
    """Load the exact historical V14 source map and verify its retained archive."""
    root = Path(__file__).resolve().parents[2]
    parent_path = root / 'work/purchase-observation/parent.json'
    archive_path = root / 'work/purchase-observation/v14-parent-sources.zip'
    parent_bytes = parent_path.read_bytes()
    archive_bytes = archive_path.read_bytes()
    if (hashlib.sha256(parent_bytes).hexdigest() != V14_PARENT_JSON_SHA256
            or hashlib.sha256(archive_bytes).hexdigest() != V14_PARENT_ARCHIVE_SHA256):
        raise ValueError('retained V14 parent source evidence differs')
    parent = json.loads(parent_bytes)
    source_map = parent.get('implementation_sha256')
    source_rows = parent.get('sources')
    if (parent.get('schema') != 'civ6rl.purchase_observation_parent.v1'
            or type(source_map) is not dict or type(source_rows) is not list):
        raise ValueError('retained V14 parent source manifest is malformed')
    with zipfile.ZipFile(io.BytesIO(archive_bytes), 'r') as archive:
        expected_names = {row.get('path') for row in source_rows if type(row) is dict}
        if set(archive.namelist()) != expected_names:
            raise ValueError('retained V14 source archive membership differs')
        by_path = {}
        for row in source_rows:
            path = row['path']
            raw = archive.read(path)
            if (type(row.get('bytes')) is not int or len(raw) != row['bytes']
                    or hashlib.sha256(raw).hexdigest() != row.get('sha256')):
                raise ValueError('retained V14 source archive file digest differs')
            by_path[path] = row['sha256']
    for name, digest in source_map.items():
        path = name if name.startswith('civ6rl/') else 'civ6rl/learning/' + name
        if name == 'civ6rl/native_movement_preview.py' and path not in by_path:
            current_path = root / path
            if hashlib.sha256(current_path.read_bytes()).hexdigest() != digest:
                raise ValueError('V14 movement preview source differs from its parent pin')
            continue
        if by_path.get(path) != digest:
            raise ValueError('V14 implementation source map differs from its archive')
    return source_map


def _migrate_v14_actor_and_optimizer(model, vocabulary, optimizer_state, *, source_learner_sha256):
    """Pad the V14 entity input with V15 purchase channels and map Adam by name."""
    from .encoding import ACTOR_V14, ACTOR_V15, actor_feature_names
    from .model import RecurrentPolicy, actor_schema_for_config
    if actor_schema_for_config(model.config) != ACTOR_V14:
        raise ValueError('public purchase actor migration requires a V14 source actor')
    old_entities = len(actor_feature_names(ACTOR_V14)[0])
    new_entities = len(actor_feature_names(ACTOR_V15)[0])
    added = new_entities - old_entities
    if added != 14 or model.config.entity_features != old_entities:
        raise ValueError('V14/V15 public entity feature transition differs')
    if type(optimizer_state) is not dict:
        raise ValueError('source actor optimizer state is malformed')
    _validate_optimizer(model, optimizer_state)
    target_config = replace(model.config, entity_features=new_entities, actor_schema=ACTOR_V15)
    target = RecurrentPolicy(target_config)
    old_state, target_state = model.state_dict(), target.state_dict()
    if set(old_state) != set(target_state):
        raise ValueError('V14/V15 actor parameter names differ')
    weight_name = 'entity.0.weight'
    old_weight, new_weight = old_state.get(weight_name), target_state.get(weight_name)
    if (not isinstance(old_weight, torch.Tensor) or not isinstance(new_weight, torch.Tensor)
            or old_weight.ndim != 2 or old_weight.shape[0] != new_weight.shape[0]
            or old_weight.shape[1] != old_entities + model.config.category_size
            or new_weight.shape[1] != new_entities + model.config.category_size):
        raise ValueError('V14/V15 entity encoder weight layout differs')
    mapped = {}
    for name, old_tensor in old_state.items():
        fresh = target_state[name]
        if name == weight_name:
            value = torch.zeros_like(fresh)
            value[:, :old_entities] = old_tensor[:, :old_entities]
            value[:, old_entities + added:] = old_tensor[:, old_entities:]
            mapped[name] = value
        else:
            if old_tensor.shape != fresh.shape or old_tensor.dtype != fresh.dtype:
                raise ValueError('V14/V15 changed a non-entity actor parameter')
            mapped[name] = old_tensor.detach().clone()
    target.load_state_dict(mapped, strict=True)
    target.eval()

    old_parameter_names = [name for name, _ in model.named_parameters()]
    new_parameter_names = [name for name, _ in target.named_parameters()]
    if old_parameter_names != new_parameter_names or weight_name not in old_parameter_names:
        raise ValueError('V14/V15 optimizer parameter names differ')
    weight_index = old_parameter_names.index(weight_name)
    mapped_optimizer = _cpu_copy(optimizer_state)
    state_row = mapped_optimizer['state'].get(weight_index)
    if state_row is not None:
        for moment_name in ('exp_avg', 'exp_avg_sq'):
            old_moment = state_row[moment_name]
            if old_moment.shape != old_weight.shape:
                raise ValueError('V14 entity Adam moment shape differs')
            fresh = torch.zeros_like(new_weight, device='cpu')
            fresh[:, :old_entities] = old_moment[:, :old_entities]
            fresh[:, old_entities + added:] = old_moment[:, old_entities:]
            state_row[moment_name] = fresh
    _validate_optimizer(target, mapped_optimizer)
    from .artifacts import policy_version
    target_version = policy_version(target, vocabulary)
    receipt = dict(
        schema='civ6rl.actor_schema_migration.v1',
        id=PUBLIC_PURCHASE_OBSERVATION_V15,
        source_learner_sha256=source_learner_sha256,
        source_evidence=dict(parent_json_sha256=V14_PARENT_JSON_SHA256,
                             source_archive_sha256=V14_PARENT_ARCHIVE_SHA256),
        source_policy_sha256=policy_version(model, vocabulary),
        target_policy_sha256=target_version,
        source_actor_schema=ACTOR_V14,
        target_actor_schema=ACTOR_V15,
        entity_weight=dict(parameter=weight_name, old_entity_columns=old_entities,
                           added_zero_entity_columns=added, category_embedding_columns_shifted=added),
        preserved=dict(non_entity_actor_tensors=True, actor_optimizer_by_parameter_name=True,
                       critic=True, critic_optimizer=True, rng=True, update_count=True,
                       closed_episode_ledger=True, last_update_context=True, collector_empty=True),
        retained_trajectories_reusable=False,
        campaign_receipts_migrated=False,
    )
    return target, mapped_optimizer, receipt


def implementation_hashes():
    folder = Path(__file__).resolve().parent
    hashes = {}
    for name in IMPLEMENTATION:
        # This sole root-level dependency is a fixed public validator, not a
        # caller-selected path. Keep its namespaced pin distinct from learner
        # modules and reject any future path that escapes these two locations.
        if name == 'civ6rl/native_movement_preview.py':
            path = folder.parent / 'native_movement_preview.py'
            if path.resolve().parent != folder.parent.resolve():
                raise ValueError('movement preview implementation path escaped civ6rl')
        else:
            path = folder / name
            if path.resolve().parent != folder.resolve():
                raise ValueError('learner implementation path escaped its package')
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def exact_source_transition(previous):
    """Select only a reviewed exact source transition; never migrate tensors."""
    current = implementation_hashes()
    if previous == current:
        return None
    for profile, changes in SOURCE_TRANSITIONS.items():
        expected={name:digest for name,digest in current.items()
                  if name not in changes or changes[name] is not None}
        expected.update({name:digest for name,digest in changes.items() if digest is not None})
        if previous == expected:
            return profile
    raise ValueError('learner implementation has no reviewed exact source transition')


def _campaign_successor_supporting_sources():
    """Pin only the reviewed transport dependencies outside IMPLEMENTATION."""
    root=Path(__file__).resolve().parents[2]
    return {name:digest for name,digest in COMPLETED_CAMPAIGN_SUPPORTING_SOURCES_V1.items()}


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone().contiguous()
    if isinstance(value, dict):
        return {k:_cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(v) for v in value)
    if value is None or type(value) in (str,bool,int,float):
        return value
    raise ValueError('unsupported learner state value')


def _rng_state(device):
    state = {'python':random.getstate(),'torch_cpu':torch.get_rng_state().clone(),'torch_mps':None}
    if device == 'mps':
        torch.mps.synchronize()
        state['torch_mps'] = torch.mps.get_rng_state().clone()
    return state


def _restore_rng(state, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch_cpu'])
    if device == 'mps':
        torch.mps.set_rng_state(state['torch_mps'])


def _validate_rng(state, device):
    if type(state) is not dict or set(state) != {'python','torch_cpu','torch_mps'}:
        raise ValueError('invalid RNG fields')
    probe = random.Random()
    probe.setstate(state['python'])
    cpu = state['torch_cpu']
    if not isinstance(cpu,torch.Tensor) or cpu.dtype != torch.uint8 or cpu.ndim != 1:
        raise ValueError('invalid CPU RNG state')
    torch.Generator(device='cpu').set_state(cpu)
    mps = state['torch_mps']
    if device == 'mps':
        if (not isinstance(mps,torch.Tensor) or mps.dtype != torch.uint8
                or mps.shape != torch.mps.get_rng_state().shape):
            raise ValueError('invalid MPS RNG state')
    elif mps is not None:
        raise ValueError('unexpected MPS state for CPU learner')


def _optimizer_state(model, optimizer, *, max_step=64):
    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ValueError('only the current single-group Adam learner can resume')
    params = list(model.parameters())
    if any(x is not y for x,y in zip(params, optimizer.param_groups[0]['params'])) or len(params) != len(optimizer.param_groups[0]['params']):
        raise ValueError('optimizer parameter order differs')
    state = _cpu_copy(optimizer.state_dict())
    _validate_optimizer(model,state,max_step=max_step)
    return state


def _validate_optimizer(model,state,*,max_step=64):
    if type(state) is not dict or set(state) != {'state','param_groups'} or len(state['param_groups']) != 1:
        raise ValueError('invalid optimizer state structure')
    group = state['param_groups'][0]
    params = list(model.parameters())
    # Preserve every backend option, while rejecting option/key drift.
    template = torch.optim.Adam(params,lr=3e-4).state_dict()['param_groups'][0]
    if set(group) != set(template) or group['params'] != list(range(len(params))):
        raise ValueError('optimizer parameter IDs or options differ')
    for key in set(template)-{'params'}:
        if group[key] != template[key]:
            raise ValueError('unsupported optimizer configuration: '+key)
    if type(state['state']) is not dict or not state['state']:
        raise ValueError('missing trained Adam moments')
    for index,row in state['state'].items():
        if type(index) is not int or not 0 <= index < len(params) or set(row) != {'step','exp_avg','exp_avg_sq'}:
            raise ValueError('invalid Adam parameter state')
        step = row['step']
        value=float(step) if isinstance(step,torch.Tensor) and step.numel()==1 else math.nan
        if not math.isfinite(value) or value != int(value) or not 0 < value <= max_step:
            raise ValueError('invalid Adam step counter')
        for name in ('exp_avg','exp_avg_sq'):
            tensor = row[name]
            if (not isinstance(tensor,torch.Tensor) or tensor.shape != params[index].shape
                    or tensor.dtype != params[index].dtype or not torch.isfinite(tensor).all()):
                raise ValueError('invalid Adam moment')
        if (row['exp_avg_sq'] < 0).any():
            raise ValueError('negative Adam second moment')


def _parameter_ids(module):
    return [id(parameter) for parameter in module.parameters()]


def _optimizer_parameter_ids(optimizer):
    return [id(parameter) for group in optimizer.param_groups for parameter in group['params']]


def _valid_sha(value):
    return (type(value) is str and len(value)==64
            and all(character in '0123456789abcdef' for character in value))


def _privileged_payload(model, actor_optimizer, *, max_step=64):
    state = getattr(model, '_privileged_state', None)
    if state is None:
        return None
    if type(state) is not dict or set(state) != {'critic','optimizer','version'}:
        raise ValueError('invalid privileged critic runtime state')
    from .privileged_critic import PrivilegedCritic, critic_bytes, critic_version
    critic, optimizer, version = state['critic'], state['optimizer'], state['version']
    if not isinstance(critic, PrivilegedCritic):
        raise ValueError('invalid privileged critic module')
    critic_ids = _parameter_ids(critic)
    actor_ids = _parameter_ids(model)
    if not critic_ids or len(set(critic_ids)) != len(critic_ids) or set(critic_ids) & set(actor_ids):
        raise ValueError('critic parameters are shared with the actor')
    optimizer_ids = _optimizer_parameter_ids(optimizer) if isinstance(optimizer, torch.optim.Optimizer) else []
    actor_optimizer_ids = _optimizer_parameter_ids(actor_optimizer)
    if set(optimizer_ids) & set(actor_optimizer_ids):
        raise ValueError('actor and critic optimizers share parameters')
    if set(optimizer_ids) & set(actor_ids):
        raise ValueError('critic optimizer owns actor parameters')
    blob = critic_bytes(critic)
    if type(blob) is not bytes or not 0 < len(blob) <= MAX_BYTES:
        raise ValueError('invalid privileged critic artifact')
    digest = hashlib.sha256(blob).hexdigest()
    calculated_version = critic_version(critic)
    if not _valid_sha(version) or version != digest or calculated_version != digest:
        raise ValueError('privileged critic version is stale')
    optimizer_state = _optimizer_state(critic, optimizer,max_step=max_step)
    return {
        'bytes': blob,
        'sha256': digest,
        'version': version,
        'optimizer': optimizer_state,
        'optimizer_parameter_names': [name for name, _ in critic.named_parameters()],
    }


def _critic_mode(model):
    from .critic_modes import mode_of
    mode = mode_of(model)
    if mode not in ('player_only', 'privileged_diagnostic', 'history_only_diagnostic'):
        raise ValueError('invalid critic mode')
    return mode


def _validate_update_context(context, critic_mode='player_only'):
    base = {'scenario_sha256','records_sha256','parent_policy_version','optimized_decisions','epochs'}
    if critic_mode == 'privileged_diagnostic':
        extra = {'privileged_sha256','parent_critic_version'}
    elif critic_mode == 'history_only_diagnostic':
        extra = {'parent_critic_version'}
    elif critic_mode == 'player_only':
        extra = set()
    else:
        raise ValueError('invalid critic mode in learner provenance')
    if type(context) is not dict or set(context) not in (base | extra,base | extra | {'multiplayer'}):
        raise ValueError('invalid learner update provenance')
    decision_limit=12
    if 'multiplayer' in context:
        multiplayer=context['multiplayer']
        multiplayer_fields={'schema','global_decisions','player_ids','diagnostic_collection'}
        if (type(multiplayer) is not dict
                or set(multiplayer) not in (multiplayer_fields,multiplayer_fields|{'public_skill_scope'})
                or multiplayer['schema']!='civ6rl.multiplayer_update_provenance998.v1'):
            raise ValueError('invalid multiplayer update provenance')
        budget=multiplayer['diagnostic_collection']
        maximum=32
        if budget is not None:
            from .diagnostic_budget998 import validate_collection
            maximum=validate_collection(budget)['max_decisions']
        minimum_players=2
        if 'public_skill_scope' in multiplayer:
            from .skill_task991 import validate_public_sources
            from .skill_context993 import SCHEMA as SKILL_SCHEMA
            scope=multiplayer['public_skill_scope']
            if (type(scope) is not dict or set(scope)!={'allow_single_public_skill_group','scenario_schema',
                                                       'public_continuity_sources'}
                    or scope['allow_single_public_skill_group'] is not True
                    or scope['scenario_schema']!=SKILL_SCHEMA or budget is None):
                raise ValueError('invalid single public skill group provenance')
            validate_public_sources(scope['public_continuity_sources'])
            minimum_players=1
        players=multiplayer['player_ids']
        if (type(players) is not list or not minimum_players<=len(players)<=4
                or any(type(pid) is not int or not 0<=pid<=63 for pid in players)
                or players!=sorted(set(players))
                or type(multiplayer['global_decisions']) is not int
                or not 2<=multiplayer['global_decisions']<=maximum):
            raise ValueError('invalid multiplayer update provenance counts')
        decision_limit=multiplayer['global_decisions']
    hashes = ['scenario_sha256','records_sha256','parent_policy_version']
    hashes.extend(sorted(extra))
    if any(not _valid_sha(context[key]) for key in hashes):
        raise ValueError('invalid learner update provenance hash')
    if (type(context['optimized_decisions']) is not int or not 1<=context['optimized_decisions']<=decision_limit
            or type(context['epochs']) is not int or not 1<=context['epochs']<=4):
        raise ValueError('invalid learner update provenance counts')


def _environment(device):
    return dict(torch_version=str(torch.__version__),system=platform.system(),machine=platform.machine(),
                device=device,threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled())


def _validate_campaign_receipt(state,context,policy_digest,critic_version,parent_learner_sha256=None):
    from .campaign import validate_state
    state=validate_state(state)
    if not state['receipts']:
        raise ValueError('campaign checkpoint requires a completed receipt')
    receipt=state['receipts'][-1]
    if (receipt['policy_version']!=policy_digest
            or receipt['parent_policy_version']!=context['parent_policy_version']
            or receipt['records_sha256']!=context['records_sha256']
            or receipt['scenario_sha256']!=context['scenario_sha256']
            or receipt['optimized_decisions']!=context['optimized_decisions']
            or receipt['epochs']!=context['epochs']):
        raise ValueError('campaign receipt differs from learner update provenance')
    multiplayer=context.get('multiplayer')
    global_decisions=multiplayer.get('global_decisions') if type(multiplayer) is dict else context['optimized_decisions']
    if receipt['global_decisions']!=global_decisions:
        raise ValueError('campaign receipt decision count differs from learner provenance')
    if receipt['parent_critic_version']!=context.get('parent_critic_version'):
        raise ValueError('campaign receipt parent critic differs from learner provenance')
    if receipt['critic_version']!=critic_version:
        raise ValueError('campaign receipt critic differs from learner artifact')
    if parent_learner_sha256 is not None and receipt['parent_learner_sha256']!=parent_learner_sha256:
        raise ValueError('campaign receipt parent learner differs from loaded checkpoint')
    return receipt


def training_bytes(runtime):
    _,vocabulary,model,memory,device = runtime
    if model._sample_audit or memory._states:
        raise ValueError('learner snapshot requires an empty completed boundary')
    count = getattr(model,'_diagnostic_updates',None)
    context = getattr(model,'_last_update_context',None)
    campaign_state=getattr(model,'_campaign_state',None)
    campaign_module=None
    if type(count) is not int or not 1 <= count or (campaign_state is None and count>4) or not context:
        raise ValueError('no completed learner update provenance')
    closed = sorted(model._closed_episodes)
    if not closed or (campaign_state is None and len(closed)>16):
        raise ValueError('invalid closed-episode ledger')
    blob = policy_bytes(model,vocabulary)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != model._policy_version:
        raise ValueError('learner policy version is stale')
    actor_optimizer = model._optimizer
    critic_mode = _critic_mode(model)
    adam_max=64
    if campaign_state is not None:
        from .campaign import validate_state
        checked=validate_state(campaign_state)
        campaign_updates=checked['origin_updates']+len(checked['receipts'])
        adam_max=64+max(campaign_updates,checked['contract']['max_updates'])*checked['contract']['max_epochs_per_update']
    critic = _privileged_payload(model, actor_optimizer,max_step=adam_max)
    if (critic_mode == 'player_only') != (critic is None):
        raise ValueError('critic mode and critic runtime state differ')
    _validate_update_context(context, critic_mode=critic_mode)
    schema=SCHEMA
    if campaign_state is not None:
        from . import campaign as campaign_module
        campaign_state=campaign_module.validate_state(campaign_state,total_updates=count,
                                                        closed_episodes=closed)
        if campaign_state['receipts']:
            _validate_campaign_receipt(campaign_state,context,digest,
                                       critic.get('version') if critic is not None else None,
                                       getattr(model,'_loaded_learner_sha256',None))
        else:
            succession=campaign_state.get('succession')
            expected_origin_sha=(succession['parent_learner_sha256'] if succession is not None
                                 else campaign_state['contract']['origin_learner_sha256'])
            if (count!=campaign_state['origin_updates']
                    or set(closed)!=set(map(tuple,campaign_state['origin_closed_episodes']))
                    or (succession is None and getattr(model,'_loaded_learner_sha256',None)!=expected_origin_sha)
                    or (succession is not None and (succession['global_completed_updates']!=count
                        or succession['parent_learner_sha256']!=campaign_state['contract']['origin_learner_sha256']))):
                raise ValueError('empty campaign boundary differs from its origin learner')
        schema=CAMPAIGN_SCHEMA
    payload = dict(schema=schema,training_eligible=False,boundary='after_completed_task_update',
        implementation_sha256=implementation_hashes(),environment=_environment(device),
        policy=blob,policy_sha256=digest,optimizer=_optimizer_state(model,actor_optimizer,max_step=adam_max),
        optimizer_parameter_names=[name for name,_ in model.named_parameters()],
        rng=_rng_state(device),updates=count,closed_episodes=closed,last_update_context=context,
        privileged_critic=critic,critic_mode=critic_mode,
        scheduler=None,collector={'pending_samples':0,'player_memory':'empty','next_game':'explicit_new_episode'})
    if campaign_state is not None:payload['campaign']=campaign_state
    stream=io.BytesIO();torch.save(payload,stream)
    data=stream.getvalue()
    if len(data)>MAX_BYTES:
        raise ValueError('learner snapshot exceeds bound')
    return data


def load_training_bytes(data,*,sha256,device='cpu',expected_vocabulary_fingerprint=None,
                        implementation_migration=None,max_completed_updates=4,campaign_contract=None,
                        candidate_family_prior=None,actor_schema_migration=None,source_restore=None,
                        campaign_succession=False):
    if type(data) is not bytes or not 0<len(data)<=MAX_BYTES or hashlib.sha256(data).hexdigest()!=sha256:
        raise ValueError('learner snapshot size or digest differs')
    if device not in ('cpu','mps'):
        raise ValueError('unsupported resume device')
    if type(max_completed_updates) is not int or not 1<=max_completed_updates<=4:
        raise ValueError('invalid resume update bound')
    campaign_module=None
    if campaign_contract is not None:
        from . import campaign as campaign_module
        campaign_contract=campaign_module.validate_contract(campaign_contract)
    if implementation_migration is not None and implementation_migration not in SOURCE_TRANSITIONS:
        raise ValueError('unknown learner implementation migration')
    if candidate_family_prior not in (None, 'inverse_legal_candidate_count_v1'):
        raise ValueError('unknown candidate family prior')
    if actor_schema_migration not in (None, PUBLIC_PURCHASE_OBSERVATION_V15):
        raise ValueError('unknown actor schema migration')
    if source_restore not in (None, V14_PARENT_SOURCE_RESTORE):
        raise ValueError('unknown learner source restore')
    if type(campaign_succession) is not bool:
        raise ValueError('campaign succession selection must be explicit boolean')
    if ((actor_schema_migration is not None or source_restore is not None)
            and (implementation_migration is not None or candidate_family_prior is not None
                 or campaign_contract is not None or campaign_succession)):
        raise ValueError('V14 parent restore/migration cannot combine with other migrations or campaigns')
    if actor_schema_migration is not None and source_restore is not None:
        raise ValueError('V14 actor migration and source-only restore are mutually exclusive')
    if candidate_family_prior is not None and implementation_migration != 'candidate_family_prior_reconfiguration_v1':
        raise ValueError('candidate family prior requires its exact source transition')
    prior_rng = _rng_state(device)
    committed=False
    try:
        payload=torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)
        if type(payload) is dict and payload.get('schema') in ('civ6rl.learner_boundary.v1',
                                                                'civ6rl.learner_boundary.v2'):
            raise ValueError('learner v1/v2 snapshots do not contain critic mode state; migration must be explicit')
        required={'schema','training_eligible','boundary','implementation_sha256','environment','policy',
                  'policy_sha256','optimizer','optimizer_parameter_names','rng','updates','closed_episodes',
                  'last_update_context','privileged_critic','critic_mode','scheduler','collector'}
        if type(payload) is dict and payload.get('schema')==CAMPAIGN_SCHEMA:
            required=required|{'campaign'}
        if (type(payload) is not dict or set(payload)!=required or payload['schema'] not in (SCHEMA,CAMPAIGN_SCHEMA)
                or payload['training_eligible'] is not False or payload['boundary']!='after_completed_task_update'):
            raise ValueError('unknown learner snapshot contract')
        campaign_state=None
        if payload['schema']==CAMPAIGN_SCHEMA:
            if campaign_module is None:
                from . import campaign as campaign_module
            if campaign_contract is None:
                raise ValueError('campaign learner resume requires its exact contract')
            campaign_state=campaign_module.validate_state(payload['campaign'])
            if not campaign_succession and campaign_state['contract']!=campaign_contract:
                raise ValueError('learner campaign contract differs')
            if campaign_succession and (not campaign_state['receipts']
                    or campaign_state['contract']==campaign_contract):
                raise ValueError('campaign succession requires a different exhausted source campaign')
        elif campaign_succession:
            raise ValueError('campaign succession requires a full learner with its prior campaign state')
        if (implementation_migration==CAMPAIGN_ORIGIN_REATTACH_V1_ID
                and (campaign_contract is None or campaign_state is not None or payload['schema']!=SCHEMA)):
            raise ValueError('campaign origin source transition applies only when attaching to an uncampaigned v3 learner')
        if ((implementation_migration==COMPLETED_CAMPAIGN_SUCCESSOR_V1_ID) != campaign_succession
                or (campaign_succession and (campaign_contract is None or campaign_state is None
                    or payload['schema']!=CAMPAIGN_SCHEMA
                    or campaign_contract['origin_learner_sha256']!=sha256))):
            raise ValueError('completed campaign successor requires its exact parent, contract and source transition')
        if candidate_family_prior is not None:
            if (payload['schema'] != SCHEMA or campaign_state is not None
                    or payload['implementation_sha256'].get('model.py') != FAMILY_PRIOR_RECONFIGURATION_V1['model.py']
                    or payload['implementation_sha256'].get('training_state.py') != FAMILY_PRIOR_RECONFIGURATION_V1['training_state.py']):
                raise ValueError('candidate family prior reconfiguration requires the exact uncampaigned V3 source')
        elif (campaign_state is None and campaign_contract is not None
              and campaign_contract['origin_learner_sha256']!=sha256):
            raise ValueError('campaign origin learner digest differs')
        current_sources=implementation_hashes()
        migration=None
        parent_source_migration = actor_schema_migration is not None or source_restore is not None
        if parent_source_migration:
            parent_sources = _pinned_v14_parent_sources()
            if (payload['schema'] != SCHEMA or payload['implementation_sha256'] != parent_sources
                    or payload.get('campaign') is not None
                    or sha256 != V14_PARENT_LEARNER_SHA256
                    or payload.get('policy_sha256') != V14_PARENT_POLICY_SHA256):
                raise ValueError('V14 parent restore requires the exact learner, actor and source bundle')
            if actor_schema_migration is not None:
                if campaign_state is not None:
                    raise ValueError('V14 actor migration does not migrate campaign receipts')
                migration=dict(profile=actor_schema_migration,source_checkpoint_sha256=sha256,
                    source_evidence=dict(parent_json_sha256=V14_PARENT_JSON_SHA256,
                                         source_archive_sha256=V14_PARENT_ARCHIVE_SHA256),
                    sources={name:dict(previous=digest,current=current_sources.get(name))
                             for name,digest in parent_sources.items()},
                    model_migration=True,optimizer_migration=True,
                    retained_trajectories_reusable=False)
            else:
                migration=dict(profile=source_restore,source_checkpoint_sha256=sha256,
                    source_evidence=dict(parent_json_sha256=V14_PARENT_JSON_SHA256,
                                         source_archive_sha256=V14_PARENT_ARCHIVE_SHA256),
                    sources={name:dict(previous=digest,current=current_sources.get(name))
                             for name,digest in parent_sources.items()},
                    model_migration=False,optimizer_migration=False,
                    retained_trajectories_reusable=False)
        elif implementation_migration is None:
            if payload['implementation_sha256']!=current_sources:
                raise ValueError('learner implementation differs; migration must be explicit')
        else:
            changes=SOURCE_TRANSITIONS[implementation_migration]
            expected_sources={name:digest for name,digest in current_sources.items()
                              if name not in changes or changes[name] is not None}
            expected_sources.update({name:digest for name,digest in changes.items() if digest is not None})
            if payload['implementation_sha256']!=expected_sources:
                raise ValueError('learner source pins do not match bounded collection resume migration')
            migration=dict(profile=implementation_migration,source_checkpoint_sha256=sha256,
                sources={name:dict(previous=previous,current=current_sources.get(name))
                         for name,previous in changes.items()},
                model_migration=False,optimizer_migration=False)
        if payload['environment']!=_environment(device):
            raise ValueError('learner backend/version/thread configuration differs')
        if (payload['scheduler'] is not None or payload['collector']!=
                {'pending_samples':0,'player_memory':'empty','next_game':'explicit_new_episode'}):
            raise ValueError('unsupported scheduler or pending collector state')
        if (type(payload['updates']) is not int or payload['updates']<1
                or (campaign_state is None and payload['updates']>max_completed_updates)):
            raise ValueError('invalid cumulative update count')
        closed=payload['closed_episodes']
        campaign_max_closed=(campaign_module.MAX_GLOBAL_CLOSED_EPISODES if campaign_state is not None else 16)
        if (type(closed) is not list or not 1<=len(closed)<=campaign_max_closed
                or any(type(row) not in (list,tuple) or len(row)!=2 or any(type(x) is not str or not 0<len(x)<=256 for x in row) for row in closed)
                or len(set(map(tuple,closed)))!=len(closed)):
            raise ValueError('invalid closed-episode ledger')
        critic_payload=payload['privileged_critic']
        critic_mode=payload['critic_mode']
        if critic_mode not in ('player_only','privileged_diagnostic','history_only_diagnostic'):
            raise ValueError('invalid checkpoint critic mode')
        if (critic_mode == 'player_only') != (critic_payload is None):
            raise ValueError('checkpoint critic mode and payload differ')
        if critic_payload is not None:
            critic_fields={'bytes','sha256','version','optimizer','optimizer_parameter_names'}
            if type(critic_payload) is not dict or set(critic_payload)!=critic_fields:
                raise ValueError('invalid privileged critic checkpoint fields')
            critic_blob=critic_payload['bytes']
            critic_hash=critic_payload['sha256']
            critic_version_saved=critic_payload['version']
            if (type(critic_blob) is not bytes or not 0<len(critic_blob)<=MAX_BYTES
                    or not _valid_sha(critic_hash) or not _valid_sha(critic_version_saved)
                    or hashlib.sha256(critic_blob).hexdigest()!=critic_hash
                    or critic_version_saved!=critic_hash):
                raise ValueError('privileged critic digest or version differs')
            if type(critic_payload['optimizer_parameter_names']) is not list:
                raise ValueError('invalid critic optimizer parameter names')
        if campaign_state is not None:
            campaign_module.validate_state(campaign_state,total_updates=payload['updates'],closed_episodes=closed)
            contract=(campaign_contract if campaign_succession else campaign_state['contract'])
            adam_max=64+max(payload['updates'],contract['max_updates'])*contract['max_epochs_per_update']
            if len(campaign_state['receipts'])>contract['max_updates']:
                if not campaign_succession:
                    raise ValueError('campaign update budget exceeded')
        else:
            adam_max=64
        context=payload['last_update_context']
        _validate_update_context(context, critic_mode=critic_mode)
        _validate_rng(payload['rng'],device)
        model,vocabulary=load_policy_bytes(payload['policy'],sha256=payload['policy_sha256'],device=device)
        if expected_vocabulary_fingerprint is not None and vocabulary.fingerprint!=expected_vocabulary_fingerprint:
            raise ValueError('resume vocabulary differs')
        if payload['optimizer_parameter_names'] != [name for name,_ in model.named_parameters()]:
            raise ValueError('optimizer parameter names differ')
        source_policy_version=policy_version(model,vocabulary)
        if source_policy_version!=payload['policy_sha256']:
            raise ValueError('restored learner policy identity differs')
        actor_migration_receipt=None
        actor_optimizer_state=payload['optimizer']
        if actor_schema_migration is not None:
            if payload['schema'] != SCHEMA or campaign_state is not None:
                raise ValueError('public purchase actor migration requires an uncampaigned learner boundary')
            model,actor_optimizer_state,actor_migration_receipt = _migrate_v14_actor_and_optimizer(
                model,vocabulary,actor_optimizer_state,source_learner_sha256=sha256)
        else:
            _validate_optimizer(model,actor_optimizer_state,max_step=adam_max)
        _validate_optimizer(model,actor_optimizer_state,max_step=adam_max)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        optimizer.load_state_dict(actor_optimizer_state)
        model._optimizer=optimizer;model._diagnostic_updates=payload['updates']
        if candidate_family_prior is not None:
            prior_before=getattr(model.config,'candidate_family_prior',None)
            if prior_before is not None:
                raise ValueError('candidate family prior is already active or unknown')
            from .model import actor_schema_for_config
            if actor_schema_for_config(model.config)!='civ6rl.recurrent_policy.v14':
                raise ValueError('candidate family prior requires the exact V14 actor')
            old_config=model.config
            model.config=replace(old_config,candidate_family_prior=candidate_family_prior)
            target_policy_version=policy_version(model,vocabulary)
            if target_policy_version==source_policy_version:
                raise ValueError('candidate family prior did not change actor identity')
            model._policy_version=target_policy_version
            model._policy_reconfiguration_receipt=dict(
                schema='civ6rl.policy_reconfiguration.v1',
                prior_id=candidate_family_prior,
                source_learner_sha256=sha256,
                source_policy_sha256=payload['policy_sha256'],
                target_policy_sha256=target_policy_version,
                source_implementation_sha256=payload['implementation_sha256'],
                target_implementation_sha256=current_sources,
                preserved=dict(actor_tensors=True,actor_optimizer=True,critic=True,
                    critic_optimizer=True,rng=True,updates=True,closed_episodes=True,
                    last_update_context=True,collector_empty=True),
                retained_trajectories_reusable=False)
        else:
            model._policy_version=(actor_migration_receipt['target_policy_sha256']
                                   if actor_migration_receipt is not None else source_policy_version)
        if actor_migration_receipt is not None:
            model._resume_actor_schema_migration=actor_migration_receipt
        if critic_payload is None:
            model._privileged_state=None
        else:
            from .privileged_critic import PrivilegedCritic, critic_bytes, critic_version, load_critic_bytes
            critic=load_critic_bytes(critic_payload['bytes'],sha256=critic_payload['sha256'],device=device)
            if not isinstance(critic,PrivilegedCritic):
                raise ValueError('loaded privileged critic has an unexpected type')
            if (critic_version(critic)!=critic_payload['version']
                    or hashlib.sha256(critic_bytes(critic)).hexdigest()!=critic_payload['sha256']):
                raise ValueError('restored privileged critic identity differs')
            if critic_payload['optimizer_parameter_names'] != [name for name,_ in critic.named_parameters()]:
                raise ValueError('critic optimizer parameter names differ')
            actor_ids=set(_parameter_ids(model));critic_ids=set(_parameter_ids(critic))
            if not critic_ids or len(critic_ids)!=len(_parameter_ids(critic)) or actor_ids & critic_ids:
                raise ValueError('restored critic parameters overlap actor')
            _validate_optimizer(critic,critic_payload['optimizer'],max_step=adam_max)
            critic_optimizer=critic.make_optimizer(learning_rate=3e-4)
            critic_optimizer.load_state_dict(critic_payload['optimizer'])
            if set(_optimizer_parameter_ids(optimizer)) & set(_optimizer_parameter_ids(critic_optimizer)):
                raise ValueError('restored actor and critic optimizers share parameters')
            model._privileged_state={'critic':critic,'optimizer':critic_optimizer,
                                     'version':critic_payload['version']}
        model._critic_mode=critic_mode
        model._sample_audit={};model._closed_episodes=set(map(tuple,closed))
        model._last_update_context=context
        model._loaded_learner_sha256=sha256
        succession_receipt=None
        if campaign_succession:
            old_state=campaign_state
            # Validate the exhausted source campaign against its last committed
            # update before replacing only its per-campaign budget metadata.
            _validate_campaign_receipt(old_state,context,payload['policy_sha256'],
                critic_payload['version'] if critic_payload is not None else None)
            transition_receipt=dict(profile=COMPLETED_CAMPAIGN_SUCCESSOR_V1_ID,
                source_checkpoint_sha256=sha256,sources=migration['sources'],
                supporting_sources={name:{'previous':digest_value,'current':hashlib.sha256(
                    (Path(__file__).resolve().parents[2]/name).read_bytes()).hexdigest()}
                    for name,digest_value in _campaign_successor_supporting_sources().items()},
                model_migration=False,optimizer_migration=False)
            critic_version_value=critic_payload['version'] if critic_payload is not None else None
            succession_receipt=campaign_module.successor_receipt(old_state,
                parent_learner_sha256=sha256,parent_policy_sha256=payload['policy_sha256'],
                parent_critic_sha256=critic_version_value,new_contract=campaign_contract,
                source_transition=transition_receipt,closed_episodes=closed)
            campaign_state=campaign_module.succeed_campaign(old_state,
                receipt=succession_receipt,closed_episodes=closed)
        if campaign_state is not None:
            _validate_campaign_receipt(campaign_state,context,payload['policy_sha256'],
                critic_payload['version'] if critic_payload is not None else None) if campaign_state['receipts'] else None
            model._campaign_state=campaign_state
        if succession_receipt is not None:
            model._resume_campaign_succession=succession_receipt
        elif campaign_state is None and campaign_contract is not None:
            model._campaign_state=campaign_module.initial_state(campaign_contract,payload['updates'],closed)
        if migration is not None:
            model._resume_implementation_migration=migration
        _restore_rng(payload['rng'],device)
        committed=True
        return torch,vocabulary,model,PlayerMemory(),device
    finally:
        if not committed:
            _restore_rng(prior_rng,device)
