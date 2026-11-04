"""Persistent, bounded JSON-lines policy worker for an SSH stdio channel."""
from __future__ import annotations

import json
import copy
import hashlib
import math
import os
import sys
import threading
import time
from typing import Any
from .wire import (MAX_LINE_BYTES, EXTENDED_MAX_LINE_BYTES,
                   EXTENDED_MAX_DECODED_BYTES, decode_line, encode_line)
from .diagnostic_budget998 import validate_collection, is_extended

MAX_REQUESTS = 16
MAX_FROZEN_REQUESTS = 65
REQUEST_WATCHDOG_SECONDS = 20
PROCESS_WATCHDOG_SECONDS = 120


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _artifact_bytes(reference, max_bytes):
    from pathlib import Path
    if type(reference) is not dict or set(reference) != {'path','sha256'}:
        raise ValueError('invalid_artifact_reference')
    path=Path(reference['path'])
    if path.is_absolute() or '..' in path.parts or path.parent!=Path('learner-artifacts'):
        raise ValueError('checkpoint_must_be_local_learner_artifact')
    if not path.resolve().is_relative_to(Path.cwd().resolve()) or path.stat().st_size>max_bytes:
        raise ValueError('checkpoint_path_or_size')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=reference['sha256']:
        raise ValueError('artifact_digest_differs')
    return data


def _publish_artifact(data, suffix):
    """Publish complete immutable bytes; interrupted writes never look committed."""
    from pathlib import Path
    import tempfile
    folder=Path('learner-artifacts');folder.mkdir(exist_ok=True)
    if folder.is_symlink():raise ValueError('artifact_directory_must_not_be_symlink')
    digest=hashlib.sha256(data).hexdigest();path=folder/(digest+suffix)
    if path.exists():
        if path.is_symlink() or path.read_bytes()!=data:raise ValueError('artifact_collision')
        return dict(path=path.as_posix(),sha256=digest)
    fd,name=tempfile.mkstemp(prefix='.pending-',dir=folder)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)
    return dict(path=path.as_posix(),sha256=digest)


def _publish_update_artifacts(policy_blob,learner_blob,commit_factory=None):
    """Publish the actor and learner before making an optional campaign commit visible."""
    policy_ref=_publish_artifact(policy_blob,'.pt')
    learner_ref=_publish_artifact(learner_blob,'.learner.pt')
    commit_ref=None
    if commit_factory is not None:
        commit_ref=_publish_artifact(commit_factory(policy_ref,learner_ref),'.commit.json')
    return policy_ref,learner_ref,commit_ref


def _validate_campaign_resume_commit(resume_ref,commit_ref,model,state):
    from . import campaign
    if (type(commit_ref) is not dict or set(commit_ref)!={'path','sha256'}
            or type(commit_ref['sha256']) is not str or len(commit_ref['sha256'])!=64
            or any(c not in '0123456789abcdef' for c in commit_ref['sha256'])
            or commit_ref['path']!='learner-artifacts/'+commit_ref['sha256']+'.commit.json'):
        raise ValueError('invalid campaign pair commit reference')
    raw=_artifact_bytes(commit_ref,1024*1024)
    commit=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique_pairs,parse_constant=_reject_constant)
    last=state['receipts'][-1]
    policy_ref=dict(path='learner-artifacts/'+model._policy_version+'.pt',sha256=model._policy_version)
    campaign.validate_commit(commit,learner=resume_ref,policy=policy_ref,
        campaign=campaign.summary(state),parent=last['parent_learner_sha256'])
    actor_data=_artifact_bytes(commit['policy'],64*1024*1024)
    if hashlib.sha256(actor_data).hexdigest()!=model._policy_version:
        raise ValueError('campaign commit actor differs from resumed learner')
    return commit


def _send(value: dict, *, extended=False) -> None:
    try:
        data = (encode_line(value, max_line_bytes=EXTENDED_MAX_LINE_BYTES,
                            max_decoded_bytes=EXTENDED_MAX_DECODED_BYTES)
                if extended else encode_line(value))
    except (TypeError, ValueError, UnicodeError):
        data = b'{"id":null,"ok":false,"error":"response_too_large"}\n'
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def _build_policy(vocabulary_serialized: str, device: str, seed: int, checkpoint=None, *, critic_mode='player_only'):
    # Kept lazy so the stdlib transport client and fake-worker tests do not
    # require PyTorch on the native host.
    import torch
    from .encoding import (ACTION_FEATURES, ENTITY_FEATURES, GLOBAL_FEATURES,
                           MAP_CHANNELS, ContentVocabulary)
    from .model import PolicyConfig, RecurrentPolicy, PlayerMemory
    from .critic_modes import MODES,SEPARATE_MODES

    vocabulary = ContentVocabulary.from_serialized(vocabulary_serialized)
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("invalid_seed")
    if critic_mode not in MODES:
        raise ValueError('invalid critic mode')
    if device not in ("cpu", "mps"):
        raise ValueError("invalid_device")
    if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise ValueError("mps_unavailable")
    torch.manual_seed(seed)
    if device == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    config = PolicyConfig(map_channels=len(MAP_CHANNELS),
                          global_features=len(GLOBAL_FEATURES),
                          entity_features=len(ENTITY_FEATURES),
                          action_features=len(ACTION_FEATURES),
                          vocabulary_size=vocabulary.vocabulary_size)
    if checkpoint is None:
        model = RecurrentPolicy(config).to(device)
    else:
        from .artifacts import load_policy_bytes
        model,loaded=load_policy_bytes(_artifact_bytes(checkpoint,64*1024*1024),sha256=checkpoint['sha256'],device=device)
        if loaded.fingerprint!=vocabulary.fingerprint:raise ValueError('checkpoint_vocabulary_mismatch')
    model.eval()
    from .artifacts import policy_version
    model._policy_version=policy_version(model,vocabulary)
    model._diagnostic_updates=0 if checkpoint is None else 1
    model._sample_audit={};model._closed_episodes=set()
    if critic_mode in SEPARATE_MODES:
        from .privileged_critic import PrivilegedCritic,PrivilegedCriticConfig,critic_version
        from .privileged_encoding import PRIVILEGED_SCHEMA,PRIVILEGED_FEATURES
        # Construct on CPU without changing the actor's sampling RNG. A plain
        # dict keeps critic parameters out of actor modules/state_dict/export.
        with torch.random.fork_rng(devices=[]):
            critic=PrivilegedCritic(PrivilegedCriticConfig(PRIVILEGED_SCHEMA,PRIVILEGED_FEATURES,
                                    model.config.hidden_size)).to(device).eval()
        model._privileged_state=dict(critic=critic,optimizer=critic.make_optimizer(),version=critic_version(critic))
    model._critic_mode=critic_mode
    return torch, vocabulary, model, PlayerMemory(), device


def _infer(native_response: dict, runtime, *, scenario=None, completed_decisions=None) -> dict[str, Any]:
    torch, vocabulary, model, memory, device = runtime
    from .task_context import encode_task_response, validate_collection_binding
    from .model import collate, actor_schema_for_config

    started = time.perf_counter()
    frame = encode_task_response(native_response, vocabulary=vocabulary,
                                 scenario=scenario, completed_decisions=completed_decisions,
                                 actor_schema=actor_schema_for_config(model.config))
    if frame["vocabulary_fingerprint"] != vocabulary.fingerprint:
        raise ValueError("vocabulary_mismatch")
    ids = frame["candidate_ids"]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("empty_or_duplicate_candidates")
    episode = (native_response["game_generation"], frame["episode_id"])
    if episode in getattr(model,'_closed_episodes',set()):raise ValueError('updated_policy_requires_new_episode')
    validate_collection_binding(model,native_response,frame,completed_decisions)
    player, sequence = frame["player_id"], frame["sequence"]
    state = memory.take(episode, player, sequence)
    batch = collate([frame], device=device)
    reset = torch.tensor([state is None], dtype=torch.bool, device=device)
    with torch.inference_mode():
        actions, log_probs, value, new_state, entropy, assignments = model.act(batch, state, reset)
        selected = int(actions.reshape(-1)[0].item())
        log_prob = float(log_probs.reshape(-1)[0].item())
        entropy = float(entropy.reshape(-1)[0].item())
        value = float(value.reshape(-1)[0].item())
    elapsed = time.perf_counter() - started
    if not all(math.isfinite(x) for x in (log_prob, entropy, value, elapsed)) or elapsed < 0:
        raise ValueError("nonfinite_inference_result")
    if selected < 0 or selected >= len(ids):
        raise ValueError("selected_candidate_out_of_range")
    candidate_id = ids[selected]
    selected_action = native_response["decisions"]["candidates"][selected]["action"]
    assignment = assignments[0] if assignments is not None else None
    if assignment is not None:
        from .policy_assignments import validate_joint_assignment
        if selected_action.get("kind") != "policies" or type(assignment) is not list:
            raise ValueError("unexpected_joint_assignment")
        slots = selected_action.get("slots")
        validate_joint_assignment(slots, assignment)
    elif selected_action.get("kind") == "policies":
        raise ValueError("policy_action_missing_joint_assignment")
    # Commit recurrent state only after every inference/result check succeeds.
    memory.put(episode, player, sequence, new_state)
    result = {"episode_id": frame["episode_id"], "player_id": player, "sequence": sequence,
            "candidate_id": candidate_id, "log_prob": log_prob, "value": value,
            "entropy": entropy, "inference_seconds": elapsed,
            "model_status": "diagnostic_updated" if getattr(model,'_diagnostic_updates',0) else "untrained",
            "training_eligible": False}
    result.update(policy_version=getattr(model,'_policy_version',None),memory_reset=state is None,
                  sampling_mode='unconditioned_policy_sample',scenario_sha256=frame['scenario_sha256'],
                  task_completed_decisions=completed_decisions)
    if getattr(model,'_privileged_state',None) is not None:
        # Version metadata is not a network input and contains no private state.
        result['critic_version']=model._privileged_state['version']
        from .critic_modes import mode_of
        result['critic_mode']=mode_of(model)
    if assignment is not None:
        result["assignments"] = assignment
    if hasattr(model,'_sample_audit'):
        key=(native_response['episode_id'],native_response['game_generation'],player,sequence)
        model._sample_audit[key]={'before':copy.deepcopy(native_response),'choice':copy.deepcopy(result)}
    return result


def _update_diagnostic(request,runtime):
    """Share update admission and atomic publication across collection shapes."""
    torch,vocabulary,model,memory,device=runtime
    from .critic_modes import mode_of
    if getattr(model,'_evaluation_only',False):raise ValueError('frozen evaluation worker refuses optimizer updates')
    collection=getattr(model,'_diagnostic_collection',None)
    if collection is not None:
        if request.get('op','update_diagnostic')!='update_multiplayer_diagnostic':
            raise ValueError('diagnostic collection permits only a multiplayer update')
        if model._diagnostic_collection_updates>=collection['max_updates']:
            raise ValueError('diagnostic collection update budget exhausted')
    campaign_state=getattr(model,'_campaign_state',None)
    campaign_module=None
    if campaign_state is not None:
        from . import campaign as campaign_module
        campaign_module.admit_update(campaign_state,request['records'],request['epochs'])
    elif getattr(model,'_diagnostic_updates',0)>=4:raise ValueError('diagnostic_update_budget')
    if request['policy_version']!=model._policy_version:raise ValueError('stale_update_version')
    private=getattr(model,'_privileged_state',None)
    critic_mode=mode_of(model)
    all_extra={'privileged_snapshots','privileged_context','critic_version'}
    required_extra=({'privileged_snapshots','privileged_context','critic_version'} if critic_mode=='privileged_diagnostic'
                    else {'critic_version'} if critic_mode=='history_only_diagnostic' else set())
    if set(request)&all_extra != required_extra:
        raise ValueError('private update fields do not match learner mode')
    from .learner import update_native_diagnostic,update_multiplayer_diagnostic
    from .artifacts import policy_bytes,policy_version
    from .training_state import training_bytes
    optimizer=getattr(model,'_optimizer',None)
    if optimizer is None:optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    started=time.perf_counter()
    options=dict(sample_audit=model._sample_audit,expected_policy_version=model._policy_version,
          epochs=request['epochs'],privileged_state=private,
          privileged_snapshots=request.get('privileged_snapshots'),privileged_context=request.get('privileged_context'),
          expected_critic_version=request.get('critic_version'),critic_mode=critic_mode)
    if request.get('op','update_diagnostic')=='update_multiplayer_diagnostic':
        context=dict(scenarios=request['scenarios'],collection=request['collection'])
        if collection is not None:options['diagnostic_collection']=collection
        report=update_multiplayer_diagnostic(model,optimizer,vocabulary,request['records'],
              request['scenarios'],request['collection'],**options)
    else:
        context=request['scenario']
        report=update_native_diagnostic(model,optimizer,vocabulary,request['records'],context,
              burn_in=request['burn_in'],**options)
    model._optimizer=optimizer;model._diagnostic_updates+=1
    prior=model._policy_version;model._policy_version=policy_version(model,vocabulary)
    if prior==model._policy_version:raise ValueError('no_diagnostic_parameter_change')
    for sample in model._sample_audit.values():
        before=sample['before'];model._closed_episodes.add((before['game_generation'],before['episode_id']))
    model._sample_audit.clear();memory._states.clear()
    def json_hash(value):
        return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    model._last_update_context=dict(scenario_sha256=json_hash(context),records_sha256=json_hash(request['records']),
                                   parent_policy_version=prior,optimized_decisions=report['optimized_decisions'],epochs=request['epochs'])
    if request.get('op','update_diagnostic')=='update_multiplayer_diagnostic':
        model._last_update_context['multiplayer']=dict(
            schema='civ6rl.multiplayer_update_provenance998.v1',
            global_decisions=len(request['records']),
            player_ids=sorted(row['player_id'] for row in request['scenarios']),
            diagnostic_collection=copy.deepcopy(collection))
        if request['collection'].get('allow_single_public_skill_group') is True:
            from .skill_context993 import SCHEMA as SKILL_SCHEMA
            model._last_update_context['multiplayer']['public_skill_scope']=dict(
                allow_single_public_skill_group=True,scenario_schema=SKILL_SCHEMA,
                public_continuity_sources=copy.deepcopy(request['collection']['public_continuity_sources']))
    if private is not None:
        model._last_update_context['parent_critic_version']=request['critic_version']
        if critic_mode=='privileged_diagnostic':
            model._last_update_context['privileged_sha256']=json_hash(dict(
                context=request['privileged_context'],snapshots=request['privileged_snapshots']))
    if campaign_state is not None:
        before=request['records'][0]['before']
        critic_after=private['version'] if private is not None else None
        campaign_state=campaign_module.append_receipt(campaign_state,dict(
            parent_learner_sha256=model._loaded_learner_sha256,
            game_generation=before['game_generation'],episode_id=before['episode_id'],
            global_decisions=len(request['records']),optimized_decisions=report['optimized_decisions'],
            epochs=request['epochs'],records_sha256=model._last_update_context['records_sha256'],
            scenario_sha256=model._last_update_context['scenario_sha256'],
            parent_policy_version=prior,policy_version=model._policy_version,
            parent_critic_version=model._last_update_context.get('parent_critic_version'),
            critic_version=critic_after))
        model._campaign_state=campaign_state
    # Validate/serialize both artifacts before publishing either immutable file.
    # A learner-checkpoint rejection must not orphan a seemingly valid actor.
    policy_blob=policy_bytes(model,vocabulary)
    learner_blob=training_bytes(runtime)
    commit_factory=None
    if campaign_state is not None:
        from .campaign import COMMIT_SCHEMA,summary as campaign_summary
        parent_learner_sha256=model._loaded_learner_sha256
        campaign_ack=campaign_summary(campaign_state)
        def commit_factory(policy_ref,resume_ref):
            return json.dumps(dict(schema=COMMIT_SCHEMA,learner=resume_ref,policy=policy_ref,
                campaign=campaign_ack,parent_learner_sha256=parent_learner_sha256),
                sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
    policy_ref,resume_ref,commit_ref=_publish_update_artifacts(policy_blob,learner_blob,commit_factory)
    if campaign_state is not None:
        model._loaded_learner_sha256=resume_ref['sha256']
    report.update(policy_version=model._policy_version,previous_policy_version=prior,
                  checkpoint=policy_ref,learner_checkpoint=resume_ref,completed_updates=model._diagnostic_updates,
                  resume_boundary='after_completed_task_update',update_seconds=time.perf_counter()-started,
                  model_status='diagnostic_updated')
    if campaign_state is not None:
        report['campaign']=campaign_module.summary(campaign_state)
        report['campaign_commit']=commit_ref
    if collection is not None:model._diagnostic_collection_updates+=1
    return report


def _handle(request: dict, runtime_box: list):
    if type(request) is not dict or type(request.get("id")) is not int or request["id"] < 1:
        raise ValueError("invalid_request")
    op = request.get("op")
    if op == "init":
        optional={'critic_mode','evaluation_only','request_budget','diagnostic_collection','episode_budget','campaign_contract','resume_commit'}
        fields=set(request)-optional
        if fields not in ({"id", "op", "vocabulary", "device", "seed"},{"id", "op", "vocabulary", "device", "seed","checkpoint"},
                                {"id", "op", "vocabulary", "device", "seed","resume"}):
            raise ValueError("invalid_init_fields")
        if runtime_box[0] is not None:
            raise ValueError("already_initialized")
        if not isinstance(request["vocabulary"], str) or not request["vocabulary"]:
            raise ValueError("vocabulary_required")
        evaluation_only=request.get('evaluation_only',False)
        if type(evaluation_only) is not bool:
            raise ValueError('evaluation_only must be boolean')
        campaign_contract=request.get('campaign_contract')
        resume_commit=request.get('resume_commit')
        if campaign_contract is not None:
            from . import campaign as campaign_module
            campaign_contract=campaign_module.validate_contract(campaign_contract)
            if ('resume' not in request or 'episode_budget' not in request or evaluation_only
                    or 'diagnostic_collection' not in request):
                raise ValueError('campaign requires learner resume, explicit episode budget, and non-frozen collection')
            if resume_commit is not None and (type(resume_commit) is not dict or set(resume_commit)!={'path','sha256'}):
                raise ValueError('invalid campaign resume commit reference')
        elif resume_commit is not None:
            raise ValueError('resume_commit requires campaign_contract')
        if evaluation_only and ('checkpoint' not in request or 'resume' in request
                                or request.get('critic_mode','player_only')!='player_only'):
            raise ValueError('frozen evaluation requires an actor-only checkpoint')
        collection=None
        episode_budget=(validate_collection(request['episode_budget'])
                        if 'episode_budget' in request else None)
        extended_budget=episode_budget is not None and is_extended(episode_budget)
        if episode_budget is not None and not extended_budget:
            raise ValueError('episode_budget requires the bounded episode schema')
        if extended_budget and type(request.get('request_budget')) is not int:
            raise ValueError('episode budget requires request_budget')
        if extended_budget and request['request_budget'] != episode_budget['max_decisions'] + 2:
            raise ValueError('episode request budget must equal max_decisions plus two')
        if 'diagnostic_collection' in request:
            collection=validate_collection(request['diagnostic_collection'])
            if evaluation_only:
                raise ValueError('diagnostic collection requires non-frozen policy')
            if request.get('critic_mode','player_only') not in ('player_only','history_only_diagnostic','privileged_diagnostic'):
                raise ValueError('invalid diagnostic critic mode')
            if extended_budget:
                if collection != episode_budget:
                    raise ValueError('diagnostic_collection must exactly equal episode_budget')
            elif is_extended(collection):
                raise ValueError('extended collection requires an episode budget handshake')
            elif (type(request.get('request_budget')) is not int
                    or request['request_budget']!=collection['max_decisions']+2):
                raise ValueError('diagnostic request budget must equal max_decisions plus two')
        elif extended_budget and not evaluation_only:
            raise ValueError('training episode budget requires diagnostic_collection')
        elif 'request_budget' in request and not extended_budget:
            budget = request['request_budget']
            checkpoint = request.get('checkpoint')
            if type(budget) is not int or not MAX_REQUESTS < budget <= MAX_FROZEN_REQUESTS:
                raise ValueError('request_budget must be an integer in [17, 65]')
            if (not evaluation_only or type(checkpoint) is not dict
                    or set(checkpoint) != {'path', 'sha256'}
                    or type(checkpoint['path']) is not str or not checkpoint['path']
                    or type(checkpoint['sha256']) is not str or len(checkpoint['sha256']) != 64
                    or any(c not in '0123456789abcdef' for c in checkpoint['sha256'])
                    or 'resume' in request or request.get('critic_mode','player_only') != 'player_only'):
                raise ValueError('extended request budget requires frozen actor-only checkpoint evaluation')
        if 'resume' in request:
            from .training_state import load_training_bytes,MAX_BYTES
            from .encoding import ContentVocabulary
            expected_vocabulary=ContentVocabulary.from_serialized(request['vocabulary']).fingerprint
            runtime=load_training_bytes(_artifact_bytes(request['resume'],MAX_BYTES),
                                        sha256=request['resume']['sha256'],device=request['device'],
                                        expected_vocabulary_fingerprint=expected_vocabulary,
                                        max_completed_updates=4 if campaign_contract is not None else (3 if collection is not None else 4),
                                        campaign_contract=campaign_contract)
            runtime_box[0]=runtime
            from .critic_modes import mode_of
            actual_mode=mode_of(runtime[2])
            if 'critic_mode' in request and request['critic_mode']!=actual_mode:
                runtime_box[0]=None
                raise ValueError('resumed critic mode differs')
        else:
            runtime_box[0] = _build_policy(request["vocabulary"], request["device"], request["seed"],request.get('checkpoint'),
                                          critic_mode=request.get('critic_mode','player_only'))
        model=runtime_box[0][2]
        campaign_state=getattr(model,'_campaign_state',None)
        if campaign_contract is not None:
            from . import campaign as campaign_module
            try:
                if campaign_state is None:
                    campaign_state=campaign_module.initial_state(campaign_contract,model._diagnostic_updates,model._closed_episodes)
                    model._campaign_state=campaign_state
                if campaign_state['contract']!=campaign_contract:
                    raise ValueError('resumed campaign contract differs')
                if campaign_state['receipts']:
                    if resume_commit is None:
                        raise ValueError('continued campaign requires pair commit reference')
                    _validate_campaign_resume_commit(request['resume'],resume_commit,model,campaign_state)
                elif resume_commit is not None:
                    raise ValueError('campaign without receipts cannot accept a pair commit')
                campaign_module.admit_episode(campaign_state,episode_budget)
            except BaseException:
                runtime_box[0]=None
                raise
        model._evaluation_only=evaluation_only
        if extended_budget:
            model._episode_budget=dict(episode_budget)
            model._episode_budget_inferences=0
            model._episode_budget_episode=None
        if collection is not None:
            model._diagnostic_collection=collection
            model._diagnostic_collection_inferences=0
            model._diagnostic_collection_updates=0
            model._diagnostic_collection_episode=None
        from .critic_modes import mode_of
        result = {"model_status": "diagnostic_updated" if model._diagnostic_updates else "untrained", "device": request["device"], "training_eligible": False,'policy_version':model._policy_version,
                'critic_mode':mode_of(model),
                'critic_version':model._privileged_state['version'] if getattr(model,'_privileged_state',None) is not None else None,
                'evaluation_only':evaluation_only,
                'completed_updates':model._diagnostic_updates,'learner_resumed':'resume' in request}
        if 'request_budget' in request: result['request_budget'] = request['request_budget']
        if extended_budget: result['episode_budget']=dict(episode_budget)
        if collection is not None:result['diagnostic_collection']=dict(collection)
        if campaign_state is not None:
            from .campaign import summary as campaign_summary
            result['campaign']=campaign_summary(campaign_state)
        return result
    if op in ('update_diagnostic','update_multiplayer_diagnostic'):
        base={'id','op','records','policy_version','epochs'}
        base.update({'scenario','burn_in'} if op=='update_diagnostic' else {'scenarios','collection'})
        if set(request) not in (base,base|{'critic_version'},base|{'privileged_snapshots','privileged_context','critic_version'}) or runtime_box[0] is None:
            raise ValueError('invalid_diagnostic_update_request')
        try:
            return _update_diagnostic(request,runtime_box[0])
        except BaseException:
            # Failed validation or persistence must not leave a partly updated
            # learner usable by a later inference request.
            runtime_box[0]=None
            raise
    if op == "infer":
        if set(request) not in ({"id", "op", "response"}, {"id", "op", "response", "scenario", "completed_decisions"}):
            raise ValueError("invalid_infer_fields")
        if runtime_box[0] is None:
            raise ValueError("not_initialized")
        if type(request["response"]) is not dict:
            raise ValueError("invalid_native_response")
        model=runtime_box[0][2]
        collection=getattr(model,'_diagnostic_collection',None)
        episode_budget=getattr(model,'_episode_budget',None)
        cap=episode_budget if episode_budget is not None else collection
        if cap is not None:
            if collection is not None and model._diagnostic_collection_updates:
                raise ValueError('diagnostic collection is closed after its update')
            used=(model._episode_budget_inferences if episode_budget is not None
                  else model._diagnostic_collection_inferences)
            if used>=cap['max_decisions']:
                raise ValueError('episode inference budget exhausted')
            native=request['response']
            if (type(native.get('game_generation')) is not str or not native['game_generation']
                    or type(native.get('episode_id')) is not str or not native['episode_id']):
                raise ValueError('diagnostic collection requires an episode and generation')
            identity=(native['game_generation'],native['episode_id'])
            # Preserve the completed-boundary ledger across process restarts.
            # This joins exact episode/generation identities, not renamed games.
            if identity in model._closed_episodes:
                raise ValueError('updated_policy_requires_new_episode')
            prior_identity=(model._episode_budget_episode if episode_budget is not None
                            else model._diagnostic_collection_episode)
            if prior_identity not in (None,identity):
                raise ValueError('diagnostic collection cannot change episode or generation')
        result=_infer(request["response"], runtime_box[0], scenario=request.get('scenario'),
                      completed_decisions=request.get('completed_decisions'))
        if episode_budget is not None:
            model._episode_budget_episode=identity
            model._episode_budget_inferences+=1
        elif collection is not None:
            model._diagnostic_collection_episode=identity
            model._diagnostic_collection_inferences+=1
        return result
    raise ValueError("unsupported_operation")


def main() -> int:
    process_started = time.monotonic()
    process_watchdog = threading.Timer(PROCESS_WATCHDOG_SECONDS, lambda: os._exit(124))
    process_watchdog.daemon = True
    process_watchdog.start()
    runtime = [None]
    requests = 0
    request_limit = MAX_REQUESTS
    episode_budget = None
    extended_active = False
    while True:
        line_limit = EXTENDED_MAX_LINE_BYTES if episode_budget is not None else MAX_LINE_BYTES
        raw = sys.stdin.buffer.readline(line_limit + 1)
        if not raw:
            process_watchdog.cancel()
            return 0
        if len(raw) > line_limit or not raw.endswith(b"\n"):
            _send({"id": None, "ok": False, "error": "framing_limit_or_missing_newline"})
            process_watchdog.cancel()
            return 2
        requests += 1
        if requests > request_limit:
            _send({"id": None, "ok": False, "error": "request_limit_exceeded"})
            process_watchdog.cancel()
            return 2
        request_id = None
        request_seconds = (episode_budget['request_seconds'] if extended_active
                           else REQUEST_WATCHDOG_SECONDS)
        watchdog = threading.Timer(request_seconds, lambda: os._exit(124))
        watchdog.daemon = True
        watchdog.start()
        try:
            request = (decode_line(raw, max_line_bytes=EXTENDED_MAX_LINE_BYTES,
                                   max_decoded_bytes=EXTENDED_MAX_DECODED_BYTES)
                       if extended_active else decode_line(raw))
            request_id = request.get("id") if isinstance(request, dict) else None
            runtime_value=runtime[0]
            model=(runtime_value[2] if isinstance(runtime_value,(tuple,list)) and len(runtime_value)>=3 else None)
            campaign_state=getattr(model,'_campaign_state',None)
            if (campaign_state is not None and request.get('op') in
                    ('update_diagnostic','update_multiplayer_diagnostic')):
                from .campaign import update_timeout_seconds
                update_seconds=update_timeout_seconds(len(request.get('records',[])))
                watchdog.cancel()
                watchdog=threading.Timer(update_seconds,lambda:os._exit(124))
                watchdog.daemon=True;watchdog.start()
            result = _handle(request, runtime)
            _send({"id": request_id, "ok": True, **result}, **({'extended':True} if extended_active else {}))
            # Only a successful typed init can raise this process-local bound.
            # Inference fields, environment variables and failed init attempts
            # never change it. An explicit episode acknowledgement also sets
            # the remaining process lifetime and subsequent request duration.
            if request.get('op') == 'init':
                request_limit = result.get('request_budget', MAX_REQUESTS)
                acknowledged = result.get('episode_budget')
                if acknowledged is not None:
                    # The init frame was decoded with default limits. Only _handle's
                    # successful schema validation and exact acknowledgement unlocks
                    # larger subsequent frames and watchdogs.
                    episode_budget = validate_collection(acknowledged)
                    declared = (validate_collection(request['episode_budget'])
                                if 'episode_budget' in request else None)
                    if not is_extended(episode_budget) or episode_budget != declared:
                        raise ValueError('invalid episode budget acknowledgement')
                    extended_active = True
                    process_watchdog.cancel()
                    runtime_value=runtime[0]
                    model=(runtime_value[2] if isinstance(runtime_value,(tuple,list))
                           and len(runtime_value)>=3 else None)
                    campaign_state=getattr(model,'_campaign_state',None)
                    process_reserve=(600 if campaign_state is not None else 0)
                    process_watchdog = threading.Timer(
                        max(0.0, process_started + episode_budget['wall_seconds'] + process_reserve - time.monotonic()),
                        lambda: os._exit(124))
                    process_watchdog.daemon = True
                    process_watchdog.start()
        except BaseException as exc:
            # Keep stdout protocol-only and avoid serializing native payloads or
            # stack traces into the response.
            error = str(exc)[:240] if isinstance(exc, (ValueError, NotImplementedError)) else type(exc).__name__
            try:
                _send({"id": request_id, "ok": False, "error": error})
            except BaseException:
                return 2
        finally:
            watchdog.cancel()


if __name__ == "__main__":
    raise SystemExit(main())
