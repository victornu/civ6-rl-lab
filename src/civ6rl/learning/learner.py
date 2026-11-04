"""Bounded native-data PPO diagnostics, never automatic training qualification.

Replays the actual player's history with the actor's current weights, verifies
behavior log probabilities/values, and derives rewards anew from the explicit
task contract. It cannot admit coverage-selected seeds or another player's
successor. Production qualification and long-run scheduling are separate.
"""
from __future__ import annotations
import copy,math
import torch
from .encoding import encode_response
from .task_context import encode_task_response, scenario_digest
from .model import collate, actor_schema_for_config
from .ppo import generalized_advantages,update
from .scenario import evaluate_step
from .action_receipt import verify_applied_action


def _prepare_native_diagnostic(model,optimizer,vocabulary,records,scenario,*,sample_audit,
                             expected_policy_version,burn_in=0,epochs=2,discount=.99,
                             privileged_state=None,privileged_snapshots=None,
                             privileged_context=None,expected_critic_version=None,critic_mode=None,delayed=False,collection_cut=False,
                             discount_unit='player_decision',decision_limit=12,public_continuity_sources=None,
                             sequence_limit=63):
    if type(sequence_limit) is not int or not 1<=sequence_limit<=256:
        raise ValueError('invalid explicit recurrent sequence bound')
    recurrent_limit=min(sequence_limit,256 if delayed else 12)
    if type(decision_limit) is not int or not 1<=decision_limit<=recurrent_limit:
        raise ValueError('invalid explicit recurrent decision limit')
    if type(records) is not list or not (1 if delayed else 2)<=len(records)<=decision_limit:
        raise ValueError('native diagnostic decision bound')
    if type(burn_in) is not int or not 0<=burn_in<len(records) or type(epochs) is not int or not 1<=epochs<=4:
        raise ValueError('unbounded diagnostic recurrent update')
    if type(discount) not in (int,float) or not math.isfinite(discount) or not 0<discount<=1:
        raise ValueError('invalid diagnostic discount')
    if discount_unit not in ('player_decision','native_turn'):
        raise ValueError('explicit known discount unit required')
    actor_ids={id(p) for p in model.parameters()}
    if not isinstance(optimizer,torch.optim.Optimizer):
        raise TypeError('actor optimizer must be a torch optimizer')
    owned=[id(p) for group in optimizer.param_groups for p in group['params']]
    if set(owned)!=actor_ids or len(owned)!=len(actor_ids):
        raise ValueError('actor optimizer must own exactly actor parameters')
    device=next(model.parameters()).device
    actor_schema=actor_schema_for_config(model.config)
    from .critic_modes import mode_of
    mode=mode_of(model)
    if critic_mode is not None and critic_mode!=mode:
        raise ValueError('critic mode differs from runtime')
    if (mode=='player_only') != (privileged_state is None):
        raise ValueError('critic mode and supplied learner state disagree')
    private_frames=None
    if privileged_state is None:
        if any(x is not None for x in (privileged_snapshots,privileged_context,expected_critic_version)):
            raise ValueError('player-only learner cannot receive privileged data')
    else:
        from .privileged_encoding import encode_privileged_snapshot,PRIVILEGED_SCHEMA,PRIVILEGED_FEATURES
        from .privileged_critic import critic_version,update_critic
        if type(privileged_state) is not dict or set(privileged_state)!={'critic','optimizer','version'}:
            raise ValueError('unknown privileged learner state')
        critic=privileged_state['critic'];critic_optimizer=privileged_state['optimizer']
        if not isinstance(critic_optimizer,torch.optim.Optimizer):
            raise TypeError('critic optimizer must be a torch optimizer')
        if (expected_critic_version!=privileged_state['version']
                or critic_version(critic)!=expected_critic_version):
            raise ValueError('stale privileged critic version')
        if (critic.config.privileged_schema!=PRIVILEGED_SCHEMA
                or critic.config.privileged_features!=PRIVILEGED_FEATURES
                or critic.config.actor_hidden_dim!=model.config.hidden_size):
            raise ValueError('privileged feature/history layout differs')
        critic_ids={id(p) for p in critic.parameters()}
        if actor_ids & critic_ids:
            raise ValueError('critic and actor share parameters')
        opt_ids=[id(p) for g in critic_optimizer.param_groups for p in g['params']]
        if set(opt_ids)!=critic_ids or len(opt_ids)!=len(critic_ids):
            raise ValueError('critic optimizer must own exactly critic parameters')
        if mode=='privileged_diagnostic':
            if type(privileged_snapshots) is not list or len(privileged_snapshots)!=len(records)+1:
                raise ValueError('privileged trajectory requires every state plus actual successor')
            private_frames=[encode_privileged_snapshot(s,r['before'],expected_context=privileged_context)
                            for s,r in zip(privileged_snapshots[:-1],records)]
        else:
            if privileged_snapshots is not None or privileged_context is not None:
                raise ValueError('history-only critic refuses privileged inputs')
            # Matched architecture and capacity, with no private-channel data
            # collected or read. All private dimensions are explicitly unknown.
            empty_private={'values':[0.]*len(PRIVILEGED_FEATURES),'known':[False]*len(PRIVILEGED_FEATURES)}
            private_frames=[empty_private for _ in records]
    task_digest=scenario_digest(scenario)
    from .skill_context993 import SCHEMA as SKILL_SCHEMA, evaluate_skill_transition,evaluate_public_skill_interval
    public_skill = delayed and public_continuity_sources is not None and scenario.get('schema') == SKILL_SCHEMA
    classify = evaluate_skill_transition if scenario.get('schema') == SKILL_SCHEMA else evaluate_step
    frames=[];actions=[];assignments=[];task_steps=[];old_logps=[];old_values=[];step_discounts=[]
    first=records[0]['before'];identity=(first['episode_id'],first['game_generation'],first['observation']['player_id'])
    previous=None
    for i,record in enumerate(records):
        expected={'before','choice','action_receipt','next_response'} if delayed else {'before','choice','after'}
        if public_skill:expected.add('public_witnesses')
        if set(record)!=expected:raise ValueError('unknown rollout record fields')
        before,choice=record['before'],record['choice']
        receipt=record['action_receipt'] if delayed else record['after']
        after=record['next_response'] if delayed else record['after']
        if (type(after) is not dict or type(after.get('observation')) is not dict) and not (public_skill and after is None):
            raise ValueError('task update requires actual same-player successor observation')
        if tuple(before[k] for k in ('episode_id','game_generation'))+(before['observation']['player_id'],)!=identity:
            raise ValueError('mixed episode/player history')
        if previous is not None and before!=previous:raise ValueError('noncontiguous recurrent history')
        for key in ('episode_id','game_generation'):
            if after is not None and after[key]!=before[key]:raise ValueError('reset spliced into native trajectory')
        if (receipt['episode_id']!=before['episode_id'] or receipt['game_generation']!=before['game_generation']
                or receipt['sequence']!=before['sequence']+1 or receipt['operation']!='act'):
            raise ValueError('wrong immediate action receipt')
        if after is not None and (after['observation']['player_id']!=identity[2] or after['operation']!='act'
                or (after['sequence']<=before['sequence'] if delayed else after['sequence']!=before['sequence']+1)):
            raise ValueError('wrong player or action successor')
        if delayed and after is not None and after['sequence']>receipt['sequence'] and scenario.get('schema')==SKILL_SCHEMA and not public_skill:
            raise ValueError('skill continuity across opponent intervals is not qualified')
        if choice.get('policy_version')!=expected_policy_version or choice.get('sampling_mode')!='unconditioned_policy_sample':
            raise ValueError('stale or coverage-conditioned behavior policy')
        if choice.get('critic_version')!=expected_critic_version:
            raise ValueError('critic version differs from behavior collection')
        if mode!='player_only' and choice.get('critic_mode')!=mode:
            raise ValueError('critic information mode differs from behavior collection')
        if choice.get('memory_reset') is not (i==0):raise ValueError('missing initial memory or unexpected reset')
        if choice.get('scenario_sha256')!=task_digest or choice.get('task_completed_decisions')!=i:
            raise ValueError('scenario or task count differs from behavior collection')
        key=(before['episode_id'],before['game_generation'],before['observation']['player_id'],before['sequence'])
        if sample_audit.get(key)!={'before':before,'choice':choice}:
            raise ValueError('rollout differs from the worker actual sampled decisions')
        assignment=verify_applied_action(before,choice,receipt)
        frame=encode_task_response(before,vocabulary=vocabulary,scenario=scenario,completed_decisions=i,
                                   actor_schema=actor_schema)
        index=frame['candidate_ids'].index(choice['candidate_id'])
        outcome=after['outcome'] if after is not None else receipt['outcome']
        if type(outcome.get('terminated')) is not bool or type(outcome.get('truncated')) is not bool:
            raise ValueError('native outcome missing explicit flags')
        # An official result is not invented for this task-specific diagnostic.
        if public_skill:
            step=evaluate_public_skill_interval(scenario,before,record['public_witnesses'],decision_count=i+1,
                sources=public_continuity_sources,next_response=after,action_receipt=receipt)
            if after is None and not step['terminated']:
                raise ValueError('nonterminal public interval requires actual same-player successor for bootstrap')
        else:
            step=classify(scenario,before['observation'],after['observation'],decision_count=i+1,
                               official_game_ended=outcome['terminated'])
        if not step['episode_valid'] or step['reward'] is None:raise ValueError('invalid native task data')
        if (step['terminated'] or step['truncated']) and i!=len(records)-1:
            raise ValueError('history extends after task end')
        if outcome['truncated'] and not collection_cut and not (step['terminated'] or step['truncated']):
            raise ValueError('host limit precedes declared task horizon')
        frames.append(frame);actions.append(index);assignments.append([assignment]);task_steps.append(step)
        end_turn=(step['public_continuity']['terminal_turn'] if public_skill and step['terminated']
                  else after['observation']['turn'])
        elapsed=end_turn-before['observation']['turn']
        if elapsed<0:raise ValueError('same-player successor goes backwards in game time')
        step_discounts.append(discount if discount_unit=='player_decision' else discount**elapsed)
        old_logps.append(choice['log_prob']);old_values.append(choice['value']);previous=after
    if not (task_steps[-1]['terminated'] or task_steps[-1]['truncated']):
        if not collection_cut:raise ValueError('diagnostic update needs a closed task or explicit horizon')
        task_steps[-1]=dict(task_steps[-1],status='truncated',truncated=True,reason='explicit_global_collection_boundary')
    batches=[collate([frame],device) for frame in frames]
    actions=torch.tensor(actions,dtype=torch.long,device=device)[:,None]
    old_logps=torch.tensor(old_logps,device=device)[:,None]
    old_values=torch.tensor(old_values,device=device)[:,None]
    values=[];logps=[];history=[];state=None
    with torch.no_grad():
        for i,batch in enumerate(batches):
            logp,value,state,_=model.evaluate_actions(batch,actions[i],assignments[i],state)
            logps.append(logp);values.append(value);history.append(tuple(x.detach().clone() for x in state))
        replay_logps=torch.stack(logps);values=torch.stack(values)
        if not torch.isfinite(old_logps).all() or not torch.isfinite(old_values).all():
            raise ValueError('nonfinite behavior statistics')
        if not torch.allclose(replay_logps,old_logps,atol=1e-4,rtol=1e-5) or not torch.allclose(values,old_values,atol=1e-4,rtol=1e-5):
            raise ValueError('replayed recurrent behavior differs from actor')
        actor_value_error=float((values-old_values).abs().max())
        if private_frames is not None:
            private_values=torch.tensor([f['values'] for f in private_frames],dtype=values.dtype,device=device)
            private_masks=torch.tensor([f['known'] for f in private_frames],dtype=torch.bool,device=device)
            history_hidden=torch.cat([h[0] for h in history]);history_cell=torch.cat([h[1] for h in history])
            values=critic(private_values,private_masks,history_hidden,history_cell)[:,None]
        if task_steps[-1]['terminated']:
            bootstrap=torch.zeros_like(values[-1])
            if mode=='privileged_diagnostic' and privileged_snapshots[-1] is not None:
                # A task may succeed while the native game still continues.
                # Validate any available successor, but NEVER bootstrap it.
                if previous is None:
                    raise ValueError('terminal without actual successor cannot receive a fabricated private successor')
                encode_privileged_snapshot(privileged_snapshots[-1],previous,expected_context=privileged_context)
        else:
            last_frame=encode_task_response(previous,vocabulary=vocabulary,scenario=scenario,
                                            completed_decisions=len(records),allow_horizon=True,
                                            actor_schema=actor_schema)
            _,bootstrap,last_state=model(collate([last_frame],device,value_only=True),state,value_only=True)
            if private_frames is not None:
                final_private=(encode_privileged_snapshot(privileged_snapshots[-1],previous,expected_context=privileged_context)
                               if mode=='privileged_diagnostic' else empty_private)
                bootstrap=critic(torch.tensor([final_private['values']],dtype=values.dtype,device=device),
                                 torch.tensor([final_private['known']],dtype=torch.bool,device=device),*last_state)
        next_values=torch.cat((values[1:],bootstrap[None]))
        rewards=torch.tensor([[s['reward']] for s in task_steps],device=device)
        terminated=torch.tensor([[s['terminated']] for s in task_steps],dtype=torch.bool,device=device)
        truncated=torch.tensor([[s['truncated']] for s in task_steps],dtype=torch.bool,device=device)
        advantages,returns=generalized_advantages(rewards,values,next_values,terminated,truncated,
            torch.tensor(step_discounts,dtype=values.dtype,device=device)[:,None])
    ppo_group=dict(batches=batches[burn_in:],actions=actions[burn_in:],old_log_probs=old_logps[burn_in:],
        advantages=advantages[burn_in:],returns=returns[burn_in:],assignments=assignments[burn_in:],
        burn_in_batches=batches[:burn_in])
    critic_batch=None if private_frames is None else dict(values=private_values[burn_in:],masks=private_masks[burn_in:],
        hidden=history_hidden[burn_in:],cell=history_cell[burn_in:],returns=returns[burn_in:,0])
    report=dict(training_eligible=False,strategic_training_ready=False,mode='diagnostic_native',
        records=len(records),burn_in=burn_in,optimized_decisions=len(records)-burn_in,scenario_sha256=task_digest,
        behavior_log_prob_max_error=float((replay_logps-old_logps).abs().max()),behavior_value_max_error=actor_value_error,
        bootstrap_value=float(bootstrap[0]),critic_mode=mode,previous_critic_version=expected_critic_version,
        critic_version=privileged_state['version'] if privileged_state is not None else None,
        critic_history='frozen_behavior_actor_hidden_cell' if private_frames is not None else None,task_steps=task_steps,
        discount_unit=discount_unit,discounts=step_discounts)
    return dict(ppo_group=ppo_group,critic_batch=critic_batch,report=report)


def _fit_critic(prepared, privileged_state, epochs, mode):
    if privileged_state is None:return {}
    from .privileged_critic import update_critic,critic_version
    rows=[p['critic_batch'] for p in prepared]
    joined={key:torch.cat([r[key] for r in rows]) for key in ('values','masks','hidden','cell','returns')}
    for _ in range(epochs):
        metrics=update_critic(privileged_state['critic'],privileged_state['optimizer'],joined['values'],joined['masks'],
            joined['hidden'],joined['cell'],joined['returns'])
    privileged_state['version']=critic_version(privileged_state['critic'])
    result=dict(separate_value_loss=metrics['value_loss'],separate_grad_norm=metrics['grad_norm'])
    if mode=='privileged_diagnostic':
        result.update(privileged_value_loss=metrics['value_loss'],privileged_grad_norm=metrics['grad_norm'])
    return result


def _validate_excluded_budget_tail(row, row_index, own_length):
    """Accept only the exact unresolved final action emitted by finish_budget().

    Such a row remains in the collection/audit but cannot contribute a reward,
    successor value, or GAE step. In particular, task endpoints and arbitrary
    missing observations are not eligible for this exclusion.
    """
    receipt = row.get('action_receipt')
    outcome = receipt.get('outcome') if type(receipt) is dict else None
    if (row_index != own_length - 1
            or row.get('truncated') is not True
            or row.get('terminated') is not False
            or row.get('truncation_reason') != 'diagnostic_budget'
            or row.get('next_response') is not None
            or row.get('encoded_next') is not None
            or row.get('bootstrap_available') is not False
            or row.get('public_successor_available') is not False
            or row.get('terminal_result') is not None
            or row.get('terminal_observation') is not None
            or row.get('reward') is not None
            or row.get('training_eligible') is not False
            or type(outcome) is not dict
            or outcome.get('terminated') is not False
            or outcome.get('truncated') is not False):
        raise ValueError('only an exact final diagnostic-budget tail may be excluded')


def update_native_diagnostic(model,optimizer,vocabulary,records,scenario,*,sample_audit,
                             expected_policy_version,burn_in=0,epochs=2,discount=.99,
                             privileged_state=None,privileged_snapshots=None,
                             privileged_context=None,expected_critic_version=None,critic_mode=None):
    prepared=_prepare_native_diagnostic(model,optimizer,vocabulary,records,scenario,sample_audit=sample_audit,
        expected_policy_version=expected_policy_version,burn_in=burn_in,epochs=epochs,discount=discount,
        privileged_state=privileged_state,privileged_snapshots=privileged_snapshots,privileged_context=privileged_context,
        expected_critic_version=expected_critic_version,critic_mode=critic_mode)
    group=prepared['ppo_group']
    metrics=update(model,optimizer,**group,epochs=epochs,value_weight=0. if privileged_state is not None else .5)
    metrics.update(_fit_critic([prepared],privileged_state,epochs,prepared['report']['critic_mode']))
    report=prepared['report'];report['metrics']=metrics
    if privileged_state is not None:report['critic_version']=privileged_state['version']
    return report


def update_multiplayer_diagnostic(model,optimizer,vocabulary,records,scenarios,collection,*,sample_audit,
        expected_policy_version,epochs=2,discount=.99,privileged_state=None,privileged_snapshots=None,
        privileged_context=None,expected_critic_version=None,critic_mode=None,diagnostic_collection=None):
    """One coordinated update from a complete, bounded global native journal.

    The immediate receipt proves the action. Only the next observation owned
    by that player supplies its reward/bootstrap. Unresolved tails are reported
    and excluded, never zero-filled or treated as losses. No global victory or
    elimination contract is inferred here. Interval-sensitive skill rewards
    require pinned public witnesses at every intervening action boundary.

    Optimizer execution is not transactional. The worker must invalidate the
    runtime on any exception and never publish a checkpoint from a partial
    actor/critic update. Input validation precedes either optimizer step.
    """
    from .multiplayer_trajectory import MultiPlayerDiagnosticTrajectory
    from .ppo import update_grouped
    from .critic_modes import mode_of
    import hashlib,json
    global_limit,player_limit=32,12
    sequence_limit=63
    extended_budget=False
    if diagnostic_collection is not None:
        from .diagnostic_budget998 import validate_collection,is_extended
        declared=validate_collection(diagnostic_collection)
        global_limit=player_limit=declared['max_decisions']
        extended_budget=is_extended(declared)
        if extended_budget:sequence_limit=declared['max_decisions']
    if type(records) is not list or not 2<=len(records)<=global_limit:
        raise ValueError('multiplayer diagnostic exceeds declared global decision bound')
    allow_single=(type(collection) is dict and collection.get('allow_single_public_skill_group') is True)
    if type(scenarios) is not list or not (1 if allow_single else 2)<=len(scenarios)<=4:
        raise ValueError('declare bounded player scenarios (one requires explicit public skill opt-in)')
    by_player={}
    for row in scenarios:
        if (type(row) is not dict or set(row)!={'player_id','scenario','burn_in'}
                or type(row['player_id']) is not int or not 0<=row['player_id']<=63
                or row['player_id'] in by_player or type(row['burn_in']) is not int or row['burn_in']<0):
            raise ValueError('invalid per-player scenario declaration')
        scenario_digest(row['scenario']);by_player[row['player_id']]=row
    collection_fields={'schema','stop_reason','last_sequence','discount_unit'}
    if (type(collection) is not dict or set(collection) not in
            (collection_fields,collection_fields|{'public_continuity_sources'},
             collection_fields|{'public_continuity_sources','allow_single_public_skill_group'})
            or collection['schema']!='civ6rl.multiplayer_collection997.v1'
            or collection['stop_reason'] not in ('caller_cap','native_cap')
            or type(collection['last_sequence']) is not int
            or collection['discount_unit'] not in ('player_decision','native_turn')):
        raise ValueError('explicit collection boundary/discount convention required')
    public_sources=collection.get('public_continuity_sources')
    if 'public_continuity_sources' in collection:
        from .skill_task991 import validate_public_sources
        validate_public_sources(public_sources)
    if 'allow_single_public_skill_group' in collection:
        from .skill_context993 import SCHEMA as SKILL_SCHEMA
        if (not allow_single or diagnostic_collection is None or public_sources is None
                or any(row['scenario'].get('schema')!=SKILL_SCHEMA for row in scenarios)):
            raise ValueError('single group requires explicit diagnostic budget and public skill continuity')
    last=records[-1]['after']
    if (last['sequence']!=collection['last_sequence'] or last['outcome']['terminated']
            or last['outcome']['truncated'] is not (collection['stop_reason']=='native_cap')):
        raise ValueError('collection boundary differs from native journal')
    observed_keys=set()
    trajectory=MultiPlayerDiagnosticTrajectory(policy_identity={'policy_id':actor_schema_for_config(model.config),'version':expected_policy_version},
        vocabulary=vocabulary,player_ids=tuple(by_player),public_continuity_sources=public_sources,
        actor_schema=actor_schema_for_config(model.config))
    for raw in records:
        raw_fields={'before','choice','after'}|({'public_witnesses'} if public_sources is not None else set())
        if type(raw) is not dict or set(raw)!=raw_fields:raise ValueError('unknown global receipt fields')
        before,choice,after=raw['before'],raw['choice'],raw['after']
        key=(before['episode_id'],before['game_generation'],before['observation']['player_id'],before['sequence'])
        if key in observed_keys or sample_audit.get(key)!={'before':before,'choice':choice}:
            raise ValueError('global journal differs from actual worker decisions')
        observed_keys.add(key)
        if after['outcome']['terminated']:raise ValueError('global result mapping is not qualified for task learning')
        trajectory.append(before,choice,after,public_witnesses=raw.get('public_witnesses'))
    if set(sample_audit)!=observed_keys:
        raise ValueError('all outstanding sampled decisions must be closed by the coordinated update')
    resolved=trajectory.finish_budget();mode=mode_of(model)
    # Privileged data are joined against the original global sequence, then
    # sliced per player. No opponent snapshot substitutes for a missing tail.
    private_lookup={}
    if mode=='privileged_diagnostic':
        from .privileged_encoding import encode_privileged_snapshot
        frames=[r['before'] for r in records]+[last]
        if type(privileged_snapshots) is not list or len(privileged_snapshots)!=len(frames):
            raise ValueError('global privileged sidecar must cover every boundary')
        for snapshot,frame in zip(privileged_snapshots,frames):
            encode_privileged_snapshot(snapshot,frame,expected_context=privileged_context)
            private_lookup[(frame['observation']['player_id'],frame['sequence'])]=snapshot
    elif privileged_snapshots is not None or privileged_context is not None:
        raise ValueError('non-privileged mode rejects sidecars')
    prepared=[];reports=[];excluded=[];excluded_groups=[]
    for pid,declaration in by_player.items():
        own=[r for r in resolved if r['player_id']==pid]
        usable=[]
        from .skill_context993 import SCHEMA as SKILL_SCHEMA,evaluate_public_skill_interval
        public_skill=public_sources is not None and declaration['scenario'].get('schema')==SKILL_SCHEMA
        for row_index,row in enumerate(own):
            public_terminal=False
            if public_skill:
                interval=evaluate_public_skill_interval(declaration['scenario'],row['before_response'],
                    row['public_witnesses'],decision_count=row_index+1,sources=public_sources,
                    next_response=row['next_response'],action_receipt=row['action_receipt'])
                if not interval.get('episode_valid') or interval.get('reward') is None:
                    raise ValueError('invalid public skill interval cannot be excluded or learned')
                public_terminal=interval['terminated']
            if row['next_response'] is None and not public_terminal:
                if row_index != len(own)-1:
                    raise ValueError('only an unresolved tail may be excluded')
                if not usable:
                    _validate_excluded_budget_tail(row,row_index,len(own))
                excluded.append(dict(player_id=pid,sequence=row['sequence'],reason='missing_same_player_successor'))
            else:
                usable.append(dict(before=row['before_response'],choice=row['choice'],
                    action_receipt=row['action_receipt'],next_response=row['next_response']))
                if public_skill:usable[-1]['public_witnesses']=row['public_witnesses']
        if not usable:
            if own:
                # The only permissible all-tail group is one sampled action
                # that remains pending at the explicit collection boundary.
                # Its raw receipt stays in the worker audit and the resolved
                # row stays in `excluded_tails`; this receipt makes the group
                # level optimizer exclusion explicit as well.
                if len(own)!=1:
                    raise ValueError('empty player group is not a single final budget tail')
                _validate_excluded_budget_tail(own[0],0,1)
                excluded_groups.append(dict(player_id=pid,sequence=own[0]['sequence'],
                    reason='no_usable_transition_budget_tail'))
            continue
        snapshots=None
        if mode=='privileged_diagnostic':
            snapshots=[private_lookup[(pid,r['before']['sequence'])] for r in usable]
            successor=usable[-1]['next_response']
            snapshots.append(private_lookup[(pid,successor['sequence'])] if successor is not None else None)
        item=_prepare_native_diagnostic(model,optimizer,vocabulary,usable,declaration['scenario'],sample_audit=sample_audit,
            expected_policy_version=expected_policy_version,burn_in=declaration['burn_in'],epochs=epochs,discount=discount,
            privileged_state=privileged_state,privileged_snapshots=snapshots,privileged_context=privileged_context,
            expected_critic_version=expected_critic_version,critic_mode=critic_mode,
            delayed=True,collection_cut=True,discount_unit=collection['discount_unit'],decision_limit=player_limit,
            public_continuity_sources=public_sources,sequence_limit=sequence_limit)
        item['report'].update(player_id=pid,global_sequences=[r['before']['sequence'] for r in usable],
            successor_sequences=[r['next_response']['sequence'] if r['next_response'] is not None else None for r in usable])
        prepared.append(item);reports.append(item['report'])
    if len(prepared)<(1 if allow_single else 2):
        raise ValueError('coordinated diagnostic requires at least two usable players unless public skill opt-in is explicit')
    # All groups, native receipts, reward targets and behavior probabilities
    # have been checked before either optimizer is allowed to mutate weights.
    grouped_options=dict(epochs=epochs,value_weight=0. if privileged_state is not None else .5)
    if extended_budget:grouped_options['maximum_steps']=declared['max_decisions']
    metrics=update_grouped(model,optimizer,[p['ppo_group'] for p in prepared],**grouped_options)
    metrics.update(_fit_critic(prepared,privileged_state,epochs,mode))
    digest=hashlib.sha256(json.dumps({'scenarios':scenarios,'collection':collection},sort_keys=True,
        ensure_ascii=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
    return dict(mode='diagnostic_native_singleplayer_skill' if len(prepared)==1 else 'diagnostic_native_multiplayer',
        training_eligible=False,strategic_training_ready=False,
        records=len(records),optimized_decisions=sum(r['optimized_decisions'] for r in reports),
        player_groups=reports,excluded_tails=excluded,excluded_player_groups=excluded_groups,
        scenario_sha256=digest,collection=copy.deepcopy(collection),
        behavior_log_prob_max_error=max(r['behavior_log_prob_max_error'] for r in reports),
        behavior_value_max_error=max(r['behavior_value_max_error'] for r in reports),
        critic_mode=mode,previous_critic_version=expected_critic_version,
        critic_version=privileged_state['version'] if privileged_state is not None else None,metrics=metrics,
        diagnostic_collection=copy.deepcopy(diagnostic_collection))
