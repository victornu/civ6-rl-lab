"""Persistent bounded learner budgets; no scenario or strategic admission."""
import copy
import hashlib
import json
import re

SCHEMA='civ6rl.learner_campaign.v1'
STATE_SCHEMA='civ6rl.learner_campaign_state.v1'
COMMIT_SCHEMA='civ6rl.learner_campaign_commit.v1'
MAX_UPDATES=1024
MAX_GLOBAL_UPDATES=4096
MAX_GLOBAL_CLOSED_EPISODES=8192
MAX_CAMPAIGN_UPDATE_SECONDS=600
CAMPAIGN_UPDATE_BASE_SECONDS=60
CAMPAIGN_UPDATE_SECONDS_PER_DECISION=2
SUCCESSION_SCHEMA='civ6rl.learner_campaign_succession.v1'
SUCCESSION_PROFILE='completed_campaign_successor_v1'
SUCCESSION_PRESERVED=dict(actor_tensors=True,actor_adam=True,critic=True,critic_adam=True,
    rng=True,global_updates=True,closed_episode_ledger=True,last_update_context=True,
    collector_empty=True)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
        ensure_ascii=True,allow_nan=False).encode()).hexdigest()


def sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def integer(value,low,high):
    return type(value) is int and low<=value<=high


def update_timeout_seconds(decisions):
    """Bound one complete campaign PPO update, including checkpointed backward."""
    if not integer(decisions,1,256):
        raise ValueError('campaign update decision count is outside the reviewed bound')
    return min(MAX_CAMPAIGN_UPDATE_SECONDS,
        CAMPAIGN_UPDATE_BASE_SECONDS+CAMPAIGN_UPDATE_SECONDS_PER_DECISION*decisions)


def campaign_session_seconds(episode_budget,contract):
    """Bound native collection wall plus one explicit successor update reserve."""
    from .diagnostic_budget998 import validate_collection,is_extended
    budget=validate_collection(episode_budget)
    validate_contract(contract)
    if not is_extended(budget):
        raise ValueError('campaign requires an extended episode wall budget')
    return budget['wall_seconds']+MAX_CAMPAIGN_UPDATE_SECONDS


def validate_contract(value):
    fields={'schema','campaign_id','origin_learner_sha256','max_updates','max_decisions','max_epochs_per_update'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!=SCHEMA
            or type(value['campaign_id']) is not str or re.fullmatch('[a-z0-9][a-z0-9-]{0,63}',value['campaign_id']) is None
            or not sha(value['origin_learner_sha256']) or not integer(value['max_updates'],1,MAX_UPDATES)
            or not integer(value['max_decisions'],2,value['max_updates']*256)
            or not integer(value['max_epochs_per_update'],1,4)):
        raise ValueError('invalid explicit learner campaign contract')
    return copy.deepcopy(value)


def identities(rows,maximum):
    if (type(rows) not in (list,tuple,set) or not 1<=len(rows)<=maximum
            or any(type(r) not in (list,tuple) or len(r)!=2 or any(type(v) is not str or not 0<len(v)<=256 for v in r) for r in rows)
            or len(set(map(tuple,rows)))!=len(rows)):
        raise ValueError('invalid campaign closed-episode ledger')
    return sorted(map(tuple,rows))


def _validate_succession_receipt(receipt,contract,origin_updates,origin_closed_episodes):
    fields={'schema','profile','parent_learner_sha256','parent_policy_sha256','parent_critic_sha256',
        'previous_contract','previous_contract_sha256','previous_state_sha256','previous_summary',
        'global_completed_updates','closed_episodes_sha256','new_contract','new_contract_sha256',
        'source_transition','preserved','retained_trajectories_reusable'}
    if type(receipt) is not dict or set(receipt)!=fields or receipt.get('schema')!=SUCCESSION_SCHEMA or receipt.get('profile')!=SUCCESSION_PROFILE:
        raise ValueError('invalid completed campaign succession receipt')
    previous=validate_contract(receipt['previous_contract']);target=validate_contract(receipt['new_contract'])
    previous_summary=validate_summary(receipt['previous_summary'],previous)
    if (not sha(receipt['parent_learner_sha256']) or not sha(receipt['parent_policy_sha256'])
            or not sha(receipt['parent_critic_sha256']) or not sha(receipt['previous_state_sha256'])
            or receipt['previous_contract_sha256']!=digest(previous)
            or receipt['new_contract_sha256']!=digest(target) or target!=contract
            or previous['campaign_id']==target['campaign_id']
            or target['origin_learner_sha256']!=receipt['parent_learner_sha256']
            or (previous_summary['remaining_updates']>0 and previous_summary['remaining_decisions']>=2)
            or type(receipt['global_completed_updates']) is not int
            or receipt['global_completed_updates']!=origin_updates
            or receipt['global_completed_updates']!=previous_summary['origin_updates']+previous_summary['completed_updates']
            or not sha(receipt['closed_episodes_sha256'])
            or receipt['closed_episodes_sha256']!=digest([list(x) for x in identities(origin_closed_episodes,MAX_GLOBAL_CLOSED_EPISODES)])
            or receipt['preserved']!=SUCCESSION_PRESERVED
            or receipt['retained_trajectories_reusable'] is not False):
        raise ValueError('completed campaign succession receipt differs from its source or target')
    transition=receipt['source_transition']
    if (type(transition) is not dict
            or set(transition)!={'profile','source_checkpoint_sha256','sources','supporting_sources','model_migration','optimizer_migration'}
            or transition.get('profile')!=SUCCESSION_PROFILE
            or transition.get('source_checkpoint_sha256')!=receipt['parent_learner_sha256']
            or type(transition.get('sources')) is not dict or not transition['sources']
            or transition.get('model_migration') is not False
            or transition.get('optimizer_migration') is not False):
        raise ValueError('campaign succession source-only transition differs')
    for name,row in transition['sources'].items():
        if (type(name) is not str or type(row) is not dict or set(row)!={'previous','current'}
                or not sha(row['previous']) or not sha(row['current']) or row['previous']==row['current']):
            raise ValueError('invalid campaign succession source pin')
    supporting=transition['supporting_sources']
    if type(supporting) is not dict or set(supporting)!={'civ6rl/learning/transport.py','civ6rl/learning/wire.py'}:
        raise ValueError('campaign succession supporting source set differs')
    for name,row in supporting.items():
        if (type(row) is not dict or set(row)!={'previous','current'}
                or not sha(row['previous']) or not sha(row['current'])):
            raise ValueError('invalid campaign succession supporting source pin')
    return copy.deepcopy(receipt)


def validate_state(value,*,total_updates=None,closed_episodes=None):
    fields={'schema','contract','origin_updates','origin_closed_episodes','receipts'}
    if type(value) is dict and set(value)==fields|{'succession'}:
        has_succession=True
    elif type(value) is dict and set(value)==fields:
        has_succession=False
    else:
        raise ValueError('invalid learner campaign state')
    if value['schema']!=STATE_SCHEMA:
        raise ValueError('invalid learner campaign state')
    contract=validate_contract(value['contract'])
    if not integer(value['origin_updates'],1,MAX_GLOBAL_UPDATES):raise ValueError('invalid campaign origin counter')
    original=identities(value['origin_closed_episodes'],MAX_GLOBAL_CLOSED_EPISODES)
    receipts=value['receipts']
    if type(receipts) is not list or len(receipts)>contract['max_updates']:
        raise ValueError('campaign update budget exceeded')
    fields={'parent_learner_sha256','game_generation','episode_id','global_decisions',
        'optimized_decisions','epochs','records_sha256','scenario_sha256','parent_policy_version',
        'policy_version','parent_critic_version','critic_version'}
    closed=set(original);decisions=0
    for index,r in enumerate(receipts):
        if type(r) is not dict or set(r)!=fields:raise ValueError('invalid campaign receipt')
        for name in ('parent_learner_sha256','records_sha256','scenario_sha256','parent_policy_version','policy_version'):
            if not sha(r[name]):raise ValueError('invalid campaign receipt hash')
        if (r['parent_critic_version'] is None)!=(r['critic_version'] is None):raise ValueError('campaign critic identity differs')
        if r['critic_version'] is not None and (not sha(r['parent_critic_version']) or not sha(r['critic_version'])):
            raise ValueError('invalid campaign critic hash')
        if (not integer(r['global_decisions'],2,256) or not integer(r['optimized_decisions'],1,r['global_decisions'])
                or not integer(r['epochs'],1,contract['max_epochs_per_update'])):
            raise ValueError('invalid campaign receipt counts')
        episode=identities([(r['game_generation'],r['episode_id'])],1)[0]
        if episode in closed:raise ValueError('campaign episode already closed')
        closed.add(episode);decisions+=r['global_decisions']
        if index and (r['parent_policy_version']!=receipts[index-1]['policy_version']
                or r['parent_critic_version']!=receipts[index-1]['critic_version']):
            raise ValueError('campaign policy or critic chain differs')
    if decisions>contract['max_decisions']:raise ValueError('campaign decision budget exceeded')
    if total_updates is not None and (type(total_updates) is not int or total_updates!=value['origin_updates']+len(receipts)):
        raise ValueError('campaign cumulative counter differs')
    if len(closed)>MAX_GLOBAL_CLOSED_EPISODES:
        raise ValueError('campaign closed-episode history exceeds its bound')
    if closed_episodes is not None and set(identities(closed_episodes,MAX_GLOBAL_CLOSED_EPISODES))!=closed:
        raise ValueError('campaign closed-episode ledger differs')
    if has_succession:
        succession=_validate_succession_receipt(value['succession'],contract,
            value['origin_updates'],original)
        if (succession['closed_episodes_sha256']!=digest([list(x) for x in original])
                or succession['global_completed_updates']!=value['origin_updates']):
            raise ValueError('campaign succession boundary differs from preserved history')
    return copy.deepcopy(value)


def initial_state(contract,origin_updates,closed_episodes):
    return validate_state(dict(schema=STATE_SCHEMA,contract=validate_contract(contract),
        origin_updates=origin_updates,origin_closed_episodes=[list(x) for x in identities(closed_episodes,MAX_GLOBAL_CLOSED_EPISODES)],receipts=[]))


def successor_receipt(state,*,parent_learner_sha256,parent_policy_sha256,parent_critic_sha256,
                      new_contract,source_transition,closed_episodes):
    """Build the only receipt that can start a successor at an exhausted boundary."""
    state=validate_state(state);new_contract=validate_contract(new_contract)
    old_summary=summary(state)
    if old_summary['remaining_updates']>0 and old_summary['remaining_decisions']>=2:
        raise ValueError('campaign still admits another update')
    if not state['receipts']:
        raise ValueError('campaign succession requires a completed campaign update')
    global_updates=state['origin_updates']+len(state['receipts'])
    if (global_updates>MAX_GLOBAL_UPDATES or new_contract['origin_learner_sha256']!=parent_learner_sha256
            or new_contract['campaign_id']==state['contract']['campaign_id']):
        raise ValueError('campaign successor must bind a distinct id to the exact parent learner')
    closed=identities(closed_episodes,MAX_GLOBAL_CLOSED_EPISODES)
    expected_closed=set(map(tuple,state['origin_closed_episodes']))|{
        (row['game_generation'],row['episode_id']) for row in state['receipts']}
    if set(closed)!=expected_closed:
        raise ValueError('campaign succession must preserve the complete closed-episode ledger')
    latest=state['receipts'][-1]
    if (latest['policy_version']!=parent_policy_sha256 or latest['critic_version']!=parent_critic_sha256):
        raise ValueError('campaign succession parent actor or critic differs from the final update')
    receipt=dict(schema=SUCCESSION_SCHEMA,profile=SUCCESSION_PROFILE,
        parent_learner_sha256=parent_learner_sha256,parent_policy_sha256=parent_policy_sha256,
        parent_critic_sha256=parent_critic_sha256,previous_contract=copy.deepcopy(state['contract']),
        previous_contract_sha256=digest(state['contract']),previous_state_sha256=digest(state),
        previous_summary=old_summary,global_completed_updates=global_updates,
        closed_episodes_sha256=digest([list(x) for x in closed]),new_contract=copy.deepcopy(new_contract),
        new_contract_sha256=digest(new_contract),source_transition=copy.deepcopy(source_transition),
        preserved=copy.deepcopy(SUCCESSION_PRESERVED),retained_trajectories_reusable=False)
    return _validate_succession_receipt(receipt,new_contract,global_updates,closed)


def succeed_campaign(state,*,receipt,closed_episodes):
    """Create a fresh per-campaign budget while retaining global learner history."""
    state=validate_state(state)
    if type(receipt) is not dict:
        raise ValueError('campaign succession receipt is malformed')
    receipt=_validate_succession_receipt(receipt,receipt.get('new_contract'),
        receipt.get('global_completed_updates'),closed_episodes)
    if (receipt['previous_state_sha256']!=digest(state)
            or receipt['previous_contract']!=state['contract']):
        raise ValueError('campaign succession receipt is stale or replayed')
    target=receipt['new_contract']
    successor=initial_state(target,receipt['global_completed_updates'],closed_episodes)
    successor['succession']=copy.deepcopy(receipt)
    return validate_state(successor)


def summary(state):
    state=validate_state(state);c=state['contract'];rows=state['receipts']
    decisions=sum(r['global_decisions'] for r in rows)
    return dict(campaign_id=c['campaign_id'],contract_sha256=digest(c),origin_updates=state['origin_updates'],
        completed_updates=len(rows),decisions=decisions,optimized_decisions=sum(r['optimized_decisions'] for r in rows),
        remaining_updates=c['max_updates']-len(rows),remaining_decisions=c['max_decisions']-decisions)


def validate_summary(value,contract):
    contract=validate_contract(contract)
    fields={'campaign_id','contract_sha256','origin_updates','completed_updates','decisions',
            'optimized_decisions','remaining_updates','remaining_decisions'}
    if type(value) is not dict or set(value)!=fields:raise ValueError('invalid campaign summary')
    if (value['campaign_id']!=contract['campaign_id'] or value['contract_sha256']!=digest(contract)
            or not integer(value['origin_updates'],1,MAX_GLOBAL_UPDATES)
            or not integer(value['completed_updates'],0,contract['max_updates'])
            or not integer(value['decisions'],2*value['completed_updates'],min(contract['max_decisions'],256*value['completed_updates']))
            or not integer(value['optimized_decisions'],value['completed_updates'],value['decisions'])
            or not integer(value['remaining_updates'],0,contract['max_updates'])
            or not integer(value['remaining_decisions'],0,contract['max_decisions'])
            or value['remaining_updates']!=contract['max_updates']-value['completed_updates']
            or value['remaining_decisions']!=contract['max_decisions']-value['decisions']):
        raise ValueError('campaign summary differs from contract')
    return copy.deepcopy(value)


def admit_episode(state,budget):
    from .diagnostic_budget998 import validate_collection,is_extended
    state=validate_state(state);budget=validate_collection(budget);s=summary(state)
    if not is_extended(budget):raise ValueError('campaign requires an explicit episode budget')
    if s['remaining_updates']<1 or budget['max_decisions']>s['remaining_decisions']:
        raise ValueError('campaign remaining budget cannot admit this episode')


def admit_update(state,records,epochs):
    state=validate_state(state);s=summary(state)
    if (s['remaining_updates']<1 or type(records) is not list or not 2<=len(records)<=min(256,s['remaining_decisions'])
            or not integer(epochs,1,state['contract']['max_epochs_per_update'])):
        raise ValueError('campaign update budget exhausted or exceeded')
    episodes={(r['before']['game_generation'],r['before']['episode_id']) for r in records}
    closed=set(map(tuple,state['origin_closed_episodes']))|{(r['game_generation'],r['episode_id']) for r in state['receipts']}
    if len(episodes)!=1 or episodes & closed:raise ValueError('campaign update requires one new episode')


def append_receipt(state,receipt):
    state=validate_state(state);state['receipts'].append(copy.deepcopy(receipt))
    return validate_state(state)


def validate_commit(value,*,learner=None,policy=None,campaign=None,parent=None):
    fields={'schema','learner','policy','campaign','parent_learner_sha256'}
    if type(value) is not dict or set(value)!=fields or value['schema']!=COMMIT_SCHEMA:
        raise ValueError('invalid campaign pair commit')
    for name,suffix in [('learner','.learner.pt'),('policy','.pt')]:
        r=value[name]
        if (type(r) is not dict or set(r)!={'path','sha256'} or not sha(r['sha256'])
                or r['path']!='learner-artifacts/'+r['sha256']+suffix):
            raise ValueError('invalid campaign commit artifact')
    if not sha(value['parent_learner_sha256']):raise ValueError('invalid campaign commit parent')
    for actual,expected in [(value['learner'],learner),(value['policy'],policy),(value['campaign'],campaign),(value['parent_learner_sha256'],parent)]:
        if expected is not None and actual!=expected:raise ValueError('campaign pair commit binding differs')
    return copy.deepcopy(value)
