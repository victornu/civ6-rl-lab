"""Player-facing trade features and exact joins to native action catalogues.

This projects the official UI's formatting, not any trade-yield or path rule.
The six-yield profile is explicit; other yield layouts are refused.
"""
import json
import math

YIELDS=('food','production','gold','science','culture','faith')
_BASE=('trade_destination_owner','trade_destination_self','trade_has_post',
       'trade_repeat_route','trade_origin_path_bonus','trade_path_steps')
TRADE_FEATURES=tuple(x for name in _BASE for x in (name,name+'_known'))+tuple(
    x for side in ('origin','destination') for name in YIELDS for x in
    (f'trade_{side}_{name}',f'trade_{side}_{name}_known',
     f'trade_{side}_{name}_icon',f'trade_{side}_{name}_icon_known'))+tuple(
    x for side in ('origin','destination') for name in ('religion','pressure')
    for x in (f'trade_{side}_{name}',f'trade_{side}_{name}_known'))


def ui_yield(value):
    """SupportFunctions.Round(value, 1), called by TradeRouteChooser."""
    if type(value) not in (int,float) or not math.isfinite(value) or abs(value)>=1e12:
        raise ValueError('Invalid trade yield for public formatting')
    return math.floor(value*10+0.5)/10


def route_features(destination,player_id):
    values=[0.0]*len(TRADE_FEATURES)
    def put(name,value):
        i=TRADE_FEATURES.index(name);values[i]=float(value);values[i+1]=1.0
    put('trade_destination_owner',destination['owner'])
    put('trade_destination_self',destination['owner']==player_id)
    put('trade_has_post',destination['has_post'])
    put('trade_repeat_route',destination['repeat_route'])
    put('trade_origin_path_bonus',destination['origin_yields']['path_bonus'])
    # The path is published by the route lens. This is its edge count, not
    # Map.GetPlotDistance or turns-to-completion, neither of which is inferred.
    put('trade_path_steps',len(destination['path'])-1)
    for side in ('origin','destination'):
        quote=destination[side+'_yields']
        if len(quote['values'])!=6:
            raise ValueError('Trade feature mapping requires the pinned six-yield profile')
        for key,raw in zip(YIELDS,quote['values']):
            put(f'trade_{side}_{key}',ui_yield(raw))
            # Nonzero values rounded to zero still have an icon in the UI.
            put(f'trade_{side}_{key}_icon',raw!=0)
        if quote['majority_religion']>0 and quote['religion_pressure']>0:
            put(f'trade_{side}_religion',quote['majority_religion'])
            put(f'trade_{side}_pressure',quote['religion_pressure'])
    return values


def origin_features(player_id):
    values=[0.0]*len(TRADE_FEATURES)
    i=TRADE_FEATURES.index('trade_destination_owner');values[i:i+2]=[float(player_id),1.0]
    i=TRADE_FEATURES.index('trade_destination_self');values[i:i+2]=[1.0,1.0]
    return values


def _key(action):
    return json.dumps(action,sort_keys=True,separators=(',',':'),allow_nan=False)


def validate_and_index(decisions,observation,*,ended=False):
    """No filtering: all native confirmed trade choices must have candidates.

    Closed wrappers retain the public catalogue but must offer no actions.
    Own-city IDs serve only as join keys. Foreign route destinations come from
    the official chooser, without inferring unseen foreign-city state.
    """
    from .trade_contract984 import validate_routes,validate_origins
    if type(ended) is not bool:
        raise ValueError('Trade ended flag must be boolean')
    offered={}
    def add(action,source,target,features):
        key=_key(action)
        if key in offered:raise ValueError('Duplicate native trade choice')
        offered[key]=dict(source=tuple(source),target=tuple(target),features=features)
    if 'trade_routes' in decisions:
        for action,trader,destination in validate_routes(decisions['trade_routes'],observation):
            add(action,(trader['x'],trader['y']),(destination['x'],destination['y']),
                route_features(destination,observation['player_id']))
    if 'trade_origins' in decisions:
        for action,unit,target in validate_origins(decisions['trade_origins'],observation):
            add(action,(unit['x'],unit['y']),(target['x'],target['y']),origin_features(observation['player_id']))
    candidates=decisions.get('candidates')
    if type(candidates) is not list:raise ValueError('Trade candidates must be a list')
    if ended:
        if candidates:raise ValueError('Closed trade decision still offers actions')
        return {}
    indexed={};seen=set()
    for row in candidates:
        if type(row) is not dict or type(row.get('action')) is not dict:
            raise ValueError('Malformed candidate in trade decision')
        action=row['action']
        if action.get('kind') not in ('make_trade_route','change_trade_origin'):continue
        if set(row)!={'candidate_id','action'} or type(row['candidate_id']) is not str or not row['candidate_id']:
            raise ValueError('Malformed trade candidate identity')
        key=_key(action)
        if key not in offered or key in seen or row['candidate_id'] in indexed:
            raise ValueError('Trade candidate differs from native catalogue or duplicates a choice')
        seen.add(key);indexed[row['candidate_id']]=offered[key]
    if seen!=set(offered):raise ValueError('Trade candidates do not exactly cover the published choices')
    return indexed
