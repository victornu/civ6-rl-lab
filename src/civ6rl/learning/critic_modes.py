"""Explicit information contract for the three value-estimation baselines."""
MODES=('player_only','history_only_diagnostic','privileged_diagnostic')
SEPARATE_MODES=('history_only_diagnostic','privileged_diagnostic')


def mode_of(model):
    state=getattr(model,'_privileged_state',None)
    # Compatibility for in-memory synthetic fixtures predating an explicit
    # mode. Persisted checkpoints always carry a declared mode/version.
    inferred='privileged_diagnostic' if state is not None else 'player_only'
    mode=getattr(model,'_critic_mode',inferred)
    if mode not in MODES:
        raise ValueError('unknown critic mode')
    if (mode=='player_only') != (state is None):
        raise ValueError('critic mode and separate state disagree')
    if state is not None and (type(state) is not dict or set(state)!={'critic','optimizer','version'}):
        raise ValueError('invalid separate critic state')
    return mode
