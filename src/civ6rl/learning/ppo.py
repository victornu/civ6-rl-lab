"""Bounded recurrent PPO updates, with explicit termination and truncation.

Sequences stay ordered. Each batch column belongs to one player's trajectory;
turns of other players must never be interleaved into that player's memory.
Task rewards and native scenario qualification belong to the collector.
"""
from __future__ import annotations
import math
import torch


def _mps_checkpoint_context_fn():
    """Preserve CPU and MPS RNG across a non-reentrant recomputation.

    PyTorch's generic checkpoint RNG helper enters ``torch.mps.device`` while
    restoring state, but MPS exposes one logical device and its public API has
    no such context manager. Its get/set RNG functions need no device switch.
    """
    from contextlib import contextmanager, nullcontext
    cpu_forward_state=torch.get_rng_state().clone()
    mps_forward_state=torch.mps.get_rng_state().clone()

    @contextmanager
    def recompute_context():
        cpu_outer_state=torch.get_rng_state().clone()
        mps_outer_state=torch.mps.get_rng_state().clone()
        try:
            torch.set_rng_state(cpu_forward_state)
            torch.mps.set_rng_state(mps_forward_state)
            yield
        finally:
            torch.set_rng_state(cpu_outer_state)
            torch.mps.set_rng_state(mps_outer_state)

    return nullcontext(),recompute_context()


def generalized_advantages(rewards, values, next_values, terminated, truncated,
                           discounts, *, gae_lambda=0.95):
    """[time,batch]; next_values are from actual successors, before resets.

Truncation bootstraps the final observation, then breaks the GAE chain.
A true terminal has no bootstrap. A crashed/invalid rollout must be discarded
by the caller and cannot be represented as a terminal loss for the agent.
"""
    shape = rewards.shape
    if len(shape) != 2 or any(x.shape != shape for x in (values, next_values, terminated, truncated, discounts)):
        raise ValueError("GAE tensors must share [time,batch] shape")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise ValueError("termination/truncation must be Boolean")
    if not 0 <= gae_lambda <= 1 or ((discounts < 0) | (discounts > 1)).any():
        raise ValueError("invalid discount")
    if any(not torch.isfinite(x).all() for x in (rewards, values, next_values, discounts)):
        raise ValueError("nonfinite trajectory")
    with torch.no_grad():
        delta = rewards + discounts * (~terminated) * next_values - values
        advantages = torch.zeros_like(rewards)
        carry = torch.zeros_like(rewards[0])
        for t in reversed(range(shape[0])):
            carry = delta[t] + discounts[t] * gae_lambda * (~(terminated[t] | truncated[t])) * carry
            advantages[t] = carry
        return advantages, advantages + values


def _objective(logps, vals, entropy, old_log_probs, advantages, returns,
               *, clip, value_weight, entropy_weight):
    """One decision-weighted PPO objective shared by both update entry points."""
    log_ratio = logps - old_log_probs
    ratio = log_ratio.exp()
    approx_kl = ((ratio - 1) - log_ratio).mean()
    policy_loss = -torch.minimum(ratio * advantages,
        ratio.clamp(1 - clip, 1 + clip) * advantages).mean()
    # A separate critic must not backpropagate through the actor value head.
    value_loss = (vals - returns).square().mean() if value_weight else vals.new_zeros(())
    loss = policy_loss + value_weight * value_loss - entropy_weight * entropy
    return loss, policy_loss, value_loss, approx_kl, ratio


def update(model, optimizer, batches, actions, old_log_probs, advantages, returns,
           *, assignments=None, initial_state=None, burn_in_batches=None, resets=None, epochs=2, clip=0.2,
           value_weight=0.5, entropy_weight=0.01, max_grad_norm=0.5, target_kl=0.03):
    if not 1 <= epochs <= 16 or not 1 <= len(batches) <= 64:
        raise ValueError("unbounded or empty update")
    burn_in_batches = [] if burn_in_batches is None else burn_in_batches
    if len(burn_in_batches) + len(batches) > 64:
        raise ValueError("recurrent prefix and update exceed the sequence bound")
    shape = (len(batches), batches[0]["global"].shape[0])
    if not 1 <= shape[1] <= 64:
        raise ValueError("diagnostic PPO batch exceeds 64 sequences")
    if any(x.shape != shape for x in (actions, old_log_probs, advantages, returns)):
        raise ValueError("recurrent trajectory dimensions differ")
    if actions.dtype != torch.long:
        raise ValueError("actions must be integer indices")
    if resets is None:
        resets = torch.zeros(shape, dtype=torch.bool, device=actions.device)
    if resets.shape != shape or resets.dtype != torch.bool:
        raise ValueError("invalid recurrent reset boundaries")
    if any(not torch.isfinite(x).all() for x in (old_log_probs, advantages, returns)):
        raise ValueError("nonfinite PPO targets")
    if assignments is None:
        assignments = [[None] * shape[1] for _ in batches]
    if len(assignments) != shape[0] or any(len(row) != shape[1] for row in assignments):
        raise ValueError("joint assignments must share [time,batch] shape")
    for t, batch in enumerate(batches):
        if (actions[t] < 0).any() or (actions[t] >= batch["action_mask"].shape[1]).any():
            raise ValueError("action outside menu")
        if not batch["action_mask"].gather(1, actions[t, :, None]).all():
            raise ValueError("recorded action was illegal")
    old_log_probs, returns = old_log_probs.detach(), returns.detach()
    advantages = advantages.detach()
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    metrics = {}
    for epoch in range(epochs):
        state = None if initial_state is None else tuple(x.detach() for x in initial_state)
        # Recompute from the same player's history at EACH parameter version;
        # a hidden state saved under older weights is not a current prefix.
        with torch.no_grad():
            for prefix in burn_in_batches:
                if prefix['global'].shape[0] != shape[1]:
                    raise ValueError('burn-in player columns differ')
                _, _, state = model(prefix, state)
        if state is not None: state = tuple(x.detach() for x in state)
        logps, vals, entropies = [], [], []
        for t, batch in enumerate(batches):
            logp, value, state, joint_entropy = model.evaluate_actions(batch, actions[t], assignments[t], state, resets[t])
            logps.append(logp); vals.append(value); entropies.append(joint_entropy)
        logps, vals, entropy = torch.stack(logps), torch.stack(vals), torch.stack(entropies).mean()
        loss, policy_loss, value_loss, approx_kl, ratio = _objective(
            logps, vals, entropy, old_log_probs, advantages, returns,
            clip=clip, value_weight=value_weight, entropy_weight=entropy_weight)
        # Stop BEFORE another destructive update, using the common PPO
        # nonnegative sample estimator; never report it as an exact KL.
        if epoch and approx_kl.item() > target_kl:
            metrics["early_stop_kl"] = 1.0
            break
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        metrics = {"loss": float(loss.detach()), "policy_loss": float(policy_loss.detach()),
                   "value_loss": float(value_loss.detach()), "entropy": float(entropy.detach()),
                   "kl": float(approx_kl.detach()), "grad_norm": float(grad_norm.detach()),
                   "clip_fraction": float(((ratio - 1).abs() > clip).float().mean().detach()),
                   "epochs_completed": float(epoch + 1), "early_stop_kl": 0.0}
    return metrics


def _grouped_targets(groups, maximum_steps=64):
    """Validate every sequence before scoring or making an optimizer step."""
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= 256:
        raise ValueError('grouped recurrent maximum_steps must be an integer in [1, 256]')
    required = {'batches', 'actions', 'old_log_probs', 'advantages', 'returns'}
    optional = {'assignments', 'initial_state', 'burn_in_batches', 'resets'}
    if type(groups) not in (list, tuple) or not 1 <= len(groups) <= 64:
        raise ValueError('grouped PPO requires between 1 and 64 sequence groups')
    prepared = []
    total_steps = total_columns = 0
    common_device = None
    for group in groups:
        if type(group) is not dict or not required <= set(group) or set(group) - required - optional:
            raise ValueError('unknown or missing grouped PPO fields')
        batches = group['batches']
        prefix = group.get('burn_in_batches')
        prefix = [] if prefix is None else prefix
        if (type(batches) not in (list, tuple) or not 1 <= len(batches) <= maximum_steps
                or type(prefix) not in (list, tuple)):
            raise ValueError('invalid grouped recurrent sequence')
        total_steps += len(batches) + len(prefix)
        if total_steps > maximum_steps:
            raise ValueError(f'grouped prefixes and optimized steps exceed {maximum_steps}')
        first = batches[0]
        if (type(first) is not dict or not isinstance(first.get('global'), torch.Tensor)
                or first['global'].ndim != 2):
            raise ValueError('grouped PPO requires two-dimensional global batches')
        columns = first['global'].shape[0]
        total_columns += columns
        if not 1 <= columns <= 64 or total_columns > 64:
            raise ValueError('grouped PPO columns exceed 64')
        shape = (len(batches), columns)
        device = first['global'].device
        if common_device is None: common_device = device
        if device != common_device:
            raise ValueError('grouped PPO devices differ')
        targets = [group[name] for name in ('actions', 'old_log_probs', 'advantages', 'returns')]
        if any(not isinstance(x, torch.Tensor) or x.shape != shape or x.device != device for x in targets):
            raise ValueError('grouped recurrent target dimensions or devices differ')
        actions, old, advantages, returns = targets
        if actions.dtype != torch.long:
            raise ValueError('grouped actions must be integer indices')
        if any(not x.is_floating_point() or not torch.isfinite(x).all() for x in targets[1:]):
            raise ValueError('nonfinite or nonfloating grouped PPO targets')
        resets = group.get('resets')
        if resets is None: resets = torch.zeros(shape, dtype=torch.bool, device=device)
        if not isinstance(resets, torch.Tensor) or resets.shape != shape or resets.dtype != torch.bool or resets.device != device:
            raise ValueError('invalid grouped recurrent reset boundaries')
        assignments = group.get('assignments')
        if assignments is None: assignments = [[None] * columns for _ in batches]
        if (type(assignments) not in (list, tuple) or len(assignments) != shape[0]
                or any(type(row) not in (list, tuple) or len(row) != columns for row in assignments)):
            raise ValueError('grouped joint assignments must share time/batch shape')
        state = group.get('initial_state')
        if state is not None:
            if (type(state) not in (list, tuple) or len(state) != 2
                    or any(not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] != columns
                           or x.device != device or not x.is_floating_point()
                           or not torch.isfinite(x).all() for x in state)
                    or state[0].shape != state[1].shape):
                raise ValueError('invalid grouped initial recurrent state')
            state = tuple(x.detach() for x in state)
        for batch in (*prefix, *batches):
            if (type(batch) is not dict or not isinstance(batch.get('global'), torch.Tensor)
                    or batch['global'].ndim != 2 or batch['global'].shape[0] != columns):
                raise ValueError('grouped batch or burn-in player columns differ')
            for value in batch.values():
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0 or value.shape[0] != columns or value.device != device:
                        raise ValueError('grouped batch tensor dimensions or devices differ')
                    if value.is_complex() or not torch.isfinite(value).all():
                        raise ValueError('nonfinite grouped observation')
            mask = batch.get('action_mask')
            if (not isinstance(mask, torch.Tensor) or mask.ndim != 2
                    or mask.dtype != torch.bool or not mask.any(dim=1).all()):
                raise ValueError('invalid grouped legal action mask')
        for t, batch in enumerate(batches):
            mask = batch['action_mask']
            if (actions[t] < 0).any() or (actions[t] >= mask.shape[1]).any():
                raise ValueError('grouped action outside menu')
            if not mask.gather(1, actions[t, :, None]).all():
                raise ValueError('recorded grouped action was illegal')
            # Native policy assignments are validated before any model call.
            indices, catalogs = batch.get('policy_action_indices'), batch.get('policy_catalogs')
            if indices is not None or catalogs is not None:
                if (type(indices) not in (list, tuple) or type(catalogs) not in (list, tuple)
                        or len(indices) != columns or len(catalogs) != columns):
                    raise ValueError('grouped policy catalogue dimensions differ')
                from .policy_assignments import validate_joint_assignment
                for col, action in enumerate(actions[t].tolist()):
                    if indices[col] == action:
                        validate_joint_assignment(catalogs[col]['slots'], assignments[t][col])
                    elif assignments[t][col] is not None:
                        raise ValueError('ordinary grouped action cannot have policy assignments')
            elif any(value is not None for value in assignments[t]):
                raise ValueError('grouped assignments require a public policy catalogue')
        prepared.append(dict(batches=batches, actions=actions, old_log_probs=old.detach(),
            advantages=advantages.detach(), returns=returns.detach(), assignments=assignments,
            initial_state=state, burn_in_batches=prefix, resets=resets))
    return prepared


def update_grouped(model, optimizer, groups, *, epochs=2, maximum_steps=64,
                   checkpoint_activations=None, clip=0.2,
                   value_weight=0.5, entropy_weight=0.01, max_grad_norm=0.5, target_kl=0.03):
    """Optimize unequal ordered sequences together without synthetic actions.

    Each group takes the sequence arguments of ``update``. The sum of sequence
    lengths including prefixes is at most ``maximum_steps`` (64 by default,
    explicitly up to 256), and total batch columns at most 64. Advantages are normalized over ALL optimized decisions; every metric
    and loss uses the same decision weighting. One optimizer step per epoch
    aggregates all groups, while recurrent states and burn-in stay independent.
    Extended limits checkpoint each complete action evaluation with PyTorch's
    non-reentrant activation checkpoint. RNG state is preserved, and recurrent
    outputs remain connected across every optimized step; the existing
    detached burn-in boundary is unchanged.
    ``checkpoint_activations`` can explicitly override this (``None`` selects
    it only when ``maximum_steps`` exceeds the historical 64-step limit).
    No trajectory identity, rewards, privileged sidecars or admission are inferred.
    """
    if type(epochs) is not int or not 1 <= epochs <= 16:
        raise ValueError('unbounded grouped epochs')
    if type(maximum_steps) is not int or not 1 <= maximum_steps <= 256:
        raise ValueError('grouped recurrent maximum_steps must be an integer in [1, 256]')
    if checkpoint_activations is None:
        # Keep historical 64-step updates unchanged. The extended sequence
        # contract uses non-reentrant activation checkpointing by default.
        checkpoint_activations = maximum_steps > 64
    elif type(checkpoint_activations) is not bool:
        raise ValueError('checkpoint_activations must be boolean or None')
    hyperparameters = (clip, value_weight, entropy_weight, max_grad_norm, target_kl)
    if (any(type(x) not in (int, float) or not math.isfinite(x) for x in hyperparameters)
            or not 0 <= clip < 1 or value_weight < 0 or entropy_weight < 0
            or max_grad_norm <= 0 or target_kl < 0):
        raise ValueError('invalid grouped PPO hyperparameters')
    groups = _grouped_targets(groups, maximum_steps=maximum_steps)
    advantages = torch.cat([g['advantages'].reshape(-1) for g in groups])
    advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    if not torch.isfinite(advantages).all():
        raise ValueError('nonfinite globally normalized advantages')
    old = torch.cat([g['old_log_probs'].reshape(-1) for g in groups])
    returns = torch.cat([g['returns'].reshape(-1) for g in groups])
    metrics = {}
    for epoch in range(epochs):
        all_logps, all_values, all_entropies = [], [], []
        for group in groups:
            state = group['initial_state']
            with torch.no_grad():
                for prefix in group['burn_in_batches']:
                    _, _, state = model(prefix, state)
            if state is not None: state = tuple(x.detach() for x in state)
            logps, values, entropies = [], [], []
            for t, batch in enumerate(group['batches']):
                evaluate = lambda recurrent_state, batch=batch, actions=group['actions'][t], \
                    assignments=group['assignments'][t], reset=group['resets'][t]: \
                    model.evaluate_actions(batch, actions, assignments, recurrent_state, reset)
                if checkpoint_activations:
                    # Pass recurrent tensors as checkpoint inputs so gradients
                    # propagate through every time step. Non-reentrant mode
                    # supports parameter gradients when the initial state is
                    # None and preserves the complete nested output structure.
                    from torch.utils.checkpoint import checkpoint
                    observation_input = batch['global']
                    if state is None:
                        checkpoint_function = lambda _observation, run=evaluate: run(None)
                        checkpoint_inputs = (observation_input,)
                    else:
                        checkpoint_function = lambda _observation, h, c, run=evaluate: run((h, c))
                        checkpoint_inputs = (observation_input, *state)
                    if observation_input.device.type == 'mps':
                        # MPS has RNG state APIs but no torch.mps.device context;
                        # preserve its single-device stream with a custom context.
                        outputs = checkpoint(checkpoint_function, *checkpoint_inputs,
                                             use_reentrant=False, preserve_rng_state=False,
                                             context_fn=_mps_checkpoint_context_fn)
                    else:
                        outputs = checkpoint(checkpoint_function, *checkpoint_inputs,
                                             use_reentrant=False, preserve_rng_state=True)
                    logp, value, state, entropy = outputs
                else:
                    logp, value, state, entropy = evaluate(state)
                expected = (group['actions'].shape[1],)
                if any(not isinstance(x, torch.Tensor) or x.shape != expected for x in (logp, value, entropy)):
                    raise ValueError('grouped policy outputs differ from decision shape')
                if (not torch.isfinite(logp).all() or not torch.isfinite(entropy).all()
                        or value_weight and not torch.isfinite(value).all()):
                    raise ValueError('nonfinite grouped policy outputs')
                logps.append(logp); values.append(value); entropies.append(entropy)
            all_logps.append(torch.stack(logps).reshape(-1))
            all_values.append(torch.stack(values).reshape(-1))
            all_entropies.append(torch.stack(entropies).reshape(-1))
        logps, values = torch.cat(all_logps), torch.cat(all_values)
        entropy = torch.cat(all_entropies).mean()
        loss, policy_loss, value_loss, approx_kl, ratio = _objective(
            logps, values, entropy, old, advantages, returns,
            clip=clip, value_weight=value_weight, entropy_weight=entropy_weight)
        if not torch.isfinite(loss) or not torch.isfinite(approx_kl):
            raise ValueError('nonfinite grouped PPO objective')
        if epoch and approx_kl.item() > target_kl:
            metrics['early_stop_kl'] = 1.0
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        metrics = {'loss': float(loss.detach()), 'policy_loss': float(policy_loss.detach()),
                   'value_loss': float(value_loss.detach()), 'entropy': float(entropy.detach()),
                   'kl': float(approx_kl.detach()), 'grad_norm': float(grad_norm.detach()),
                   'clip_fraction': float(((ratio - 1).abs() > clip).float().mean().detach()),
                   'epochs_completed': float(epoch + 1), 'early_stop_kl': 0.0}
    return metrics
