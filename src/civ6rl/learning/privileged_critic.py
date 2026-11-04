"""Isolated value critic prototype for privileged training observations.

The actor's detached LSTM hidden/cell pair stands in for public history. This
is an approximation: it is neither a second history encoder nor a claim that
the actor state is a sufficient statistic. Privileged inputs affect value
only; this module exposes no action or policy head.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io

import torch
from torch import nn


@dataclass(frozen=True)
class PrivilegedCriticConfig:
    """Frozen feature contract for one critic checkpoint."""

    privileged_schema: str
    privileged_features: tuple[str, ...]
    actor_hidden_dim: int
    hidden_dim: int = 128
    version: str = "civ6rl.privileged_critic.v1"

    def __post_init__(self):
        if self.version != "civ6rl.privileged_critic.v1":
            raise ValueError("unsupported privileged critic version")
        if not isinstance(self.privileged_schema, str) or not self.privileged_schema.strip():
            raise ValueError("a declared privileged feature schema is required")
        if (not isinstance(self.privileged_features, tuple) or not self.privileged_features
                or any(not isinstance(name, str) or not name for name in self.privileged_features)
                or len(set(self.privileged_features)) != len(self.privileged_features)):
            raise ValueError("privileged feature names must be a nonempty unique tuple")
        if (type(self.actor_hidden_dim) is not int or self.actor_hidden_dim <= 0
                or type(self.hidden_dim) is not int or self.hidden_dim <= 0):
            raise ValueError("critic dimensions must be positive integers")


class PrivilegedCritic(nn.Module):
    """Value-only network with no actor parameters or shared optimizer state."""

    history_representation = "detached_actor_lstm_hc_approx_v1"

    def __init__(self, config: PrivilegedCriticConfig):
        super().__init__()
        if not isinstance(config, PrivilegedCriticConfig):
            raise TypeError("config must be PrivilegedCriticConfig")
        self.config = config
        input_dim = 2 * config.actor_hidden_dim + 2 * len(config.privileged_features)
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, privileged_current, privileged_mask, actor_hidden, actor_cell):
        """Return `[B]` values; `privileged_mask` distinguishes unknown from zero.

        Unknown privileged cells must use canonical numeric zero and a false
        mask. A missing schema, mask, or field is never inferred as known.
        Actor state is detached before it enters the critic graph.
        """
        values = privileged_current
        mask = privileged_mask
        hidden = actor_hidden
        cell = actor_cell
        if not all(isinstance(x, torch.Tensor) for x in (values, mask, hidden, cell)):
            raise TypeError("critic inputs must be tensors, including an explicit privileged mask")
        if values.ndim != 2 or values.shape[1] != len(self.config.privileged_features):
            raise ValueError("privileged feature layout differs from declared schema")
        batch = values.shape[0]
        if batch < 1 or mask.shape != values.shape or mask.dtype != torch.bool:
            raise ValueError("privileged mask must be Boolean and match [B,D]")
        expected_state = (batch, self.config.actor_hidden_dim)
        if hidden.shape != expected_state or cell.shape != expected_state:
            raise ValueError("detached actor hidden/cell layout differs")
        tensors = (values, hidden, cell)
        if any(not x.is_floating_point() for x in tensors):
            raise TypeError("numeric critic inputs must be floating point")
        if any(x.device != values.device or x.dtype != values.dtype for x in (hidden, cell)) or mask.device != values.device:
            raise ValueError("critic inputs must share device and floating dtype")
        parameter = next(self.parameters())
        if parameter.device != values.device or parameter.dtype != values.dtype:
            raise ValueError("critic parameters and inputs must share device and floating dtype")
        if any(not torch.isfinite(x).all() for x in tensors):
            raise ValueError("nonfinite privileged critic input")
        if torch.count_nonzero(values.masked_select(~mask)):
            raise ValueError("unknown privileged values must be canonical zero under a false mask")

        known = mask.to(dtype=values.dtype)
        masked_values = values.detach() * known
        features = torch.cat((
            masked_values,
            known,
            hidden.detach(),
            cell.detach(),
        ), dim=-1)
        result = self.value_net(features).squeeze(-1)
        if result.shape != (batch,) or not torch.isfinite(result).all():
            raise ValueError("critic produced an invalid value tensor")
        return result

    def make_optimizer(self, learning_rate: float = 3e-4):
        if not isinstance(learning_rate, (int, float)) or isinstance(learning_rate, bool):
            raise ValueError("critic learning rate must be numeric")
        if not 0 < learning_rate < 1 or not torch.isfinite(torch.tensor(float(learning_rate))):
            raise ValueError("critic learning rate must be finite and in (0,1)")
        return torch.optim.Adam(self.parameters(), lr=float(learning_rate))

    def checkpoint_metadata(self):
        return {
            "schema": self.config.version,
            "config": asdict(self.config),
            "history_representation": self.history_representation,
            "value_only": True,
            "privileged_features_are_policy_inputs": False,
            "training_eligible": False,
        }


def update_critic(critic: PrivilegedCritic, optimizer, privileged_current,
                  privileged_mask, actor_hidden, actor_cell, target_value, *,
                  max_grad_norm: float = 1.0):
    """Perform one detached MSE update, rejecting any optimizer parameter sharing."""
    if not isinstance(critic, PrivilegedCritic):
        raise TypeError("critic must be PrivilegedCritic")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("critic optimizer is required")
    if (not isinstance(max_grad_norm, (int, float)) or isinstance(max_grad_norm, bool)
            or not 0 < max_grad_norm < float("inf")):
        raise ValueError("gradient bound must be finite and positive")
    critic_parameters = list(critic.parameters())
    critic_ids = [id(parameter) for parameter in critic_parameters]
    optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    if len(set(optimizer_ids)) != len(optimizer_ids) or set(optimizer_ids) != set(critic_ids):
        raise ValueError("optimizer must own exactly the critic parameters and no actor parameters")
    if not isinstance(target_value, torch.Tensor) or target_value.ndim != 1:
        raise ValueError("critic targets must have shape [B]")
    if target_value.shape[0] != privileged_current.shape[0]:
        raise ValueError("critic target batch differs")
    if (not target_value.is_floating_point() or target_value.device != privileged_current.device
            or target_value.dtype != privileged_current.dtype or not torch.isfinite(target_value).all()):
        raise ValueError("critic targets must be finite and match input device/dtype")

    target = target_value.detach()
    optimizer.zero_grad(set_to_none=True)
    prediction = critic(privileged_current, privileged_mask, actor_hidden, actor_cell)
    loss = (prediction - target).square().mean()
    if not torch.isfinite(loss):
        raise ValueError("nonfinite critic loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(critic_parameters, max_grad_norm,
                                               error_if_nonfinite=True)
    optimizer.step()
    return {"value_loss": float(loss.detach()), "grad_norm": float(grad_norm.detach())}


def critic_bytes(critic):
    if not isinstance(critic, PrivilegedCritic):
        raise TypeError('expected separate privileged critic')
    payload = critic.checkpoint_metadata()
    payload['state_dict'] = {k: v.detach().cpu().contiguous() for k, v in critic.state_dict().items()}
    if any(not torch.isfinite(v).all() for v in payload['state_dict'].values()):
        raise ValueError('nonfinite critic parameters')
    stream = io.BytesIO()
    torch.save(payload, stream)
    return stream.getvalue()


def critic_version(critic):
    return hashlib.sha256(critic_bytes(critic)).hexdigest()


def load_critic_bytes(data, *, sha256, device='cpu'):
    if (type(data) is not bytes or not 0 < len(data) <= 64 * 1024 * 1024
            or hashlib.sha256(data).hexdigest() != sha256):
        raise ValueError('critic artifact size or digest differs')
    if device not in ('cpu', 'mps'):
        raise ValueError('unsupported critic device')
    payload = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
    if type(payload) is not dict or set(payload) != {
            'schema', 'config', 'history_representation', 'value_only',
            'privileged_features_are_policy_inputs', 'training_eligible', 'state_dict'}:
        raise ValueError('unknown critic artifact fields')
    config = PrivilegedCriticConfig(**payload['config'])
    if (config.actor_hidden_dim > 1024 or config.hidden_dim > 1024
            or len(config.privileged_features) > 4096):
        raise ValueError('critic dimensions exceed bounds')
    critic = PrivilegedCritic(config)
    if {k: v for k, v in payload.items() if k != 'state_dict'} != critic.checkpoint_metadata():
        raise ValueError('critic metadata differs')
    critic.load_state_dict(payload['state_dict'], strict=True)
    if any(not torch.isfinite(p).all() for p in critic.parameters()):
        raise ValueError('nonfinite critic parameters')
    return critic.to(device).eval()
