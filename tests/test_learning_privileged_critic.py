"""Synthetic tests for the isolated privileged value-critic prototype."""
import unittest

try:
    import torch
    from torch import nn
except ImportError:
    torch = None

if torch is not None:
    from civ6rl.learning.privileged_critic import (
        PrivilegedCritic,
        PrivilegedCriticConfig,
        update_critic,
    )


def config():
    return PrivilegedCriticConfig(
        privileged_schema="synthetic.hidden-state.fixture.v1",
        privileged_features=("unseen_enemy_count", "hidden_resource_value"),
        actor_hidden_dim=4,
        hidden_dim=12,
    )


def inputs(device="cpu", batch=3):
    torch.manual_seed(21)
    return (
        torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 3.0]], device=device),
        torch.tensor([[True, False], [False, True], [True, True]], device=device),
        torch.randn(batch, 4, device=device),
        torch.randn(batch, 4, device=device),
    )


def set_probe_weights(critic):
    """Make the first privileged value and its known-mask bit observable."""
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.zero_()
        critic.value_net[0].weight[0, 0] = 1.0
        critic.value_net[0].weight[0, 2] = 1.0
        critic.value_net[2].weight[0, 0] = 1.0
        critic.value_net[4].weight[0, 0] = 1.0


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class PrivilegedCriticTests(unittest.TestCase):
    def test_config_requires_declared_stable_schema_and_feature_names(self):
        with self.assertRaises(ValueError):
            PrivilegedCriticConfig("", ("hidden",), 4)
        with self.assertRaises(ValueError):
            PrivilegedCriticConfig("fixture.v1", ("same", "same"), 4)
        with self.assertRaises(ValueError):
            PrivilegedCriticConfig("fixture.v1", ("hidden",), 0)

    def test_value_changes_with_privileged_feature_and_unknown_mask_is_explicit(self):
        critic = PrivilegedCritic(config()).eval()
        set_probe_weights(critic)
        values, mask, hidden, cell = inputs()
        changed = values.clone()
        changed[0, 0] += 5.0
        first = critic(values, mask, hidden, cell)
        second = critic(changed, mask, hidden, cell)
        self.assertEqual(first.shape, (3,))
        self.assertFalse(torch.allclose(first[0], second[0]))
        # Unknown feature is represented by both canonical zero and a false mask;
        # known numeric zero remains distinguishable by the mask bit.
        zeros = torch.zeros(1, 2)
        unknown = torch.tensor([[False, False]])
        known_zero = torch.tensor([[True, False]])
        h = torch.zeros(1, 4)
        unknown_value = critic(zeros, unknown, h, h)
        known_zero_value = critic(zeros, known_zero, h, h)
        self.assertFalse(torch.allclose(unknown_value, known_zero_value))

    def test_invalid_shapes_nonfinite_and_noncanonical_missing_values_refuse(self):
        critic = PrivilegedCritic(config())
        values, mask, hidden, cell = inputs()
        with self.assertRaises(TypeError):
            critic(values, None, hidden, cell)
        with self.assertRaises(ValueError):
            critic(values[:, :1], mask[:, :1], hidden, cell)
        bad = values.clone()
        bad[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            critic(bad, mask, hidden, cell)
        bad_missing = values.clone()
        bad_missing[0, 1] = 9.0
        with self.assertRaisesRegex(ValueError, "canonical zero"):
            critic(bad_missing, mask, hidden, cell)

    def test_update_changes_critic_only_and_detaches_actor_history_memory(self):
        torch.manual_seed(5)
        critic = PrivilegedCritic(config())
        optimizer = critic.make_optimizer(1e-2)
        actor = nn.Linear(5, 8)
        actor_before = [parameter.detach().clone() for parameter in actor.parameters()]
        actor_input = torch.randn(3, 5)
        state = actor(actor_input)
        hidden, cell = state[:, :4], state[:, 4:]
        hidden.retain_grad();cell.retain_grad()
        memory_before = (hidden.detach().clone(), cell.detach().clone())
        critic_before = [parameter.detach().clone() for parameter in critic.parameters()]
        values, mask, _, _ = inputs()
        target = torch.tensor([4.0, -3.0, 2.0])

        metrics = update_critic(critic, optimizer, values, mask, hidden, cell, target)

        self.assertGreater(metrics["value_loss"], 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(
            critic_before, critic.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actor_before, actor.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in actor.parameters()))
        self.assertIsNone(hidden.grad)
        self.assertIsNone(cell.grad)
        self.assertTrue(torch.equal(hidden.detach(), memory_before[0]))
        self.assertTrue(torch.equal(cell.detach(), memory_before[1]))

    def test_shared_actor_optimizer_is_rejected_before_any_update(self):
        critic = PrivilegedCritic(config())
        actor = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(list(critic.parameters()) + list(actor.parameters()), lr=1e-3)
        values, mask, hidden, cell = inputs()
        before = [parameter.detach().clone() for parameter in critic.parameters()]
        with self.assertRaisesRegex(ValueError, "exactly the critic parameters"):
            update_critic(critic, optimizer, values, mask, hidden, cell, torch.zeros(3))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, critic.parameters())))

    def test_cpu_and_available_mps_forward_and_update(self):
        devices = ["cpu"]
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            devices.append("mps")
        for device in devices:
            with self.subTest(device=device):
                critic = PrivilegedCritic(config()).to(device)
                optimizer = critic.make_optimizer(1e-3)
                values, mask, hidden, cell = inputs(device)
                target = torch.tensor([0.25, -0.5, 1.0], device=device)
                value = critic(values, mask, hidden, cell)
                self.assertEqual(value.device.type, device)
                self.assertTrue(torch.isfinite(value).all())
                update_critic(critic, optimizer, values, mask, hidden, cell, target)
                if device == "mps":
                    torch.mps.synchronize()
                self.assertTrue(torch.isfinite(critic(values, mask, hidden, cell)).all())

    def test_checkpoint_metadata_declares_approximate_history_and_value_only_scope(self):
        metadata = PrivilegedCritic(config()).checkpoint_metadata()
        self.assertEqual(metadata["history_representation"], "detached_actor_lstm_hc_approx_v1")
        self.assertTrue(metadata["value_only"])
        self.assertFalse(metadata["privileged_features_are_policy_inputs"])
        self.assertFalse(metadata["training_eligible"])


if __name__ == "__main__":
    unittest.main()
