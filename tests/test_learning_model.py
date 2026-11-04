import copy
import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional learning dependency")
class LearningModelTests(unittest.TestCase):
    def setUp(self):
        from civ6rl.learning.encoding import ContentVocabulary, encode_response, MAP_CHANNELS, GLOBAL_FEATURES, ENTITY_FEATURES, ACTION_FEATURES
        from civ6rl.learning.model import RecurrentPolicy, PolicyConfig, collate
        from tests.test_learning_encoding import response
        torch.manual_seed(7); torch.set_num_threads(2)
        self.raw = response(candidates=[
            {"candidate_id": "a", "action": {"kind": "end_turn"}},
            {"candidate_id": "b", "action": {"kind": "skip_unit", "unit_id": 101}},
        ])
        self.vocab = ContentVocabulary.fit(self.raw)
        self.frame = encode_response(self.raw, vocabulary=self.vocab)
        self.collate = collate
        self.model = RecurrentPolicy(PolicyConfig(len(MAP_CHANNELS), len(GLOBAL_FEATURES),
            len(ENTITY_FEATURES), len(ACTION_FEATURES), self.vocab.vocabulary_size, hidden_size=32, spatial_size=8))

    def test_mask_assigns_exact_zero_probability_and_rejects_empty_menu(self):
        from civ6rl.learning.model import masked_distribution
        logits = torch.tensor([[100., 1., 2.]], requires_grad=True)
        dist = masked_distribution(logits, torch.tensor([[False, True, True]]))
        self.assertEqual(dist.probs[0, 0].item(), 0)
        self.assertTrue(torch.isfinite(dist.entropy()).all())
        (-dist.log_prob(torch.tensor([1]))).backward()
        self.assertEqual(logits.grad[0, 0].item(), 0)
        with self.assertRaises(ValueError):
            masked_distribution(logits, torch.zeros_like(logits, dtype=torch.bool))

    def test_padding_invariance_and_candidate_permutation(self):
        single = self.collate([self.frame])
        large = copy.deepcopy(self.frame)
        large["map"] = [[[0.] * 9 for _ in range(8)] for _ in large["map"]]
        large["map_categorical"] = [[[0] * 9 for _ in range(8)] for _ in large["map_categorical"]]
        large["entities"] += large["entities"] * 2
        large["entity_mask"] += large["entity_mask"] * 2
        large["entity_categorical"] += large["entity_categorical"] * 2
        dist, value, _ = self.model(single)
        padded_dist, padded_value, _ = self.model(self.collate([self.frame, large]))
        torch.testing.assert_close(dist.probs[0], padded_dist.probs[0], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(value[0], padded_value[0], atol=1e-6, rtol=1e-5)
        perm = copy.deepcopy(self.frame)
        for key in ("candidates", "action_categorical", "action_mask", "candidate_ids"):
            perm[key] = list(reversed(perm[key]))
        perm_dist, _, _ = self.model(self.collate([perm]))
        torch.testing.assert_close(dist.probs.flip(-1), perm_dist.probs)

    def test_lstm_has_memory_and_explicit_reset(self):
        batch = self.collate([self.frame])
        _, first, state = self.model(batch)
        _, repeated, _ = self.model(batch, state)
        self.assertFalse(torch.equal(first, repeated))
        _, reset, _ = self.model(batch, state, torch.tensor([True]))
        torch.testing.assert_close(first, reset)
        repeated.sum().backward()
        self.assertGreater(self.model.memory.weight_hh.grad.abs().sum().item(), 0)

    def test_player_memory_isolation_and_replay_refusal(self):
        from civ6rl.learning.model import PlayerMemory
        memory = PlayerMemory()
        memory.put("A", 0, 2, self.model.initial_state(1))
        self.assertIsNone(memory.take("A", 1, 3))
        self.assertIsNone(memory.take("B", 0, 3))
        with self.assertRaises(ValueError): memory.take("A", 0, 2)
        memory.reset_episode("A")
        self.assertIsNone(memory.take("A", 0, 0))

    def test_gae_bootstraps_truncation_but_not_termination_or_next_episode(self):
        from civ6rl.learning.ppo import generalized_advantages
        rewards = torch.tensor([[1., 1.], [99., 99.]])
        vals = torch.zeros_like(rewards)
        next_vals = torch.tensor([[10., 10.], [0., 0.]])
        term = torch.tensor([[True, False], [True, True]])
        trunc = torch.tensor([[False, True], [False, False]])
        adv, _ = generalized_advantages(rewards, vals, next_vals, term, trunc, torch.full_like(vals, .9))
        torch.testing.assert_close(adv[0], torch.tensor([1., 10.]))

    def test_recurrent_ppo_updates_weights_and_rejects_illegal_recorded_action(self):
        from civ6rl.learning.ppo import update
        batches = [self.collate([self.frame, self.frame]) for _ in range(3)]
        state = None; logps = []; actions = []
        with torch.no_grad():
            for batch in batches:
                dist, _, state = self.model(batch, state)
                chosen = torch.tensor([0, 1]); actions.append(chosen); logps.append(dist.log_prob(chosen))
        before = self.model.memory.weight_ih.detach().clone()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        adv = torch.tensor([[1., -1.], [0.5, -0.5], [-1., 1.]])
        metrics = update(self.model, optimizer, batches, torch.stack(actions), torch.stack(logps), adv, adv)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in metrics.values()))
        self.assertFalse(torch.equal(before, self.model.memory.weight_ih))
        batches[0]["action_mask"][0, 0] = False
        with self.assertRaises(ValueError):
            update(self.model, optimizer, batches, torch.stack(actions), torch.stack(logps), adv, adv)

    def test_weights_only_checkpoint_roundtrip(self):
        import io
        from civ6rl.learning.model import checkpoint_payload, PolicyConfig, RecurrentPolicy
        stream = io.BytesIO(); torch.save(checkpoint_payload(self.model), stream); stream.seek(0)
        payload = torch.load(stream, weights_only=True)
        other = RecurrentPolicy(PolicyConfig(**payload["config"]))
        other.load_state_dict(payload["state_dict"])
        a = self.model(self.collate([self.frame]))[0].probs
        b = other(self.collate([self.frame]))[0].probs
        torch.testing.assert_close(a, b)


if __name__ == "__main__": unittest.main()
