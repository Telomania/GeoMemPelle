"""Mechanism checks; run with the same SF fork and dependencies as training."""
import unittest

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from experiments.geomem.train import (
    MiniGridCore, make_actor_critic, make_env, parse_args, register_components,
)
from geomem_models.lin_intrmotiv import (
    CategoricalDGEncoder, DGProjection_batchnorm_relu, LinSequenceCore,
)
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu as SourceDG
from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
import gymnasium as gym


class SharedBackboneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        register_components()
        cls.cfg = parse_args(["--seed=7", "--DG_BN_intercept=0.0"])

    def test_impulse_and_nilpotency(self):
        core = LinSequenceCore(features=1, repeat=2, length=3, bypass_size=0)
        state = core.initial_state(1)
        outputs = []
        for value in [1.0, 0, 0, 0, 0]:
            state, _ = core(torch.tensor([[value]]), state)
            outputs.append(state[0].tolist())
        self.assertEqual(outputs, [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],
                                   [0, 0, 0, 1], [0, 0, 0, 0]])
        w = core.recurrent_matrix()
        torch.testing.assert_close(torch.linalg.matrix_power(w, 4), torch.zeros_like(w))
        self.assertEqual(list(core.parameters()), [])

    def test_single_sequence_packed_and_upstream_match(self):
        cfg = self.cfg
        width = cfg.Hippo_n_feature + 980
        shared = MiniGridCore(cfg, width)
        upstream = SimpleSequenceWithBypassCore(cfg, width)
        images = torch.zeros(6, 3, 7, 7, 3, dtype=torch.long)
        images[..., 0] = torch.randint(0, 11, images[..., 0].shape)
        encoder = CategoricalDGEncoder(features=cfg.Hippo_n_feature, threshold=0)
        encoded = encoder(images.flatten(0, 1)).reshape(6, 3, width)
        initial = torch.rand(3, cfg.rnn_size)
        actual, final = shared.shared.forward_sequence(encoded, initial)
        state = initial
        for step in range(6):
            expected, state = upstream(encoded[step], state)
            torch.testing.assert_close(actual[step], expected)
        torch.testing.assert_close(final, state)
        lengths = torch.tensor([6, 2, 4])  # deliberately unsorted
        packed = pack_padded_sequence(encoded, lengths, enforce_sorted=False)
        replay, final = shared(packed, initial)
        replay, _ = pad_packed_sequence(replay)
        for batch, length in enumerate(lengths.tolist()):
            torch.testing.assert_close(replay[:length, batch], actual[:length, batch])
            torch.testing.assert_close(final[batch], actual[length - 1, batch])
        # Reset means zero initial state, irrespective of an earlier episode.
        zero = shared.shared.initial_state(3)
        fresh, _ = shared(encoded[0], zero)
        again, _ = shared(encoded[0], zero)
        torch.testing.assert_close(fresh, again)
        self.assertFalse(torch.equal(fresh, actual[0]))

    def test_normalization_equivalence_and_readonly_inference(self):
        local = DGProjection_batchnorm_relu(8, 4, batchnorm_semantics="running_poststep_atomic")
        source = SourceDG(8, 4, batchnorm_semantics="running_poststep_atomic")
        source.load_state_dict(local.state_dict())
        x = torch.randn(10, 8)
        for model in (local, source):
            with model.running_stats_update(True):
                model(x)
            model.post_step_update_running_stats()
        torch.testing.assert_close(local(x), source(x))
        before = {key: value.clone() for key, value in local.state_dict().items()}
        # Actor batch size one and learner minibatch use the same moments.
        torch.testing.assert_close(local(x[:1]), local(x)[:1])
        for key, value in local.state_dict().items():
            torch.testing.assert_close(value, before[key])

    def test_real_actor_gradient_boundary_and_dg_objective(self):
        env = make_env("geomem_minigrid", self.cfg)
        try:
            image, _ = env.reset(seed=7)
            obs_space = gym.spaces.Dict({"obs": env.observation_space})
            actor = make_actor_critic(self.cfg, obs_space, env.action_space)
            images = torch.as_tensor(image).unsqueeze(0).repeat(4, 1, 1, 1)
            features = actor.forward_head({"obs": images})
            states, _ = actor.forward_core(features, torch.zeros(4, self.cfg.rnn_size))
            result = actor.forward_tail(states, values_only=False, sample_actions=False)
            (result["action_logits"].square().mean() + result["values"].square().mean()).backward()
            dg = actor.encoder.DG_projection.linear.weight
            self.assertTrue(dg.grad is None or dg.grad.abs().sum() == 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in actor.decoder.parameters()))
            self.assertEqual(list(actor.core.parameters()), [])
            actor.zero_grad(set_to_none=True)
            features = actor.forward_head({"obs": images})
            mask = torch.ones(4, self.cfg.Hippo_n_feature)
            loss = DistanceLearnerReward._encoder_loss(
                type("LearnerView", (), {"cfg": self.cfg})(), features,
                torch.ones(4), mask, torch.ones(4, dtype=torch.bool), 0)
            loss.backward()
            self.assertIsNotNone(dg.grad)
            self.assertGreater(float(dg.grad.abs().sum()), 0)
            self.assertTrue(all(p.grad is None for p in actor.decoder.parameters()))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
