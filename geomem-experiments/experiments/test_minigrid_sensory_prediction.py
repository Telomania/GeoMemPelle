"""Focused tests of causal routing and train/evaluation separation (no Gym needed)."""

import unittest

import numpy as np
import torch

from minigrid_sensory_prediction import SensoryPredictor, episode_tensors, evaluate, fit
from geomem_env.gym_trajectories import Episode


class SensoryPredictionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8)
        images = np.zeros((5, 2, 2, 3), dtype=np.int64)
        images[:, ..., 0] = np.arange(5)[:, None, None]
        self.episode = Episode(images, np.array([0, 1, 2, 0]), np.array([False] * 4), np.array([False, False, False, True]), 1)

    def model(self, **kwargs):
        return SensoryPredictor((2, 2, 3), 3, features=3, repeat=2, length=3, threshold=0.0, **kwargs)

    def test_prediction_alignment(self):
        images, actions, targets = episode_tensors(self.episode)
        self.assertEqual(images[:, 0, 0, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(targets[:, 0, 0, 0].tolist(), [1, 2, 3, 4])
        self.assertEqual(actions.tolist(), [0, 1, 2, 0])

    def test_future_inputs_do_not_change_past_and_episode_resets(self):
        model = self.model().eval()
        images, actions, _ = episode_tensors(self.episode)
        original = model(images, actions).detach()
        changed = images.clone()
        changed[-1, ..., 0] = 9
        changed_actions = actions.clone()
        changed_actions[-1] = 2
        torch.testing.assert_close(model(changed, changed_actions)[:-1], original[:-1])
        torch.testing.assert_close(model(images, actions), original)

    def test_memoryless_has_identical_parameters_but_no_history(self):
        recurrent, memoryless = self.model(), self.model(memory=False)
        memoryless.load_state_dict(recurrent.state_dict())
        images, actions, _ = episode_tensors(self.episode)
        batch = memoryless(images, actions)
        individual = torch.cat([memoryless(image[None], action[None]) for image, action in zip(images, actions)])
        torch.testing.assert_close(batch, individual)
        self.assertEqual(sum(p.numel() for p in recurrent.parameters()), sum(p.numel() for p in memoryless.parameters()))

    def test_evaluation_never_updates_moments(self):
        model = self.model()
        fit(model, [episode_tensors(self.episode)], epochs=1, learning_rate=0.001, seed=1)
        torch.testing.assert_close(model.encoder.DG_projection.linear.weight.norm(dim=1), torch.ones(3))
        before = {name: value.clone() for name, value in model.state_dict().items()}
        metrics = evaluate(model, [self.episode])
        self.assertTrue(np.isfinite(metrics["cross_entropy"]))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_frozen_encoder_weights_and_initial_calibration(self):
        model = self.model(encoder_learning="frozen")
        projection = model.encoder.DG_projection
        initial = projection.linear.weight.detach().clone()
        fit(model, [episode_tensors(self.episode)], epochs=2, learning_rate=0.001, seed=1)
        torch.testing.assert_close(projection.linear.weight, initial)
        self.assertEqual(int(projection.batchnorm1d.num_batches_tracked), 1)

    def test_persistence_and_empty_change_mask(self):
        episode = Episode(np.zeros_like(self.episode.observations), self.episode.actions, self.episode.terminated, self.episode.truncated, 1)
        metrics = evaluate(None, [episode])
        self.assertEqual(metrics["token_accuracy"], 1.0)
        self.assertIsNone(metrics["changed_token_accuracy"])
        self.assertIsNone(metrics["cross_entropy"])

    def test_single_step_episodes_use_training_calibration_pool(self):
        model = self.model()
        images, actions, targets = episode_tensors(self.episode)
        episodes = [(images[i:i+1], actions[i:i+1], targets[i:i+1]) for i in range(2)]
        fit(model, episodes, epochs=1, learning_rate=0.001, seed=1)
        self.assertEqual(int(model.encoder.DG_projection.batchnorm1d.num_batches_tracked), 3)


if __name__ == "__main__":
    unittest.main()
