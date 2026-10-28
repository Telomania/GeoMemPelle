"""Boundary and alignment tests without requiring a Gym installation."""

import unittest
from types import SimpleNamespace

import numpy as np

from geomem_env.gym_trajectories import collect_episodes, minigrid_image


class CounterEnv:
    action_space = SimpleNamespace(n=3)

    def __init__(self, terminal_at=2, legacy=False, timeout=False):
        self.terminal_at, self.legacy, self.timeout = terminal_at, legacy, timeout
        self.reset_seeds = []

    def reset(self, seed=None):
        self.reset_seeds.append(seed)
        self.value, self.steps = 0, 0
        observation = np.array([self.value])
        return observation if self.legacy else (observation, {})

    def step(self, action):
        self.value += action + 1
        self.steps += 1
        done = self.steps == self.terminal_at
        observation = np.array([self.value])
        if self.legacy:
            return observation, 0, done, {"TimeLimit.truncated": done and self.timeout}
        return observation, 0, done and not self.timeout, done and self.timeout, {}


class GymTrajectoryTests(unittest.TestCase):
    def collect(self, env, **kwargs):
        return collect_episodes(env, 2, kwargs.pop("max_steps", 5), seed=10, observation_fn=np.asarray, **kwargs)

    def test_alignment_and_terminal_transition(self):
        env = CounterEnv()
        episodes = self.collect(env, action_ids=[2])
        self.assertEqual(env.reset_seeds, [10, 11])
        for episode in episodes:
            np.testing.assert_array_equal(episode.observations[:, 0], [0, 3, 6])
            np.testing.assert_array_equal(episode.actions, [2, 2])
            np.testing.assert_array_equal(episode.terminated, [False, True])
            self.assertFalse(episode.truncated.any())

    def test_environment_and_collector_time_limits(self):
        for env, max_steps in ((CounterEnv(timeout=True), 5), (CounterEnv(terminal_at=20), 3)):
            episode = self.collect(env, max_steps=max_steps)[0]
            self.assertTrue(episode.truncated[-1])
            self.assertFalse(episode.terminated.any())
            self.assertEqual(len(episode.observations), len(episode.actions) + 1)

    def test_legacy_timeout_is_not_termination(self):
        episode = self.collect(CounterEnv(legacy=True, timeout=True))[0]
        self.assertTrue(episode.truncated[-1])
        self.assertFalse(episode.terminated.any())

    def test_seeded_actions_and_policy(self):
        first, second = self.collect(CounterEnv()), self.collect(CounterEnv())
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.actions, right.actions)
        episode = self.collect(CounterEnv(), policy=lambda observation, rng: 1)[0]
        np.testing.assert_array_equal(episode.actions, [1, 1])

    def test_observation_copy_and_validation(self):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        result = minigrid_image({"image": source, "direction": 1, "mission": "unused"})
        source[0, 0, 0] = 3
        self.assertEqual(result[0, 0, 0], 0)
        for invalid in (np.zeros((2, 2)), np.full((2, 2, 3), 99), np.zeros((2, 2, 3), dtype=float)):
            with self.assertRaises(ValueError):
                minigrid_image({"image": invalid})

    def test_invalid_action_and_empty_limits(self):
        with self.assertRaises(ValueError):
            self.collect(CounterEnv(), action_ids=[3])
        with self.assertRaises(ValueError):
            self.collect(CounterEnv(), max_steps=0)


if __name__ == "__main__":
    unittest.main()
