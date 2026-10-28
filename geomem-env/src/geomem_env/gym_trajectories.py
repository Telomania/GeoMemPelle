"""Small, offline trajectory collector; Gym itself is an optional dependency.

An episode has T actions and T+1 observations. Thus observations[t], actions[t]
predict observations[t+1], including the final transition but never a reset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

MINIGRID_CARDINALITIES = (11, 6, 3)  # object, color, state; MiniGrid 2.x


@dataclass(frozen=True)
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    seed: int


def minigrid_image(observation: dict[str, Any]) -> np.ndarray:
    """Copy only the partial categorical image (exclude mission and direction)."""
    image = np.asarray(observation["image"])
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected a MiniGrid image with shape (width, height, 3)")
    if not np.issubdtype(image.dtype, np.integer):
        raise ValueError("MiniGrid image must contain integer categories")
    for channel, cardinality in enumerate(MINIGRID_CARDINALITIES):
        if np.any(image[..., channel] < 0) or np.any(image[..., channel] >= cardinality):
            raise ValueError(f"Invalid category in MiniGrid channel {channel}")
    return image.astype(np.int64, copy=True)


def collect_episodes(
    env: Any,
    num_episodes: int,
    max_steps: int,
    seed: int = 0,
    observation_fn: Callable[[Any], np.ndarray] = minigrid_image,
    action_ids: list[int] | tuple[int, ...] | None = None,
    policy: Callable[[Any, np.random.Generator], int] | None = None,
) -> list[Episode]:
    """Collect complete bounded episodes from one non-vector, non-autoreset env.

    Supports Gymnasium and legacy Gym reset/step signatures with Discrete actions.
    A policy receives the current raw observation and a seeded random generator.
    Without a policy, sample uniformly from action_ids (default: all actions).
    Collector time limits set truncated=True; environment terminal flags are kept.
    The caller owns env.close(). Replace observation_fn to use another environment.
    """
    if num_episodes < 1 or max_steps < 1:
        raise ValueError("num_episodes and max_steps must be positive")
    n_actions = int(env.action_space.n)
    available = tuple(range(n_actions)) if action_ids is None else tuple(action_ids)
    if not available or any(a < 0 or a >= n_actions for a in available):
        raise ValueError("action_ids must contain valid Discrete action indices")
    episodes = []
    for episode_index in range(num_episodes):
        episode_seed = seed + episode_index
        rng = np.random.default_rng(episode_seed)
        try:
            reset = env.reset(seed=episode_seed)
        except TypeError:  # old Gym versions use env.seed()
            env.seed(episode_seed)
            reset = env.reset()
        observation = reset[0] if isinstance(reset, tuple) and len(reset) == 2 and isinstance(reset[1], dict) else reset
        observations = [np.array(observation_fn(observation), copy=True)]
        actions, terminated_flags, truncated_flags = [], [], []
        for step in range(max_steps):
            action = int(policy(observation, rng) if policy else rng.choice(available))
            if not 0 <= action < n_actions:
                raise ValueError("Policy returned an invalid Discrete action")
            result = env.step(action)
            if len(result) == 5:
                observation, _, terminated, truncated, _ = result
            else:
                observation, _, done, info = result
                truncated = bool(info.get("TimeLimit.truncated", False))
                terminated = bool(done) and not truncated
            truncated = bool(truncated) or (step + 1 == max_steps and not terminated)
            observations.append(np.array(observation_fn(observation), copy=True))
            actions.append(action)
            terminated_flags.append(bool(terminated))
            truncated_flags.append(truncated)
            if terminated or truncated:
                break
        episodes.append(Episode(
            np.stack(observations), np.asarray(actions, dtype=np.int64),
            np.asarray(terminated_flags, dtype=bool), np.asarray(truncated_flags, dtype=bool),
            episode_seed,
        ))
    return episodes
