"""Standalone offline sensory prediction: collect -> split episodes -> fit -> score.

Run from geomem-experiments: python experiments/minigrid_sensory_prediction.py
See minigrid_sensory_prediction.md for the short conceptual guide.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
for package in ("geomem-env", "geomem-models"):
    sys.path.insert(0, str(ROOT.parent / package / "src"))

from geomem_env.gym_trajectories import MINIGRID_CARDINALITIES, Episode, collect_episodes
from geomem_models.lin_intrmotiv import CategoricalDGEncoder, LinSequenceCore


def episode_tensors(episode: Episode) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each input uses only image[t], action[t]; targets are image[t+1]."""
    images = torch.as_tensor(episode.observations, dtype=torch.long)
    return images[:-1], torch.as_tensor(episode.actions), images[1:]


class SensoryPredictor(nn.Module):
    """Same DG encoder and CA3 core as APPO; action enters only this task head."""

    def __init__(self, image_shape, n_actions, features=16, repeat=2, length=8, threshold=2.0, memory=True, encoder_learning="prediction"):
        super().__init__()
        self.n_actions, self.memory = n_actions, memory
        self.encoder_learning = encoder_learning
        self.encoder = CategoricalDGEncoder(image_shape, features, threshold)
        self.core = LinSequenceCore(features, repeat, length, self.encoder.bypass_size)
        self.head = nn.Linear(self.core.state_size + n_actions, self.encoder.bypass_size)
        if encoder_learning == "frozen":
            self.encoder.requires_grad_(False)

    def forward(self, images: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(images)
        if self.memory:
            representation, _ = self.core.forward_sequence(encoded[:, None, :])
            representation = representation[:, 0, :]
        else:
            # Identical encoder, core and head, but reset memory before EACH step.
            representation, _ = self.core(encoded, self.core.initial_state(len(images)))
        action = F.one_hot(actions, self.n_actions).float()
        return self.head(torch.cat((representation, action), dim=-1))


def sensory_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean categorical cross-entropy over cells, time, and the three channels."""
    heads = logits.reshape(*targets.shape[:-1], sum(MINIGRID_CARDINALITIES)).split(MINIGRID_CARDINALITIES, dim=-1)
    return torch.stack([
        F.cross_entropy(head.reshape(-1, size), targets[..., channel].reshape(-1))
        for channel, (head, size) in enumerate(zip(heads, MINIGRID_CARDINALITIES))
    ]).mean()


def fit(model: SensoryPredictor, episodes: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], epochs: int, learning_rate: float, seed: int) -> None:
    """One full episode per optimizer step; no padding or state carried across resets."""
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    rng = np.random.default_rng(seed)
    model.train()
    projection = model.encoder.DG_projection
    calibration = torch.cat([images for images, _, _ in episodes])
    if len(calibration) < 2:
        raise ValueError("DG calibration needs at least two training transitions")
    # Training observations only; frozen-DG control keeps these moments forever.
    with torch.no_grad(), projection.running_stats_update(True):
        model.encoder(calibration)
    projection.post_step_update_running_stats()
    for _ in range(epochs):
        for index in rng.permutation(len(episodes)):
            images, actions, targets = episodes[index]
            optimizer.zero_grad()
            with projection.running_stats_update(model.encoder_learning == "prediction"):
                loss = sensory_loss(model(images, actions), targets)
            loss.backward()
            optimizer.step()
            if model.encoder_learning == "prediction":
                if len(images) < 2:
                    # A one-step episode is valid data, but cannot calibrate variance.
                    with torch.no_grad(), projection.running_stats_update(True):
                        model.encoder(calibration)
                projection.post_step_update_running_stats()


@torch.no_grad()
def evaluate(model: SensoryPredictor | None, episodes: list[Episode]) -> dict[str, float | None]:
    """None is persistence: copy the current image as the next image prediction."""
    predictions, targets, current = [], [], []
    ce_sum, transitions = 0.0, 0
    if model is not None:
        model.eval()
    for episode in episodes:
        images, actions, target = episode_tensors(episode)
        if model is None:
            prediction = torch.as_tensor(episode.observations[:-1])
        else:
            logits = model(images, actions)
            ce_sum += float(sensory_loss(logits, target)) * len(target)
            heads = logits.reshape(*target.shape[:-1], sum(MINIGRID_CARDINALITIES)).split(MINIGRID_CARDINALITIES, dim=-1)
            prediction = torch.stack([head.argmax(dim=-1) for head in heads], dim=-1)
        transitions += len(target)
        predictions.append(prediction)
        targets.append(target)
        current.append(torch.as_tensor(episode.observations[:-1]))
    prediction, target, previous = (torch.cat(items) for items in (predictions, targets, current))
    correct = prediction == target
    changed = target != previous
    return {
        "cross_entropy": ce_sum / transitions if model is not None else None,
        "token_accuracy": float(correct.float().mean()),
        "object_accuracy": float(correct[..., 0].float().mean()),
        "color_accuracy": float(correct[..., 1].float().mean()),
        "state_accuracy": float(correct[..., 2].float().mean()),
        "cell_accuracy": float(correct.all(dim=-1).float().mean()),
        "changed_token_accuracy": float(correct[changed].float().mean()) if changed.any() else None,
        "changed_token_fraction": float(changed.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="MiniGrid-Empty-Random-6x6-v0")
    parser.add_argument("--train-episodes", type=int, default=24)
    parser.add_argument("--test-episodes", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--length", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--encoder-learning", choices=("prediction", "frozen"), default="prediction")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actions", type=int, nargs="+", default=[0, 1, 2], help="MiniGrid: left, right, forward; use 0 1 2 3 4 5 6 for all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures" / "minigrid_sensory_prediction")
    args = parser.parse_args()
    if min(args.train_episodes, args.test_episodes, args.max_steps, args.epochs, args.features, args.repeat, args.length) < 1 or args.learning_rate <= 0:
        parser.error("Episode counts, steps, epochs, core dimensions and learning rate must be positive")
    try:
        import gymnasium as gym
        import minigrid  # registers MiniGrid environments
    except ImportError as error:
        raise SystemExit("Install the optional dependencies: python -m pip install torch numpy gymnasium minigrid") from error

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    env = gym.make(args.env)
    try:
        episodes = collect_episodes(env, args.train_episodes + args.test_episodes, args.max_steps, args.seed, action_ids=args.actions)
        n_actions = env.action_space.n
    finally:
        env.close()
    train, test = episodes[:args.train_episodes], episodes[args.train_episodes:]
    encoded = [episode_tensors(episode) for episode in train]
    models = {}
    for name, memory in (("memoryless", False), ("lin_sequence", True)):
        torch.manual_seed(args.seed)
        models[name] = SensoryPredictor(train[0].observations.shape[1:], n_actions, args.features, args.repeat, args.length, args.threshold, memory, args.encoder_learning)
    rows = []
    for name, model in (("persistence", None), *models.items()):
        if model is not None:
            fit(model, encoded, args.epochs, args.learning_rate, args.seed)
        row = {"model": name, **evaluate(model, test)}
        rows.append(row)
        print(json.dumps(row, allow_nan=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    w = models["lin_sequence"].core.recurrent_matrix()
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "gymnasium": gym.__version__, "minigrid": minigrid.__version__},
        "train_seeds": [episode.seed for episode in train], "test_seeds": [episode.seed for episode in test],
        "train_transitions": sum(len(episode.actions) for episode in train), "test_transitions": sum(len(episode.actions) for episode in test),
        "core_diagnostics": {
            "normality_commutator": float(torch.linalg.matrix_norm(w.T @ w - w @ w.T) / w.square().sum().clamp_min(1e-12)),
            "spectral_norm": float(torch.linalg.matrix_norm(w, ord=2)),
            "spectral_radius": float(torch.linalg.eigvals(w).abs().max()),
            "frobenius_norm": float(torch.linalg.matrix_norm(w)),
        },
        "trainable_parameters": {name: sum(p.numel() for p in model.parameters() if p.requires_grad) for name, model in models.items()},
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Keep raw episodes so a different model can use precisely the same data.
    arrays = {}
    for index, episode in enumerate(episodes):
        for field in ("observations", "actions", "terminated", "truncated", "seed"):
            arrays[f"episode_{index}_{field}"] = np.asarray(getattr(episode, field))
    np.savez_compressed(args.output_dir / "trajectories.npz", **arrays)
    torch.save({name: model.state_dict() for name, model in models.items()}, args.output_dir / "models.pt")
    print(f"Saved metrics, configuration, trajectories, and weights to {args.output_dir}")


if __name__ == "__main__":
    main()
