"""Run from geomem-experiments: python -m experiments.geomem.train."""
from __future__ import annotations
import os
from pathlib import Path
import sys

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE / "geomem-models" / "src"))
if os.environ.get("SF_HIPPOSLAM_ROOT"):
    sys.path.insert(0, os.environ["SF_HIPPOSLAM_ROOT"])

import gymnasium as gym
import minigrid  # registers MiniGrid environments
import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence
from geomem_models.lin_intrmotiv import CategoricalDGEncoder, LinSequenceCore
from sample_factory.algo.utils.model_context import global_learner_factory, global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.encoder import Encoder
from sample_factory.train import run_rl
from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import make_hipposlam_actor_critic
from sf_working_directories.IntrMotiv.dmlab.custom_core import SimpleSequenceWithBypassCore
from sf_working_directories.IntrMotiv.dmlab.custom_decoder import make_hipposlam_decoder
from sf_working_directories.IntrMotiv.dmlab.custom_learner import make_hipposlam_learner
from sf_working_directories.IntrMotiv.dmlab.custom_params import add_hipposlam_env_args, hipposlam_override_defaults


class MiniGridImage(gym.ObservationWrapper):
    """Partial categorical view only: no mission, compass, pose, or true depth."""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict({"obs": env.observation_space["image"]})

    def observation(self, observation):
        return {"obs": observation["image"]}


def make_env(full_env_name, cfg, env_config=None, render_mode=None):
    return MiniGridImage(gym.make(cfg.minigrid_env, render_mode=render_mode,
                                 max_steps=cfg.minigrid_max_steps))


class MiniGridEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        self.shared = CategoricalDGEncoder(obs_space["obs"].shape,
                                           cfg.Hippo_n_feature, cfg.DG_BN_intercept)

    @property
    def DG_projection(self):
        return self.shared.DG_projection

    def projection_input(self, obs_dict):
        return self.shared.projection_input(obs_dict["obs"])

    def forward(self, obs_dict):
        return self.shared(obs_dict["obs"])

    def get_out_size(self):
        return self.shared.get_out_size()


class MiniGridCore(SimpleSequenceWithBypassCore):
    """Upstream contract with the same pure core used by prediction.

    SF splits replay at episode boundaries and supplies each sequence's stored
    initial state. Optional HRL/context modes are deliberately unsupported.
    """
    def __init__(self, cfg, input_size):
        super().__init__(cfg, input_size)
        self.shared = LinSequenceCore(cfg.Hippo_n_feature, cfg.Hippo_R,
                                      cfg.Hippo_L, input_size - cfg.Hippo_n_feature)

    def forward(self, head_output, rnn_states, **kwargs):
        if kwargs:
            raise ValueError("Graph replay conditioning is disabled in this adapter")
        if isinstance(head_output, PackedSequence):
            padded, lengths = pad_packed_sequence(head_output)
            state = rnn_states
            outputs = []
            for step, frame in enumerate(padded):
                candidate, _ = self.shared(frame, state)
                active = (lengths > step).to(frame.device)[:, None]
                state = torch.where(active, candidate, state)
                outputs.append(state)
            packed = pack_padded_sequence(torch.stack(outputs), lengths.cpu(),
                                          enforce_sorted=head_output.sorted_indices is None)
            return packed, state
        self.last_dg_activity = head_output[:, :self.Hippo_n_feature]
        return self.shared(head_output, rnn_states)


def make_encoder(cfg, obs_space):
    return MiniGridEncoder(cfg, obs_space)


def make_core(cfg, input_size):
    return MiniGridCore(cfg, input_size)


def register_components():
    register_env("geomem_minigrid", make_env)
    factory = global_model_factory()
    factory.register_encoder_factory(make_encoder)
    factory.register_model_core_factory(make_core)
    factory.register_decoder_factory(make_hipposlam_decoder)
    factory.register_actor_critic_factory(make_hipposlam_actor_critic)
    global_learner_factory().register_learner_factory(make_hipposlam_learner)


def parse_args(argv=None):
    parser, _ = parse_sf_args(argv)
    add_hipposlam_env_args(parser)
    hipposlam_override_defaults(parser)
    parser.add_argument("--minigrid_env", default="MiniGrid-Empty-6x6-v0")
    parser.add_argument("--minigrid_max_steps", type=int, default=128)
    parser.set_defaults(
        env="geomem_minigrid", algo="APPO", device="cpu", async_rl=True,
        num_workers=2, num_envs_per_worker=2, worker_num_splits=1,
        rollout=32, recurrence=32, batch_size=128, num_batches_per_epoch=1, num_epochs=1,
        train_for_env_steps=100000, experiment="geomem_minigrid",
        train_dir=str(WORKSPACE / "geomem-experiments" / "train_dir"),
        Hippo_n_feature=16, Hippo_R=2, Hippo_L=8, DG_name="batchnorm_relu",
        DG_BN_intercept=2.0, dg_batchnorm_semantics="running_poststep_atomic",
        core_name="BypassSS", use_rnn=True, rnn_type="gru", rnn_num_layers=1,
        normalize_input=False, normalize_returns=False, obs_scale=1.0,
        obs_subtract_mean=0.0, env_frameskip=1, decoder_mlp_layers=[64, 64],
        actor_critic_share_weights=True, distance_learning=True, double_value=False,
        ppo_dg_gradient="stop", advantage_reward_source="internal", use_internal=True,
        use_external=False, extra_encoder_losses=True, extra_decoder_loss=False,
        online_spatial_telemetry=False, hrl_controllable_graph=False,
        save_initial_checkpoint=True, checkpoint_frame_targets="",
        with_wandb=False, with_pbt=False, experiment_summaries_interval=5,
    )
    cfg = parse_full_cfg(parser, argv)
    required = {
        "core_name": "BypassSS", "distance_learning": True,
        "hrl_controllable_graph": False, "dg_context_feedback": "none",
        "dg_goal_input": "none", "intrinsic_goal_mode": "none",
        "ca3_state_readout_mode": "off", "controller_learning": "ppo",
        "dg_orthogonal_recruitment": False, "actor_critic_share_weights": True,
        "dg_batchnorm_semantics": "running_poststep_atomic",
        "normalize_input": False, "obs_scale": 1.0, "env_frameskip": 1,
    }
    for key, expected in required.items():
        if getattr(cfg, key) != expected:
            raise ValueError(f"MiniGrid baseline requires {key}={expected!r}")
    if min(cfg.Hippo_n_feature, cfg.Hippo_R, cfg.Hippo_L, cfg.minigrid_max_steps) < 1:
        raise ValueError("Feature count, repeat, length, and max steps must be positive")
    probe = make_env(cfg.env, cfg)
    try:
        backbone = CategoricalDGEncoder(probe.observation_space["obs"].shape,
                                         cfg.Hippo_n_feature, cfg.DG_BN_intercept)
        cfg.rnn_size = cfg.Hippo_n_feature * (cfg.Hippo_R + cfg.Hippo_L - 1) + backbone.bypass_size
    finally:
        probe.close()
    # rnn_type sizes SF's state buffer only; this model contains NO GRU.
    cfg.rnn_type, cfg.rnn_num_layers, cfg.use_rnn = "gru", 1, True
    cfg.rnn_persistent_state_size = 0
    cfg.head_l1_size = cfg.Hippo_n_feature
    cfg.extra_policy_output_shapes = ()
    cfg.cli_args.update(rnn_size=cfg.rnn_size, rnn_type="gru", rnn_num_layers=1, use_rnn=True)
    return cfg


def main(argv=None):
    register_components()
    return run_rl(parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
