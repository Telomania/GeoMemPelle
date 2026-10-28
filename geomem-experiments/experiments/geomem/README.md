# Pelle: MiniGrid with the Lin/IntrMotiv APPO agent

This is the **full reinforcement-learning path**: Sample Factory APPO, the actual IntrMotiv actor–critic/learner, and the shared Lin sequence backbone. The [small sensory-prediction task](../MINIGRID_SENSORY_PREDICTION.md) imports the same encoder and core, replacing the policy/value heads and learning objective.

## Run

Use the existing `SF_hipposlam` fork, not an unrelated PyPI Sample Factory version. Its IntrMotiv modules and learner extensions are runtime dependencies; nothing in that checkout is modified. [PROVENANCE.json](PROVENANCE.json) records the inspected commit and source hashes.

From `geomem-experiments`, in an environment containing that fork, PyTorch, Gymnasium and MiniGrid:

```bash
python -m experiments.geomem.train --seed=0
```

The default uses two asynchronous CPU workers, two environments each, and `MiniGrid-Empty-Random-6x6-v0`. Results/checkpoints go to `train_dir/geomem_minigrid/`. Set `SF_HIPPOSLAM_ROOT` to the fork checkout if it is not already importable. On this host the verified interpreter is `/home/xiaoxiong/miniforge3/envs/SF_git/bin/python`; MiniGrid is currently available through `PYTHONPATH=/tmp/geomem-minigrid-deps` (temporary installation; install `minigrid==2.3.1` in your environment for durable use).

Quick asynchronous wiring check:

```bash
python -m experiments.geomem.train --seed=0 --experiment=smoke \
  --train_for_env_steps=512 --batch_size=64 --rollout=16 --recurrence=16 \
  --decorrelate_experience_max_seconds=0 --set_workers_cpu_affinity=False
```

For single-process debugging add `--serial_mode=True --async_rl=False --num_workers=1 --num_envs_per_worker=1`. This changes execution mode, not the model or learner.

## Input and state

`observation['image'] → fixed categorical one-hot → sparse DG → additive CA3 register → policy/value MLPs`. The **same current one-hot observation also bypasses CA3** and joins the readout. MiniGrid's own `ImgObsWrapper` removes mission/direction; Sample Factory supplies its standard dictionary and rollout wrappers. The local SF fork has no MiniGrid-specific integration, so no old SF1 wrapper is reimplemented.

The categorical view is $7\times7\times3$ (object/color/state), giving 980 one-hot features. This replaces the fixed RGB/ResNet frontend. The bypass replaces DMLab's depth/instruction stream with the current categorical view; it is not a simulated depth sensor. Absolute pose, compass, reward and future action do not enter DG. All seven native MiniGrid actions remain available to the policy. The mission-free frontend suits fixed-task environments; instruction-dependent tasks need a frontend extension.

Open the shared implementation at `geomem-models/src/geomem_models/lin_intrmotiv.py`. With $F=16$, $R=2$, $L=8$, the CA3 state has $E=R+L-1=9$ slots per feature:

$$
d_t=\operatorname{ReLU}\!\left((A x_t-\mu)/\sqrt{v+\epsilon}-\theta\right),\qquad
H_t[f,k]=\mathbf{1}_{k>0}H_{t-1}[f,k-1]+\mathbf{1}_{k<R}d_t[f].
$$

The state is flattened feature-first (144 CA3 entries), followed by the current 980-entry bypass. **CA3 has no trainable recurrent weights and no tanh**: the fixed nilpotent shift matrix is available from `LinSequenceCore.recurrent_matrix()`. Environment resets clear state. Learner replay receives stored initial states and splits sequences at episode boundaries. The `rnn_type=gru` config value only selects SF's state-buffer sizing convention; no GRU exists in this model.

Change `--Hippo_n_feature`, `--Hippo_R`, `--Hippo_L`, or `--DG_BN_intercept` for common representation settings. Change `LinSequenceCore` once to test a different recurrence in **both** workflows; keep dimensions, input scaling, and training data controlled when attributing effects to normality.

## What learns

| Component | Update |
| --- | --- |
| Categorical frontend and CA3 shift | Fixed; CA3 activity advances each observation. |
| DG projection $A$ | Upstream temporal-event credit, separate from PPO. Default `encourage` rewards an active dominant arrival DG unit using predecessor-distance credit; differentiable unused-unit recruitment uses pre-threshold logits. |
| DG normalization | Same running moments for actor and learner; no inference updates. After each optimizer step, source learner normalizes DG rows and calibrates moments from valid cached inputs, under the policy lock. |
| Policy/value MLPs | APPO clipped policy loss, value loss and entropy term. Default `ppo_dg_gradient=stop` blocks this loss from changing DG. |

The default worker uses **intrinsic temporal-transition rewards**, preserving the flat IntrMotiv setting; MiniGrid's goal reward is recorded separately. To train the worker on the environment's reward, use `--advantage_reward_source=external --use_external=True --use_internal=False`; DG still uses its temporal objective. Optional HRL/graph managers, recruitment, contextual feedback, DMLab sensors and robot telemetry are outside this baseline.

Prediction uses the same routing and recurrence, but its default cross-entropy trains DG directly and adds the candidate action only at the prediction readout. This objective and its fixed collection policy differ from APPO. Matching architecture alone does not prove transfer of a curvature–performance relationship. For a stricter representation control, load the prediction export after SF initialization and freeze both DG weights and moments:

```bash
python -m experiments.geomem.train --experiment=frozen_control \
  --shared_encoder_checkpoint=figures/minigrid_sensory_prediction_frozen_smoke/shared_encoder.pt \
  --transfer_freeze_dg=True
```

This checks the complete representation configuration and loads all DG tensors/statistics exactly; a matching seed alone would not ensure matching initialization. Align environment and action support too: prediction defaults to random navigation actions 0/1/2, while APPO selects among all seven; use prediction's `--actions 0 1 2 3 4 5 6` when appropriate.

## Changing environments and checking changes

`--minigrid_env=...` changes MiniGrid tasks. For another Gymnasium environment, pass `--env_factory=your_module:make_env` (callable takes `cfg, render_mode=None`). Reuse its existing SF integration if available. For a different observation encoding, pass `--representation_factory=your_module:Encoder`; the small pure-PyTorch interface is `CategoricalDGEncoder` (constructor `image_shape, features, threshold`; `DG_projection`, `projection_input`, `forward`, `bypass_size`, `get_out_size`). Both prediction and RL should use that same frontend; neither the core nor learner needs an environment-specific rewrite. The supplied categorical frontend is **not** suitable for arbitrary RGB or continuous observations without adaptation.

```bash
python -m unittest discover -s experiments/geomem/tests -v
```

Tests check source/shared DG equivalence, single-step and packed replay, impulse lifetime, state reset and gradient ownership. [VALIDATION.json](VALIDATION.json) records real asynchronous training and exact frozen-transfer checks. These short runs establish functioning updates, not task mastery or a curvature result.
