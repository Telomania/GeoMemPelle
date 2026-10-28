# MiniGrid sensory prediction: Pelle's minimal entry point

This is standalone supervised learning on offline trajectories. It does not import Sample Factory. Its encoder and recurrent core are the **same shared classes** used by the full [MiniGrid APPO adaptation](geomem/README.md); only the task head and learning objective differ.

## Run

From `geomem-experiments`, with Python 3.10+ and CPU PyTorch:

```bash
python -m pip install numpy torch gymnasium minigrid
python experiments/minigrid_sensory_prediction.py
# Fast execution check:
python experiments/minigrid_sensory_prediction.py --train-episodes 4 --test-episodes 2 --max-steps 24 --epochs 2
# Fixed encoder control:
python experiments/minigrid_sensory_prediction.py --encoder-learning frozen --output-dir figures/minigrid_prediction_frozen
```

The default is `MiniGrid-Empty-Random-6x6-v0`; random collection uses left/right/forward. `--actions 0 1 2 3 4 5 6` includes pickup/drop/toggle/done for other tasks. Episodes reset on termination or truncation. The training and test episodes use disjoint reset seeds; this holds out sampled trajectories, not necessarily distinct layouts. The first environment is deliberately simple, so memory may offer no advantage.

## Follow the data and gradients

1. `collect_episodes` stores $T+1$ categorical images and $T$ actions. Only the partial image's object/color/state channels enter the model; mission, absolute direction, rewards and simulator state are excluded. Input image $o_t$ and action $a_t$ predict $o_{t+1}$. Targets never enter the current forward pass; hidden state resets between episodes.
2. `CategoricalDGEncoder` one-hot encodes each image cell (11/6/3 categories), applies a learned DG projection, running normalization and thresholded ReLU, and preserves the current one-hot image as a sensory bypass. `LinSequenceCore` shifts a fixed finite register and adds each DG activation into its first `repeat` slots. Defaults match APPO: 16 DG features, repeat 2, length 8, threshold 2. This is a fixed nonnormal shift core, not a trainable generic RNN.
3. `SensoryPredictor` concatenates the core output with the selected action, then uses one linear head to predict the next image's three categorical distributions. The loss is mean cross-entropy across time, cells and channels. Actions enter the prediction head only, so the shared representation has the same observation-only routing as APPO.

By default, cross-entropy updates the DG projection and readout through the fixed core; the shift itself has no trainable parameters. Full APPO instead uses its own temporal-event DG objective and policy/value objectives. **Shared architecture does not make these learning objectives equivalent.** The optional `--encoder-learning frozen` holds DG weights and normalization fixed after initial training-only calibration. This isolates readout learning and is the simplest cross-task representation control.

Normalization uses stored moments during every forward pass. Training-only observations initialize them; after each learned-DG update, projection rows are normalized to unit length, then cached current inputs refresh moments using the updated weights, matching IntrMotiv's update order. Test episodes never update moments. A single-transition training episode uses the training calibration pool for that refresh. Full episode backpropagation is used here; APPO uses rollout sequences.

## Read the output; change only what you need

`figures/minigrid_sensory_prediction/` contains `metrics.csv`, `run.json`, saved `trajectories.npz` and `models.pt`. The memoryless control uses identical encoder/core/head dimensions and initialization but resets the core **before every timestep**. Persistence copies the current image. All three receive the same held-out episodes and report token, object/color/state, whole-cell and changed-token accuracy. Learned predictors also report cross-entropy; deterministic persistence has no finite comparable cross-entropy. The changed-token fraction exposes static-pixel dominance; no changes gives a blank changed-token score, not zero.

`run.json` records seeds, versions, counts and the CA3 matrix diagnostic $\|W^T W-WW^T\|_F/\|W\|_F^2$, excluding the sensory bypass. The fixed shift has zero spectral radius but nonzero operator norm. These are infrastructure diagnostics, not a curvature result or a normality sweep.

- Change collection/environment in `geomem_env/gym_trajectories.py`; the collector supports Gymnasium and legacy Gym with discrete actions. Other observation types need a replacement observation adapter and encoder/head.
- Change encoder/core once in `geomem_models/lin_intrmotiv.py` to affect both supervised and APPO paths. Tune `--features`, `--repeat`, `--length`, `--threshold` consistently across both.
- Change the prediction target or loss in this script. Keep episode splits and action alignment intact.

`models.pt` exposes `encoder` (weights **and** normalization moments), `representation_config` (image shape and all four core/encoder settings), and both predictors' complete state dictionaries. `shared_encoder.pt` contains only the encoder and config. APPO can load either with `--shared_encoder_checkpoint /absolute/path/shared_encoder.pt --transfer_freeze_dg True` for a matched frozen comparison. A shared seed alone is insufficient because trainers may initialize modules differently. Checkpoints are local outputs ignored by Git; preserve them when comparing tasks.
