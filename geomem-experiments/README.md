# geomem-experiments

Runnable experiment scripts and generated artifacts for GeoMem.

## Pelle's MiniGrid Workflows

The two workflows share the categorical DG encoder and Lin sequence core in `geomem-models/src/geomem_models/lin_intrmotiv.py`.

- [Standalone sensory prediction](experiments/minigrid_sensory_prediction.md): collect trajectories and train a supervised next-observation predictor without Sample Factory.
- [Full APPO agent](experiments/geomem/README.md): adapt the IntrMotiv agent to MiniGrid while retaining Sample Factory and its separate encoder/worker learning paths.
- [Research direction](../geomem-paper/reports/pelle_research_direction.md): hypothesis, comparison constraints, and pending curvature definitions.

## Owns

- Benchmark and sweep scripts under `experiments/`.
- Generated CSV/SVG outputs under `figures/`.
- Experiment-facing documentation.

## Dependency Direction

This subproject is allowed to depend on `geomem-env` and `geomem-models`. The first migration pass keeps mirrored legacy scripts runnable in place; new scripts should prefer the package APIs from the sibling subprojects.

## Fast Smoke Tests

```bash
python3 experiments/temporal_basis_benchmark.py
python3 experiments/commutativity_probe.py
python3 experiments/observation_ambiguity.py
python3 experiments/transition_ambiguity.py
python3 -m py_compile experiments/*.py
```

Do not run the expensive RNN sweeps as part of routine migration checks.
