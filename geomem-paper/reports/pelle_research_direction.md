# Pelle: discrete environments and sensory prediction

Research direction agreed on 2026-10-28. This is a plan, not a new empirical result.

## Question and first stage

Pelle, a master's student, is working on the relationship between environment/task structure and useful reservoir structure. He has identified promising measures of effective curvature. Their exact definitions and implementations still need to be linked here; no particular curvature measure is selected by this note.

The working hypothesis is that an environment/task measure, ideally effective curvature, predicts which reservoir weight structure works well, especially how normal or nonnormal the recurrent matrix is. The relationship, its direction, and the suitability of effective curvature remain open questions.

Start with simple discrete environments, initially MiniGrid through a Gym-compatible interface. First use sensory prediction on collected trajectories to make the feedback loop faster and more controllable. Establish a reproducible pattern here before moving to larger embodied tasks such as DMLab. Success on sensory prediction alone will not establish a benefit for navigation or reinforcement learning.

## Minimal supervised experiment contract

- Collect seeded episodes and retain observations, actions, and separate termination/truncation flags. Keep episode boundaries explicit; split training and evaluation by episode, with held-out environment seeds where applicable.
- Predict the next sensory observation from the current sensory history and the action leading to the target. Never feed the target or privileged simulator state into the predictor. Use the same collected trajectories and target encoding across model comparisons.
- Start with MiniGrid's compact categorical observation. Document which observation fields enter the model and which are excluded. Make replacing the environment, collection policy, observation adapter, and model straightforward.
- Compare against a memoryless observation-plus-action predictor and a simple persistence predictor before interpreting recurrent performance. One-step prediction may not require useful memory in every environment; measure that limitation before increasing task complexity or prediction horizon.
- Separate environment/task measurements from model measurements. Effective curvature, observation ambiguity, action commutativity, and memory requirements are related candidates, not interchangeable quantities. State graph construction, sampling policy, resolution, and scale when attaching a curvature value to a condition.
- Measure recurrent normality explicitly. For a real matrix $W$, normality means $W^T W = W W^T$. Record the diagnostic's normalization and distinguish fixed recurrent structure from learned parameters and evolving recurrent state. Control reservoir size, recurrent scale, input routing, data, and training budget when comparing structures; record unmatched quantities rather than attributing everything to nonnormality.

The initial implementation is infrastructure, not a curvature sweep. Once Pelle supplies the measure definitions, attach them to environment conditions and compare prediction error and memory benefit across matched reservoir families and seeds. Any new scientific analysis belongs in the existing results synthesis and relevant comparison figures. Evaluate new variants on the same applicable metrics; explain any omitted legacy metric whose task or target is incompatible.

## Two parallel implementation assignments

| Workstream | Deliverable | Completion criteria |
| --- | --- | --- |
| Minimal sensory prediction | A standalone supervised Gym-compatible trajectory collector and readable MiniGrid sensory-prediction training script, with no Sample Factory dependency, sharing the full agent's encoder and recurrent core. | A reproducible CPU smoke run; explicit episode splits and action/target alignment; minimal dependencies; clear places for Pelle to change the environment, observation, model, and loss. |
| Full IntrMotiv / Lin adaptation | A dedicated `geomem` experiment folder retaining the APPO/Sample Factory robotic-agent training setting, using the Lin 2026 model lineage and useful, verified IntrMotiv improvements, adapted to MiniGrid. | Inspect the actual source before selecting changes; preserve the actor-critic and APPO training path; document provenance and retained/changed/omitted mechanisms; replace environment-specific visual/depth input paths appropriately; run training; provide a concise technical guide covering input routing, core state/update rules, learned versus fixed parameters, and run commands. |

Keep reusable environment code in `geomem-env`, reusable model code in `geomem-models`, and runnable experiments in `geomem-experiments`. The dedicated adaptation folder is `geomem-experiments/experiments/geomem/`. Preserve the full version's Sample Factory APPO training infrastructure and defining model mechanisms; remove or adapt environment-specific components explicitly. Do not substitute a supervised trainer for the full version or call a generic reservoir an equivalent Lin/IntrMotiv model.

The two workstreams have different objectives: supervised sensory prediction for the minimal version, and APPO reinforcement learning for the full version. Share compatible observation conventions and environment utilities where useful, but do not force identical losses or metrics across these tasks. Within each task, compare variants on the same applicable metrics and data/evaluation protocol. Full-agent checks must include the observation-to-action path, recurrent episode resets, rollout/learner consistency, and an actual Sample Factory training smoke run.

## Architecture comparability across tasks

The minimal version must use an encoder and recurrent core comparable to the full RL agent, so findings can inform the later RL experiments. Implement these components once in `geomem-models`, without Sample Factory imports, and use them in both versions. Match sensory encoding, input routing, encoder configuration, core state shape, transition rule, initialization, and episode resets. Use a supervised sensory-prediction head in the minimal version and policy/value heads in the full version. The minimal training loop must remain independent of Sample Factory.

Do not use a generic reservoir as the minimal default while the full agent uses a different Lin/IntrMotiv encoder and core. Future recurrent-matrix interventions should be available in both paths with the same definitions. Record which parameters each objective updates: shared architecture alone does not guarantee matched learned representations. Keep learning objectives, encoder update routes, and offline versus policy-driven data collection explicit, and use matched or frozen-encoder controls when isolating core effects. Transfer of a finding is a hypothesis until checked in the full RL setting.

## Reuse and environment extensibility

Reuse Sample Factory's existing MiniGrid integration wherever compatible with the shared sensory interface. Check existing factories, wrappers, and registration before adding replacements; explain any small adapter required for the shared encoder. The standalone supervised version must remain independent of Sample Factory.

Both versions should be able to extend to other environments. Keep Gym environment construction and observation adaptation separate from the shared encoder/core and training orchestration. For a new sensory modality, replace the observation frontend and, for supervised learning, the target encoding/loss while retaining the recurrent mechanism. Do not claim arbitrary RGB, vector, or continuous-action support merely because MiniGrid runs; document the supported interface and exact extension points. Prefer small explicit functions and existing integrations over a new framework.

## Pending scientific inputs

Pelle's curvature definitions/code, the first environment family that varies those measures, and the matched reservoir comparison protocol remain to be specified. Implementation can proceed with a replaceable measurement interface and without inventing curvature values or claiming the desired pattern already exists.

## Implementation entry points

- [Standalone supervised guide](../../geomem-experiments/experiments/minigrid_sensory_prediction.md): collection, prediction, controls, and encoder export.
- [Full APPO guide](../../geomem-experiments/experiments/geomem/README.md): reused IntrMotiv learning path, MiniGrid adaptation, and frozen-encoder transfer.
- [Shared encoder and core](../../geomem-models/src/geomem_models/lin_intrmotiv.py): common representation code, independent of Sample Factory.
- [Verification in the results synthesis](current_results_synthesis.md#minigrid-infrastructure-verification): small execution checks, separate from scientific evidence.
