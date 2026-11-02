# Configuration

The Concertina policy type is `concertina` (`--policy.type=concertina`). Its configuration, `ConcertinaConfig`
(`src/concertina/policy/configuration_concertina.py`), is a LeRobot `PreTrainedConfig` made of two sub-configurations and the
optimizer presets:

| Field | Type | Contents |
|---|---|---|
| `model` | `ConcertinaModelConfig` | backbone, action layout, training objective, decoding recipe (`model.inference`) |
| `data` | `ConcertinaDataConfig` | dataset mixture, normalization, gripper calibration, data quality |
| `optimizer_*`, `scheduler_*` | | AdamW and cosine-with-warmup presets |

Both sub-configurations are saved in the checkpoint's `config.json`, so an evaluation rebuilds exactly the model and
the per-dataset processing of training. Every field can be set in a YAML file (`--config_path`) or on the command
line, for example `--policy.model.chunk_size=20` or `--policy.model.inference.remask_steps=0`.

The defaults of `model` are the released model. The options that are not constants exist for one of three reasons:
they are needed to train or run the model on other data or hardware, they are implementation branches studied in the
paper's ablations and analyses (see [Ablation branches](#ablation-branches)), or they are decoding strategies kept
for experimentation. The kept branches were checked against the original implementation of the paper
experiments (see [reproduction.md](reproduction.md)).

[Ablation branches](#ablation-branches) names the paper section that studies each branch, and
[Paper terms](#paper-terms) maps the paper's notation to the configuration fields.

## Top level

| Field | Default | Meaning |
|---|---|---|
| `normalization_mapping` | `VISUAL: IDENTITY, STATE: MEAN_STD, ACTION: MEAN_STD` | LeRobot normalization modes; Concertina supports `MEAN_STD` and `IDENTITY` for state and action |
| `dtype` | `null` | parameter storage dtype (LeRobot's field): `null` keeps bfloat16; training uses `float32` (see [Precision](#precision)) |
| `push_to_hub` | `false` | upstream defaults to `true`, which requires a Hub repo id |
| `optimizer_lr` | `2.5e-5` | AdamW peak learning rate |
| `optimizer_betas` / `optimizer_eps` / `optimizer_weight_decay` | `(0.9, 0.95)` / `1e-8` / `0.01` | AdamW |
| `optimizer_grad_clip_norm` | `1.0` | gradient clipping |
| `scheduler_warmup_steps` / `scheduler_decay_steps` / `scheduler_decay_lr` | `1000` / `100000` / `2.5e-6` | cosine decay with linear warmup |

The presets apply unless the training configuration sets `use_policy_training_preset: false` and its own
`optimizer` / `scheduler`. `chunk_size` and `n_action_steps` are read from `model.chunk_size` and
`model.inference.n_action_steps`.

### Precision

| | Parameter storage | Compute |
|---|---|---|
| Training | `policy.dtype: float32` (fp32 weights and optimizer state) | bf16 autocast (`accelerator.mixed_precision: bf16`) |
| Inference and evaluation | bf16 (released checkpoints are bf16) | bf16 |

Decoding always runs with bf16 weights: a policy that stores fp32 weights (a `concertina-train` checkpoint) decodes
with bf16 copies of them, so it matches the bf16 weights the released checkpoints were evaluated with. Training
with bf16 storage (`policy.dtype` unset) loses most updates to bf16 rounding: in a single-GPU check with the 4B
model, a training update changed 27.9% of the trainable weights with bf16 storage against 99.993% with fp32
storage. Every configuration in `configs/train/` sets `dtype: float32` and `accelerator.mixed_precision: bf16`.

## Model (`policy.model`)

### Backbone and kernels

| Field | Default | Meaning |
|---|---|---|
| `backbone_path` | `""` | path or Hub id of the Qwen3-VL-layout backbone. Empty in exported checkpoints, which bundle `backbone/` (configuration, tokenizer and image processor; weights come from `model.safetensors`) |
| `vision_attn_implementation` | `sdpa` | `sdpa` or `flash_attention_2` (needs flash-attn) for the vision tower |
| `training_attn_implementation` | `cudnn_sdpa_native_gqa` | text attention in train mode: cuDNN with native grouped-query attention, or `sdpa`. Decoding always uses SDPA |
| `gradient_checkpointing` | `true` | |
| `use_liger_kernels` | `true` | Liger RMSNorm/RoPE/SwiGLU kernels in the text decoder (the `liger` extra), bound to this model instance only. They are Triton GPU kernels: set `false` to run the model on CPU |

### Inputs and action layout

Actions and states use a canonical 20-dimensional layout: two arms x (xyz position, 6D rotation, gripper). The
gripper channels are indices 9 and 19; single-arm data fills the first arm. Images are resized to 224 x 224.

| Field | Default | Meaning |
|---|---|---|
| `camera_keys` | `[observation.images.head]` | canonical camera keys in prompt order; datasets map their cameras to them (`DatasetSpec.camera_map`) |
| `chunk_size` | `10` | actions predicted per chunk; a multiple of `block_size` |
| `block_size` | `10` | actions per diffusion block; blocks are generated one after another, the actions of a block jointly |
| `gripper_indices` | `[9, 19]` | dimensions predicted as gripper-closure logits and trained with a binary cross-entropy; empty regresses the grippers like the other dimensions |
| `gripper_target` | `probability` | `probability`: calibrated closure targets in [0, 1] (`data` calibration); `binary`: source values thresholded at 0.5 |
| `action_dim_loss_weights` | 500 (xyz), 10 (rotation), 1 (gripper), per arm | per-dimension weights of the action reconstruction loss (smooth L1 on continuous dimensions) |
| `state_mask_dims`, `state_mask_prob`, `state_mask_ratio_range`, `state_mask_fill_value` | `[2, 12]`, `0.5`, `(1.0, 1.0)`, `0.0` | training-time state dropout: with probability `state_mask_prob`, a fraction (drawn from the range) of `state_mask_dims` is set to the fill value |
| `random_resized_crop` | `true` | training-time random resized crop (scale 0.9-1.0) |

### Training objective

| Field | Default | Meaning |
|---|---|---|
| `corruption.mode` | `log_linear_mask` | `log_linear_mask`: each action token is masked with probability t (block diffusion time), supervised tokens weighted by 1/t. `bard_mixed`: each token stays clean, receives a visible corruption or is masked, with the BARD state probabilities; visible and masked tokens are supervised |
| `corruption.visible_corruption` | `uniform` | `bard_mixed` only: `uniform` (a draw from `visible_uniform_range`) or `gaussian` (clean value + N(0, `visible_noise_std`)) |
| `corruption.visible_noise_std`, `corruption.visible_uniform_range`, `corruption.probability_floor` | `1.0`, `(-1, 1)`, `1e-3` | `bard_mixed` parameters |
| `corruption.loss_weighting` | `none` | `bard_mixed` only: `none` or `inverse_t` |
| `mask_value` | `zeros` | value of masked action tokens: `zeros`, `ones` or `random` |
| `full_mask_probability` | `0.0` | probability that a sample has all its action tokens masked (`log_linear_mask` only) |
| `visible_action_context.*` | `strategy: none` | Gaussian perturbation of the action context that later blocks attend to; see below |
| `token_risk` | `true` | token-risk head: predicts each action token's log reconstruction loss (auxiliary loss, weight 0.05); its negation is the `token_risk` decoding confidence |
| `action_history.*` | `enabled: false` | condition each chunk on previously executed actions; see below |
| `freeze_action_heads`, `freeze_backbone_layers` | `false`, `[]` | freeze the action projector and head, and text decoder layers by index |
| `rng_seed` | `0` | seed of the per-sample training randomness; `concertina-train` sets it to the run's `seed` |

Training randomness (augmentation, masking, corruption) is drawn from generators keyed by
`(rng_seed, training step, dataset_index, index)`, so it does not depend on the number of workers or GPUs. The
training-step counter is saved with the checkpoint; `concertina-train` restarts it when fine-tuning and keeps it when
resuming.

**Visible action context** (`visible_action_context`): with probability `apply_probability`, the clean actions of
earlier blocks and the unmasked tokens of the current block are perturbed with Gaussian noise of standard deviation
`noise_std` (one value, or one per dimension; gripper dimensions are never perturbed). `strategy` draws the noise
per token (`shared_gaussian`) or per block (`block_shared_gaussian`). `repair_loss_weight` adds a regression loss of
the clean continuous dimensions on the perturbed tokens; `apply_to_history` also perturbs the action history.

**Action history** (`action_history`): `horizon` executed actions (oldest first) condition each chunk. `mode:
prefix` appends them after the state token of the conditioning prefix; `mode: native` encodes them like clean action
blocks before the first generated block (`horizon` a multiple of `block_size`). Slots before the episode start are
hidden (`missing_history: mask`) or filled with N(0, `noise_std`) noise (`random_noise`). Training reads the
history from the first `horizon` steps of the action window. Exclusive with real-time chunking.

## Decoding recipe (`policy.model.inference`)

The recipe saved with a checkpoint is the one of its configuration; the evaluation YAMLs in `configs/eval/` replace
it with the recipe of the reported results of each benchmark (see [eval.md](eval.md)).

| Field | Default | Meaning |
|---|---|---|
| `n_action_steps` | `10` | actions executed from each decoded chunk |
| `draft_steps` | `4` | draft rounds per block; every masked token is committed by the last round |
| `confidence` | `token_risk` | per-token confidence: `token_risk`, `hidden_norm`, `prediction_stability`, `combined` or `random` |
| `commit_strategy` | `dynamic` | tokens committed per round: `dynamic` (the round budget, raised to the tokens above `commit_confidence_threshold`, at most `commit_max_ratio` of the block), `static` or `sequential` (left to right) |
| `commit_confidence_threshold`, `commit_max_ratio` | `0.0`, `0.5` | `dynamic` commit parameters |
| `draft_ranking` | `confidence` | which tokens a round commits: `confidence`, `spread_risk` (temporal coverage) or another score (`random`, `hidden_norm`, `hidden_norm_small`, `prediction_stability`, `combined`, `*_fp32`, `stability_norm_bootstrap`) |
| `draft_seed` | `null` | seed of the `random` ranking |
| `visible_revision`, `revision_margin` | `true`, `0.0` | replace a committed token when a later round is more confident |
| `remask_steps`, `remask_ratio` | `2`, `0.2` | remasking rounds per finished block (0 disables) and the fraction remasked per round |
| `remask_cooldown_steps`, `remask_decay_budget`, `remask_accept_if_improved`, `remask_improvement_margin`, `remask_early_stop_patience`, `remask_keep_best`, `remask_no_repeat` | `1`, `true`, `true`, `0.0`, `2`, `true`, `false` | remasking schedule and acceptance |
| `remask_keep_draft_translation` | `false` | keep the drafted xyz of both arms during remasking (10-action blocks, first 10 actions only) |
| `rtc`, `rtc_prefix_source`, `rtc_execution_horizon` | `false`, `executed_tail`, `0` | real-time chunking: each chunk is decoded with actions of the previous chunk fixed as its prefix |

Recipes used for the reported results:

| Recipe | `draft_steps` | `visible_revision` | `remask_steps` / `remask_ratio` | `remask_cooldown_steps` / `decay_budget` / `accept_if_improved` / `keep_best` |
|---|---|---|---|---|
| CALVIN and LIBERO results (`configs/eval/`) | 4 | `false` | 5 / 0.15 | 0 / `false` / `false` / `false` |
| Default, saved with the training configurations | 4 | `true` | 2 / 0.2 | 1 / `true` / `true` / `true` |

Both use `n_action_steps: 10`, `confidence: token_risk` and dynamic commits (threshold 0, at most half a block per
round). One-pass decoding of every block from all-masked tokens is `draft_steps: 1, remask_steps: 0`.

## Ablation branches

Paper sections are given by title; "App." marks an appendix section.

| Branch | Options | Released model | Paper section |
|---|---|---|---|
| Block topology | `chunk_size`, `block_size` (for example chunks of 10, 20, 50 or 100 actions in blocks of 5 or 10) | pretraining 50/10, then 20/10; fine-tuning 10/10 | Blockwise Horizon Partition, Pretraining Weighting, and Execution Horizon |
| Execution horizon | `inference.n_action_steps` | 10 | Blockwise Horizon Partition, Pretraining Weighting, and Execution Horizon; App. Decoding Budget and Closed-Loop Replanning Frequency |
| Corruption process | `corruption.mode`: `log_linear_mask` (mask-only) or `bard_mixed` (three-state mixed, with `visible_corruption` `uniform` or `gaussian`) | `log_linear_mask` | Blockwise Horizon Partition, Pretraining Weighting, and Execution Horizon |
| Reconstruction weighting | `corruption.loss_weighting`: `none` or `inverse_t` (`bard_mixed`) | 1/t weighting of `log_linear_mask` | Blockwise Horizon Partition, Pretraining Weighting, and Execution Horizon |
| Pretraining weighting and regularization | `action_dim_loss_weights`, `state_mask_*`, `freeze_backbone_layers`, `random_resized_crop` | 500/10/1 per arm; state masking in pretraining and the CALVIN fine-tuning; see the configurations | Blockwise Horizon Partition, Pretraining Weighting, and Execution Horizon |
| Gripper objective | `gripper_target: probability` (calibrated closure BCE), `binary` (binarized BCE), or `gripper_indices: []` (Smooth-L1 regression) | `probability` | Co-Design of Action Representation and Objective in the Shared Hidden Space; App. Probability-Valued Gripper Supervision |
| Trainable action interfaces and backbone | `freeze_action_heads`, `freeze_backbone_layers` | see the configurations | Co-Design of Action Representation and Objective in the Shared Hidden Space |
| Decoding budget | `inference.draft_steps`, `inference.remask_steps` (one-pass decoding: `draft_steps: 1, remask_steps: 0`) | 4 / 5 in the evaluation recipes | App. Decoding Budget and Closed-Loop Replanning Frequency |
| Training stages | `configs/train/pretrain_60k.yaml`, `pretrain_90k.yaml`, `calvin.yaml`, `libero.yaml` | - | App. Pretraining Mixture and Training Stages |
| Visible action context | `visible_action_context.strategy` and its noise and repair loss | `none` | not studied in the paper; kept for experimentation |

The auxiliary coordinate objective of the paper's ablations is not an option of this implementation: that ablation
was run with an earlier model of the research code (DreamVLA/MDLM-based), and the objective has no effect on
Concertina's native block-diffusion path.

The "teacher-random" mode of the generation-exposure analysis is not part of this implementation either: it was an
offline analysis script. The one-pass all-mask mode of that analysis is the decoding option above
(`draft_steps: 1, remask_steps: 0`).

Further options kept for experimentation: the other decoding strategies of `model.inference`, `mask_value`,
`full_mask_probability`, `token_risk`, real-time chunking and action history, and the data-quality filter.

## Paper terms

| Paper | Configuration |
|---|---|
| horizon `H` | `model.chunk_size` |
| block size `K`; number of blocks `M = H / K` | `model.block_size`; `chunk_size / block_size` |
| executed horizon `H_exec` (`E`) | `model.inference.n_action_steps` |
| canonical width `D = 20` (two arms x [xyz, 6D rotation, gripper]); coordinate validity `q` | fixed layout; `DatasetSpec.valid_action_dim` (10 or 20) and the processor's valid-dimension masks |
| coordinate weights `ω_d = (500, 10, 1)` (translation, rotation, gripper) | `model.action_dim_loss_weights` (`[500]*3 + [10]*6 + [1]` per arm) |
| proprioceptive state masking | `model.state_mask_prob`, `state_mask_dims`, `state_mask_ratio_range`, `state_mask_fill_value` |
| early-layer freeze (layers 0-1) | `model.freeze_backbone_layers: [0, 1]` |
| frozen vs. trainable action interfaces `E_φ`, `D_ψ` | `model.freeze_action_heads` |
| visual random-resized crop (RRC) | `model.random_resized_crop` |
| mask-only corruption with inverse-`t` weighting; `ε = 10^-3` | `model.corruption.mode: log_linear_mask`; constant `NOISE_EPS` |
| three-state mixed corruption (clean / visible substitution / mask) | `model.corruption.mode: bard_mixed`, `visible_corruption`, `probability_floor`, `loss_weighting: none \| inverse_t` |
| unknown-action sentinel (zero vector) | `model.mask_value: zeros` |
| probability-valued gripper BCE vs. Smooth-L1 on all coordinates | `model.gripper_indices` with `gripper_target: probability` (calibrated by `data`) vs. `gripper_indices: []` |
| token-risk head (LayerNorm + two-layer MLP, width 256, loss weight 0.05) | `model.token_risk` (constants `TOKEN_RISK_HIDDEN_DIM`, `TOKEN_RISK_LOSS_WEIGHT`) |
| risk-guided decoding: proposal budget `D`, refinement steps `R` | `model.inference.confidence: token_risk`, `draft_steps`, `remask_steps` (with `remask_ratio` and the other `remask_*` fields) |
| single-pass parallel decoding `D = 1, R = 0` | `draft_steps: 1, remask_steps: 0` |

## Data (`policy.data`)

| Field | Default | Meaning |
|---|---|---|
| `datasets` | `[]` | the mixture, a list of `DatasetSpec`; the list position is the `dataset_index` of a dataset's samples |
| `drop_n_first_frames`, `drop_n_last_frames` | `0`, `0` | frames at the start/end of every episode that are never sampled |
| `fps_align_target_fps` | `null` | resample the actions of datasets recorded at another frame rate to this rate |
| `resize_images` | `false` | resize every camera to 224 x 224 in the dataset (bilinear, no padding); required when datasets have different camera sizes, otherwise the policy resizes with aspect-preserving padding |
| `normalization` | `none` | `none`, `global` (one set of statistics; one source schema) or `per_embodiment` (statistics pooled per `DatasetSpec.embodiment`) |
| `gripper_range_tolerance` | `1e-6` | tolerance of the gripper calibration endpoints |
| `custom_gripper_profiles` | `[]` | `GripperProfile`s for sources not covered by the builtin profiles |
| `inference_dataset_index` | `0` | the dataset whose conversion, statistics and calibration evaluation uses |
| `quality.require_release_ready` | `true` | refuse quality packages compiled from a partial scan or with unresolved blockers |

**Sampling.** With one dataset, each epoch is one seeded permutation of its valid frames (wrapped around to a whole
number of global batches). With several, each sample first draws a dataset by `weight` (with replacement), then a
valid frame of it uniformly; if no dataset sets a weight, all valid frames of the mixture are equally likely. The
order is a function of `(seed, epoch)` only, so a resumed run continues it exactly when the number of GPUs and the
batch size are unchanged.

### `DatasetSpec`

| Field | Default | Meaning |
|---|---|---|
| `path` | required | local LeRobot v3 dataset directory; relative paths are resolved against `dataset.root` |
| `source` | required | source schema: `droid`, `agibot`, `robomind`, `libero` or `calvin` |
| `embodiment` | required | normalization group for `per_embodiment` |
| `adapter` | `null` | RoboMIND only: `robomind_agilex`, `robomind_single` or `robomind_fr3_dual` |
| `weight` | `null` | relative sampling weight (all or none of the datasets set one) |
| `source_fps` | `null` | frame rate for `fps_align_target_fps`; `null` reads the dataset metadata |
| `camera_map` | `{}` | canonical camera key -> candidate dataset camera keys (the first present is used) |
| `valid_action_dim` | `20` | `10` for single-arm data (the second arm is padding), `20` for dual-arm data |
| `gripper_profile` | `null` | calibration profile id; `null` uses the builtin profile of `source` |
| `rotation_stats_path` | `null` | statistics file of `concertina-compute-rotation-stats` (an explicit path that does not exist is an error); `null` reads `<path>/meta/rotation_stats.json`, and if the dataset has none, training computes it once into `$CONCERTINA_CACHE_DIR/rotation_stats/<dataset>-<fingerprint>.json` (default `~/.cache/concertina`) and leaves the dataset untouched |
| `episodes` | `null` | use only these episodes |
| `quality_package` | `null` | package of `concertina-data-quality compile`: removes quarantined episodes and excluded frames, masks action targets |

### Gripper calibration

With `gripper_target: probability`, each dataset's gripper channels are mapped to a closure in [0, 1] from declared
open/closed endpoints, and the postprocessor maps predicted closure probabilities back to the source convention.

| Builtin profile | Open | Closed | Source convention |
|---|---|---|---|
| `droid` | 0 | 1 | closure = 1 - width / max width |
| `agibot` | 0 | 1 | effector position, 0 open / 1 closed |
| `calvin` | 1 | -1 | +1 open / -1 closed |
| `libero` | -1 | 1 | robosuite gripper command, -1 open / +1 closed |

RoboMIND datasets need a `custom_gripper_profiles` entry (`id`, `indices`, `open_values`, `closed_values`,
`source: robomind`, `adapter`, `out_of_range: error | clip`, `provenance`).

## Processors

The pre/post-processors are built from `data` and saved with the checkpoint:

- preprocessor: source binding (evaluation inputs without `dataset_index` are converted with the adapter of
  `inference_dataset_index`), canonical-layout check, valid-dimension masks, gripper calibration, per-embodiment
  normalization, renaming, device placement, gripper targets, global normalization;
- postprocessor: the inverse steps, ending with the conversion of the canonical action to the source action
  (for example CALVIN's xyz + Euler angles + gripper command).

## Example

The policy part of `configs/train/libero.yaml` (abridged), the LIBERO fine-tuning of the paper:

```yaml
dataset:
  repo_id: concertina/libero  # a name only
  root: /path/to/datasets  # holds libero/
policy:
  type: concertina
  dtype: float32
  pretrained_path: checkpoints/concertina-pretrain-030000
  model:
    backbone_path: ''  # the backbone bundled with the pretrained checkpoint
    camera_keys: [observation.images.image, observation.images.image2]
    chunk_size: 10
    block_size: 10
    gripper_indices: [9]
    gripper_target: probability
    state_mask_dims: [2]
    # inference: the default recipe; configs/eval/libero.yaml replaces it for evaluation
  data:
    datasets:
      - path: libero
        source: libero
        embodiment: libero
        valid_action_dim: 10
        gripper_profile: libero
    drop_n_last_frames: 9
    normalization: global
  scheduler_warmup_steps: 80
  scheduler_decay_steps: 5000
batch_size: 56
steps: 5000
seed: 42
parallelism:
  dp_shard: -1  # FSDP2 over all GPUs
accelerator:
  mixed_precision: bf16
```

`configs/train/pretrain_60k.yaml` shows a weighted mixture: 39 datasets of three sources with sampling weights,
camera maps onto the canonical cameras `observation.images.head`, `observation.images.wrist_left` and
`observation.images.wrist_right`, frame-rate alignment and per-embodiment normalization.
