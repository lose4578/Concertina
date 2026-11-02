# Reproducing the paper results

This release is a re-implementation of the code used for the paper's experiments ("the original implementation")
as a plugin of an unmodified, pinned LeRobot. The model was ported to Transformers 5 and training moved to LeRobot's
training loop with FSDP2. This document summarizes how the release was verified against the original
implementation, and lists every known difference and every setting a reproduction has to match.

The paper's results next to the release's evaluation of the released checkpoints and of a fresh `concertina-train`
fine-tuning (LIBERO only). CALVIN: 1000 sequences of `configs/eval/calvin_sequences_1000.json`, seed 0. LIBERO: 4
suites x 10 tasks x 50 episodes, seeds from 1000.

| Benchmark | Metric | Paper | Released checkpoint, release stack | Fine-tuned with `concertina-train` |
|---|---|---|---|---|
| CALVIN ABC -> D | average length (1000 sequences) | 4.142 (tasks 1-5 in a row: 98.6 / 93.8 / 83.2 / 73.7 / 64.9%; four workers, dynamic sequence order) | 4.153 (98.0 / 93.0 / 83.8 / 75.1 / 65.4%; sequential protocol on one GPU, `configs/eval/calvin_abc_d.yaml` unmodified) | not re-run (1) |
| LIBERO | success rate (4 suites x 500 episodes) | 98.15% (1963/2000; Spatial 99.0, Object 100.0, Goal 98.8, Long 94.8%) | 97.80% (1956/2000; Spatial 99.4, Object 100.0, Goal 98.2, Long 93.6%) | 97.45% (1949/2000; Spatial 98.8, Object 99.2, Goal 95.4, Long 96.4%) |

(1) The release re-trained only the LIBERO fine-tuning with `concertina-train`.

On CALVIN, the release-stack run and the paper run agree on 848 of the 1000 sequences; a paired bootstrap of the
average-length difference over sequences (10,000 resamples) gives +0.011, 95% CI [-0.054, +0.076]. The two runs also
differ in sequence order (sequential vs. four workers), whose effect is described in
[eval.md](eval.md#calvin).

The LIBERO fine-tuning started from a local copy of the 90k pretraining checkpoint with `configs/train/libero.yaml`
on 4 GPUs: per-GPU batch 14 with 8-fold gradient accumulation (global batch 448, as in the paper), `steps`, warmup,
decay and `save_freq` multiplied by 8 (5,000 updates, checkpoint `040000`), and gradient checkpointing on. Gradient
accumulation changes the per-sample training randomness (see [Known
differences](#known-differences-from-the-original-implementation)).

## Software stacks

| | Original implementation | Release |
|---|---|---|
| LeRobot | modified fork | unmodified, pinned commit (`pyproject.toml`) |
| Transformers | 4.57.1 | 5.5.4 |
| PyTorch | 2.11 (CUDA 12.8) | 2.11.0 (CUDA 12.8) |
| Liger-Kernel | 0.8.2 | 0.8.5 |
| Distributed training | DeepSpeed ZeRO-2 | FSDP2 through LeRobot / Accelerate |
| Training precision | bf16 compute, fp32 master weights and optimizer state | bf16 autocast, fp32 parameters and optimizer state (`policy.dtype: float32`) |
| Inference precision | bf16 | bf16 |
| Backbone | `fudan-generative-ai/Bard-VL-B4-Mask-4B-Instruct` | the same files (every copy used for the paper's runs is byte-identical to the Hub release) |
| Fine-tuning data | local copies of the CALVIN and LIBERO LeRobot datasets | the same datasets from the Hub: `Traly/calvin_abc_d-lerobot`, and `lerobot/libero`, which the paper trained on (data and videos identical; see [datasets.md](datasets.md)) |

The exact release environment is `tools/env/constraints.txt`. The verification below ran on NVIDIA RTX PRO 6000
Blackwell GPUs. The paper's checkpoints were trained on 8 GPUs: pretraining and the CALVIN fine-tuning on NVIDIA
L20X (per-GPU batch 192), the LIBERO fine-tuning on NVIDIA H20 (per-GPU batch 56). CALVIN was evaluated on L20X,
LIBERO on 7 GPUs of the same H20 machine that trained it.

## Verification

### Model

The model checks run both implementations on the same checkpoint and the same inputs with TF32 off and
deterministic algorithms on, and compare, for chunks of 10, 20 and 50 actions and every gripper objective
(calibrated probability, binary, regression): the decoded actions of the checkpoint's own recipe and of the CALVIN/LIBERO recipe,
the hidden states of every block read during decoding (all draft and remask rounds), the training loss and its
components, gradients and a bit-pattern fingerprint of every parameter gradient, and the trainable parameters,
their order and the optimizer groups.

| Check | Result |
|---|---|
| Original vs release model code, both on Transformers 4.57.1 | bitwise equal for every quantity |
| Original implementation on Transformers 4.57.1 vs release on Transformers 5.5.4 (SDPA training attention) | bitwise equal for every quantity; same number of block reads, hence identical commit and remask decisions |
| Same, with the cuDNN grouped-query training attention (the default) | forward, loss and decoding bitwise equal; gradients differ, as between two runs of the original implementation (cuDNN has no deterministic backward for dense masks) |
| Six model configurations (one- and two-arm, three cameras incl. 240x320, frozen layers, no gripper key, mixed embodiments in one batch, full-mask mixture) | bitwise equal (72-84 checks each) |
| The kept implementation branches (14 configurations: `bard_mixed` corruptions, visible action context, pure-pretraining settings, 25 decoding recipes, mask values, state masking, real-time chunking, the four action-history variants) | bitwise equal (362 checks) |
| A converted CALVIN checkpoint on real CALVIN batches through `ConcertinaPolicy` (decoded actions of both recipes, losses, all per-parameter gradient norms) | bitwise equal |
| FSDP2 on 2 GPUs: sharded training step, decoding on the sharded policy, collective save and strict reload | reloaded policy decodes the same actions (max abs difference 0) |

In total, the final setting and all config-switchable branches (21 variants) are bitwise identical to the original
implementation on Transformers 5.5.4.

No numeric tolerance was needed for the Transformers 5 port. The components that could have shifted were checked
individually and are equal: image preprocessing (pixel values and grids), chat-template rendering and tokenization,
multimodal RoPE positions, the attention masks handed to SDPA, the RoPE frequency dtype, and the Liger RMSNorm,
RoPE and SwiGLU forward math.

### Data and training

| Check | Result |
|---|---|
| Real CALVIN samples decoded by `ConcertinaMixtureDataset`, the full training preprocessor, and a converted checkpoint's preprocessing with its saved statistics | bitwise equal to the original stack |
| Single-dataset sampler order (two epochs, and the per-rank shards of Accelerate) | bitwise equal to the original stack |
| `concertina-compute-rotation-stats` | reproduces the original statistics file exactly |
| 2-GPU FSDP2 training (fp32 parameters, bf16 autocast) resumed at step 100 vs an uninterrupted run to step 200, deterministic kernels | bitwise identical: sample order, all 100 losses, all 736 final tensors |
| The same without deterministic kernels | runs differ through nondeterministic GPU kernels; two identical fresh runs already differ at step 2 |
| One AdamW step at learning rate 2.5e-5 | changes 99.95% of the sampled trainable elements with fp32 parameters, against 28% that remain visible in bf16 |

Unit and integration tests in `tests/` cover the properties that the paper training relies on: dataset sampling
frequencies follow the configured weights, the sample order is a pure function of `(seed, epoch)`, and resuming at
any step under data-parallel sharding continues the uninterrupted order sample by sample; `concertina-train` trains,
saves, resumes and fine-tunes through LeRobot's training loop.

### Evaluation

The CALVIN evaluation reproduces the original evaluation stack exactly on a converted CALVIN checkpoint: bitwise
equal per-step actions on identical observations, and identical outcomes for all 1000 sequences when the original
run's sequence order is replayed. The comparison, the score under CALVIN's sequential protocol, and the evaluation
noise to expect between runs are in [eval.md](eval.md#validation-against-the-original-implementation).

The LIBERO evaluations were compared per task with the paper run (40 tasks x 50 episodes, `configs/eval/libero.yaml`):

| Run | Spatial | Object | Goal | Long | Total | Tasks differing from the paper run |
|---|---:|---:|---:|---:|---:|---|
| Paper | 495 | 500 | 494 | 474 | 1963 | - |
| Released checkpoint, release stack | 497 (+2) | 500 (0) | 491 (-3) | 468 (-6) | 1956 (-7) | 15 (9 lower, 6 higher; at most 6 episodes) |
| Fine-tuned with `concertina-train` | 494 (-1) | 496 (-4) | 477 (-17) | 482 (+8) | 1949 (-14) | 19 (14 lower, 5 higher; at most 13 episodes) |

Both release-stack runs score 44/50 on `libero_goal` task 2 (paper run: 50/50). Per-episode outcomes cannot be
compared with the paper run, whose LeRobot version advanced the init state after a successful episode (see [Known
differences](#known-differences-from-the-original-implementation)); suite and task success counts can.

## Known differences from the original implementation

### Training

- **Distributed training.** The paper checkpoints were trained with DeepSpeed ZeRO-2; the release trains with
  FSDP2. Gradient reduction order and optimizer-state sharding differ, so training runs are not bitwise
  comparable across the two, and neither is a run with a different number of GPUs.
- **Precision.** Training keeps the original's precision split: fp32 parameters and optimizer state, bf16
  compute (`policy.dtype: float32` with `accelerator.mixed_precision: bf16`). Do not train with bf16 parameter
  storage: with the 4B model on one GPU, a training update changed only 27.9% of the trainable weights with bf16
  storage, against 99.993% with fp32 storage. Inference always uses bf16 weights; released checkpoints are stored
  in bf16, and `concertina-train` checkpoints (fp32) decode with bf16 copies of their weights.
- **Training nondeterminism.** The default training attention (cuDNN, grouped-query) has a nondeterministic
  backward pass, so two training runs with the same seed differ slightly (already at step 2). The model checks above ran
  deterministically with `model.training_attn_implementation: sdpa`, TF32 off and
  `torch.use_deterministic_algorithms(True)`.
- **Image augmentation.** When `dataset.image_transforms` is enabled, the release uses LeRobot's current batched
  `ImageTransforms`, whose random-number consumption and arithmetic differ from the version used for the paper
  (same family of color, sharpness and affine augmentations; outputs differ by up to 0.29 for the same seed). It is
  also applied only to the cameras the model uses, where the original applied it to every video key of a dataset.
  Among the paper's runs only pretraining enabled it; the fine-tuning runs had it disabled.
- **Single-dataset sample order.** The original fine-tuning visited the frames of a single dataset in an unseeded
  random order; the release uses a permutation seeded by `seed + epoch`. The orders are equivalent in
  distribution only.
- **Per-sample training randomness** (augmentation, masking, corruption) is keyed by the policy's training-step
  counter, saved with the checkpoint; the original used the optimizer step injected by its training loop. The two
  coincide without gradient accumulation. The counter counts micro-batches, so a run with gradient accumulation draws
  different noise: it is statistically, not bitwise, equivalent to the run without.
- **Dataset checks.** The original repaired non-dense episode metadata at training time and tolerated partially
  copied datasets. The release requires a dense episode layout (`concertina-normalize-episode-metadata` repairs it
  offline) and the data and video files of every camera the model reads (or an explicit `DatasetSpec.episodes`
  subset). As in the original, missing rotation-aware normalization statistics are computed at training time; the release
writes them to the Concertina cache (`$CONCERTINA_CACHE_DIR/rotation_stats/`) instead of the dataset.
- **Training loop features.** The release logs to Weights & Biases (offline with `WANDB_MODE=offline`) instead of
  TensorBoard, counts training length in `steps` rather than epochs, and does not report per-group gradient metrics.

### Inference and evaluation

- **Decoding recipes differ per benchmark** and are not taken from the checkpoint: use the YAMLs in
  `configs/eval/` (see [configuration.md](configuration.md#decoding-recipe-policymodelinference)).
- **Vision attention.** CALVIN and LIBERO were evaluated with SDPA attention in the vision tower, the release
  default.
- **TF32.** The CALVIN configuration disables TF32 matmuls; the LIBERO configuration enables them, as in the
  reference runs.
- **GPU model and kernels.** The bitwise results above hold for one GPU model and software stack. Other GPUs
  select other kernels; because decoding commits and remasks tokens by confidence, a tiny numeric difference can
  change a decision and the rest of an episode. Expect per-episode outcomes to differ across hardware while
  aggregate scores stay within evaluation noise.
- **Liger kernels.** The reported results used the Liger RMSNorm, RoPE and SwiGLU kernels
  (`model.use_liger_kernels`, the `liger` extra). Without them the text decoder uses the Transformers kernels,
  which are not bitwise identical to them.
- **CALVIN sequence order and noise.** The CALVIN environment carries state from one sequence to the next, so
  outcomes depend on the order in which a worker evaluates the sequences. The release protocol is CALVIN's
  sequential one (one GPU); the paper's CALVIN result came from four workers that claimed sequences dynamically,
  an order that cannot be replayed. [eval.md](eval.md#calvin) gives the measured effect of the order and the sampling error of 1000
  sequences, against which CALVIN scores should be compared.
- **LIBERO init states.** The reference runs used a LeRobot version whose LIBERO environment advanced the init
  state an extra time after a successful episode; current LeRobot uses init state `k` for episode `k`. Suite
  success rates are comparable, per-episode outcomes are not.
- **Simulator versions.** See [eval.md](eval.md#reproduction-notes) for CALVIN rendering and the LIBERO simulator
  versions.

## Checklist

1. Install the pinned environment (`tools/env/constraints.txt`) with the `liger` extra.
2. Evaluate with the benchmark's YAML in `configs/eval/`, unmodified except for paths, from the repository root.
3. CALVIN: one GPU (sequential protocol), EGL rendering, the bundled sequence manifest.
4. Report the number of evaluated episodes, the seeds and the GPU model with the scores.
