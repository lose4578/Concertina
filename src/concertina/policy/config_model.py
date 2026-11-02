"""Model sub-configuration for the Concertina policy.

The defaults are the released model. The other values of the options below are implementation branches kept for
the paper's ablations and analyses (block topology, corruption, visible action context, decoding strategies,
real-time chunking, action history). Every remaining architectural or objective choice is a constant in
`concertina.model` (see `concertina.model.constants`). This module is plain dataclasses with no LeRobot dependency
so that `concertina.model` can consume it directly.
"""

import math
from dataclasses import dataclass, field

VISION_ATTN_IMPLEMENTATIONS = ("sdpa", "flash_attention_2")
GRIPPER_TARGETS = ("probability", "binary")
TRAINING_ATTN_IMPLEMENTATIONS = ("cudnn_sdpa_native_gqa", "sdpa")
MASK_VALUES = ("zeros", "ones", "random")
CORRUPTION_MODES = ("log_linear_mask", "bard_mixed")
VISIBLE_CORRUPTIONS = ("uniform", "gaussian")
LOSS_WEIGHTINGS = ("none", "inverse_t")
VISIBLE_CONTEXT_STRATEGIES = ("none", "shared_gaussian", "block_shared_gaussian")
HISTORY_MODES = ("prefix", "native")
MISSING_HISTORY = ("mask", "random_noise")
CONFIDENCES = ("token_risk", "hidden_norm", "prediction_stability", "combined", "random")
COMMIT_STRATEGIES = ("dynamic", "static", "sequential")
DRAFT_RANKINGS = (
    "confidence",
    "spread_risk",
    "random",
    "hidden_norm",
    "hidden_norm_small",
    "hidden_norm_fp32",
    "prediction_stability",
    "combined",
    "combined_fp32",
    "stability_norm_bootstrap",
)
RTC_PREFIX_SOURCES = ("executed_tail", "leftover_tail")


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


@dataclass
class ConcertinaInferenceConfig:
    """Block-diffusion decoding recipe.

    The defaults are the recipe stored with the released checkpoints. LIBERO and CALVIN use `draft_steps=4,
    visible_revision=False, remask_steps=5, remask_ratio=0.15, remask_cooldown_steps=0, remask_decay_budget=False,
    remask_accept_if_improved=False, remask_keep_best=False`. One-pass decoding of each block from all-masked
    tokens (the "one-pass all-mask" mode of the generation-exposure analysis) is `draft_steps=1, remask_steps=0`.
    """

    # Number of actions executed from each predicted chunk.
    n_action_steps: int = 10
    # Draft rounds per action block; every masked token is committed by the last round.
    draft_steps: int = 4
    # Per-token decoding confidence: "token_risk" (negated predicted log loss; needs the token-risk head),
    # "hidden_norm" (L2 norm of the action hidden state), "prediction_stability" (inverse change from the previous
    # proposal), "combined" (equal mix of the last two, each normalized per sample) or "random".
    confidence: str = "token_risk"
    # How many pending tokens a draft round commits: "dynamic" (the round budget, raised to the number of tokens
    # whose confidence exceeds `commit_confidence_threshold`, at most `commit_max_ratio` of the block), "static"
    # (the round budget) or "sequential" (the round budget, taken left to right).
    commit_strategy: str = "dynamic"
    commit_confidence_threshold: float = 0.0
    commit_max_ratio: float = 0.5
    # Which pending tokens a round commits: "confidence" (the most confident), "spread_risk" (greedy temporal
    # coverage of the block, ties broken by confidence), or the highest score of another ranking: "random",
    # "hidden_norm", "hidden_norm_small" (lowest norm first), "prediction_stability", "combined", their "_fp32"
    # variants (reductions in float32) or "stability_norm_bootstrap" (hidden norm in the first round, stability
    # after). The count still follows `commit_strategy` and the confidence.
    draft_ranking: str = "confidence"
    # Seed of the "random" ranking; successive decodes use seed, seed + 1, ... None draws each decode's seed from
    # the global RNG.
    draft_seed: int | None = None
    # Replace an already committed (still mutable) token when a later draft round is more confident.
    visible_revision: bool = True
    revision_margin: float = 0.0
    # Confidence-guided remasking rounds applied to each finished block (0 disables refinement).
    remask_steps: int = 2
    remask_ratio: float = 0.2
    remask_cooldown_steps: int = 1
    remask_decay_budget: bool = True
    remask_accept_if_improved: bool = True
    remask_improvement_margin: float = 0.0
    remask_early_stop_patience: int = 2
    remask_keep_best: bool = True
    # Remask each token of a block at most once.
    remask_no_repeat: bool = False
    # Keep the drafted translation (xyz of both arms) and take only the other dimensions from remasking. Only for
    # decoding the first 10 actions with 10-action blocks.
    remask_keep_draft_translation: bool = False
    # Real-time chunking: every chunk is decoded with actions of the previous chunk fixed as its prefix, and
    # `rtc_execution_horizon` of its actions (0: `n_action_steps`) are executed before the next decode. The prefix is
    # the last executed actions ("executed_tail") or the not yet executed rest of the chunk ("leftover_tail").
    rtc: bool = False
    rtc_prefix_source: str = "executed_tail"
    rtc_execution_horizon: int = 0

    def __post_init__(self) -> None:
        if self.n_action_steps <= 0:
            raise ValueError(f"n_action_steps must be > 0, got {self.n_action_steps}")
        if self.draft_steps <= 0:
            raise ValueError(f"draft_steps must be > 0, got {self.draft_steps}")
        _check_choice("confidence", self.confidence, CONFIDENCES)
        _check_choice("commit_strategy", self.commit_strategy, COMMIT_STRATEGIES)
        _check_choice("draft_ranking", self.draft_ranking, DRAFT_RANKINGS)
        if self.commit_strategy == "sequential" and self.draft_ranking != "confidence":
            raise ValueError(
                "commit_strategy='sequential' fixes the commit order; draft_ranking must be 'confidence'"
            )
        if self.draft_seed is not None and self.draft_seed < 0:
            raise ValueError(f"draft_seed must be >= 0, got {self.draft_seed}")
        if not 0.0 < self.commit_max_ratio <= 1.0:
            raise ValueError(f"commit_max_ratio must be in (0, 1], got {self.commit_max_ratio}")
        if self.revision_margin < 0:
            raise ValueError(f"revision_margin must be >= 0, got {self.revision_margin}")
        if self.remask_steps < 0:
            raise ValueError(f"remask_steps must be >= 0, got {self.remask_steps}")
        if not 0.0 <= self.remask_ratio <= 1.0:
            raise ValueError(f"remask_ratio must be in [0, 1], got {self.remask_ratio}")
        if self.remask_cooldown_steps < 0:
            raise ValueError(f"remask_cooldown_steps must be >= 0, got {self.remask_cooldown_steps}")
        if self.remask_improvement_margin < 0:
            raise ValueError(f"remask_improvement_margin must be >= 0, got {self.remask_improvement_margin}")
        if self.remask_early_stop_patience <= 0:
            raise ValueError(f"remask_early_stop_patience must be > 0, got {self.remask_early_stop_patience}")
        _check_choice("rtc_prefix_source", self.rtc_prefix_source, RTC_PREFIX_SOURCES)
        if self.rtc_execution_horizon < 0:
            raise ValueError(f"rtc_execution_horizon must be >= 0, got {self.rtc_execution_horizon}")
        if self.rtc and self.remask_keep_draft_translation:
            raise ValueError("remask_keep_draft_translation does not support real-time chunking")


@dataclass
class ConcertinaCorruptionConfig:
    """Training corruption of the action tokens of each block at its diffusion time `t`."""

    # "log_linear_mask": each token is masked with probability t; supervised tokens are weighted by 1 / t.
    # "bard_mixed": each token stays clean, is replaced by a visible corruption, or is masked, with the BARD
    # probabilities 1 - sin(pi t / 2), sin + cos - 1 and 1 - cos(pi t / 2); visible and masked tokens are supervised.
    mode: str = "log_linear_mask"
    # bard_mixed: the visible corruption, "uniform" (a draw from `visible_uniform_range`) or "gaussian" (the clean
    # value plus N(0, visible_noise_std)).
    visible_corruption: str = "uniform"
    visible_noise_std: float = 1.0
    visible_uniform_range: tuple[float, float] = (-1.0, 1.0)
    # bard_mixed: lower bound of each state's unnormalized probability.
    probability_floor: float = 1e-3
    # bard_mixed: weighting of the supervised tokens, "none" or "inverse_t" (1 / t).
    loss_weighting: str = "none"

    def __post_init__(self) -> None:
        _check_choice("corruption.mode", self.mode, CORRUPTION_MODES)
        _check_choice("corruption.visible_corruption", self.visible_corruption, VISIBLE_CORRUPTIONS)
        _check_choice("corruption.loss_weighting", self.loss_weighting, LOSS_WEIGHTINGS)
        self.visible_uniform_range = tuple(self.visible_uniform_range)
        if self.visible_noise_std < 0:
            raise ValueError(f"corruption.visible_noise_std must be >= 0, got {self.visible_noise_std}")
        low, high = self.visible_uniform_range
        if low >= high:
            raise ValueError(
                f"corruption.visible_uniform_range must be increasing, got {self.visible_uniform_range}"
            )
        if not 0.0 <= self.probability_floor < 1.0:
            raise ValueError(f"corruption.probability_floor must be in [0, 1), got {self.probability_floor}")


@dataclass
class ConcertinaVisibleContextConfig:
    """Training-time Gaussian perturbation of the action context that later blocks see (the clean stream and the
    unmasked tokens of the noisy stream), with an optional loss that regresses the clean actions from it."""

    # "none", "shared_gaussian" (one draw per token) or "block_shared_gaussian" (one draw per block, shared by its
    # tokens).
    strategy: str = "none"
    # Probability that a sample's context is perturbed.
    apply_probability: float = 0.0
    # Noise standard deviation, one value for every action dimension or one per dimension; gripper dimensions are
    # never perturbed.
    noise_std: list[float] = field(default_factory=lambda: [0.0])
    # Weight of the regression loss of the clean continuous dimensions on the perturbed visible tokens.
    repair_loss_weight: float = 0.0
    # Also perturb the action history (with its own random streams and without a repair loss).
    apply_to_history: bool = False

    def __post_init__(self) -> None:
        _check_choice("visible_action_context.strategy", self.strategy, VISIBLE_CONTEXT_STRATEGIES)
        self.noise_std = [float(value) for value in self.noise_std]
        if not self.noise_std or any(not math.isfinite(value) or value < 0 for value in self.noise_std):
            raise ValueError(
                f"visible_action_context.noise_std must be finite and >= 0, got {self.noise_std}"
            )
        if not 0.0 <= self.apply_probability <= 1.0:
            raise ValueError(
                f"visible_action_context.apply_probability must be in [0, 1], got {self.apply_probability}"
            )
        if self.repair_loss_weight < 0:
            raise ValueError(
                f"visible_action_context.repair_loss_weight must be >= 0, got {self.repair_loss_weight}"
            )
        has_noise = any(value > 0 for value in self.noise_std)
        if self.strategy == "none":
            if self.apply_probability or has_noise or self.repair_loss_weight or self.apply_to_history:
                raise ValueError(
                    "visible_action_context.strategy='none' takes no noise, repair or history settings"
                )
        elif self.apply_probability <= 0 or not has_noise:
            raise ValueError(
                f"visible_action_context.strategy={self.strategy!r} needs apply_probability > 0 and a positive "
                "noise_std"
            )

    @property
    def enabled(self) -> bool:
        return self.strategy != "none"


@dataclass
class ConcertinaActionHistoryConfig:
    """Condition each chunk on the last `horizon` executed actions (oldest first).

    Training batches carry the history as the first `horizon` steps of the action window.
    """

    enabled: bool = False
    horizon: int = 20
    # "prefix": history tokens follow the state token in the conditioning prefix. "native": history is encoded like
    # clean action blocks before the first generated block (`horizon` must be a multiple of `block_size`).
    mode: str = "prefix"
    # Missing history slots (before the episode start): "mask" hides them; "random_noise" fills them with
    # N(0, noise_std) values marked by the missing-history role embedding.
    missing_history: str = "mask"
    noise_std: float = 1.0

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ValueError(f"action_history.horizon must be > 0, got {self.horizon}")
        _check_choice("action_history.mode", self.mode, HISTORY_MODES)
        _check_choice("action_history.missing_history", self.missing_history, MISSING_HISTORY)
        if not math.isfinite(self.noise_std) or self.noise_std < 0:
            raise ValueError(f"action_history.noise_std must be finite and >= 0, got {self.noise_std}")


@dataclass
class ConcertinaModelConfig:
    # Path or Hub id of the Qwen3-VL-layout vision-language backbone (weights, tokenizer and image processor).
    backbone_path: str = ""
    # Attention kernel of the vision tower ("flash_attention_2" requires flash-attn).
    vision_attn_implementation: str = "sdpa"
    # Text attention kernel used in train mode; eval mode and KV-cache decoding always use SDPA.
    training_attn_implementation: str = "cudnn_sdpa_native_gqa"
    gradient_checkpointing: bool = True
    # Instance-scoped Liger RMSNorm/RoPE/SwiGLU kernels for the text decoder (requires liger-kernel).
    use_liger_kernels: bool = True

    # Canonical camera keys, in prompt order.
    camera_keys: list[str] = field(default_factory=lambda: ["observation.images.head"])
    # Predicted actions per chunk; must be a multiple of `block_size`.
    chunk_size: int = 10
    # Actions per diffusion block: blocks are generated one after another, the actions of a block jointly.
    block_size: int = 10

    # Action dimensions predicted as gripper-closure logits and trained with a binary cross-entropy objective
    # against the batch's gripper targets; decoded values are probabilities. Empty: grippers are regressed with
    # smooth-L1 like every other dimension.
    gripper_indices: list[int] = field(default_factory=lambda: [9, 19])
    # "probability": targets in [0, 1], Bernoulli KL computed in float32.
    # "binary": targets in {0, 1}, cross-entropy computed in the model dtype.
    gripper_target: str = "probability"
    # Per-dimension weights of the action reconstruction loss.
    action_dim_loss_weights: list[float] = field(
        default_factory=lambda: (
            [500.0, 500.0, 500.0] + [10.0] * 6 + [1.0] + [500.0, 500.0, 500.0] + [10.0] * 6 + [1.0]
        )
    )
    # Training-time state dropout: with probability `state_mask_prob`, set a fraction drawn uniformly from
    # `state_mask_ratio_range` of `state_mask_dims` (at least one) of a sample to `state_mask_fill_value`.
    state_mask_dims: list[int] = field(default_factory=lambda: [2, 12])
    state_mask_prob: float = 0.5
    state_mask_ratio_range: tuple[float, float] = (1.0, 1.0)
    state_mask_fill_value: float = 0.0
    # Training-time random resized crop of the camera images.
    random_resized_crop: bool = True

    # Value of masked action tokens: "zeros", "ones" or "random" (standard-normal draws from the global RNG).
    mask_value: str = "zeros"
    corruption: ConcertinaCorruptionConfig = field(default_factory=ConcertinaCorruptionConfig)
    # Probability that a training sample has all of its action tokens masked (with unit loss weight) instead of
    # the time-dependent random masking (log_linear_mask only).
    full_mask_probability: float = 0.0
    visible_action_context: ConcertinaVisibleContextConfig = field(
        default_factory=ConcertinaVisibleContextConfig
    )
    # Token-risk head: predicts the log reconstruction loss of each action token from its hidden state (auxiliary
    # loss); its negation is the "token_risk" decoding confidence.
    token_risk: bool = True
    action_history: ConcertinaActionHistoryConfig = field(default_factory=ConcertinaActionHistoryConfig)

    # Parameter freezing for finetuning: the action projector and head (with the history role embedding), and text
    # decoder layers by index.
    freeze_action_heads: bool = False
    freeze_backbone_layers: list[int] = field(default_factory=list)

    # Seed of the sample-keyed training randomness (augmentation, masking); set it to the training seed.
    rng_seed: int = 0

    inference: ConcertinaInferenceConfig = field(default_factory=ConcertinaInferenceConfig)

    def __post_init__(self) -> None:
        for name, cls in (
            ("inference", ConcertinaInferenceConfig),
            ("corruption", ConcertinaCorruptionConfig),
            ("visible_action_context", ConcertinaVisibleContextConfig),
            ("action_history", ConcertinaActionHistoryConfig),
        ):
            if isinstance(getattr(self, name), dict):
                setattr(self, name, cls(**getattr(self, name)))
        self.state_mask_ratio_range = tuple(self.state_mask_ratio_range)
        _check_choice(
            "vision_attn_implementation", self.vision_attn_implementation, VISION_ATTN_IMPLEMENTATIONS
        )
        _check_choice(
            "training_attn_implementation", self.training_attn_implementation, TRAINING_ATTN_IMPLEMENTATIONS
        )
        if not self.camera_keys:
            raise ValueError("camera_keys must not be empty")
        if self.chunk_size <= 0 or self.block_size <= 0 or self.chunk_size % self.block_size:
            raise ValueError(
                f"chunk_size ({self.chunk_size}) must be a positive multiple of block_size ({self.block_size})"
            )
        if len(set(self.gripper_indices)) != len(self.gripper_indices):
            raise ValueError(f"gripper_indices contains duplicates: {self.gripper_indices}")
        _check_choice("gripper_target", self.gripper_target, GRIPPER_TARGETS)
        _check_choice("mask_value", self.mask_value, MASK_VALUES)
        if not 0.0 <= self.full_mask_probability <= 1.0:
            raise ValueError(f"full_mask_probability must be in [0, 1], got {self.full_mask_probability}")
        if self.full_mask_probability > 0 and self.corruption.mode != "log_linear_mask":
            raise ValueError("full_mask_probability requires corruption.mode='log_linear_mask'")
        if not 0.0 <= self.state_mask_prob <= 1.0:
            raise ValueError(f"state_mask_prob must be in [0, 1], got {self.state_mask_prob}")
        low, high = self.state_mask_ratio_range
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(f"state_mask_ratio_range must satisfy 0 <= low <= high <= 1, got {(low, high)}")
        if not self.token_risk and self.inference.confidence == "token_risk":
            raise ValueError("inference.confidence='token_risk' requires token_risk=True")
        history = self.action_history
        if history.enabled and history.mode == "native" and history.horizon % self.block_size:
            raise ValueError(
                f"native action history needs horizon ({history.horizon}) to be a multiple of block_size "
                f"({self.block_size})"
            )
        if history.enabled and self.inference.rtc:
            raise ValueError("action_history and inference.rtc are mutually exclusive")
        if self.inference.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.inference.n_action_steps}) cannot exceed chunk_size ({self.chunk_size})"
            )
        if self.inference.rtc_execution_horizon > self.chunk_size:
            raise ValueError(
                f"rtc_execution_horizon ({self.inference.rtc_execution_horizon}) cannot exceed chunk_size "
                f"({self.chunk_size})"
            )
