"""Masked block-diffusion primitives: noise schedule, token corruption, commit budgets and loss reduction."""

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from concertina.model.rng import SampleKeyedRNG

if TYPE_CHECKING:
    from concertina.policy.config_model import ConcertinaCorruptionConfig, ConcertinaVisibleContextConfig


def action_block_ranges(action_length: int, block_size: int) -> list[tuple[int, int]]:
    if action_length <= 0 or action_length % block_size != 0:
        raise ValueError(
            f"action_length={action_length} must be a positive multiple of block_size={block_size}"
        )
    return [(start, start + block_size) for start in range(0, action_length, block_size)]


def mask_values(mode: str, shape: tuple[int, ...], *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Values of masked action tokens; "random" draws from the global RNG."""
    if mode == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    if mode == "ones":
        return torch.ones(shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def sample_block_timesteps(
    rng: SampleKeyedRNG,
    *,
    num_blocks: int,
    block_size: int,
    eps: float,
    device: torch.device,
) -> Tensor:
    """Stratified per-block diffusion times, repeated over the tokens of each block, shape `(B, A)`.

    Each sample draws one time from each of `num_blocks` equal strata of (0, 1]; a random permutation decouples
    the temporal block index from the noise level.
    """
    offsets = rng.rand((num_blocks,), stream="native_dlm.timestep.offset", device=device)
    permutation_scores = rng.rand((num_blocks,), stream="native_dlm.timestep.permutation", device=device)
    strata = torch.arange(num_blocks, device=device, dtype=torch.float32).unsqueeze(0)
    times = (offsets + strata) / float(num_blocks)
    permutation = torch.argsort(permutation_scores, dim=1)
    times = times.gather(1, permutation).clamp(min=eps, max=1.0)
    return times.repeat_interleave(block_size, dim=1)


def _bard_state_probabilities(timesteps: Tensor, floor: float) -> tuple[Tensor, Tensor]:
    """Normalized BARD probabilities of the clean and visible-corruption states."""
    phi = timesteps.clamp(0.0, 1.0) * (math.pi / 2.0)
    mask_weight = (1.0 - torch.cos(phi)).clamp(min=floor, max=1.0)
    visible_weight = (torch.sin(phi) + torch.cos(phi) - 1.0).clamp(min=floor, max=1.0)
    clean_weight = (1.0 - torch.sin(phi)).clamp(min=floor, max=1.0)
    normalizer = clean_weight + visible_weight + mask_weight
    return clean_weight / normalizer, visible_weight / normalizer


def corrupt_action_tokens(
    action_tokens: Tensor,
    timesteps: Tensor,
    action_valid: Tensor,
    rng: SampleKeyedRNG,
    *,
    masked_values: Tensor,
    corruption: "ConcertinaCorruptionConfig",
    eps: float,
    full_mask_probability: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Corrupt the action tokens of each block at its diffusion time (see `ConcertinaCorruptionConfig`).

    With probability `full_mask_probability` a sample is instead masked entirely, with unit loss weights.
    Returns the noisy tokens, the boolean supervision mask, the loss weights and the boolean mask of the tokens
    replaced by a visible corruption.
    """
    move_chance = timesteps.clamp_min(eps)
    loss_weights = (-1 / move_chance).abs()
    if corruption.mode == "log_linear_mask":
        uniform = rng.rand(
            move_chance.shape[1:], stream="native_dlm.log_mask.categorical", device=move_chance.device
        )
        masked = uniform <= move_chance
        visible = torch.zeros_like(masked)
        noisy_tokens = torch.where(masked.unsqueeze(-1), masked_values, action_tokens)
    else:
        if corruption.visible_corruption == "uniform":
            low, high = corruption.visible_uniform_range
            draw = rng.rand(
                action_tokens.shape[1:], stream="native_dlm.visible.uniform", device=action_tokens.device
            )
            visible_values = low + draw.to(dtype=action_tokens.dtype) * (high - low)
        else:
            draw = rng.randn(
                action_tokens.shape[1:], stream="native_dlm.visible.gaussian", device=action_tokens.device
            )
            visible_values = action_tokens + draw.to(dtype=action_tokens.dtype) * corruption.visible_noise_std
        uniform = rng.rand(timesteps.shape[1:], stream="native_dlm.bard.categorical", device=timesteps.device)
        clean_probability, visible_probability = _bard_state_probabilities(
            timesteps, corruption.probability_floor
        )
        clean = uniform < clean_probability
        visible = (uniform < clean_probability + visible_probability) & ~clean
        masked = ~(clean | visible)
        noisy_tokens = torch.where(
            clean.unsqueeze(-1),
            action_tokens,
            torch.where(visible.unsqueeze(-1), visible_values, masked_values),
        )
        masked = masked | visible
        if corruption.loss_weighting == "none":
            loss_weights = torch.ones_like(loss_weights)
    if full_mask_probability > 0:
        branch = rng.rand((), stream="native_dlm.mask_mixture.branch", device=move_chance.device)
        full = (branch < full_mask_probability).unsqueeze(1)
        masked = masked | full
        noisy_tokens = torch.where(full.unsqueeze(-1), masked_values, noisy_tokens)
        loss_weights = torch.where(full, torch.ones_like(loss_weights), loss_weights)
    supervised = masked & action_valid
    noisy_tokens = torch.where(action_valid.unsqueeze(-1), noisy_tokens, action_tokens)
    return noisy_tokens, supervised, loss_weights, visible


def perturb_visible_context(
    action_tokens: Tensor,
    action_valid: Tensor,
    valid_dim_mask: Tensor | None,
    rng: SampleKeyedRNG,
    *,
    config: "ConcertinaVisibleContextConfig",
    block_size: int,
    gripper_indices: list[int],
    stream_prefix: str,
) -> tuple[Tensor, Tensor]:
    """Add Gaussian noise to the continuous valid dimensions of the valid tokens of selected samples.

    Returns the perturbed tokens and the boolean `(B,)` mask of the selected samples.
    """
    batch_size, action_length, action_dim = action_tokens.shape
    device = action_tokens.device
    selected = rng.rand((), stream=f"{stream_prefix}.apply", device=device) < float(config.apply_probability)
    epsilon = rng.randn(action_tokens.shape[1:], stream=f"{stream_prefix}.gaussian", device=device).to(
        dtype=action_tokens.dtype
    )
    if config.strategy == "block_shared_gaussian":
        block_starts = torch.arange(action_length, device=device) // block_size * block_size
        epsilon = epsilon.index_select(1, block_starts)
    if len(config.noise_std) == 1:
        std = action_tokens.new_full((action_dim,), float(config.noise_std[0]))
    else:
        std = action_tokens.new_tensor(config.noise_std)
    if std.shape != (action_dim,):
        raise ValueError(
            f"visible_action_context.noise_std has {std.numel()} values for {action_dim} dimensions"
        )
    continuous = torch.ones(batch_size, action_dim, device=device, dtype=torch.bool)
    if valid_dim_mask is not None:
        continuous &= valid_dim_mask.to(device=device, dtype=torch.bool)
    continuous[:, gripper_indices] = False
    noise_mask = (
        selected[:, None, None]
        & action_valid[:, :, None]
        & continuous[:, None, :]
        & (std > 0.0)[None, None, :]
    )
    noise = torch.where(
        noise_mask,
        epsilon * std.view(1, 1, action_dim),
        torch.zeros((), device=device, dtype=action_tokens.dtype),
    )
    return action_tokens + noise, selected


def reduce_masked_loss(
    per_token_loss: Tensor,
    supervised: Tensor,
    loss_weights: Tensor,
    action_valid_count: Tensor,
) -> Tensor:
    """Per-sample weighted sum over supervised tokens, normalized by the sample's valid-token count."""
    weights = loss_weights.to(per_token_loss.dtype)
    per_sample = torch.sum(per_token_loss * weights * supervised.to(per_token_loss.dtype), dim=1)
    return (per_sample / action_valid_count.to(per_token_loss.dtype).clamp_min(1.0)).mean()


def per_sample_commit_budgets(pending: Tensor, num_rounds: int) -> Tensor:
    """Split each sample's pending-token count across `num_rounds` as evenly as possible, shape `(B, rounds)`."""
    pending_counts = pending.sum(dim=1, dtype=torch.int64)
    base = torch.div(pending_counts, num_rounds, rounding_mode="floor")
    remainder = pending_counts.remainder(num_rounds)
    round_indices = torch.arange(num_rounds, device=pending.device).unsqueeze(0)
    return base.unsqueeze(1) + (round_indices < remainder.unsqueeze(1)).to(torch.int64)
