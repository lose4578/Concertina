"""Confidence-guided decoding of one action block: draft rounds with commit budgets, then remask refinement."""

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from concertina.model.constants import CONFIDENCE_EPS, STABILITY_WEIGHT

if TYPE_CHECKING:
    from concertina.policy.config_model import ConcertinaInferenceConfig

# Maps block tokens `(B, block, D)` and the previous proposal (or None) to the action hidden states, the proposal
# `(B, block, D)` and its confidence `(B, block)`.
ScoreFn = Callable[[Tensor, Tensor | None], tuple[Tensor, Tensor, Tensor]]
# Maps (hidden, proposal, previous proposal or None, current block tokens) to draft-ranking scores `(B, block)`.
RankFn = Callable[[Tensor, Tensor, Tensor | None, Tensor], Tensor]


def _finite_scores(scores: Tensor) -> Tensor:
    return torch.nan_to_num(
        scores, nan=-torch.inf, posinf=torch.finfo(scores.dtype).max, neginf=torch.finfo(scores.dtype).min
    )


def _stability(prediction: Tensor, reference: Tensor | None) -> Tensor:
    if reference is None:
        return torch.ones(prediction.shape[:2], device=prediction.device)
    return 1.0 / ((prediction - reference).abs().mean(dim=-1) + CONFIDENCE_EPS)


def _combined(hidden: Tensor, prediction: Tensor, reference: Tensor) -> Tensor:
    """Equal mix of hidden norm and stability (against `reference`), each normalized by its per-sample maximum."""
    hidden_score = hidden.norm(dim=-1)
    reference = reference.to(device=prediction.device, dtype=prediction.dtype)
    stability = 1.0 / ((prediction - reference).abs().mean(dim=-1) + CONFIDENCE_EPS)
    hidden_score = hidden_score / hidden_score.amax(dim=1, keepdim=True).clamp(min=CONFIDENCE_EPS)
    stability = stability / stability.amax(dim=1, keepdim=True).clamp(min=CONFIDENCE_EPS)
    return STABILITY_WEIGHT * stability + (1 - STABILITY_WEIGHT) * hidden_score


def token_confidence(
    name: str,
    hidden: Tensor,
    prediction: Tensor,
    previous: Tensor | None,
    inputs: Tensor,
    risk: Callable[[Tensor], Tensor] | None,
) -> Tensor:
    """Decoding confidence of each proposed token (see `ConcertinaInferenceConfig.confidence`).

    `previous` is the previous proposal of the block (None in its first draft round) and `inputs` the block tokens
    the proposal was computed from. "random" draws from the global RNG.
    """
    if name == "token_risk":
        if risk is None:
            raise ValueError("The token_risk confidence needs the token-risk head")
        confidence = -risk(hidden)
    elif name == "hidden_norm":
        confidence = hidden.norm(dim=-1)
    elif name == "prediction_stability":
        confidence = _stability(prediction, previous)
    elif name == "combined":
        confidence = _combined(hidden, prediction, previous if previous is not None else inputs)
    else:
        confidence = torch.rand(prediction.shape[:2], device=prediction.device)
    return _finite_scores(confidence)


def draft_ranking(
    name: str,
    hidden: Tensor,
    prediction: Tensor,
    previous: Tensor | None,
    current: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    """Scores that order the pending tokens of a draft round (see `ConcertinaInferenceConfig.draft_ranking`)."""
    if name == "random":
        scores = torch.rand(prediction.shape[:2], device=prediction.device, generator=generator)
        return _finite_scores(scores)
    if name in ("hidden_norm_fp32", "combined_fp32", "stability_norm_bootstrap"):
        hidden, prediction, current = hidden.float(), prediction.float(), current.float()
        previous = previous.float() if previous is not None else None
    if name == "stability_norm_bootstrap":
        name = "hidden_norm" if previous is None else "prediction_stability"
    if name in ("hidden_norm", "hidden_norm_small", "hidden_norm_fp32"):
        scores = hidden.norm(dim=-1)
        if name == "hidden_norm_small":
            scores = -scores
    elif name == "prediction_stability":
        scores = _stability(prediction, previous)
    else:
        scores = _combined(hidden, prediction, previous if previous is not None else current)
    return _finite_scores(scores)


def _spread_positions(confidence: Tensor, pending: Tensor, k: int) -> Tensor:
    """Greedy temporal coverage of one block row: repeatedly take the pending position farthest from every
    committed or chosen one, breaking ties by the highest confidence, then by the lowest index."""
    remaining = torch.where(pending)[0]
    anchors = torch.where(~pending)[0]
    chosen = []
    for _ in range(min(k, int(remaining.numel()))):
        candidates = remaining
        if anchors.numel() > 0:
            distances = (remaining[:, None] - anchors[None, :]).abs().amin(dim=1)
            candidates = remaining[distances == distances.max()]
        scores = confidence[candidates]
        position = candidates[scores == scores.max()][0]
        chosen.append(position)
        remaining = remaining[remaining != position]
        anchors = torch.cat((anchors, position.reshape(1)))
    return torch.stack(chosen) if chosen else remaining[:0]


def select_commits(
    confidence: Tensor,
    pending: Tensor,
    budget: Tensor,
    *,
    strategy: str = "dynamic",
    threshold: float = 0.0,
    max_ratio: float = 1.0,
    ranking: Tensor | None = None,
    spread: bool = False,
) -> Tensor:
    """Choose which pending tokens to commit this round (see `ConcertinaInferenceConfig.commit_strategy`).

    Each sample commits its round budget of pending tokens; with the "dynamic" strategy, tokens above `threshold`
    may raise the count up to `max_ratio` of the block. Tokens are taken left to right ("sequential"), by temporal
    coverage (`spread`), or by the highest `ranking` (default: the confidence).
    """
    selected = torch.zeros_like(pending)
    block_length = pending.shape[1]
    scores = confidence if ranking is None else ranking
    for b in range(pending.shape[0]):
        pending_indices = torch.where(pending[b])[0]
        num_pending = int(pending_indices.numel())
        if num_pending == 0:
            continue
        k = min(int(budget[b].item()), num_pending)
        if strategy == "dynamic":
            num_confident = int((confidence[b, pending_indices] > threshold).sum().item())
            burst_cap = max(k, math.ceil(max_ratio * block_length))
            k = min(num_pending, burst_cap, max(k, num_confident))
        if k <= 0:
            continue
        if strategy == "sequential":
            chosen = pending_indices[:k]
        elif spread:
            chosen = _spread_positions(confidence[b], pending[b], k)
        else:
            chosen = pending_indices[torch.topk(scores[b, pending_indices], k=k, largest=True).indices]
        selected[b, chosen] = True
    return selected


def draft_block(
    tokens: Tensor,
    confidence: Tensor,
    pending: Tensor,
    budgets: Tensor,
    score_fn: ScoreFn,
    recipe: "ConcertinaInferenceConfig",
    rank_fn: RankFn | None = None,
) -> Tensor:
    """Unmask the pending tokens of a block over `recipe.draft_steps` rounds.

    `tokens` is the block's slice of the chunk buffer (pending tokens hold mask values) and is updated in place;
    each round scores that slice. Tokens that are not pending initially (real-time-chunking prefix) never change.
    Returns the confidence of the decoded tokens.
    """
    num_rounds = recipe.draft_steps
    fixed = ~pending
    previous = None
    for round_index in range(num_rounds):
        active = pending.any(dim=1)
        if not bool(active.any()):
            break
        hidden, prediction, round_confidence = score_fn(tokens, previous)
        if not bool(torch.isfinite(prediction[active]).all()):
            raise FloatingPointError("Non-finite action prediction during block decoding")

        current = tokens
        if recipe.visible_revision:
            # Committed tokens stay mutable until the block is finalized; revise them when more confident.
            mutable = ~pending & ~fixed & active.unsqueeze(1)
            revise = mutable & (round_confidence > confidence + recipe.revision_margin)
            if bool(revise.any()):
                current = torch.where(revise.unsqueeze(-1), prediction, current)
                confidence = torch.where(revise, round_confidence, confidence)

        commit = select_commits(
            round_confidence,
            pending,
            budgets[:, round_index],
            strategy=recipe.commit_strategy,
            threshold=recipe.commit_confidence_threshold,
            max_ratio=recipe.commit_max_ratio,
            ranking=rank_fn(hidden, prediction, previous, current) if rank_fn is not None else None,
            spread=recipe.draft_ranking == "spread_risk",
        )
        if round_index == num_rounds - 1:
            commit = commit | pending
        tokens.copy_(torch.where(commit.unsqueeze(-1), prediction, current))
        confidence = torch.where(commit, round_confidence, confidence)
        pending = pending & ~commit
        previous = prediction
    if bool(pending.any()):
        raise RuntimeError("Block decoding finished with uncommitted tokens")
    return confidence


def refine_block(
    tokens: Tensor,
    confidence: Tensor,
    eligible: Tensor,
    score_fn: ScoreFn,
    recipe: "ConcertinaInferenceConfig",
    mask_values: Callable[[], Tensor],
) -> tuple[Tensor, Tensor]:
    """Iteratively remask the least-confident `eligible` tokens of a decoded block and re-predict them.

    `mask_values()` provides the masked-token values once refinement starts.
    """
    if recipe.remask_steps <= 0 or recipe.remask_ratio <= 0 or not bool(eligible.any()):
        return tokens, confidence
    batch_size, block_length = confidence.shape
    remasked = torch.zeros_like(eligible) if recipe.remask_no_repeat else None
    best_tokens = tokens.clone()
    best_confidence = confidence.clone()
    cooldown = torch.zeros((batch_size, block_length), dtype=torch.long, device=tokens.device)
    active = eligible.any(dim=1)
    stalled = torch.zeros(batch_size, dtype=torch.long, device=tokens.device)
    masked_values = mask_values()
    for step in range(recipe.remask_steps):
        if remasked is not None:
            active = active & (eligible & ~remasked).any(dim=1)
        if not bool(active.any()):
            break
        selected = torch.zeros((batch_size, block_length), dtype=torch.bool, device=tokens.device)
        decay = (recipe.remask_steps - step) / recipe.remask_steps if recipe.remask_decay_budget else 1.0
        for b in range(batch_size):
            if not bool(active[b]):
                continue
            candidate_mask = eligible[b] & (cooldown[b] == 0)
            if remasked is not None:
                candidate_mask = candidate_mask & ~remasked[b]
            candidates = torch.where(candidate_mask)[0]
            if candidates.numel() == 0:
                continue
            k = max(1, int(recipe.remask_ratio * int(eligible[b].sum()) * decay))
            k = min(k, int(candidates.numel()))
            lowest = torch.topk(confidence[b, candidates], k=k, largest=False).indices
            selected[b, candidates[lowest]] = True
        if not bool(selected.any()):
            # Cooling-down tokens become candidates again in a later round.
            cooling = active.unsqueeze(1) & eligible & (cooldown > 0)
            if remasked is not None:
                cooling = cooling & ~remasked
            if bool(cooling.any()):
                cooldown = torch.clamp(cooldown - 1, min=0)
                continue
            break

        if remasked is not None:
            remasked = remasked | selected
        draft = torch.where(selected.unsqueeze(-1), masked_values, tokens)
        _, proposal, proposal_confidence = score_fn(draft, tokens)
        accept = selected
        if recipe.remask_accept_if_improved:
            accept = accept & (proposal_confidence > confidence + recipe.remask_improvement_margin)
        tokens = torch.where(accept.unsqueeze(-1), proposal, tokens)
        confidence = torch.where(accept, proposal_confidence, confidence)
        improved = confidence > best_confidence
        best_tokens = torch.where(improved.unsqueeze(-1), tokens, best_tokens)
        best_confidence = torch.maximum(best_confidence, confidence)

        accepted_any = accept.any(dim=1)
        stalled[accepted_any] = 0
        stalled[active & selected.any(dim=1) & ~accepted_any] += 1
        cooldown = torch.clamp(cooldown - 1, min=0)
        if recipe.remask_cooldown_steps > 0:
            cooldown[selected] = recipe.remask_cooldown_steps
        active = active & (stalled < recipe.remask_early_stop_patience)

    if recipe.remask_keep_best:
        return best_tokens, best_confidence
    return tokens, confidence
