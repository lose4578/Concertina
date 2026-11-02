"""Action input projector, action output head and token-risk head."""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


class MLP(nn.Sequential):
    """`num_layers` linear layers with SiLU between them (none after the last)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int) -> None:
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: list[nn.Module] = []
        for index in range(num_layers):
            layers.append(nn.Linear(dims[index], dims[index + 1]))
            if index < num_layers - 1:
                layers.append(nn.SiLU())
        super().__init__(*layers)


class TokenRiskHead(nn.Sequential):
    """Predicts the log reconstruction loss of each action token from its hidden state."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return super().forward(hidden).squeeze(-1)


def bernoulli_kl_with_logits(logits: Tensor, target: Tensor) -> Tensor:
    """KL(Bernoulli(target) || Bernoulli(sigmoid(logits))) in float32, elementwise."""
    logits = logits.float()
    target = target.float()
    cross_entropy = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    target_entropy = torch.special.entr(target) + torch.special.entr(1.0 - target)
    kl = cross_entropy - target_entropy
    # Clamp tiny negative values from cancellation at the optimum without changing the BCE gradient.
    return kl + (kl.clamp_min(0.0) - kl).detach()


def _weighted_dim_mean(loss: Tensor, dim_weights: Tensor, valid_dim_mask: Tensor | None) -> Tensor:
    dim_weights = dim_weights.to(dtype=loss.dtype, device=loss.device)
    loss = loss * dim_weights
    if valid_dim_mask is None:
        return loss.sum(dim=-1) / dim_weights.sum(dim=-1).clamp_min(1e-8)
    valid_dim_mask = valid_dim_mask.to(dtype=loss.dtype, device=loss.device).unsqueeze(1)
    return (loss * valid_dim_mask).sum(dim=-1) / (valid_dim_mask * dim_weights).sum(dim=-1).clamp_min(1e-8)


def action_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    gripper_target: Tensor | None,
    *,
    gripper_indices: list[int],
    gripper_target_mode: str,
    dim_weights: Tensor,
    valid_dim_mask: Tensor | None,
    regression_dim_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor | None]:
    """Per-token action loss, shape `(B, T)`, and optionally its regression view.

    Dimensions use smooth-L1 against `target`, except gripper dimensions, which use a binary cross-entropy
    between the predicted logits and `gripper_target` (that must also be the gripper values of `target`):
    a Bernoulli KL in float32 for probability targets, plain cross-entropy for binary targets. Dimensions are
    combined by a weighted mean over the valid dimensions. The regression view is the weighted smooth-L1 mean over
    `regression_dim_mask` (`(B, D)`), returned when that mask is given.
    """
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    regression = (
        None if regression_dim_mask is None else _weighted_dim_mean(loss, dim_weights, regression_dim_mask)
    )
    if gripper_indices:
        if gripper_target.shape[-1] == prediction.shape[-1]:
            gripper_target = gripper_target[..., gripper_indices]
        # Compared in the target's dtype: under autocast the prediction may be in a lower precision.
        if not torch.allclose(target[..., gripper_indices], gripper_target.to(target), atol=1e-6, rtol=0.0):
            raise ValueError("The gripper channels of the action target must equal the gripper targets")
        gripper_target = gripper_target.to(dtype=prediction.dtype, device=prediction.device)
        gripper_logits = prediction[..., gripper_indices]
        if gripper_target_mode == "probability":
            if bool(((gripper_target < -1e-6) | (gripper_target > 1.0 + 1e-6)).any()):
                raise ValueError("Gripper targets must be probabilities in [0, 1]")
            gripper_target = gripper_target.clamp(0.0, 1.0)
        else:
            zeros, ones = torch.zeros_like(gripper_target), torch.ones_like(gripper_target)
            is_binary = torch.isclose(gripper_target, zeros, atol=1e-6, rtol=0.0) | torch.isclose(
                gripper_target, ones, atol=1e-6, rtol=0.0
            )
            if not bool(is_binary.all()):
                raise ValueError("Binary gripper targets must be 0 or 1")
        if gripper_target_mode == "probability":
            loss = loss.float().clone()
            loss[..., gripper_indices] = bernoulli_kl_with_logits(gripper_logits, gripper_target)
        else:
            loss = loss.clone()
            loss[..., gripper_indices] = F.binary_cross_entropy_with_logits(
                gripper_logits, gripper_target, reduction="none"
            )
    return _weighted_dim_mean(loss, dim_weights, valid_dim_mask), regression


def action_logits_to_values(raw: Tensor, gripper_indices: list[int]) -> Tensor:
    """Map head outputs to action-token values: gripper logits become probabilities."""
    if not gripper_indices:
        return raw
    values = raw.clone()
    values[..., gripper_indices] = torch.sigmoid(values[..., gripper_indices])
    return values
