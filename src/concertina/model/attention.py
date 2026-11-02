"""Attention masks of conditional block diffusion and the cuDNN native-GQA training attention kernel."""

from typing import Any

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

CUDNN_GQA_ATTENTION = "concertina_cudnn_sdpa_native_gqa"


def cudnn_gqa_attention_forward(
    module: torch.nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Tensor, None]:
    """Dense-mask attention through cuDNN with native GQA.

    Transformers' generic SDPA adapter repeats K/V heads when a 4D mask is given; cuDNN consumes grouped K/V
    directly. cuDNN is the only enabled backend, so an unsupported input raises instead of silently falling back.
    """
    if not module.training:
        raise RuntimeError("The cuDNN GQA attention kernel is training-only; switch to eval() for decoding")
    if not isinstance(attention_mask, Tensor) or attention_mask.ndim != 4:
        raise TypeError("The cuDNN GQA attention kernel requires a dense 4D additive attention mask")
    # Under autocast, SDPA casts the query, keys, values and mask to the autocast dtype itself.
    if attention_mask.dtype != query.dtype and not torch.is_autocast_enabled(query.device.type):
        raise TypeError(f"Attention mask dtype {attention_mask.dtype} differs from query dtype {query.dtype}")
    if dropout != 0.0 or is_causal:
        raise ValueError(
            "The cuDNN GQA attention kernel requires dropout=0 and an explicit (non-causal) mask"
        )
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
            enable_gqa=True,
        )
    return output.transpose(1, 2).contiguous(), None


def register_cudnn_gqa_attention() -> None:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if not torch.backends.cudnn.is_available() or not torch.backends.cuda.cudnn_sdp_enabled():
        raise RuntimeError("cuDNN SDPA is not available; use training_attn_implementation='sdpa'")
    ALL_ATTENTION_FUNCTIONS.register(CUDNN_GQA_ATTENTION, cudnn_gqa_attention_forward)


def _additive(allowed: Tensor, dtype: torch.dtype) -> Tensor:
    additive = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    additive.masked_fill_(~allowed, torch.finfo(dtype).min)
    return additive


def build_training_attention_mask(
    prefix_valid: Tensor,
    action_valid: Tensor,
    block_size: int,
    *,
    dtype: torch.dtype,
    history_length: int = 0,
) -> Tensor:
    """Additive `(B, 1, L, L)` mask over `[prefix, noisy actions, clean actions]`.

    The prefix may end with `history_length` native action-history tokens (see `ConcertinaActionHistoryConfig`).

    - the other prefix queries attend causally to valid prefix keys before the history;
    - history block `h` attends to the valid prefix before the history and to history blocks up to `h`;
    - noisy block `b` attends to the valid prefix, its own noisy block and clean blocks before `b`;
    - clean block `b` attends to the valid prefix and clean blocks up to and including `b`.

    Invalid keys are hidden. Each invalid query attends only to itself so no attention row is empty; such
    positions are never supervised.
    """
    batch_size, prefix_length = prefix_valid.shape
    action_length = action_valid.shape[1]
    device = prefix_valid.device
    total_length = prefix_length + 2 * action_length
    context_length = prefix_length - history_length
    noisy_start = prefix_length
    clean_start = prefix_length + action_length

    allowed = torch.zeros((batch_size, total_length, total_length), device=device, dtype=torch.bool)
    context_valid = prefix_valid[:, :context_length]
    causal = torch.tril(torch.ones((context_length, context_length), device=device, dtype=torch.bool))
    allowed[:, :context_length, :context_length] = (
        causal.unsqueeze(0) & context_valid[:, :, None] & context_valid[:, None, :]
    )
    if history_length:
        history_valid = prefix_valid[:, context_length:]
        history_blocks = torch.div(
            torch.arange(history_length, device=device), block_size, rounding_mode="floor"
        )
        history_up_to_block = history_blocks[:, None] >= history_blocks[None, :]
        allowed[:, context_length:prefix_length, :context_length] = (
            history_valid[:, :, None] & context_valid[:, None, :]
        )
        allowed[:, context_length:prefix_length, context_length:prefix_length] = (
            history_up_to_block.unsqueeze(0) & history_valid[:, :, None] & history_valid[:, None, :]
        )

    blocks = torch.div(torch.arange(action_length, device=device), block_size, rounding_mode="floor")
    same_block = blocks[:, None] == blocks[None, :]
    earlier_block = blocks[:, None] > blocks[None, :]
    up_to_block = blocks[:, None] >= blocks[None, :]
    query_valid = action_valid[:, :, None]
    key_valid = action_valid[:, None, :]
    prefix_keys = query_valid & prefix_valid[:, None, :]

    allowed[:, noisy_start:clean_start, :prefix_length] = prefix_keys
    allowed[:, noisy_start:clean_start, noisy_start:clean_start] = (
        same_block.unsqueeze(0) & query_valid & key_valid
    )
    allowed[:, noisy_start:clean_start, clean_start:] = earlier_block.unsqueeze(0) & query_valid & key_valid
    allowed[:, clean_start:, :prefix_length] = prefix_keys
    allowed[:, clean_start:, clean_start:] = up_to_block.unsqueeze(0) & query_valid & key_valid

    sequence_valid = torch.cat([prefix_valid, action_valid, action_valid], dim=1)
    invalid = (~sequence_valid).nonzero(as_tuple=False)
    allowed[invalid[:, 0], invalid[:, 1], invalid[:, 1]] = True
    return _additive(allowed.unsqueeze(1), dtype)


def build_history_block_attention_mask(
    context_valid: Tensor,
    committed_valid: Tensor,
    block_valid: Tensor,
    *,
    dtype: torch.dtype,
) -> Tensor:
    """Additive mask for appending one native action-history block to the KV cache.

    Valid queries see the valid context keys, the valid earlier history keys and the valid keys of their own
    block; each invalid query attends only to itself.
    """
    key_valid = torch.cat([context_valid, committed_valid, block_valid], dim=1).to(torch.bool)
    block_valid = block_valid.to(torch.bool)
    allowed = block_valid[:, :, None] & key_valid[:, None, :]
    batch_indices, query_indices = (~block_valid).nonzero(as_tuple=True)
    block_start = context_valid.shape[1] + committed_valid.shape[1]
    allowed[batch_indices, query_indices, block_start + query_indices] = True
    return _additive(allowed.unsqueeze(1), dtype)


def build_block_attention_mask(
    prefix_valid: Tensor,
    *,
    committed_tokens: int,
    block_tokens: int,
    dtype: torch.dtype,
) -> Tensor:
    """Additive `(B, 1, block, cache + block)` mask for decoding one action block against the KV cache.

    Every query of the current block sees all valid prefix keys, all committed action keys and the whole
    current block.
    """
    batch_size = prefix_valid.shape[0]
    key_valid = torch.cat(
        [
            prefix_valid.to(dtype=torch.bool),
            torch.ones(
                batch_size, committed_tokens + block_tokens, dtype=torch.bool, device=prefix_valid.device
            ),
        ],
        dim=1,
    )
    allowed = key_valid[:, None, None, :].expand(batch_size, 1, block_tokens, key_valid.shape[1])
    return _additive(allowed, dtype)
