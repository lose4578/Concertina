# Copyright 2026 The Concertina Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from Transformers (src/transformers/models/qwen3_vl/modeling_qwen3_vl.py,
# `Qwen3VLTextAttention.forward`), Copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0,
# and from Liger-Kernel (src/liger_kernel/transformers/monkey_patch.py, `apply_liger_kernel_to_qwen3_vl`),
# Copyright 2024 LinkedIn Corporation, BSD 2-Clause License. Modified: the attention forward applies Liger's
# rotary-embedding kernel, and the kernels are bound to the modules of one model instead of patching module-level
# Transformers symbols. See NOTICE.
"""Instance-scoped Liger kernels for the Qwen3-VL text decoder.

Upstream Liger patches module-level Transformers symbols; here the Liger RMSNorm, RoPE and SwiGLU forwards are
bound only to the modules of one loaded model, so several backbones can coexist in a process. The vision tower,
the language-model head and the loss are untouched.
"""

import hashlib
import inspect
import textwrap
from importlib.metadata import version
from types import MethodType
from typing import Any

import torch
from torch import Tensor

# `_liger_rope_attention_forward` replicates `Qwen3VLTextAttention.forward` of this Transformers release; the
# digest of that upstream source is checked at install time so a changed upstream forward fails loudly.
_REPLICATED_TRANSFORMERS_VERSION = "5.5.4"
_REPLICATED_ATTENTION_FORWARD_SHA256 = "5d6148ad4fd8d7fffc8447033dd784acfca33d412778363bd9010db8b87bf2b2"
_PATCH_MARKER = "_concertina_liger_patch"


def _liger_rms_norm_forward(self: torch.nn.Module, hidden_states: Tensor) -> Tensor:
    from liger_kernel.transformers.rms_norm import LigerRMSNorm

    return LigerRMSNorm.forward(self, hidden_states)


def _liger_swiglu_forward(self: torch.nn.Module, hidden_states: Tensor) -> Tensor:
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP

    return LigerSwiGLUMLP.forward(self, hidden_states)


def _liger_rope_attention_forward(
    self: torch.nn.Module,
    hidden_states: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    attention_mask: Tensor | None,
    past_key_values: Any = None,
    **kwargs: Any,
) -> tuple[Tensor, Tensor | None]:
    """`Qwen3VLTextAttention.forward` with only the rotary embedding replaced by Liger's kernel."""
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from transformers.models.qwen3_vl import modeling_qwen3_vl

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = modeling_qwen3_vl.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, modeling_qwen3_vl.eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    return self.o_proj(attn_output), attn_weights


def _bind(module: torch.nn.Module, forward: Any) -> None:
    if hasattr(module, _PATCH_MARKER):
        raise RuntimeError(f"{type(module).__name__} already has a Liger patch")
    module.forward = MethodType(forward, module)
    setattr(module, _PATCH_MARKER, True)


def upstream_attention_forward_sha256() -> str:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention

    source = textwrap.dedent(inspect.getsource(Qwen3VLTextAttention.forward))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def install_liger_kernels(text_model: torch.nn.Module) -> None:
    """Patch the RMSNorms, attention RoPE and MLPs of one `Qwen3VLTextModel` instance."""
    import transformers
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention,
        Qwen3VLTextDecoderLayer,
        Qwen3VLTextMLP,
        Qwen3VLTextRMSNorm,
    )

    version("liger-kernel")  # Fail early with PackageNotFoundError when Liger is missing.
    if upstream_attention_forward_sha256() != _REPLICATED_ATTENTION_FORWARD_SHA256:
        raise RuntimeError(
            "The Liger attention adapter replicates Qwen3VLTextAttention.forward of transformers=="
            f"{_REPLICATED_TRANSFORMERS_VERSION}, but transformers=={transformers.__version__} ships a different "
            "forward; re-audit concertina.model.liger before updating the recorded digest"
        )

    rms_modules: list[torch.nn.Module] = [text_model.norm]
    attention_modules: list[torch.nn.Module] = []
    mlp_modules: list[torch.nn.Module] = []
    for index, layer in enumerate(text_model.layers):
        if (
            type(layer) is not Qwen3VLTextDecoderLayer
            or type(layer.self_attn) is not Qwen3VLTextAttention
            or type(layer.mlp) is not Qwen3VLTextMLP
        ):
            raise TypeError(f"Unexpected Qwen3-VL decoder layer structure at index {index}")
        rms_modules.extend(
            (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            )
        )
        attention_modules.append(layer.self_attn)
        mlp_modules.append(layer.mlp)
    if any(type(module) is not Qwen3VLTextRMSNorm for module in rms_modules):
        raise TypeError("Unexpected RMSNorm type in the Qwen3-VL text model")

    for module in rms_modules:
        # Attributes read by `LigerRMSNorm.forward` for Llama-style casting.
        module.offset = 0.0
        module.casting_mode = "llama"
        module.in_place = True
        module.row_mode = None
        _bind(module, _liger_rms_norm_forward)
    for module in attention_modules:
        _bind(module, _liger_rope_attention_forward)
    for module in mlp_modules:
        _bind(module, _liger_swiglu_forward)
