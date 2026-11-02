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
# Portions adapted from Transformers (src/transformers/models/qwen3_vl/processing_qwen3_vl.py,
# `Qwen3VLProcessor.__call__`), Copyright 2025 The Qwen Team and The HuggingFace Inc. team, Apache License 2.0.
# Modified: `_expand_image_tokens` expands the image placeholders of one prompt from cached image grids. See NOTICE.
"""Qwen3-VL vision-language backbone with conditional block-diffusion entry points.

The upstream Transformers Qwen3-VL implementation is used unchanged (vision tower, DeepStack, MRoPE,
decoder). This module only builds the multimodal prompt prefix, runs the joint noisy/clean training sequence
and exposes KV-cache prefill/read/commit for block decoding.
"""

import copy
from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

import torch
from torch import Tensor, nn
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    PreTrainedConfig,
    Qwen3VLConfig,
    Qwen3VLProcessor,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.initialization import no_init_weights

from concertina.model.attention import (
    CUDNN_GQA_ATTENTION,
    build_history_block_attention_mask,
    register_cudnn_gqa_attention,
)
from concertina.model.constants import PROMPT_LAYOUT_CACHE_CAPACITY
from concertina.model.layout_cache import LayoutCache
from concertina.model.liger import install_liger_kernels

# Checkpoints that rename the Qwen3-VL model types but keep its tensor layout.
_QWEN3_VL_COMPATIBLE_MODEL_TYPES = {"bard_vl"}

ImageGrid = tuple[tuple[int, int, int], ...]
# (task text, number of views, per-view image grid)
PromptKey = tuple[str, int, ImageGrid]


@dataclass(frozen=True)
class _PromptLayout:
    input_ids: Tensor
    attention_mask: Tensor


@dataclass(frozen=True)
class _MRoPELayout:
    """Positions of the active prompt followed by the conditioning and action slots, shape `(3, L)`."""

    position_ids: Tensor
    prompt_length: int


@dataclass
class PreparedPrefix:
    """Multimodal prompt + conditioning prefix, and the MRoPE positions reserved for the action tokens.

    The prefix ends with `history_length` native action-history tokens, cached in blocks of `history_block_size`.
    """

    inputs_embeds: Tensor
    attention_mask: Tensor
    position_ids: Tensor
    action_position_ids: Tensor
    pixel_values: Tensor
    image_grid_thw: Tensor
    history_length: int = 0
    history_block_size: int = 0

    @property
    def length(self) -> int:
        return int(self.inputs_embeds.shape[1])


class ReadOnlyPrefixCache(Cache):
    """Non-mutating view over a filled `DynamicCache`.

    Qwen attention always calls `cache.update`; this view returns the committed prefix K/V concatenated with the
    current block's K/V without writing, so a draft block can be scored repeatedly against the same prefix.
    """

    def __init__(self, source: DynamicCache) -> None:
        super().__init__(layers=source.layers)
        self._source = source

    def update(
        self, key_states: Tensor, value_states: Tensor, layer_idx: int, *args: Any, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        layer = self._source.layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)


def _native_qwen3_vl_config(model_path: str) -> Qwen3VLConfig | None:
    """Map a renamed Qwen3-VL-layout config back to the native Qwen3-VL classes; `None` for native configs."""
    raw_config, _ = PreTrainedConfig.get_config_dict(model_path)
    if raw_config.get("model_type") not in _QWEN3_VL_COMPATIBLE_MODEL_TYPES:
        return None
    config = copy.deepcopy(raw_config)
    config["model_type"] = "qwen3_vl"
    config["architectures"] = ["Qwen3VLForConditionalGeneration"]
    config.pop("auto_map", None)
    for key, model_type in (("text_config", "qwen3_vl_text"), ("vision_config", "qwen3_vl")):
        if isinstance(config.get(key), dict):
            config[key]["model_type"] = model_type
            config[key].pop("auto_map", None)
    return Qwen3VLConfig.from_dict(config)


def _load_processor(model_path: str):
    try:
        return AutoProcessor.from_pretrained(model_path)
    except (KeyError, OSError, ValueError):
        # Renamed checkpoints may have an unknown outer model type but ship standard Qwen3-VL processor assets.
        return Qwen3VLProcessor.from_pretrained(model_path)


class QwenVLBackbone(nn.Module):
    def __init__(
        self,
        model_path: str,
        *,
        vision_attn_implementation: str = "sdpa",
        training_attn_implementation: str = "cudnn_sdpa_native_gqa",
        gradient_checkpointing: bool = True,
        use_liger_kernels: bool = True,
        load_weights: bool = True,
    ) -> None:
        super().__init__()
        if not model_path:
            raise ValueError("A backbone path is required")
        if vision_attn_implementation.startswith("flash_attention") and find_spec("flash_attn") is None:
            raise ValueError(
                f"vision_attn_implementation={vision_attn_implementation!r} requires the flash-attn package; "
                "install it or set vision_attn_implementation='sdpa'"
            )
        self.processor = _load_processor(model_path)
        if getattr(self.processor.image_processor, "backend", None) != "torchvision":
            raise ValueError("The backbone processor must provide a torchvision-backed image processor")
        native_config = _native_qwen3_vl_config(model_path)
        if load_weights:
            load_kwargs: dict[str, Any] = {
                "dtype": "auto",
                "low_cpu_mem_usage": True,
                "attn_implementation": "sdpa",
            }
            if native_config is not None:
                load_kwargs["config"] = native_config
            self.full_model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
        else:
            # The policy checkpoint supplies every weight; only the architecture, tokenizer and image processor
            # are taken from the backbone directory, which therefore needs no weight files.
            config = native_config if native_config is not None else Qwen3VLConfig.from_pretrained(model_path)
            with no_init_weights():
                self.full_model = AutoModelForImageTextToText.from_config(
                    config, dtype=torch.bfloat16, attn_implementation="sdpa"
                )
            # `from_pretrained` ties the LM head to the token embedding after loading; do the same here so the
            # checkpoint's single copy of the embedding fills both names.
            self.full_model.tie_weights()
            lm_head, embeddings = (
                self.full_model.get_output_embeddings(),
                self.full_model.get_input_embeddings(),
            )
            if config.tie_word_embeddings and lm_head.weight is not embeddings.weight:
                raise RuntimeError("Failed to tie the language-model head to the token embedding")
        if self.full_model.config.model_type != "qwen3_vl":
            raise ValueError(
                f"Expected a Qwen3-VL checkpoint, got model_type={self.full_model.config.model_type!r}"
            )

        if use_liger_kernels:
            install_liger_kernels(self.full_model.model.language_model)
        self.full_model.model.visual.config._attn_implementation_internal = vision_attn_implementation
        if gradient_checkpointing:
            self.full_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self._prompt_cache: LayoutCache[PromptKey, _PromptLayout] = LayoutCache(PROMPT_LAYOUT_CACHE_CAPACITY)
        self._mrope_cache: LayoutCache[tuple[PromptKey, int, int], _MRoPELayout] = LayoutCache(
            PROMPT_LAYOUT_CACHE_CAPACITY
        )

        if training_attn_implementation == "cudnn_sdpa_native_gqa":
            register_cudnn_gqa_attention()
            self._training_attn_implementation = CUDNN_GQA_ATTENTION
        else:
            self._training_attn_implementation = "sdpa"
        self._set_text_attn_implementation(self._training_attn_implementation if self.training else "sdpa")

    @property
    def hidden_size(self) -> int:
        return int(self.full_model.config.text_config.hidden_size)

    @property
    def language_model(self) -> nn.Module:
        return self.full_model.model.language_model

    def train(self, mode: bool = True) -> "QwenVLBackbone":
        super().train(mode)
        # The training kernel only supports the dense training mask; cache decoding always uses SDPA.
        self._set_text_attn_implementation(self._training_attn_implementation if mode else "sdpa")
        return self

    def _set_text_attn_implementation(self, implementation: str) -> None:
        configs = [self.full_model.config.text_config, self.language_model.config]
        configs += [layer.self_attn.config for layer in self.language_model.layers]
        for config in {id(config): config for config in configs}.values():
            config._attn_implementation_internal = implementation

    # Prompt construction.

    def _placeholder_token_id(self) -> int:
        tokenizer = self.processor.tokenizer
        for candidate in (tokenizer.pad_token_id, tokenizer.eos_token_id):
            if candidate is not None:
                return int(candidate)
        raise ValueError("The tokenizer must define pad_token_id or eos_token_id")

    def _build_prompt(self, task: str, num_images: int) -> str:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in range(num_images)]
        content.append({"type": "text", "text": task})
        messages = [{"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _expand_image_tokens(self, prompt: str, image_grid: ImageGrid) -> str:
        image_token = str(self.processor.image_token)
        merge_length = int(self.processor.image_processor.merge_size) ** 2
        for grid in image_grid:
            num_tokens = int(grid[0] * grid[1] * grid[2]) // merge_length
            prompt = prompt.replace(image_token, "<|placeholder|>" * num_tokens, 1)
        if image_token in prompt:
            raise ValueError("The chat template contains more image placeholders than images")
        return prompt.replace("<|placeholder|>", image_token)

    def _prompt_layouts(self, keys: list[PromptKey]) -> list[_PromptLayout]:
        missing = list(dict.fromkeys(key for key in keys if self._prompt_cache.get(key) is None))
        resolved: dict[PromptKey, _PromptLayout] = {}
        if missing:
            prompts = [
                self._expand_image_tokens(self._build_prompt(task, views), grid)
                for task, views, grid in missing
            ]
            tokenized = self.processor(text=prompts, padding=True, return_tensors="pt")
            token_ids = tokenized["input_ids"].to(device="cpu", dtype=torch.long)
            token_mask = tokenized["attention_mask"].to(device="cpu", dtype=torch.long)
            for index, key in enumerate(missing):
                active = token_mask[index].bool()
                layout = _PromptLayout(
                    input_ids=token_ids[index, active].contiguous().clone(),
                    attention_mask=token_mask[index, active].contiguous().clone(),
                )
                resolved[key] = layout
                self._prompt_cache.put(key, layout)
        for key in keys:
            if key not in resolved:
                resolved[key] = self._prompt_cache.get(key)
        return [resolved[key] for key in keys]

    def _pad_prompt_layouts(self, layouts: list[_PromptLayout]) -> tuple[Tensor, Tensor]:
        max_length = max(int(layout.input_ids.numel()) for layout in layouts)
        tokenizer = self.processor.tokenizer
        pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self._placeholder_token_id()
        )
        input_ids = torch.full((len(layouts), max_length), int(pad_token_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        left_padding = tokenizer.padding_side == "left"
        for index, layout in enumerate(layouts):
            length = int(layout.input_ids.numel())
            target = slice(max_length - length, max_length) if left_padding else slice(0, length)
            input_ids[index, target] = layout.input_ids
            attention_mask[index, target] = layout.attention_mask
        return input_ids, attention_mask

    def _mrope_positions(
        self,
        keys: list[PromptKey],
        input_ids: Tensor,
        attention_mask: Tensor,
        extension_length: int,
        num_action_tokens: int,
    ) -> Tensor:
        """Padded `(3, B, prompt + extension)` positions, computed per unpadded sample and cached on CPU."""
        batch_size, prompt_length = input_ids.shape
        placeholder_id = self._placeholder_token_id()
        image_token_id = int(self.full_model.config.image_token_id)
        positions = torch.ones((3, batch_size, prompt_length + extension_length), dtype=torch.long)
        for index, key in enumerate(keys):
            active = attention_mask[index].bool()
            cache_key = (key, extension_length - num_action_tokens, num_action_tokens)
            layout = self._mrope_cache.get(cache_key)
            if layout is None:
                sample_ids = input_ids[index, active]
                extended_ids = torch.cat(
                    [sample_ids, torch.full((extension_length,), placeholder_id, dtype=torch.long)]
                ).unsqueeze(0)
                grid = torch.tensor(key[2], dtype=torch.long)
                # Modality type ids: 0 for text (including the conditioning/action placeholders), 1 for image tokens.
                mm_token_type_ids = (extended_ids == image_token_id).to(torch.int32)
                position_ids, _ = self.full_model.model.get_rope_index(
                    extended_ids,
                    mm_token_type_ids,
                    image_grid_thw=grid,
                    attention_mask=torch.ones_like(extended_ids),
                )
                layout = _MRoPELayout(
                    position_ids=position_ids[:, 0].to(device="cpu", dtype=torch.long).contiguous().clone(),
                    prompt_length=int(sample_ids.numel()),
                )
                self._mrope_cache.put(cache_key, layout)
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            positions[:, index, active_indices] = layout.position_ids[:, : layout.prompt_length]
            positions[:, index, prompt_length:] = layout.position_ids[:, layout.prompt_length :]
        return positions

    @staticmethod
    def _images_to_uint8(pixel_values: Tensor) -> list[list[Tensor]]:
        """`(B, C, H, W)` or `(B, V, C, H, W)` float images in [0, 1] to per-sample lists of uint8 images.

        The quantization matches `torchvision.transforms.functional.to_pil_image` on float tensors.
        """
        images = pixel_values.detach()
        if images.ndim == 4:
            images = images.unsqueeze(1)
        images = (images.float() * 255).to(torch.uint8).contiguous()
        return [[images[b, v] for v in range(images.shape[1])] for b in range(images.shape[0])]

    def prepare_prefix(
        self,
        pixel_values: Tensor,
        tasks: list[str],
        conditioning_embeds: Tensor,
        num_action_tokens: int,
        *,
        conditioning_valid: Tensor | None = None,
        history_embeds: Tensor | None = None,
        history_valid: Tensor | None = None,
        history_block_size: int = 0,
    ) -> PreparedPrefix:
        """Embed `[chat prompt with images, conditioning tokens]` and reserve positions for the action tokens.

        `conditioning_valid` hides conditioning tokens from attention without changing any position. Native
        action-history tokens (`history_embeds`, valid where `history_valid`) take the positions just before the
        action tokens.
        """
        embedding = self.full_model.get_input_embeddings()
        device = embedding.weight.device
        images = self._images_to_uint8(pixel_values)
        if len(tasks) != len(images):
            raise ValueError(f"Got {len(tasks)} task strings for {len(images)} samples")
        processed = self.processor.image_processor(images=images, return_tensors="pt", device=device)
        image_grid_thw = processed["image_grid_thw"]
        cpu_grid = image_grid_thw.to(device="cpu", dtype=torch.long)
        grids: list[ImageGrid] = []
        start = 0
        for sample_images in images:
            rows = cpu_grid[start : start + len(sample_images)].tolist()
            grids.append(tuple(tuple(int(value) for value in row) for row in rows))
            start += len(sample_images)
        keys = [
            (str(task), len(sample_images), grid)
            for task, sample_images, grid in zip(tasks, images, grids, strict=True)
        ]
        input_ids, prompt_mask = self._pad_prompt_layouts(self._prompt_layouts(keys))

        num_conditioning = conditioning_embeds.shape[1]
        history_length = 0 if history_embeds is None else history_embeds.shape[1]
        if history_length and (history_block_size <= 0 or history_length % history_block_size):
            raise ValueError(
                f"history length {history_length} must be a multiple of history_block_size={history_block_size}"
            )
        reserved_tokens = history_length + num_action_tokens
        positions = self._mrope_positions(
            keys, input_ids, prompt_mask, num_conditioning + reserved_tokens, reserved_tokens
        ).to(device)
        prompt_embeds = embedding(input_ids.to(device))
        inputs_embeds = torch.cat(
            [prompt_embeds, conditioning_embeds.to(device=device, dtype=prompt_embeds.dtype)], 1
        )
        prefix_length = inputs_embeds.shape[1]
        attention_mask = torch.cat(
            [
                prompt_mask.to(device),
                torch.ones((prompt_mask.shape[0], num_conditioning), dtype=prompt_mask.dtype, device=device),
            ],
            dim=1,
        )
        if conditioning_valid is not None:
            attention_mask[:, prefix_length - num_conditioning :] = conditioning_valid.to(
                device=device, dtype=attention_mask.dtype
            )
        position_ids = positions[..., :prefix_length]
        action_position_ids = positions[..., prefix_length : prefix_length + reserved_tokens]
        if history_length:
            inputs_embeds = torch.cat(
                [inputs_embeds, history_embeds.to(device=device, dtype=inputs_embeds.dtype)], 1
            )
            attention_mask = torch.cat(
                [attention_mask, history_valid.to(device=device, dtype=attention_mask.dtype)], dim=1
            )
            position_ids = torch.cat([position_ids, action_position_ids[..., :history_length]], dim=-1)
            action_position_ids = action_position_ids[..., history_length:]
        return PreparedPrefix(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            action_position_ids=action_position_ids,
            pixel_values=processed["pixel_values"].to(device),
            image_grid_thw=image_grid_thw.to(device),
            history_length=history_length,
            history_block_size=history_block_size,
        )

    # Model entry points.

    def training_forward(
        self,
        prefix: PreparedPrefix,
        noisy_action_embeds: Tensor,
        clean_action_embeds: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Run `[prefix, noisy actions, clean actions]` jointly; return the hidden states of the noisy actions."""
        dtype = prefix.inputs_embeds.dtype
        inputs_embeds = torch.cat(
            [prefix.inputs_embeds, noisy_action_embeds.to(dtype), clean_action_embeds.to(dtype)], dim=1
        )
        position_ids = torch.cat(
            [prefix.position_ids, prefix.action_position_ids, prefix.action_position_ids], dim=-1
        )
        outputs = self.full_model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=prefix.pixel_values,
            image_grid_thw=prefix.image_grid_thw,
            use_cache=False,
            return_dict=True,
        )
        num_actions = noisy_action_embeds.shape[1]
        return outputs.last_hidden_state[:, prefix.length : prefix.length + num_actions]

    def _require_eval(self) -> None:
        if self.training and self._training_attn_implementation != "sdpa":
            raise RuntimeError("KV-cache decoding requires eval mode")

    def prefill(self, prefix: PreparedPrefix) -> DynamicCache:
        """KV cache of the prefix; native action history is appended block by block, like committed actions."""
        self._require_eval()
        context_length = prefix.length - prefix.history_length
        outputs = self.full_model.model(
            inputs_embeds=prefix.inputs_embeds[:, :context_length],
            attention_mask=prefix.attention_mask[:, :context_length],
            position_ids=prefix.position_ids[..., :context_length],
            pixel_values=prefix.pixel_values,
            image_grid_thw=prefix.image_grid_thw,
            use_cache=True,
            return_dict=True,
        )
        cache = outputs.past_key_values
        context_valid = prefix.attention_mask[:, :context_length].to(torch.bool)
        history_valid = prefix.attention_mask[:, context_length:].to(torch.bool)
        for block_start in range(0, prefix.history_length, prefix.history_block_size or 1):
            block_end = block_start + prefix.history_block_size
            attention_mask = build_history_block_attention_mask(
                context_valid,
                history_valid[:, :block_start],
                history_valid[:, block_start:block_end],
                dtype=prefix.inputs_embeds.dtype,
            )
            storage = slice(context_length + block_start, context_length + block_end)
            cache = self.commit_block(
                prefix.inputs_embeds[:, storage], prefix.position_ids[..., storage], attention_mask, cache
            )
        return cache

    def _cached_forward(
        self,
        action_embeds: Tensor,
        action_position_ids: Tensor,
        attention_mask: Tensor,
        cache: Cache,
    ):
        self._require_eval()
        return self.full_model.model(
            inputs_embeds=action_embeds,
            attention_mask=attention_mask,
            position_ids=action_position_ids,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    def read_block(
        self,
        action_embeds: Tensor,
        action_position_ids: Tensor,
        attention_mask: Tensor,
        cache: DynamicCache,
    ) -> Tensor:
        """Hidden states of a draft block given the committed cache, which is left unchanged."""
        outputs = self._cached_forward(
            action_embeds, action_position_ids, attention_mask, ReadOnlyPrefixCache(cache)
        )
        return outputs.last_hidden_state

    def commit_block(
        self,
        action_embeds: Tensor,
        action_position_ids: Tensor,
        attention_mask: Tensor,
        cache: DynamicCache,
    ) -> DynamicCache:
        """Append the K/V of a finalized block to the cache."""
        outputs = self._cached_forward(action_embeds, action_position_ids, attention_mask, cache)
        return outputs.past_key_values
