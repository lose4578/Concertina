"""Concertina policy: the LeRobot `PreTrainedPolicy` wrapper around the LeRobot-independent
`concertina.model.ConcertinaModel`."""

from __future__ import annotations

import builtins
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import Tensor, nn

from concertina.model import ConcertinaModel
from concertina.model.constants import ACTION_DIM, TASK
from concertina.model.rtc import RTCActionQueue, stack_rtc_prefix
from concertina.policy.configuration_concertina import ConcertinaConfig

T = TypeVar("T", bound="ConcertinaPolicy")

# Subdirectory of a saved policy holding the backbone's config, tokenizer and image-processor files (no weights);
# used when `model.backbone_path` is empty.
BUNDLED_BACKBONE_DIR = "backbone"


class ConcertinaPolicy(PreTrainedPolicy):
    config_class = ConcertinaConfig
    name = "concertina"
    # FSDP2 wrap units: the text decoder layers and vision blocks of the Qwen3-VL backbone.
    _fsdp_wrap_modules = ["Qwen3VLTextDecoderLayer", "Qwen3VLVisionBlock"]

    def __init__(
        self, config: ConcertinaConfig, *, load_backbone_weights: bool | None = None, **kwargs: Any
    ) -> None:
        """`kwargs` absorbs the dataset metadata and statistics passed by the LeRobot policy factory; Concertina
        normalizes in its processors, so the model itself does not consume them.

        By default the backbone weights are read only for a policy without `config.pretrained_path`: a policy built
        for a checkpoint gets every weight from it (`from_pretrained`, or a sharded checkpoint loaded after
        construction when resuming), so its backbone directory needs no weight files. An empty
        `model.backbone_path` refers to the `backbone/` directory bundled with `config.pretrained_path` (a local
        directory, or a Hub repository id at `config.pretrained_revision`).

        `config.dtype` is the parameter storage dtype; None keeps bfloat16, the dtype the model computes in. Training
        sets float32: every parameter (and the optimizer state) then has fp32 precision while the training loop's
        bf16 autocast (`accelerator.mixed_precision=bf16`) computes in bf16. Decoding always uses bf16 weights.
        """
        super().__init__(config)
        if config.pretrained_path:
            _resolve_bundled_backbone(config, config.pretrained_path, revision=config.pretrained_revision)
        if load_backbone_weights is None:
            load_backbone_weights = not config.pretrained_path
        self.model = ConcertinaModel(config.model, load_backbone_weights=load_backbone_weights)
        if config.dtype is not None:
            self.model.to(config.dtype)
        full_model = self.model.backbone.full_model
        if full_model.config.tie_word_embeddings:
            # The tied LM head is never saved: full state dicts gathered from sharded (FSDP2) parameters hold a
            # separate copy per name, which safetensors then rejects as a duplicate of one shared tensor.
            self._tied_lm_head = (
                "model.backbone.full_model.lm_head.weight",
                "model.backbone.full_model.model.language_model.embed_tokens.weight",
            )
            self.register_state_dict_post_hook(_drop_tied_lm_head)
            self.register_load_state_dict_pre_hook(_fill_tied_lm_head)
        else:
            self._tied_lm_head = None
        self.reset()

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> T:
        """Load a saved policy.

        Unlike the base class, the weights must match the model exactly by default, and they are read only from
        the policy checkpoint: the backbone directory contributes the architecture, tokenizer and image processor.
        An empty `model.backbone_path` refers to the `backbone/` directory bundled with the checkpoint, a local
        directory or a Hub repository id (downloaded into the Hugging Face cache with the given revision).
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        _resolve_bundled_backbone(
            config,
            pretrained_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
        )
        return super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            load_backbone_weights=False,
            **kwargs,
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        return self.model.optim_param_groups()

    def reset(self) -> None:
        model_config = self.config.model
        self._action_queue: deque[Tensor] = deque(maxlen=model_config.inference.n_action_steps)
        self._rtc_queues: list[RTCActionQueue] = []
        self._action_history: deque[Tensor] = deque(maxlen=model_config.action_history.horizon)

    def drop_queued_actions(self) -> None:
        self._action_queue.clear()
        for queue in self._rtc_queues:
            queue.drop()

    def count_queued_actions(self) -> int:
        return len(self._rtc_queues[0]) if self._rtc_queues else len(self._action_queue)

    def forward(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        loss, metrics = self.model(batch)
        return loss, {key: value.item() for key, value in metrics.items()}

    def record_executed_action(self, action: Tensor) -> None:
        """Record a normalized action `(B, D)` once it is executed; `select_action` does so for every action it
        returns. Callers that execute decoded chunks without `select_action` must call this once per action."""
        if not self.config.model.action_history.enabled:
            return
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
            raise ValueError(
                f"Expected one action of dimension {ACTION_DIM} per sample, got {tuple(action.shape)}"
            )
        if self._action_history and self._action_history[-1].shape[0] != action.shape[0]:
            self._action_history.clear()
        self._action_history.append(action.detach().clone())

    def _history_inputs(self, batch_size: int) -> dict[str, Tensor]:
        """The recorded history, left-padded to the horizon, as `predict_action_chunk` keyword arguments."""
        config = self.config.model.action_history
        if not config.enabled:
            return {}
        device, dtype = self.model.device, self.model.dtype
        history = torch.zeros((batch_size, config.horizon, ACTION_DIM), device=device, dtype=dtype)
        valid = torch.zeros((batch_size, config.horizon), device=device, dtype=torch.bool)
        if self._action_history:
            if any(entry.shape[0] != batch_size for entry in self._action_history):
                raise RuntimeError("The recorded action history does not match the batch size")
            recorded = torch.stack(
                [entry.to(device=device, dtype=dtype) for entry in self._action_history], dim=1
            )
            history[:, -recorded.shape[1] :] = recorded
            valid[:, -recorded.shape[1] :] = True
        return {"history": history, "history_valid": valid}

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], *, noise: Tensor | None = None) -> Tensor:
        """The full normalized action chunk, `(B, chunk_size, action_dim)`."""
        _reject_noise(noise)
        self.eval()
        with _bfloat16_weights(self.model):
            return self.model.predict_action_chunk(batch, **self._history_inputs(len(batch[TASK])))

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Any], *, noise: Tensor | None = None, inference_delay: int | Sequence[int] = 0
    ) -> Tensor:
        """One normalized action `(B, action_dim)`; a new chunk is decoded whenever the action queue is empty.

        With real-time chunking, `inference_delay` (per sample or shared) is the number of actions executed while
        a chunk was being decoded.
        """
        _reject_noise(noise)
        self.eval()
        if self.config.model.inference.rtc:
            return self._select_action_rtc(batch, inference_delay)
        if not self._action_queue:
            n_action_steps = self.config.model.inference.n_action_steps
            with _bfloat16_weights(self.model):
                actions = self.model.predict_action_chunk(
                    batch, num_actions=n_action_steps, **self._history_inputs(len(batch[TASK]))
                )
            self._action_queue.extend(actions.transpose(0, 1))
        action = self._action_queue.popleft()
        self.record_executed_action(action)
        return action

    def _select_action_rtc(self, batch: dict[str, Any], inference_delay: int | Sequence[int]) -> Tensor:
        recipe = self.config.model.inference
        batch_size = len(batch[TASK])
        if len(self._rtc_queues) != batch_size:
            horizon = recipe.rtc_execution_horizon or recipe.n_action_steps
            self._rtc_queues = [RTCActionQueue(horizon) for _ in range(batch_size)]
        empty = [queue.empty() for queue in self._rtc_queues]
        if any(empty) and not all(empty):
            raise RuntimeError("RTC action queues are out of sync across the batch")
        if all(empty):
            delays = (
                [inference_delay] * batch_size if isinstance(inference_delay, int) else list(inference_delay)
            )
            if len(delays) == 1:
                delays = delays * batch_size
            if len(delays) != batch_size:
                raise ValueError(f"Expected 1 or {batch_size} inference delays, got {len(delays)}")
            source = recipe.rtc_prefix_source
            prefix = stack_rtc_prefix(self._rtc_queues, source, self.config.model.chunk_size)
            with _bfloat16_weights(self.model):
                actions = self.model.predict_action_chunk(batch, prefix_actions=prefix)
            prefix_length = prefix.shape[1] if prefix is not None and source == "executed_tail" else 0
            for queue, chunk, delay in zip(self._rtc_queues, actions, delays, strict=True):
                queue.replace(chunk, int(delay), prefix_length)
        return torch.stack([queue.get() for queue in self._rtc_queues], dim=0)


@contextmanager
def _bfloat16_weights(module: nn.Module) -> Iterator[None]:
    """Run with bf16 copies of the floating parameters and buffers that are stored in another dtype (fp32 training
    weights), so that decoding matches the bf16 weights the released checkpoints were evaluated with.

    Inference-only (the copies are detached); sharded (FSDP2) modules are left to their own mixed precision.
    """
    from torch.distributed.fsdp import FSDPModule

    swapped = []
    if not any(isinstance(owner, FSDPModule) for owner in module.modules()):
        copies: dict[int, Tensor] = {}
        for owner in module.modules():
            for store, wrap in ((owner._parameters, True), (owner._buffers, False)):
                for name, tensor in store.items():
                    if tensor is None or not tensor.is_floating_point() or tensor.dtype == torch.bfloat16:
                        continue
                    if id(tensor) not in copies:  # Tied tensors keep sharing one copy.
                        copy = tensor.detach().to(torch.bfloat16)
                        copies[id(tensor)] = nn.Parameter(copy, requires_grad=False) if wrap else copy
                    swapped.append((store, name, tensor))
                    store[name] = copies[id(tensor)]
    try:
        yield
    finally:
        for store, name, tensor in reversed(swapped):
            store[name] = tensor


def _resolve_bundled_backbone(config: ConcertinaConfig, checkpoint: str | Path, **hub_kwargs: Any) -> None:
    """Point an empty `model.backbone_path` at the `backbone/` directory bundled with `checkpoint`.

    `checkpoint` is a local policy directory or a Hub repository id; for a repository id the bundled files are
    downloaded into the Hugging Face cache (`hub_kwargs`: revision, cache_dir, token, ... of `snapshot_download`).
    """
    if config.model.backbone_path:
        return
    if Path(checkpoint).is_dir():
        root = Path(checkpoint)
    else:
        try:
            root = Path(
                snapshot_download(str(checkpoint), allow_patterns=f"{BUNDLED_BACKBONE_DIR}/*", **hub_kwargs)
            )
        except HFValidationError as error:
            raise FileNotFoundError(
                f"{checkpoint} is neither a local directory nor a Hub repository id"
            ) from error
    bundled = root / BUNDLED_BACKBONE_DIR
    if not bundled.is_dir():
        raise FileNotFoundError(
            f"model.backbone_path is empty and {checkpoint} bundles no '{BUNDLED_BACKBONE_DIR}' directory"
        )
    config.model.backbone_path = str(bundled)


def _reject_noise(noise: Tensor | None) -> None:
    if noise is not None:
        raise ValueError("Concertina decoding is deterministic and does not take a noise tensor")


def _drop_tied_lm_head(policy: ConcertinaPolicy, state_dict: dict, prefix: str, local_metadata: dict) -> None:
    lm_head, _ = policy._tied_lm_head
    state_dict.pop(prefix + lm_head, None)


def _fill_tied_lm_head(policy: ConcertinaPolicy, state_dict: dict, prefix: str, *args: Any) -> None:
    lm_head, embeddings = policy._tied_lm_head
    if prefix + lm_head not in state_dict and prefix + embeddings in state_dict:
        state_dict[prefix + lm_head] = state_dict[prefix + embeddings]
