"""Concertina: a vision-language backbone that denoises continuous action blocks by masked block diffusion."""

import math
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from concertina.model.attention import build_block_attention_mask, build_training_attention_mask
from concertina.model.backbone import PreparedPrefix, QwenVLBackbone
from concertina.model.constants import (
    ACTION,
    ACTION_DIM,
    ACTION_ENCODER_HIDDEN_DIM,
    ACTION_ENCODER_LAYERS,
    ACTION_HEAD_HIDDEN_DIM,
    ACTION_HEAD_LAYERS,
    ACTION_HISTORY_UNAVAILABLE,
    ACTION_IS_PAD,
    ACTION_VALID_DIM_MASK,
    DATASET_INDEX,
    GRIPPER_TARGET,
    NOISE_EPS,
    OBS_STATE,
    SAMPLE_INDEX,
    STATE_VALID_DIM_MASK,
    TASK,
    TOKEN_RISK_EPS,
    TOKEN_RISK_HIDDEN_DIM,
    TOKEN_RISK_LOSS_WEIGHT,
    TRANSLATION_DIMS,
)
from concertina.model.diffusion import (
    action_block_ranges,
    corrupt_action_tokens,
    mask_values,
    per_sample_commit_budgets,
    perturb_visible_context,
    reduce_masked_loss,
    sample_block_timesteps,
)
from concertina.model.heads import MLP, TokenRiskHead, action_logits_to_values, action_reconstruction_loss
from concertina.model.inputs import mask_state_dims, prepare_images, random_resized_crop
from concertina.model.rng import SampleKeyedRNG
from concertina.model.sampler import draft_block, draft_ranking, refine_block, token_confidence

if TYPE_CHECKING:
    from concertina.policy.config_model import ConcertinaModelConfig


def _valid_dim_mask(mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor | None:
    """Normalize a `(D,)` or `(B, D)` valid-dimension mask to boolean `(B, D)`."""
    if mask is None:
        return None
    mask = mask.to(device=device, dtype=torch.bool)
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1 and batch_size > 1:
        mask = mask.expand(batch_size, -1)
    return mask


def _valid_action_dim_mask(mask: Tensor | None, batch_size: int, device: torch.device) -> Tensor | None:
    mask = _valid_dim_mask(mask, batch_size, device)
    if mask is not None and not bool(mask.any(dim=1).all()):
        raise ValueError("Every sample must keep at least one valid action dimension")
    return mask


def _apply_dim_mask(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values
    return values.masked_fill(~mask, 0) if values.ndim == 2 else values.masked_fill(~mask[:, None, :], 0)


def _batch_flag(value: object) -> bool:
    if value is None:
        return False
    flag = torch.as_tensor(value).to(torch.bool)
    if bool(flag.all()):
        return True
    if not bool(flag.any()):
        return False
    raise ValueError(f"{ACTION_HISTORY_UNAVAILABLE} must be uniform across the batch")


def split_history_window(batch: dict, horizon: int, chunk_size: int) -> tuple[dict, Tensor, Tensor]:
    """Split a packed `[history, chunk]` action window (and its padding and gripper targets) at the history length.

    Returns the batch of the predicted chunk, the history `(B, horizon, D)` (zero where padded) and its validity
    `(B, horizon)`. A batch flagged with `ACTION_HISTORY_UNAVAILABLE` holds only the chunk; its history is missing.
    """
    action = batch[ACTION]
    batch_size, steps, action_dim = action.shape
    if _batch_flag(batch.get(ACTION_HISTORY_UNAVAILABLE)):
        if steps != chunk_size:
            raise ValueError(f"A batch without history needs a {chunk_size}-step action window, got {steps}")
        history_valid = torch.zeros((batch_size, horizon), device=action.device, dtype=torch.bool)
        return batch, action.new_zeros((batch_size, horizon, action_dim)), history_valid
    if steps != horizon + chunk_size:
        raise ValueError(
            f"Action history needs a packed window of {horizon} + {chunk_size} steps, got {steps}; batches without "
            f"history must set {ACTION_HISTORY_UNAVAILABLE}"
        )
    chunk_batch = dict(batch)
    history_pad = None
    for key in (ACTION, ACTION_IS_PAD, GRIPPER_TARGET):
        value = batch.get(key)
        if value is None:
            continue
        if tuple(value.shape[:2]) != (batch_size, steps):
            raise ValueError(f"{key} has shape {tuple(value.shape)}; expected ({batch_size}, {steps}, ...)")
        chunk_batch[key] = value[:, horizon:]
        if key == ACTION_IS_PAD:
            history_pad = value[:, :horizon]
    history_valid = (
        torch.ones((batch_size, horizon), device=action.device, dtype=torch.bool)
        if history_pad is None
        else ~history_pad.to(device=action.device, dtype=torch.bool)
    )
    history = action[:, :horizon].masked_fill(~history_valid.unsqueeze(-1), 0)
    return chunk_batch, history, history_valid


class ConcertinaModel(nn.Module):
    def __init__(self, config: "ConcertinaModelConfig", *, load_backbone_weights: bool = True) -> None:
        """`load_backbone_weights=False` builds the backbone without reading its weight files (a policy checkpoint
        that holds every weight is loaded on top)."""
        super().__init__()
        self.config = config
        self.backbone = QwenVLBackbone(
            config.backbone_path,
            vision_attn_implementation=config.vision_attn_implementation,
            training_attn_implementation=config.training_attn_implementation,
            gradient_checkpointing=config.gradient_checkpointing,
            use_liger_kernels=config.use_liger_kernels,
            load_weights=load_backbone_weights,
        )
        d_model = self.backbone.hidden_size
        # Registration order (head before projector) fixes the parameter order seen by gradient clipping.
        self.action_head = MLP(d_model, ACTION_DIM, ACTION_HEAD_HIDDEN_DIM, ACTION_HEAD_LAYERS)
        # Shared projector of action tokens and of the proprioceptive state (same canonical layout).
        self.action_in_proj = MLP(ACTION_DIM, d_model, ACTION_ENCODER_HIDDEN_DIM, ACTION_ENCODER_LAYERS)
        self.token_risk_head = TokenRiskHead(d_model, TOKEN_RISK_HIDDEN_DIM) if config.token_risk else None
        self.action_history_role_embedding = None
        if config.action_history.enabled:
            # Role 0: a recorded past action; role 1: a missing-history placeholder.
            self.action_history_role_embedding = nn.Embedding(2, d_model)
            nn.init.normal_(self.action_history_role_embedding.weight, std=0.02)
        self.register_buffer(
            "action_dim_loss_weights",
            torch.tensor(config.action_dim_loss_weights, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )
        # Number of training forwards so far; keys the sample-keyed randomness (step 1 is the first forward).
        self.register_buffer("training_step", torch.zeros((), dtype=torch.long))
        # Decodes so far per configured "random" draft-ranking seed.
        self._draft_seed_calls: dict[int, int] = {}
        self.to(torch.bfloat16)
        self._apply_parameter_freeze()

    def _apply_parameter_freeze(self) -> None:
        if self.config.freeze_action_heads:
            for module in (self.action_in_proj, self.action_head, self.action_history_role_embedding):
                if module is not None:
                    module.requires_grad_(False)
        layers = self.backbone.language_model.layers
        for index in self.config.freeze_backbone_layers:
            layers[index].requires_grad_(False)

    def optim_param_groups(self) -> list[dict]:
        """Trainable parameters split into "backbone", "action_heads" (projector, head and history role embedding)
        and "other"."""
        groups: dict[str, list[nn.Parameter]] = {"backbone": [], "action_heads": [], "other": []}
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                groups["backbone"].append(parameter)
            elif name.startswith(("action_head.", "action_in_proj.", "action_history_role_embedding.")):
                groups["action_heads"].append(parameter)
            else:
                groups["other"].append(parameter)
        return [{"name": name, "params": params} for name, params in groups.items() if params]

    @property
    def gripper_indices(self) -> list[int]:
        return list(self.config.gripper_indices)

    @property
    def dtype(self) -> torch.dtype:
        return self.action_head[0].weight.dtype

    @property
    def device(self) -> torch.device:
        return self.action_head[0].weight.device

    def _history_embeds(
        self,
        history: Tensor | None,
        history_valid: Tensor | None,
        batch_size: int,
        action_dim_mask: Tensor | None,
        rng: SampleKeyedRNG | None,
    ) -> tuple[Tensor, Tensor]:
        """Embed the action history (oldest first); returns the embeddings and which slots are attended."""
        config = self.config.action_history
        shape = (batch_size, config.horizon, ACTION_DIM)
        device, dtype = self.device, self.dtype
        if history is None:
            if history_valid is not None and bool(history_valid.any()):
                raise ValueError("history_valid marks recorded actions but no history was given")
            values = torch.zeros(shape, device=device, dtype=dtype)
        else:
            if tuple(history.shape) != shape or history_valid is None:
                raise ValueError(f"history must have shape {shape} and come with history_valid")
            values = history.to(device=device, dtype=dtype)
        if history_valid is None:
            real = torch.zeros(shape[:2], device=device, dtype=torch.bool)
        else:
            if tuple(history_valid.shape) != shape[:2]:
                raise ValueError(f"history_valid must have shape {shape[:2]}")
            real = history_valid.to(device=device, dtype=torch.bool)
        dim_valid = None if action_dim_mask is None else action_dim_mask.unsqueeze(1)
        value_valid = real.unsqueeze(-1) if dim_valid is None else real.unsqueeze(-1) & dim_valid
        values = torch.where(value_valid, values, torch.zeros_like(values))
        if not bool(torch.isfinite(values).all()):
            raise ValueError("The action history has non-finite values in valid dimensions")
        context = self.config.visible_action_context
        if self.training and context.apply_to_history and bool(real.any()):
            values, _ = perturb_visible_context(
                values,
                real,
                action_dim_mask,
                rng,
                config=context,
                block_size=self.config.block_size,
                gripper_indices=self.gripper_indices,
                stream_prefix="action_history.visible_action_context",
            )
        attended = real
        if config.missing_history == "random_noise":
            if not bool(real.all()):
                if rng is None:
                    noise = torch.randn(shape, device=device, dtype=dtype)
                else:
                    noise = rng.randn(
                        shape[1:], stream="action_history.missing_noise", device=device, dtype=dtype
                    )
                noise = noise * float(config.noise_std)
                if dim_valid is not None:
                    noise = torch.where(dim_valid, noise, torch.zeros_like(noise))
                values = torch.where(real.unsqueeze(-1), values, noise)
            attended = torch.ones_like(real)
        embeds = self.action_in_proj(values)
        roles = self.action_history_role_embedding((~real).long()).to(embeds.dtype)
        if config.mode == "native":
            # Recorded native history is represented like clean action blocks; only placeholders carry a role.
            roles = torch.where(
                real.unsqueeze(-1), torch.zeros((), device=embeds.device, dtype=embeds.dtype), roles
            )
        embeds = embeds + roles
        return torch.where(attended.unsqueeze(-1), embeds, torch.zeros_like(embeds)), attended

    def _prefix(
        self,
        pixel_values: Tensor,
        tasks: list[str],
        state: Tensor,
        history: Tensor | None = None,
        history_valid: Tensor | None = None,
        action_dim_mask: Tensor | None = None,
        rng: SampleKeyedRNG | None = None,
    ) -> PreparedPrefix:
        # State `(B, D)` gives one conditioning token; `(B, T, D)` (observation steps) gives T tokens.
        conditioning = self.action_in_proj(state)
        if conditioning.ndim == 2:
            conditioning = conditioning.unsqueeze(1)
        history_kwargs: dict = {}
        if self.config.action_history.enabled:
            embeds, attended = self._history_embeds(
                history, history_valid, state.shape[0], action_dim_mask, rng
            )
            if self.config.action_history.mode == "prefix":
                state_valid = torch.ones(conditioning.shape[:2], device=attended.device, dtype=torch.bool)
                history_kwargs["conditioning_valid"] = torch.cat([state_valid, attended], dim=1)
                conditioning = torch.cat([conditioning, embeds], dim=1)
            else:
                history_kwargs.update(
                    history_embeds=embeds, history_valid=attended, history_block_size=self.config.block_size
                )
        return self.backbone.prepare_prefix(
            pixel_values, tasks, conditioning, self.config.chunk_size, **history_kwargs
        )

    def forward(self, batch: dict) -> tuple[Tensor, dict[str, Tensor]]:
        """Masked block-diffusion training loss on a preprocessed batch."""
        config = self.config
        history = history_valid = None
        if config.action_history.enabled:
            batch, history, history_valid = split_history_window(
                batch, config.action_history.horizon, config.chunk_size
            )
        action = batch[ACTION]
        batch_size, chunk_size, _ = action.shape
        device = action.device
        dtype = self.dtype
        if self.training:
            self.training_step += 1
        rng = SampleKeyedRNG.from_values(
            batch[SAMPLE_INDEX],
            dataset_ids=batch.get(DATASET_INDEX),
            seed=config.rng_seed,
            step=int(self.training_step),
        )
        if rng.batch_size != batch_size:
            raise ValueError(f"Got {rng.batch_size} sample indices for a batch of {batch_size}")

        action = action.to(dtype)
        action_valid = None
        if batch.get(ACTION_IS_PAD) is not None:
            action_valid = ~batch[ACTION_IS_PAD].to(device=device, dtype=torch.bool)
        gripper_target = batch[GRIPPER_TARGET] if self.gripper_indices else None
        if gripper_target is not None and action_valid is not None:
            # Padded steps carry a zero gripper target, so their gripper channels are aligned with it.
            invalid = ~action_valid.unsqueeze(-1)
            indices = self.gripper_indices
            gripper_target = gripper_target.to(device=device)
            if gripper_target.shape[-1] == action.shape[-1]:
                gripper_target = gripper_target.clone()
                selected = gripper_target[..., indices].masked_fill(invalid, 0)
                gripper_target[..., indices] = selected
            else:
                gripper_target = gripper_target.masked_fill(invalid, 0)
                selected = gripper_target
            action = action.clone()
            action[..., indices] = torch.where(invalid, selected.to(dtype=action.dtype), action[..., indices])

        state = batch[OBS_STATE]
        pixel_values = prepare_images(batch, config.camera_keys)
        if self.training:
            state = mask_state_dims(
                state,
                config.state_mask_dims,
                config.state_mask_prob,
                rng,
                ratio_range=config.state_mask_ratio_range,
                fill_value=config.state_mask_fill_value,
            )
            if config.random_resized_crop:
                pixel_values = random_resized_crop(pixel_values, rng)

        action_dim_mask = _valid_action_dim_mask(batch.get(ACTION_VALID_DIM_MASK), batch_size, device)
        state_dim_mask = _valid_dim_mask(batch.get(STATE_VALID_DIM_MASK), batch_size, device)
        if gripper_target is not None:
            gripper_target = gripper_target.to(device=device, dtype=dtype)
        action_tokens = _apply_dim_mask(action, action_dim_mask)
        state = _apply_dim_mask(state.to(dtype), state_dim_mask)
        if action_valid is None:
            action_valid = torch.ones(batch_size, chunk_size, device=device, dtype=torch.bool)

        timesteps = sample_block_timesteps(
            rng,
            num_blocks=chunk_size // config.block_size,
            block_size=config.block_size,
            eps=NOISE_EPS,
            device=device,
        )
        # Later blocks see the clean stream and the unmasked noisy tokens as context; it may be perturbed.
        context = config.visible_action_context
        context_tokens = action_tokens
        if context.enabled:
            context_tokens, context_selected = perturb_visible_context(
                action_tokens,
                action_valid,
                action_dim_mask,
                rng,
                config=context,
                block_size=config.block_size,
                gripper_indices=self.gripper_indices,
                stream_prefix="visible_action_context",
            )
        noisy_tokens, supervised, loss_weights, visible_corrupt = corrupt_action_tokens(
            context_tokens,
            timesteps,
            action_valid,
            rng,
            masked_values=mask_values(
                config.mask_value, tuple(context_tokens.shape), device=device, dtype=dtype
            ),
            corruption=config.corruption,
            eps=NOISE_EPS,
            full_mask_probability=config.full_mask_probability,
        )
        prefix = self._prefix(
            pixel_values, list(batch[TASK]), state, history, history_valid, action_dim_mask, rng
        )
        attention_mask = build_training_attention_mask(
            prefix.attention_mask.to(device=device, dtype=torch.bool),
            action_valid,
            config.block_size,
            dtype=prefix.inputs_embeds.dtype,
            history_length=prefix.history_length,
        )
        hidden = self.backbone.training_forward(
            prefix,
            noisy_action_embeds=self.action_in_proj(noisy_tokens),
            clean_action_embeds=self.action_in_proj(context_tokens),
            attention_mask=attention_mask,
        ).to(dtype)

        repair_dim_mask = None
        if context.enabled:
            repair_dim_mask = (
                torch.ones(batch_size, ACTION_DIM, device=device, dtype=torch.bool)
                if action_dim_mask is None
                else action_dim_mask.clone()
            )
            repair_dim_mask[:, self.gripper_indices] = False
        per_token_loss, repair_per_token_loss = action_reconstruction_loss(
            self.action_head(hidden),
            action_tokens,
            gripper_target,
            gripper_indices=self.gripper_indices,
            gripper_target_mode=config.gripper_target,
            dim_weights=self.action_dim_loss_weights,
            valid_dim_mask=action_dim_mask,
            regression_dim_mask=repair_dim_mask,
        )
        valid_count = action_valid.sum(dim=1)
        if config.corruption.mode == "bard_mixed":
            # Masked and visibly corrupted tokens are reduced separately, then added.
            reconstruction_loss = reduce_masked_loss(
                per_token_loss, supervised & ~visible_corrupt, loss_weights, valid_count
            ) + reduce_masked_loss(per_token_loss, supervised & visible_corrupt, loss_weights, valid_count)
        else:
            reconstruction_loss = reduce_masked_loss(per_token_loss, supervised, loss_weights, valid_count)
        loss = reconstruction_loss
        num_valid = action_valid.sum().clamp_min(1)
        metrics = {
            "loss_action_reconstruction": reconstruction_loss.detach(),
            "masked_ratio": ((supervised & ~visible_corrupt).sum().float() / num_valid).detach(),
        }
        if config.corruption.mode == "bard_mixed":
            metrics["visible_corrupt_ratio"] = (
                (supervised & visible_corrupt).sum().float() / num_valid
            ).detach()

        if context.enabled:
            # Repair: regress the clean continuous dimensions of the unmasked tokens of perturbed samples.
            repair_tokens = (
                context_selected[:, None] & action_valid & ~supervised & repair_dim_mask.any(dim=-1)[:, None]
            )
            repair_loss = reduce_masked_loss(
                repair_per_token_loss, repair_tokens, torch.ones_like(loss_weights), valid_count
            )
            weighted_repair_loss = context.repair_loss_weight * repair_loss
            loss = loss + weighted_repair_loss
            metrics["loss_visible_repair"] = repair_loss.detach()
            metrics["loss_visible_repair_weighted"] = weighted_repair_loss.detach()

        if self.token_risk_head is not None:
            # The risk head regresses the (detached) log reconstruction loss of each supervised token.
            if bool(supervised.any()):
                predicted_log_risk = self.token_risk_head(hidden.detach())
                with torch.no_grad():
                    target_log_risk = torch.log(per_token_loss.detach().clamp_min(0.0) + TOKEN_RISK_EPS)
                risk_loss = nn.functional.smooth_l1_loss(
                    predicted_log_risk[supervised], target_log_risk[supervised]
                )
            else:
                risk_loss = hidden.new_tensor(0.0)
            weighted_risk_loss = TOKEN_RISK_LOSS_WEIGHT * risk_loss
            loss = loss + weighted_risk_loss
            metrics["loss_token_risk"] = risk_loss.detach()
            metrics["loss_token_risk_weighted"] = weighted_risk_loss.detach()
        metrics["loss"] = loss.detach()
        return loss, metrics

    def _score_block(
        self,
        tokens: Tensor,
        previous: Tensor | None,
        *,
        positions: Tensor,
        attention_mask: Tensor,
        cache,
        action_dim_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.backbone.read_block(self.action_in_proj(tokens), positions, attention_mask, cache)
        prediction = _apply_dim_mask(
            action_logits_to_values(self.action_head(hidden), self.gripper_indices), action_dim_mask
        )
        confidence = token_confidence(
            self.config.inference.confidence, hidden, prediction, previous, tokens, self.token_risk_head
        )
        return hidden, prediction, confidence

    def _decodes_prefix_exactly(self) -> bool:
        """Whether decoding only the leading blocks yields the prefix of the full chunk; random mask values and
        random confidences draw from the global RNG in every block."""
        return self.config.mask_value != "random" and self.config.inference.confidence != "random"

    def _draft_generator(self, device: torch.device) -> torch.Generator | None:
        recipe = self.config.inference
        if recipe.draft_ranking != "random":
            return None
        if recipe.draft_seed is None:
            seed = int(torch.empty((), device=device, dtype=torch.int64).random_().item())
        else:
            calls = self._draft_seed_calls.get(recipe.draft_seed, 0)
            self._draft_seed_calls[recipe.draft_seed] = calls + 1
            seed = (recipe.draft_seed + calls) % (1 << 64)
        return torch.Generator(device=device).manual_seed(seed)

    @torch.no_grad()
    def predict_action_chunk(
        self,
        batch: dict,
        num_actions: int | None = None,
        *,
        prefix_actions: Tensor | None = None,
        history: Tensor | None = None,
        history_valid: Tensor | None = None,
    ) -> Tensor:
        """Decode an action chunk, `(B, num_actions, ACTION_DIM)` float32 in the normalized action space.

        Only the action blocks covering the first `num_actions` actions are decoded when that yields the
        corresponding prefix of the full chunk. `prefix_actions` `(B or 1, P, ACTION_DIM)` fixes the first actions
        of the chunk (real-time chunking). `history` `(B, horizon, ACTION_DIM)` holds the executed actions, oldest
        first, recorded where `history_valid`.
        """
        config = self.config
        chunk_size, block_size = config.chunk_size, config.block_size
        num_actions = chunk_size if num_actions is None else num_actions
        if not 0 < num_actions <= chunk_size:
            raise ValueError(f"num_actions must be in [1, {chunk_size}], got {num_actions}")
        if prefix_actions is not None and config.action_history.enabled:
            raise ValueError("prefix_actions (real-time chunking) cannot be combined with action history")
        recipe = config.inference
        keep_translation = recipe.remask_keep_draft_translation
        if keep_translation and (
            num_actions != 10
            or block_size != 10
            or prefix_actions is not None
            or not self._decodes_prefix_exactly()
        ):
            raise ValueError(
                "remask_keep_draft_translation decodes exactly 10 actions with 10-action blocks, without prefix "
                "actions and without random mask values or confidences"
            )
        device = self.device
        dtype = self.dtype
        generator = self._draft_generator(device)

        pixel_values = prepare_images(batch, config.camera_keys).to(device)
        batch_size = pixel_values.shape[0]
        state_dim_mask = _valid_dim_mask(batch.get(STATE_VALID_DIM_MASK), batch_size, device)
        action_dim_mask = _valid_action_dim_mask(batch.get(ACTION_VALID_DIM_MASK), batch_size, device)
        if keep_translation and action_dim_mask is None:
            raise ValueError("remask_keep_draft_translation needs the batch's action_valid_dim_mask")
        state = _apply_dim_mask(batch[OBS_STATE].to(device=device, dtype=dtype), state_dim_mask)

        prefix = self._prefix(pixel_values, list(batch[TASK]), state, history, history_valid, action_dim_mask)
        cache = self.backbone.prefill(prefix)
        prefix_valid = prefix.attention_mask.to(device=device, dtype=torch.bool)

        # Blocks are decoded in place. Draft rounds embed the strided block slice of this buffer, for which
        # `F.linear` adds the bias outside the GEMM; the released checkpoints were evaluated with that rounding.
        actions = mask_values(
            config.mask_value, (batch_size, chunk_size, ACTION_DIM), device=device, dtype=dtype
        )
        fixed = torch.zeros(batch_size, chunk_size, device=device, dtype=torch.bool)
        if prefix_actions is not None:
            if prefix_actions.ndim == 2:
                prefix_actions = prefix_actions.unsqueeze(0)
            if prefix_actions.ndim != 3 or prefix_actions.shape[0] not in (1, batch_size):
                raise ValueError(f"prefix_actions must be (B or 1, P, D), got {tuple(prefix_actions.shape)}")
            prefix_actions = prefix_actions.expand(batch_size, -1, -1)
            length = min(int(prefix_actions.shape[1]), chunk_size)
            if length > 0:
                actions[:, :length] = prefix_actions[:, :length].to(device=device, dtype=dtype)
                fixed[:, :length] = True
            if action_dim_mask is not None:
                actions = torch.where(fixed.unsqueeze(-1), _apply_dim_mask(actions, action_dim_mask), actions)
        confidence = torch.where(
            fixed, torch.tensor(torch.inf, device=device), torch.tensor(-torch.inf, device=device)
        )

        blocks = action_block_ranges(chunk_size, block_size)
        if self._decodes_prefix_exactly():
            blocks = blocks[: math.ceil(num_actions / block_size)]
        rank_fn = None
        if recipe.draft_ranking not in ("confidence", "spread_risk"):
            rank_fn = partial(draft_ranking, recipe.draft_ranking, generator=generator)
        translation_mask = None
        if keep_translation:
            dims = list(TRANSLATION_DIMS)
            translation_mask = torch.zeros_like(action_dim_mask)
            translation_mask[:, dims] = action_dim_mask[:, dims]
        for block_index, (start, end) in enumerate(blocks):
            positions = prefix.action_position_ids[..., start:end]
            attention_mask = build_block_attention_mask(
                prefix_valid, committed_tokens=start, block_tokens=end - start, dtype=dtype
            )
            score_fn = partial(
                self._score_block,
                positions=positions,
                attention_mask=attention_mask,
                cache=cache,
                action_dim_mask=action_dim_mask,
            )
            block = actions[:, start:end]
            pending = ~fixed[:, start:end]
            block_confidence = draft_block(
                block,
                confidence[:, start:end],
                pending,
                per_sample_commit_budgets(pending, recipe.draft_steps),
                score_fn,
                recipe,
                rank_fn,
            )
            draft = block.clone() if translation_mask is not None else None
            tokens, _ = refine_block(
                block,
                block_confidence,
                pending,
                score_fn,
                recipe,
                partial(mask_values, config.mask_value, tuple(block.shape), device=device, dtype=dtype),
            )
            if not bool(torch.isfinite(tokens).all()):
                raise FloatingPointError("Non-finite decoded action block")
            if draft is not None:
                tokens = torch.where(translation_mask.unsqueeze(1), draft, tokens)
            block.copy_(tokens)
            if block_index + 1 < len(blocks):
                cache = self.backbone.commit_block(
                    self.action_in_proj(tokens), positions, attention_mask, cache
                )
        return actions[:, :num_actions].float()
