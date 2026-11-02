"""Concertina pre/post-processor pipelines.

The dataset delivers canonical, unnormalized samples tagged with `dataset_index` (see `concertina.data.mixture`); the
pipelines apply the batched, invertible transforms that are saved with a checkpoint:

preprocessor:  source binding (evaluation only) -> canonical check -> valid-dimension masks -> gripper calibration
               -> per-embodiment normalization -> rename -> device -> gripper target -> global normalization
               -> gripper probability tokens
postprocessor: source binding -> masks -> probability snapshot -> global unnormalization -> probability restore
               -> cpu -> per-embodiment unnormalization -> inverse calibration -> source action conversion

The source binding steps make a saved checkpoint usable on raw environment observations: inputs without a
`dataset_index` are converted with the adapter of ``config.data.inference_dataset_index`` and bound to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from lerobot.configs.types import FeatureType, NormalizationMode, PipelineFeatureType, PolicyFeature
from lerobot.processor import (
    DeviceProcessorStep,
    EnvTransition,
    NormalizerProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    ProcessorStep,
    ProcessorStepRegistry,
    RenameObservationsProcessorStep,
    TransitionKey,
    UnnormalizerProcessorStep,
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_STATE,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

from concertina.data.embodiments import CANONICAL_DIM, make_source_adapter, resize_image
from concertina.data.stats import embodiment_stats
from concertina.model.constants import (
    ACTION_IS_PAD,
    ACTION_VALID_DIM_MASK,
    DATASET_INDEX,
    GRIPPER_TARGET,
    IMAGE_SIZE,
    STATE_VALID_DIM_MASK,
)
from concertina.policy.config_data import resolve_gripper_profiles

logger = logging.getLogger(__name__)

OBSERVATION = TransitionKey.OBSERVATION
COMPLEMENTARY = TransitionKey.COMPLEMENTARY_DATA
RAW_GRIPPER_KEY = "action_gripper_raw_before_normalization"
# Binary gripper targets: the source gripper value thresholded here (no calibration).
BINARY_TARGET_THRESHOLD = 0.5
CLOSURE_PROBABILITY_KEY = "gripper_closure_before_unnormalization"
EVALUABLE_SOURCES = ("libero", "calvin")


def concertina_batch_to_transition(batch: dict[str, Any]) -> EnvTransition:
    """`batch_to_transition` that also carries `dataset_index` in the complementary data."""
    transition = batch_to_transition(batch)
    if DATASET_INDEX in batch:
        complementary = dict(transition[COMPLEMENTARY] or {})
        complementary[DATASET_INDEX] = batch[DATASET_INDEX]
        transition[COMPLEMENTARY] = complementary
    return transition


def _dataset_indices(complementary: dict[str, Any]) -> torch.Tensor:
    value = complementary.get(DATASET_INDEX)
    if value is None:
        raise ValueError(f"Concertina processing needs `{DATASET_INDEX}` in the complementary data")
    indices = torch.as_tensor(value).reshape(-1)
    if indices.is_floating_point() or indices.dtype == torch.bool:
        raise ValueError(f"`{DATASET_INDEX}` must hold integers, got {indices.dtype}")
    return indices


def _rows_by_value(values: list[Any]) -> dict[Any, list[int]]:
    groups: dict[Any, list[int]] = {}
    for row, value in enumerate(values):
        groups.setdefault(value, []).append(row)
    return groups


def _apply_by_rows(tensor: torch.Tensor, groups: dict[Any, list[int]], fn) -> torch.Tensor:
    """Apply ``fn(key, rows_tensor)`` to the rows of each group (the whole tensor when one group covers it)."""
    if len(groups) == 1:
        return fn(next(iter(groups)), tensor)
    output = tensor.clone()
    for key, rows in groups.items():
        index = torch.as_tensor(rows, dtype=torch.long, device=tensor.device)
        output.index_copy_(0, index, fn(key, tensor.index_select(0, index)))
    return output


class _ConcertinaStep(ProcessorStep):
    """Steps that leave the feature description unchanged."""

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features


@ProcessorStepRegistry.register("concertina_source_binding")
@dataclass
class SourceBindingStep(_ConcertinaStep):
    """Bind inputs without a `dataset_index` (environment observations, policy actions) to one dataset spec.

    Observations are converted from the spec's source schema to the canonical layout (state, camera mapping,
    optional resize and missing-camera placeholders). Inputs that already carry a `dataset_index` come from the
    Concertina dataset, are canonical, and pass through unchanged.
    """

    dataset_index: int = 0
    source: str = ""
    adapter: str | None = None
    camera_keys: list[str] = field(default_factory=list)
    camera_map: dict[str, list[str]] = field(default_factory=dict)
    resize_hw: list[int] | None = None
    placeholder_hw: list[int] = field(default_factory=lambda: list(IMAGE_SIZE))

    def __post_init__(self) -> None:
        self._adapter = make_source_adapter(self.source, self.adapter, {})

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = dict(transition.get(COMPLEMENTARY) or {})
        if DATASET_INDEX in complementary:
            return transition
        observation = transition.get(OBSERVATION)
        action = transition.get(TransitionKey.ACTION)
        if observation:
            observation, batch_size, device = self._bind_observation(dict(observation))
        elif isinstance(action, torch.Tensor):
            batch_size = action.shape[0] if action.ndim > 1 else 1
            device = action.device
        else:
            raise ValueError("Source binding needs an observation or an action")
        complementary[DATASET_INDEX] = torch.full(
            (batch_size,), self.dataset_index, dtype=torch.long, device=device
        )
        return {**transition, OBSERVATION: observation, COMPLEMENTARY: complementary}

    def _bind_observation(self, observation: dict[str, Any]) -> tuple[dict[str, Any], int, torch.device]:
        state_keys = self._adapter.state_keys
        source_state = {key: observation[key] for key in state_keys if key in observation}
        if not source_state and len(state_keys) == 1 and OBS_STATE in observation:
            source_state = {state_keys[0]: observation[OBS_STATE]}
        if source_state:
            if len(source_state) != len(state_keys):
                raise ValueError(
                    f"{self.source} observations need {list(state_keys)}, got {sorted(source_state)}"
                )
            for key in state_keys:
                observation.pop(key, None)
            observation[OBS_STATE] = self._adapter.state(source_state)
        images = {}
        for camera in self.camera_keys:
            # Environments may already name their cameras like the model.
            candidates = [*self.camera_map.get(camera, []), camera]
            source = next((key for key in candidates if key in observation), None)
            if source is not None:
                image = torch.as_tensor(observation[source])
                # Same convention as the training loop: uint8 frames are scaled to [0, 1].
                image = image.float() / 255.0 if image.dtype == torch.uint8 else image.to(dtype=torch.float32)
                images[camera] = (
                    image if self.resize_hw is None else resize_image(image, tuple(self.resize_hw))
                )
        reference = next(iter(images.values()), observation.get(OBS_STATE))
        if reference is None:
            raise ValueError("Source binding needs a state or a camera observation")
        batched = reference.ndim == (4 if images else 2)
        batch_size = reference.shape[0] if batched else 1
        for camera in self.camera_keys:
            for key in self.camera_map.get(camera, []):
                if key not in self.camera_keys:
                    observation.pop(key, None)
            missing = camera not in images
            if missing:
                hw = self.resize_hw or self.placeholder_hw
                shape = (batch_size, 3, *hw) if batched else (3, *hw)
                images[camera] = torch.zeros(shape, dtype=torch.float32, device=reference.device)
            observation[camera] = images[camera]
            flag = torch.full((batch_size,), missing, dtype=torch.bool, device=reference.device)
            observation[f"{camera}_is_missing"] = flag if batched else flag[0]
        return observation, batch_size, reference.device

    def get_config(self) -> dict[str, Any]:
        return {
            "dataset_index": self.dataset_index,
            "source": self.source,
            "adapter": self.adapter,
            "camera_keys": list(self.camera_keys),
            "camera_map": dict(self.camera_map),
            "resize_hw": None if self.resize_hw is None else list(self.resize_hw),
            "placeholder_hw": list(self.placeholder_hw),
        }


@ProcessorStepRegistry.register("concertina_canonical_check")
@dataclass
class CanonicalCheckStep(_ConcertinaStep):
    """Reject actions/states that are not in the canonical layout."""

    dim: int = CANONICAL_DIM

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        state = (transition.get(OBSERVATION) or {}).get(OBS_STATE)
        for name, value in ((ACTION, action), (OBS_STATE, state)):
            if value is not None and value.shape[-1] != self.dim:
                raise ValueError(f"{name} must be {self.dim}D canonical, got shape {tuple(value.shape)}")
        return transition

    def get_config(self) -> dict[str, Any]:
        return {"dim": self.dim}


@ProcessorStepRegistry.register("concertina_valid_dim_masks")
@dataclass
class ValidDimMaskStep(_ConcertinaStep):
    """Per-sample masks of the valid canonical action/state dimensions, routed by `dataset_index`."""

    valid_action_dims: list[int] = field(default_factory=list)
    action_dim: int = CANONICAL_DIM

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        complementary = dict(transition.get(COMPLEMENTARY) or {})
        indices = _dataset_indices(complementary)
        if int(indices.min()) < 0 or int(indices.max()) >= len(self.valid_action_dims):
            raise ValueError(
                f"`{DATASET_INDEX}` outside the {len(self.valid_action_dims)} configured datasets"
            )
        valid = torch.as_tensor(self.valid_action_dims, dtype=torch.long, device=indices.device)[indices]
        positions = torch.arange(self.action_dim, device=indices.device)
        complementary[ACTION_VALID_DIM_MASK] = positions.unsqueeze(0) < valid.unsqueeze(1)
        state = (transition.get(OBSERVATION) or {}).get(OBS_STATE)
        if state is not None:
            state_positions = torch.arange(state.shape[-1], device=indices.device)
            complementary[STATE_VALID_DIM_MASK] = state_positions.unsqueeze(0) < valid.unsqueeze(1)
        return {**transition, COMPLEMENTARY: complementary}

    def get_config(self) -> dict[str, Any]:
        return {"valid_action_dims": list(self.valid_action_dims), "action_dim": self.action_dim}


def _action_validity(action: torch.Tensor, complementary: dict[str, Any], *, inverse: bool) -> torch.Tensor:
    valid = torch.ones_like(action, dtype=torch.bool)
    dim_mask = complementary.get(ACTION_VALID_DIM_MASK)
    if dim_mask is not None:
        mask = torch.as_tensor(dim_mask, dtype=torch.bool, device=action.device)
        while mask.ndim < action.ndim:
            mask = mask.unsqueeze(-2)
        valid &= mask.expand_as(action)
    # Padding marks training labels; generated actions are always valid.
    padding = None if inverse else complementary.get(ACTION_IS_PAD)
    if padding is not None:
        mask = ~torch.as_tensor(padding, dtype=torch.bool, device=action.device)
        if mask.ndim < action.ndim:
            mask = mask.unsqueeze(-1)
        valid &= mask.expand_as(action)
    return valid


@ProcessorStepRegistry.register("concertina_gripper_calibration")
@dataclass
class GripperCalibrationStep(_ConcertinaStep):
    """Map gripper channels between declared source endpoints and closure in [0, 1] (0 open, 1 closed).

    Forward maps valid source values to closure; ``inverse`` maps closure probabilities (taken from the snapshot
    under ``probability_key``) back to source values. Invalid entries (padding, invalid dimensions) become zero.
    """

    profiles: list[dict[str, Any]] = field(default_factory=list)
    dataset_profiles: list[str] = field(default_factory=list)
    gripper_indices: list[int] = field(default_factory=list)
    inverse: bool = False
    range_tolerance: float = 1e-6
    probability_key: str | None = None

    def __post_init__(self) -> None:
        self._profiles = {profile["id"]: profile for profile in self.profiles}
        if any(profile_id not in self._profiles for profile_id in self.dataset_profiles):
            raise ValueError("gripper calibration routes a dataset to a missing profile")
        if not self.gripper_indices or len(set(self.gripper_indices)) != len(self.gripper_indices):
            raise ValueError("gripper calibration requires unique gripper_indices")

    def _apply(self, profile_id: str, action: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        result = action.clone()
        profile = self._profiles[profile_id]
        calibrated = profile["indices"]
        for index in self.gripper_indices:
            if index not in calibrated:
                if bool(valid[..., index].any()):
                    raise ValueError(f"gripper profile {profile_id!r} does not calibrate valid index {index}")
                result[..., index] = 0
        mask = valid[..., calibrated]
        values = action[..., calibrated].to(torch.float64)
        opened = torch.tensor(profile["open_values"], dtype=torch.float64, device=action.device)
        closed = torch.tensor(profile["closed_values"], dtype=torch.float64, device=action.device)
        invalid = ~torch.isfinite(values)
        if not self.inverse and profile["out_of_range"] == "clip":
            values = torch.maximum(
                torch.minimum(values, torch.maximum(opened, closed)), torch.minimum(opened, closed)
            )
        closure = values if self.inverse else (values - opened) / (closed - opened)
        invalid |= ~torch.isfinite(closure)
        if self.inverse or profile["out_of_range"] == "error":
            invalid |= (closure < -self.range_tolerance) | (closure > 1 + self.range_tolerance)
        if bool((invalid & mask).any()):
            raise ValueError(
                f"gripper calibration {profile_id!r}: values are non-finite or outside the declared endpoints"
            )
        closure = closure.clamp(0, 1)
        mapped = opened + closure * (closed - opened) if self.inverse else closure
        result[..., calibrated] = torch.where(mask, mapped, 0).to(dtype=action.dtype)
        return result

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        complementary = transition.get(COMPLEMENTARY) or {}
        if self.inverse and self.probability_key is not None:
            probability = complementary.get(self.probability_key)
            if probability is None:
                raise ValueError("inverse gripper calibration needs the probability snapshot")
            action = action.clone()
            action[..., self.gripper_indices] = torch.as_tensor(
                probability, dtype=action.dtype, device=action.device
            )
        valid = _action_validity(action, complementary, inverse=self.inverse)
        rows = _dataset_indices(complementary).tolist()
        if action.ndim == 1:
            if len(rows) != 1:
                raise ValueError("an unbatched action needs exactly one dataset_index")
            return {
                **transition,
                TransitionKey.ACTION: self._apply(self.dataset_profiles[rows[0]], action, valid),
            }
        if len(rows) != action.shape[0]:
            raise ValueError(f"{len(rows)} dataset indices for a batch of {action.shape[0]} actions")
        groups = _rows_by_value([self.dataset_profiles[row] for row in rows])
        if len(groups) == 1:
            calibrated = self._apply(next(iter(groups)), action, valid)
        else:
            calibrated = action.clone()
            for profile_id, members in groups.items():
                index = torch.as_tensor(members, dtype=torch.long, device=action.device)
                mapped = self._apply(profile_id, action.index_select(0, index), valid.index_select(0, index))
                calibrated.index_copy_(0, index, mapped)
        return {**transition, TransitionKey.ACTION: calibrated}

    def get_config(self) -> dict[str, Any]:
        return {
            "profiles": self.profiles,
            "dataset_profiles": list(self.dataset_profiles),
            "gripper_indices": list(self.gripper_indices),
            "inverse": self.inverse,
            "range_tolerance": self.range_tolerance,
            "probability_key": self.probability_key,
        }


@ProcessorStepRegistry.register("concertina_embodiment_normalize")
@dataclass
class EmbodimentNormalizeStep(_ConcertinaStep):
    """Mean/std (un)normalization of action and state with the statistics of each sample's embodiment.

    ``preserve_gripper_indices`` keep their pre-step values: forward stores them under ``preserved_gripper_key``
    (the gripper targets are extracted from them), inverse writes them back (they are probabilities already).
    """

    dataset_embodiments: list[str] = field(default_factory=list)
    inverse: bool = False
    preserve_gripper_indices: list[int] = field(default_factory=list)
    preserved_gripper_key: str = RAW_GRIPPER_KEY
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self._stats: dict[str, dict[str, dict[str, torch.Tensor]]] = {}

    def set_stats(self, stats: dict[str, dict[str, dict[str, Any]]]) -> None:
        missing = sorted(set(self.dataset_embodiments) - set(stats))
        if missing:
            raise ValueError(f"embodiment normalization lacks statistics for {missing}")
        self._stats = {
            embodiment: {
                feature: {
                    name: torch.as_tensor(stats[embodiment][feature][name], dtype=torch.float32)
                    for name in ("mean", "std")
                }
                for feature in (ACTION, OBS_STATE)
            }
            for embodiment in dict.fromkeys(self.dataset_embodiments)
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            f"{embodiment}/{feature}/{name}": tensor.cpu()
            for embodiment, features in self._stats.items()
            for feature, values in features.items()
            for name, tensor in values.items()
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        stats: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
        for key, tensor in state.items():
            embodiment, feature, name = key.split("/")
            stats.setdefault(embodiment, {}).setdefault(feature, {})[name] = tensor
        self.set_stats(stats)

    def _transform(self, tensor: torch.Tensor, feature: str, embodiments: list[str]) -> torch.Tensor:
        def apply(embodiment: str, values: torch.Tensor) -> torch.Tensor:
            stats = self._stats[embodiment][feature]
            mean = stats["mean"].to(device=values.device, dtype=values.dtype)
            std = stats["std"].to(device=values.device, dtype=values.dtype)
            return values * std + mean if self.inverse else (values - mean) / (std + self.eps)

        if tensor.ndim >= 2 and len(embodiments) == tensor.shape[0]:
            return _apply_by_rows(tensor, _rows_by_value(embodiments), apply)
        if len(set(embodiments)) != 1:
            raise ValueError("an unbatched tensor needs a single embodiment")
        return apply(embodiments[0], tensor)

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self._stats:
            raise ValueError("embodiment normalization has no statistics")
        complementary = dict(transition.get(COMPLEMENTARY) or {})
        embodiments = [self.dataset_embodiments[i] for i in _dataset_indices(complementary).tolist()]
        action = transition.get(TransitionKey.ACTION)
        observation = transition.get(OBSERVATION)
        if action is not None:
            preserved = (
                action[..., self.preserve_gripper_indices].clone() if self.preserve_gripper_indices else None
            )
            if preserved is not None and not self.inverse:
                complementary[self.preserved_gripper_key] = preserved
            action = self._transform(action, ACTION, embodiments)
            if preserved is not None and self.inverse:
                action = action.clone()
                action[..., self.preserve_gripper_indices] = preserved
        if observation is not None and observation.get(OBS_STATE) is not None:
            observation = {
                **observation,
                OBS_STATE: self._transform(observation[OBS_STATE], OBS_STATE, embodiments),
            }
        return {
            **transition,
            OBSERVATION: observation,
            TransitionKey.ACTION: action,
            COMPLEMENTARY: complementary,
        }

    def get_config(self) -> dict[str, Any]:
        return {
            "dataset_embodiments": list(self.dataset_embodiments),
            "inverse": self.inverse,
            "preserve_gripper_indices": list(self.preserve_gripper_indices),
            "preserved_gripper_key": self.preserved_gripper_key,
            "eps": self.eps,
        }


@ProcessorStepRegistry.register("concertina_gripper_target")
@dataclass
class GripperTargetStep(_ConcertinaStep):
    """Extract the gripper BCE targets from the unnormalized action.

    ``target_mode="probability"``: the calibrated closure in [0, 1]. ``"binary"``: whether the source gripper value
    exceeds ``threshold`` (no calibration).
    """

    gripper_indices: list[int] = field(default_factory=list)
    target_mode: str = "probability"
    threshold: float = BINARY_TARGET_THRESHOLD
    key: str = GRIPPER_TARGET
    raw_key: str = RAW_GRIPPER_KEY

    def __post_init__(self) -> None:
        if self.target_mode not in ("probability", "binary"):
            raise ValueError(f"target_mode must be 'probability' or 'binary', got {self.target_mode!r}")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        complementary = dict(transition.get(COMPLEMENTARY) or {})
        raw = complementary.get(self.raw_key)
        values = (
            action[..., self.gripper_indices]
            if raw is None
            else raw.to(device=action.device, dtype=action.dtype)
        )
        if not torch.isfinite(values).all():
            raise ValueError("gripper targets contain non-finite values")
        if self.target_mode == "binary":
            complementary[self.key] = (values > self.threshold).to(dtype=action.dtype)
            return {**transition, COMPLEMENTARY: complementary}
        if bool(((values < -1e-6) | (values > 1.0 + 1e-6)).any()):
            raise ValueError(
                f"probability gripper targets must be in [0, 1], got [{values.min().item():.6g}, "
                f"{values.max().item():.6g}]"
            )
        complementary[self.key] = values.clamp(0.0, 1.0)
        return {**transition, COMPLEMENTARY: complementary}

    def get_config(self) -> dict[str, Any]:
        return {
            "gripper_indices": list(self.gripper_indices),
            "target_mode": self.target_mode,
            "threshold": self.threshold,
            "key": self.key,
            "raw_key": self.raw_key,
        }


@ProcessorStepRegistry.register("concertina_gripper_probability_tokens")
@dataclass
class GripperProbabilityTokenStep(_ConcertinaStep):
    """Write the gripper targets into the normalized action, so gripper tokens live in probability space."""

    gripper_indices: list[int] = field(default_factory=list)
    key: str = GRIPPER_TARGET

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        complementary = transition.get(COMPLEMENTARY) or {}
        if self.key not in complementary:
            raise ValueError(
                f"gripper probability tokens need {self.key!r}; run the gripper target step first"
            )
        target = complementary[self.key].to(device=action.device, dtype=action.dtype)
        if target.shape != (*action.shape[:-1], len(self.gripper_indices)):
            raise ValueError(
                f"gripper target shape {tuple(target.shape)} does not match action {tuple(action.shape)}"
            )
        action = action.clone()
        action[..., self.gripper_indices] = target.clamp(0.0, 1.0)
        return {**transition, TransitionKey.ACTION: action}

    def get_config(self) -> dict[str, Any]:
        return {"gripper_indices": list(self.gripper_indices), "key": self.key}


@ProcessorStepRegistry.register("concertina_gripper_probability_snapshot")
@dataclass
class GripperProbabilitySnapshotStep(_ConcertinaStep):
    """Keep the predicted gripper probabilities before unnormalization touches them."""

    gripper_indices: list[int] = field(default_factory=list)
    key: str = CLOSURE_PROBABILITY_KEY

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        complementary = dict(transition.get(COMPLEMENTARY) or {})
        complementary[self.key] = action[..., self.gripper_indices].clone()
        return {**transition, COMPLEMENTARY: complementary}

    def get_config(self) -> dict[str, Any]:
        return {"gripper_indices": list(self.gripper_indices), "key": self.key}


@ProcessorStepRegistry.register("concertina_gripper_probability_restore")
@dataclass
class GripperProbabilityRestoreStep(_ConcertinaStep):
    """Undo the global mean/std unnormalization on gripper channels, which hold probabilities, and clip to [0, 1]."""

    gripper_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._stats: dict[str, torch.Tensor] = {}

    def set_action_stats(self, action_stats: dict[str, Any]) -> None:
        index = torch.as_tensor(self.gripper_indices, dtype=torch.long)
        self._stats = {
            name: torch.as_tensor(action_stats[name], dtype=torch.float32).reshape(-1).index_select(0, index)
            for name in ("mean", "std")
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.cpu() for name, tensor in self._stats.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._stats = {name: tensor.to(dtype=torch.float32) for name, tensor in state.items()}

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        restored = action.clone()
        values = restored[..., self.gripper_indices]
        mean = self._stats["mean"].to(device=values.device, dtype=values.dtype)
        std = self._stats["std"].to(device=values.device, dtype=values.dtype)
        # The unnormalizer applied ``value * std + mean`` without epsilon; a zero-std channel collapsed to its mean,
        # which is then also its only possible probability.
        zero_std = std == 0
        recovered = (values - mean) / torch.where(zero_std, torch.ones_like(std), std)
        restored[..., self.gripper_indices] = torch.where(zero_std, mean, recovered).clamp(0.0, 1.0)
        return {**transition, TransitionKey.ACTION: restored}

    def get_config(self) -> dict[str, Any]:
        return {"gripper_indices": list(self.gripper_indices)}


@ProcessorStepRegistry.register("concertina_source_action")
@dataclass
class SourceActionStep(_ConcertinaStep):
    """Convert canonical actions to the native action of the bound evaluation source."""

    source: str = ""
    binarize_gripper: bool = True
    gripper_threshold: float = 0.0

    def __post_init__(self) -> None:
        self._adapter = make_source_adapter(self.source, None, {})

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition
        converted = self._adapter.source_action(
            action, binarize=self.binarize_gripper, threshold=self.gripper_threshold
        )
        return {**transition, TransitionKey.ACTION: converted}

    def get_config(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "binarize_gripper": self.binarize_gripper,
            "gripper_threshold": self.gripper_threshold,
        }


def _source_action_step(source: str, gripper_objective: str) -> SourceActionStep | None:
    """Native action conversion; ``gripper_objective`` is "closure", "binary" or "regression"."""
    if source not in EVALUABLE_SOURCES:
        return None
    # CALVIN/LIBERO grippers are +-1 commands: source values (after inverse calibration, or regressed) are
    # thresholded at 0, binary-target probabilities at 0.5.
    threshold = BINARY_TARGET_THRESHOLD if gripper_objective == "binary" else 0.0
    return SourceActionStep(source=source, binarize_gripper=True, gripper_threshold=threshold)


def make_concertina_pre_post_processors(
    config: Any,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Build the pipelines from ``config.data`` and the statistics packed by the Concertina dataset."""
    data = config.data
    specs = data.datasets
    if not specs:
        raise ValueError("config.data.datasets is empty")
    # Gripper channels are trained with a BCE: on calibrated closure probabilities ("probability", the released
    # setting) or on binarized source values ("binary"); without gripper channels they are regressed.
    gripper_indices = list(config.model.gripper_indices)
    target_mode = config.model.gripper_target
    calibrated = bool(gripper_indices) and target_mode == "probability"
    gripper_objective = ("closure" if calibrated else "binary") if gripper_indices else "regression"
    norm_map = {
        FeatureType(key): NormalizationMode(mode) for key, mode in config.normalization_mapping.items()
    }
    for feature_type in (FeatureType.STATE, FeatureType.ACTION):
        mode = norm_map.get(feature_type, NormalizationMode.IDENTITY)
        if data.normalization != "none" and mode not in (
            NormalizationMode.MEAN_STD,
            NormalizationMode.IDENTITY,
        ):
            raise ValueError(
                f"Concertina normalizes {feature_type.value} with MEAN_STD or IDENTITY, got {mode}"
            )
    if data.normalization != "none" and dataset_stats is None:
        raise ValueError(f"normalization={data.normalization!r} needs dataset statistics")

    bound = specs[data.inference_dataset_index]
    binding = SourceBindingStep(
        dataset_index=data.inference_dataset_index,
        source=bound.source,
        adapter=bound.adapter,
        camera_keys=list(config.model.camera_keys),
        camera_map=dict(bound.camera_map),
        resize_hw=list(IMAGE_SIZE) if data.resize_images else None,
        placeholder_hw=list(IMAGE_SIZE),
    )
    masks = ValidDimMaskStep(valid_action_dims=[spec.valid_action_dim for spec in specs])
    calibration = None
    if calibrated:
        profiles, dataset_profiles = resolve_gripper_profiles(data, gripper_indices)
        calibration = {
            "profiles": profiles,
            "dataset_profiles": dataset_profiles,
            "gripper_indices": gripper_indices,
            "range_tolerance": data.gripper_range_tolerance,
        }

    pre_steps: list[ProcessorStep] = [binding, CanonicalCheckStep(), masks]
    post_steps: list[ProcessorStep] = [binding, masks]
    if calibration:
        pre_steps.append(GripperCalibrationStep(**calibration))
        post_steps.append(GripperProbabilitySnapshotStep(gripper_indices=gripper_indices))

    embodiment_steps = []
    if data.normalization == "per_embodiment":
        per_embodiment = embodiment_stats(dataset_stats, [spec.embodiment for spec in specs])
        for inverse in (False, True):
            step = EmbodimentNormalizeStep(
                dataset_embodiments=[spec.embodiment for spec in specs],
                inverse=inverse,
                preserve_gripper_indices=gripper_indices,
            )
            step.set_stats(per_embodiment)
            embodiment_steps.append(step)
        pre_steps.append(embodiment_steps[0])

    pre_steps += [RenameObservationsProcessorStep(rename_map={}), DeviceProcessorStep(device=config.device)]
    if gripper_indices:
        pre_steps.append(GripperTargetStep(gripper_indices=gripper_indices, target_mode=target_mode))
    if data.normalization == "global":
        stats = {key: dataset_stats[key] for key in (ACTION, OBS_STATE)}
        features = {**config.input_features, **config.output_features}
        pre_steps.append(NormalizerProcessorStep(stats=stats, features=features, norm_map=norm_map))
        post_steps.append(
            UnnormalizerProcessorStep(stats=stats, features=dict(config.output_features), norm_map=norm_map)
        )
        if gripper_indices and norm_map.get(FeatureType.ACTION) == NormalizationMode.MEAN_STD:
            restore = GripperProbabilityRestoreStep(gripper_indices=gripper_indices)
            restore.set_action_stats(stats[ACTION])
            post_steps.append(restore)
    if gripper_indices:
        pre_steps.append(GripperProbabilityTokenStep(gripper_indices=gripper_indices))

    post_steps.append(DeviceProcessorStep(device="cpu"))
    if embodiment_steps:
        post_steps.append(embodiment_steps[1])
    if calibration:
        post_steps.append(
            GripperCalibrationStep(**calibration, inverse=True, probability_key=CLOSURE_PROBABILITY_KEY)
        )
    source_action = _source_action_step(bound.source, gripper_objective)
    if source_action is not None:
        post_steps.append(source_action)

    preprocessor = PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
        steps=pre_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=concertina_batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocessor = PolicyProcessorPipeline[PolicyAction, PolicyAction](
        steps=post_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocessor, postprocessor


def _step_keys(config: dict[str, Any]) -> set[str]:
    return {step.get("registry_name") or step["class"].rsplit(".", 1)[-1] for step in config["steps"]}


def _rebuild_with_overrides(pipeline: PolicyProcessorPipeline, overrides: dict[str, Any] | None):
    """Re-instantiate a pipeline with constructor overrides for the steps it contains."""
    config = pipeline.get_config()
    overrides = {key: value for key, value in (overrides or {}).items() if key in _step_keys(config)}
    if not overrides:
        return pipeline
    return PolicyProcessorPipeline.from_config(
        config,
        state_dict=pipeline.state_dict(),
        overrides=overrides,
        to_transition=pipeline.to_transition,
        to_output=pipeline.to_output,
    )


def make_concertina_pre_post_processors_from_pretrained(
    config: Any,
    pretrained_path: str | Path,
    *,
    revision: str | None,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None,
    dataset_meta: Any | None,
    preprocessor_overrides: dict[str, Any] | None,
    postprocessor_overrides: dict[str, Any] | None,
    preprocessor_config_filename: str,
    postprocessor_config_filename: str,
) -> tuple[Any, Any]:
    """Pipelines for a run that starts from a checkpoint.

    With ``dataset_stats`` (fine-tuning) the checkpoint's pipelines are ignored and rebuilt for the current
    ``config.data`` and data; only device/rename overrides apply, since the statistics come from the dataset.
    Without them (resume, evaluation) the saved pipelines are loaded; overrides for steps they do not contain are
    dropped.
    """
    if dataset_stats is not None:
        preprocessor, postprocessor = make_concertina_pre_post_processors(config, dataset_stats)
        runtime = {"device_processor", "rename_observations_processor"}
        preprocessor = _rebuild_with_overrides(
            preprocessor, {k: v for k, v in (preprocessor_overrides or {}).items() if k in runtime}
        )
        postprocessor = _rebuild_with_overrides(
            postprocessor, {k: v for k, v in (postprocessor_overrides or {}).items() if k in runtime}
        )
        return preprocessor, postprocessor

    loaded = []
    for filename, overrides, to_transition, to_output in (
        (
            preprocessor_config_filename,
            preprocessor_overrides,
            concertina_batch_to_transition,
            transition_to_batch,
        ),
        (
            postprocessor_config_filename,
            postprocessor_overrides,
            policy_action_to_transition,
            transition_to_policy_action,
        ),
    ):
        pipeline = PolicyProcessorPipeline.from_pretrained(
            pretrained_path,
            config_filename=filename,
            revision=revision,
            to_transition=to_transition,
            to_output=to_output,
        )
        loaded.append(_rebuild_with_overrides(pipeline, overrides))
    return loaded[0], loaded[1]
