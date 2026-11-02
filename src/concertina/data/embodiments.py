"""Per-source conversion of LeRobot samples to the canonical Concertina layout.

Canonical action/state: two arms x (xyz position, rotate6d, gripper) = 20 dims; single-arm sources fill the first
arm and zero the second. Each adapter declares the source columns it reads and converts any leading shape
``[..., D_source] -> [..., 20]``, so the same code serves dataset samples (``[T, D]``) and evaluation batches
(``[B, D]``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.constants import ACTION, OBS_STATE
from scipy.interpolate import interp1d

from concertina.data.rotation import (
    axis_angle_to_rotate6d,
    euler_to_rotate6d,
    normalize_quaternion,
    quat_to_rotate6d,
    rotate6d_to_axis_angle,
    rotate6d_to_euler,
)

CANONICAL_DIM = 20
ARM_DIM = 10

Item = Mapping[str, torch.Tensor]


def _float(value) -> torch.Tensor:
    return torch.as_tensor(value, dtype=torch.float32)


def _gripper_column(gripper, reference: torch.Tensor) -> torch.Tensor:
    """Shape a gripper value like ``reference[..., :1]``."""
    g = _float(gripper)
    if g.ndim == 0:
        g = g.unsqueeze(0)
    while g.ndim < reference.ndim:
        g = g.unsqueeze(-1)
    return g[..., :1] if g.shape[-1] != 1 else g


def _pad_single_arm(arm: torch.Tensor) -> torch.Tensor:
    if arm.shape[-1] != ARM_DIM:
        raise ValueError(f"single-arm canonical vector must have {ARM_DIM} dims, got {arm.shape[-1]}")
    return torch.cat([arm, torch.zeros_like(arm)], dim=-1)


def _check_dim(value: torch.Tensor, expected: int, what: str) -> None:
    if value.shape[-1] != expected:
        raise ValueError(f"{what} must have {expected} dims, got shape {tuple(value.shape)}")


class SourceAdapter:
    """Conversion between one source schema and the canonical layout."""

    source: str
    action_keys: tuple[str, ...]
    state_keys: tuple[str, ...]

    def action(self, item: Item) -> torch.Tensor:
        raise NotImplementedError

    def state(self, item: Item) -> torch.Tensor:
        raise NotImplementedError

    def source_action(self, action: torch.Tensor, *, binarize: bool, threshold: float) -> torch.Tensor:
        """Canonical action ``[..., 20]`` -> the source's native action (evaluation only)."""
        raise NotImplementedError(f"{type(self).__name__} has no evaluation action conversion")


class DroidAdapter(SourceAdapter):
    """DROID: cartesian position (xyz + extrinsic xyz Euler) and gripper position, single arm."""

    source = "droid"
    action_keys = ("action.cartesian_position", "action.gripper_position")
    state_keys = ("observation.state.cartesian_position", "observation.state.gripper_position")

    @staticmethod
    def _convert(cartesian, gripper) -> torch.Tensor:
        cartesian = _float(cartesian)
        pos = cartesian[..., :3]
        arm = torch.cat([pos, euler_to_rotate6d(cartesian[..., 3:6]), _gripper_column(gripper, pos)], dim=-1)
        return _pad_single_arm(arm)

    def action(self, item: Item) -> torch.Tensor:
        return self._convert(item[self.action_keys[0]], item[self.action_keys[1]])

    def state(self, item: Item) -> torch.Tensor:
        return self._convert(item[self.state_keys[0]], item[self.state_keys[1]])


class AgiBotAdapter(SourceAdapter):
    """AgiBot World: per-arm end-effector position, quaternion orientation and effector position."""

    source = "agibot"
    action_keys = ("actions.end.position", "actions.end.orientation", "actions.effector.position")
    state_keys = (
        "observation.states.end.position",
        "observation.states.end.orientation",
        "observation.states.effector.position",
    )

    @staticmethod
    def _convert(position, orientation, effector) -> torch.Tensor:
        pos = _float(position)
        ori = normalize_quaternion(_float(orientation))
        grip = _float(effector)
        left_pos, right_pos = pos[..., 0, :], pos[..., 1, :]
        return torch.cat(
            [
                left_pos,
                quat_to_rotate6d(ori[..., 0, :]),
                _gripper_column(grip[..., 0], left_pos),
                right_pos,
                quat_to_rotate6d(ori[..., 1, :]),
                _gripper_column(grip[..., 1], right_pos),
            ],
            dim=-1,
        )

    def action(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.action_keys))

    def state(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.state_keys))


class RoboMindAgilexAdapter(SourceAdapter):
    """RoboMIND AgileX: per-arm 7D pose (xyz + quaternion) and gripper."""

    source = "robomind"
    action_keys = (
        "actions.end_effector_left",
        "actions.end_effector_right",
        "actions.gripper_left",
        "actions.gripper_right",
    )
    state_keys = (
        "observation.states.end_effector_left",
        "observation.states.end_effector_right",
        "observation.states.gripper_left",
        "observation.states.gripper_right",
    )

    @staticmethod
    def _convert(left_pose, right_pose, left_gripper, right_gripper) -> torch.Tensor:
        left_pose, right_pose = _float(left_pose), _float(right_pose)
        left_pos, right_pos = left_pose[..., :3], right_pose[..., :3]
        return torch.cat(
            [
                left_pos,
                quat_to_rotate6d(normalize_quaternion(left_pose[..., 3:7])),
                _gripper_column(left_gripper, left_pos),
                right_pos,
                quat_to_rotate6d(normalize_quaternion(right_pose[..., 3:7])),
                _gripper_column(right_gripper, right_pos),
            ],
            dim=-1,
        )

    def action(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.action_keys))

    def state(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.state_keys))


class RoboMindSingleArmAdapter(SourceAdapter):
    """RoboMIND single arm (Franka/UR): 6D pose (xyz + Euler) and gripper."""

    source = "robomind"
    action_keys = ("actions.end_effector", "actions.gripper")
    state_keys = ("observation.states.end_effector", "observation.states.gripper")

    @staticmethod
    def _convert(pose, gripper) -> torch.Tensor:
        pose = _float(pose)
        pos = pose[..., :3]
        arm = torch.cat([pos, euler_to_rotate6d(pose[..., 3:6]), _gripper_column(gripper, pos)], dim=-1)
        return _pad_single_arm(arm)

    def action(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.action_keys))

    def state(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.state_keys))


class RoboMindFr3DualAdapter(SourceAdapter):
    """RoboMIND dual FR3: 12D pose (two xyz + Euler) and two grippers."""

    source = "robomind"
    action_keys = ("actions.end_effector", "actions.gripper")
    state_keys = ("observation.states.end_effector", "observation.states.gripper")

    @staticmethod
    def _convert(pose, gripper) -> torch.Tensor:
        pose, grip = _float(pose), _float(gripper)
        left_pos, right_pos = pose[..., :3], pose[..., 6:9]
        return torch.cat(
            [
                left_pos,
                euler_to_rotate6d(pose[..., 3:6]),
                _gripper_column(grip[..., 0], left_pos),
                right_pos,
                euler_to_rotate6d(pose[..., 9:12]),
                _gripper_column(grip[..., 1], right_pos),
            ],
            dim=-1,
        )

    def action(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.action_keys))

    def state(self, item: Item) -> torch.Tensor:
        return self._convert(*(item[key] for key in self.state_keys))


class LiberoAdapter(SourceAdapter):
    """LIBERO: action = delta xyz + delta axis-angle + gripper (7D); state = xyz + axis-angle + 2 finger joints."""

    source = "libero"
    action_keys = (ACTION,)
    state_keys = (OBS_STATE,)

    def action(self, item: Item) -> torch.Tensor:
        action = _float(item[ACTION])
        _check_dim(action, 7, "LIBERO action")
        arm = torch.cat([action[..., :3], axis_angle_to_rotate6d(action[..., 3:6]), action[..., 6:7]], dim=-1)
        return _pad_single_arm(arm)

    def state(self, item: Item) -> torch.Tensor:
        state = _float(item[OBS_STATE])
        _check_dim(state, 8, "LIBERO state")
        arm = torch.cat([state[..., :3], axis_angle_to_rotate6d(state[..., 3:6]), state[..., 6:7]], dim=-1)
        return _pad_single_arm(arm)

    def source_action(self, action: torch.Tensor, *, binarize: bool, threshold: float) -> torch.Tensor:
        _check_dim(action, CANONICAL_DIM, "canonical action")
        gripper = action[..., 9:10]
        if binarize:
            gripper = torch.where(gripper > threshold, torch.ones_like(gripper), -torch.ones_like(gripper))
        return torch.cat([action[..., :3], rotate6d_to_axis_angle(action[..., 3:9]), gripper], dim=-1)


class CalvinAdapter(SourceAdapter):
    """CALVIN: action = xyz + Euler + gripper (7D, +1 open / -1 closed); state = TCP xyz + Euler + width + 7
    joints + last gripper action (15D)."""

    source = "calvin"
    state_keys = (OBS_STATE,)

    def __init__(self, action_key: str = "action.absolute") -> None:
        self.action_keys = (action_key,)

    @staticmethod
    def _convert(pose: torch.Tensor, gripper: torch.Tensor) -> torch.Tensor:
        arm = torch.cat([pose[..., :3], euler_to_rotate6d(pose[..., 3:6]), gripper], dim=-1)
        return _pad_single_arm(arm)

    def action(self, item: Item) -> torch.Tensor:
        action = _float(item[self.action_keys[0]])
        _check_dim(action, 7, "CALVIN action")
        return self._convert(action, action[..., 6:7])

    def state(self, item: Item) -> torch.Tensor:
        state = _float(item[OBS_STATE])
        _check_dim(state, 15, "CALVIN state")
        return self._convert(state, state[..., 14:15])

    def source_action(self, action: torch.Tensor, *, binarize: bool, threshold: float) -> torch.Tensor:
        _check_dim(action, CANONICAL_DIM, "canonical action")
        gripper = action[..., 9:10]
        if binarize:
            gripper = torch.where(gripper > threshold, torch.ones_like(gripper), -torch.ones_like(gripper))
        return torch.cat([action[..., :3], rotate6d_to_euler(action[..., 3:9]), gripper], dim=-1)


_ROBOMIND_ADAPTERS = {
    "robomind_agilex": RoboMindAgilexAdapter,
    "robomind_single": RoboMindSingleArmAdapter,
    "robomind_fr3_dual": RoboMindFr3DualAdapter,
}


def make_source_adapter(source: str, adapter: str | None, features: Mapping[str, dict]) -> SourceAdapter:
    """Adapter of a dataset spec; ``features`` are the dataset's LeRobot features (selects the CALVIN action)."""
    if source == "robomind":
        return _ROBOMIND_ADAPTERS[adapter]()
    if source == "calvin":
        return CalvinAdapter("action.absolute" if "action.absolute" in features else ACTION)
    return {
        "droid": DroidAdapter,
        "agibot": AgiBotAdapter,
        "libero": LiberoAdapter,
    }[source]()


def interpolate_sequence(sequence: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Linearly resample ``[T, ...]`` sampled at ``source_fps`` onto ``T`` points at ``target_fps``.

    The output keeps the length; the first sample stays in place and samples past the source window repeat the
    last value.
    """
    steps = int(sequence.shape[0])
    if sequence.ndim < 2 or steps <= 1:
        return sequence
    values = sequence.detach().cpu().numpy().astype(np.float64, copy=False)
    flat = values.reshape(steps, -1)
    times = np.arange(steps, dtype=np.float64) / float(source_fps)
    queries = np.linspace(0.0, (steps - 1) / float(target_fps), steps, dtype=np.float64)
    interpolator = interp1d(
        times, flat, axis=0, kind="linear", bounds_error=False, fill_value=(flat[0], flat[-1])
    )
    return torch.as_tensor(interpolator(queries).reshape(values.shape), dtype=sequence.dtype)


def interpolate_padding(padding: torch.Tensor, source_fps: float, target_fps: float) -> torch.Tensor:
    """Padding mask matching :func:`interpolate_sequence`: a resampled point is padded if it used a padded value."""
    if padding.ndim == 0 or padding.numel() <= 1:
        return padding
    steps = padding.numel()
    times = np.arange(steps, dtype=np.float64) / float(source_fps)
    queries = np.linspace(0.0, (steps - 1) / float(target_fps), steps, dtype=np.float64)
    right = np.searchsorted(times, queries, side="left").clip(0, steps - 1)
    left = np.maximum(right - 1, 0)
    mask = padding.detach().cpu().numpy()
    result = mask[right] | ((queries != times[right]) & mask[left])
    result |= (queries < times[0]) | (queries > times[-1])
    return torch.as_tensor(result, dtype=torch.bool)


def resize_image(image: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of a ``[C, H, W]`` or ``[B, C, H, W]`` image (no aspect-preserving padding)."""
    if tuple(image.shape[-2:]) == tuple(hw):
        return image
    if image.ndim == 3:
        return F.interpolate(image.unsqueeze(0), size=hw, mode="bilinear", align_corners=False).squeeze(0)
    return F.interpolate(image, size=hw, mode="bilinear", align_corners=False)


def merge_padding(
    item: Item, value: torch.Tensor, canonical_key: str, source_keys: tuple[str, ...]
) -> torch.Tensor:
    """Union of the ``*_is_pad`` masks of the columns a canonical vector was built from."""
    expected = value.shape[:-1]
    result = torch.zeros(expected, dtype=torch.bool)
    for key in (canonical_key, *source_keys):
        mask = item.get(f"{key}_is_pad")
        if mask is None:
            continue
        mask = torch.as_tensor(mask, dtype=torch.bool)
        if mask.numel() == result.numel() == 1:
            mask = mask.reshape(expected)
        if mask.shape != expected:
            raise ValueError(
                f"{key}_is_pad shape {tuple(mask.shape)} does not match {canonical_key} {tuple(expected)}"
            )
        result |= mask
    return result


@dataclass
class SampleConverter:
    """Converts raw samples of one dataset spec to canonical, unnormalized samples.

    Attributes:
        adapter: Source schema conversion.
        camera_sources: Canonical camera key -> source camera key, or None for a camera the dataset lacks (filled
            with zeros and flagged by ``<key>_is_missing``).
        resize_hw: Resize every camera to this size; None keeps native sizes.
        placeholder_hw: Size of missing-camera placeholders when ``resize_hw`` is None.
        source_fps / target_fps: Resample actions when both are set and differ.
    """

    adapter: SourceAdapter
    camera_sources: dict[str, str | None]
    resize_hw: tuple[int, int] | None
    placeholder_hw: tuple[int, int]
    source_fps: float | None = None
    target_fps: float | None = None

    def __call__(self, item: Item) -> dict[str, torch.Tensor]:
        action = self.adapter.action(item)
        state = self.adapter.state(item)
        sample: dict[str, torch.Tensor] = {
            ACTION: action,
            OBS_STATE: state,
            f"{ACTION}_is_pad": merge_padding(item, action, ACTION, self.adapter.action_keys),
            f"{OBS_STATE}_is_pad": merge_padding(item, state, OBS_STATE, self.adapter.state_keys),
        }
        for camera, source_key in self.camera_sources.items():
            if source_key is None:
                hw = self.resize_hw or self.placeholder_hw
                sample[camera] = torch.zeros((3, *hw), dtype=torch.float32)
            else:
                image = item[source_key].to(dtype=torch.float32)
                sample[camera] = image if self.resize_hw is None else resize_image(image, self.resize_hw)
            sample[f"{camera}_is_missing"] = torch.tensor(source_key is None, dtype=torch.bool)
        if self.resamples:
            for key in (ACTION, OBS_STATE):
                sample[key] = interpolate_sequence(sample[key], self.source_fps, self.target_fps)
                sample[f"{key}_is_pad"] = interpolate_padding(
                    sample[f"{key}_is_pad"], self.source_fps, self.target_fps
                )
        return sample

    @property
    def resamples(self) -> bool:
        return (
            self.source_fps is not None
            and self.target_fps is not None
            and not np.isclose(self.source_fps, self.target_fps)
        )
