"""Scan configuration: which checks run and their source-independent thresholds.

Source-specific thresholds live in the contracts (`concertina.data_quality.contracts`). Two presets reproduce the
released scans: ``pretrain`` (DROID, AgiBot, RoboMIND; the input of the pause-cleaning policy) and ``audit``
(CALVIN; dense residual and lag checks, no visual evidence). A YAML file may override any field::

    contract: droid
    trajectory:
      pause_duration_seconds: [1.0, 2.0, 3.0, 5.0]
    video:
      sample_fraction: 0.05
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from concertina.data_quality.contracts import get_source_contract

AUDIT_CONTRACTS = frozenset({"calvin"})


@dataclass(frozen=True)
class StructuralConfig:
    enabled: bool = True
    verify_required_columns: bool = True
    verify_feature_dimensions: bool = True
    verify_nonfinite: bool = True
    verify_episode_contiguity: bool = True
    verify_frame_indices: bool = True
    verify_timestamps: bool = True


@dataclass(frozen=True)
class TrajectoryConfig:
    enabled: bool = True
    # Near-static runs at least this long (seconds) are reported; the shortest one is the review minimum.
    pause_duration_seconds: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)
    jump_mad_multiplier: float = 6.0
    minimum_corroborating_signals: int = 2
    check_velocity: bool = False
    check_acceleration: bool = False
    check_jerk: bool = False
    check_exact_duplicates: bool = True

    def __post_init__(self) -> None:
        if not self.pause_duration_seconds or min(self.pause_duration_seconds) <= 0:
            raise ValueError("trajectory.pause_duration_seconds must list positive durations")
        if self.jump_mad_multiplier <= 0 or self.minimum_corroborating_signals < 1:
            raise ValueError("trajectory.jump_mad_multiplier must be > 0, minimum_corroborating_signals >= 1")


@dataclass(frozen=True)
class AlignmentConfig:
    enabled: bool = True
    check_absolute_next: bool = False
    check_lag: bool = False
    lag_window_seconds: float = 0.5
    directional_fail_below: float = 0.6
    directional_pass_at_or_above: float = 0.7
    # None uses the contract's threshold profile.
    abs_next_candidate_translation_m: float | None = None
    abs_next_candidate_rotation_rad: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.directional_fail_below < self.directional_pass_at_or_above <= 1:
            raise ValueError(
                "alignment requires 0 <= directional_fail_below < directional_pass_at_or_above <= 1"
            )


@dataclass(frozen=True)
class VideoConfig:
    enabled: bool = True
    # Decode failures, shape or frame-rate mismatches of the sampled files block the release.
    required_for_release: bool = True
    # Fraction of the MP4 files of each camera that are probed (at least one per camera).
    sample_fraction: float = 0.01
    sample_count_per_file: int = 3
    minimum_consecutive_frames: int = 2
    black_pixel_value: int = 5
    white_pixel_value: int = 250
    monochrome_pixel_fraction: float = 0.995

    def __post_init__(self) -> None:
        if not 0 < self.sample_fraction <= 1:
            raise ValueError("video.sample_fraction must be in (0, 1]")
        if self.sample_count_per_file < 1 or self.minimum_consecutive_frames < 1:
            raise ValueError("video.sample_count_per_file and minimum_consecutive_frames must be >= 1")


@dataclass(frozen=True)
class VisualMotionConfig:
    """Camera evidence for canonical-action pause candidates (FFmpeg freezedetect + DIS/Farneback flow)."""

    enabled: bool = True
    minimum_candidate_seconds: float = 3.0
    sample_fps: float = 10.0
    decode_padding_seconds: float = 0.25
    union_gap_seconds: float = 0.25
    ffmpeg_binary: str = "ffmpeg"
    ffmpeg_noise_db: float = -42.0
    ffmpeg_min_duration_seconds: float = 0.25
    flow_max_dimension: int = 320
    flow_quantile: float = 0.95
    flow_threshold_px: float = 1.0
    flow_active_pixel_threshold_px: float = 0.5
    flow_active_area_fraction: float = 0.01
    minimum_motion_run_seconds: float = 0.4
    minimum_total_motion_seconds: float = 0.4
    motion_gap_seconds: float = 0.15
    # Motion seen by this many external cameras (or one ego camera) protects a pause candidate.
    external_corroborating_cameras: int = 2
    ego_camera_patterns: tuple[str, ...] = ("wrist", "gripper", "hand", "in_hand", "eye_in_hand")
    # Without the FFmpeg executable no candidate gets camera evidence and the cleaning policy keeps every pause:
    # the scan then warns and records it in its summary, or fails when this is set.
    require_ffmpeg: bool = False

    def __post_init__(self) -> None:
        positive = (
            "minimum_candidate_seconds",
            "sample_fps",
            "flow_threshold_px",
            "ffmpeg_min_duration_seconds",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"visual_motion.{name} must be finite and > 0")
        if not 0 < self.flow_quantile < 1 or not 0 < self.flow_active_area_fraction <= 1:
            raise ValueError("visual_motion.flow_quantile and flow_active_area_fraction must be in (0, 1)")
        if self.external_corroborating_cameras < 1 or self.flow_max_dimension < 16:
            raise ValueError("visual_motion.external_corroborating_cameras >= 1, flow_max_dimension >= 16")


@dataclass(frozen=True)
class ScanConfig:
    contract: str
    structural: StructuralConfig = field(default_factory=StructuralConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visual_motion: VisualMotionConfig = field(default_factory=VisualMotionConfig)

    def __post_init__(self) -> None:
        get_source_contract(self.contract)

    @property
    def source_contract(self):
        return get_source_contract(self.contract)

    @classmethod
    def preset(cls, contract: str) -> ScanConfig:
        name = get_source_contract(contract).name
        if name not in AUDIT_CONTRACTS:
            return cls(contract=name)
        return cls(
            contract=name,
            trajectory=TrajectoryConfig(
                pause_duration_seconds=(0.2, 0.5, 1.0, 2.0, 5.0),
                check_velocity=True,
                check_acceleration=True,
                check_jerk=True,
            ),
            alignment=AlignmentConfig(check_absolute_next=True, check_lag=True),
            visual_motion=VisualMotionConfig(enabled=False),
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _override(instance, values: dict[str, Any], where: str):
    known = {item.name: item for item in dataclasses.fields(instance)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"Unknown {where} options {unknown}")
    updates = {}
    for name, value in values.items():
        current = getattr(instance, name)
        if dataclasses.is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{where}.{name} must be a mapping")
            updates[name] = _override(current, value, f"{where}.{name}")
        elif isinstance(current, tuple):
            updates[name] = tuple(value)
        else:
            updates[name] = value
    return dataclasses.replace(instance, **updates)


def load_scan_config(path: str | Path | None, *, contract: str | None = None) -> ScanConfig:
    """Preset of the contract, overridden by the YAML file (whose ``contract`` key wins over ``contract``)."""
    values: dict[str, Any] = {}
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping")
        values = dict(loaded)
    name = values.pop("contract", None) or contract
    if name is None:
        raise ValueError("The scan needs a source contract (`contract:` in the config or --contract)")
    if contract is not None and get_source_contract(contract).name != get_source_contract(name).name:
        raise ValueError(f"--contract {contract!r} disagrees with the config contract {name!r}")
    return _override(ScanConfig.preset(name), values, "scan")
