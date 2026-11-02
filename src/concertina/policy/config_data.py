"""Data sub-configuration of the Concertina policy: dataset mixture, normalization and gripper calibration.

Plain dataclasses without a LeRobot dependency. `ConcertinaConfig` composes `ConcertinaDataConfig` as
`config.data`; it is saved with the checkpoint so that evaluation rebuilds the same per-dataset processing.
"""

import math
from dataclasses import asdict, dataclass, field

from concertina.data.quality import QualityConfig

SOURCES = ("droid", "agibot", "robomind", "libero", "calvin")
ROBOMIND_ADAPTERS = ("robomind_agilex", "robomind_single", "robomind_fr3_dual")
NORMALIZATION_MODES = ("none", "global", "per_embodiment")
SINGLE_ARM_DIM = 10
DUAL_ARM_DIM = 20
CANONICAL_GRIPPER_INDICES = (9, 19)

# Declared gripper endpoints of each source after conversion to the canonical layout (open value, closed value).
BUILTIN_GRIPPER_ENDPOINTS: dict[str, tuple[float, float, str]] = {
    "droid": (0.0, 1.0, "DROID: closure = 1 - width / max_width (action.gripper_position)"),
    "agibot": (0.0, 1.0, "AgiBot World: actions.effector.position, 0 open / 1 closed"),
    "calvin": (1.0, -1.0, "CALVIN action gripper: +1 open / -1 closed"),
    "libero": (-1.0, 1.0, "robosuite PandaGripper.format_action: -1 open / +1 closed"),
}


@dataclass
class DatasetSpec:
    """One LeRobot dataset of the mixture; its list position is the `dataset_index` of its samples."""

    # Local dataset directory; relative paths are resolved against `cfg.dataset.root`.
    path: str
    # Source schema of the action/state columns: one of SOURCES.
    source: str
    # Normalization group (per-embodiment statistics are pooled over all datasets of an embodiment).
    embodiment: str
    # Column mapping variant; required for RoboMIND only (one of ROBOMIND_ADAPTERS).
    adapter: str | None = None
    # Relative sampling weight; leaving every weight unset samples all valid frames uniformly.
    weight: float | None = None
    # Frame rate of the source data for `fps_align_target_fps`; None reads the dataset metadata.
    source_fps: float | None = None
    # Canonical camera key -> candidate source camera keys (first one present wins). Unlisted keys map to
    # themselves.
    camera_map: dict[str, list[str]] = field(default_factory=dict)
    # Number of valid leading canonical action dimensions: 10 (single arm, right arm padded) or 20.
    valid_action_dim: int = DUAL_ARM_DIM
    # Gripper calibration profile id (builtin source id or a `custom_gripper_profiles` id); None uses the
    # builtin profile of `source`.
    gripper_profile: str | None = None
    # `rotation_stats.json` produced by `concertina-compute-rotation-stats`; None reads
    # `<path>/meta/rotation_stats.json`, or computes the statistics into the user cache when the dataset has none.
    rotation_stats_path: str | None = None
    # Use only these episodes (for example the complete prefix of a partially copied dataset); None uses all.
    episodes: list[int] | None = None
    # Data-quality package (`concertina-data-quality compile`) of this dataset: its quarantines, frame exclusions and
    # field masks remove samples and mask action targets.
    quality_package: str | None = None


@dataclass
class GripperProfile:
    """Declared gripper endpoints, in canonical action coordinates, of one data source."""

    id: str
    indices: list[int]
    open_values: list[float]
    closed_values: list[float]
    source: str
    adapter: str | None = None
    # "error" rejects values outside the endpoints; "clip" saturates them (declared controller overshoot).
    out_of_range: str = "error"
    provenance: str = ""


@dataclass
class ConcertinaDataConfig:
    datasets: list[DatasetSpec] = field(default_factory=list)
    drop_n_first_frames: int = 0
    drop_n_last_frames: int = 0
    # Resample actions of datasets recorded at another frame rate to this rate (None disables).
    fps_align_target_fps: float | None = None
    # Resize every camera to the model image size inside the dataset (bilinear, no aspect padding). Required when
    # datasets have different native camera sizes; otherwise the policy resizes with aspect-preserving padding.
    resize_images: bool = False
    # "none", "global" (one statistics set for all datasets) or "per_embodiment".
    normalization: str = "none"
    # Gripper channels (`model.gripper_indices`) with probability targets are calibrated to closure in [0, 1] with
    # this tolerance and these profiles; binary targets and regressed grippers are not calibrated.
    gripper_range_tolerance: float = 1e-6
    custom_gripper_profiles: list[GripperProfile] = field(default_factory=list)
    # Dataset whose conversion, statistics and calibration evaluation uses.
    inference_dataset_index: int = 0
    # Options of the per-dataset quality packages (`DatasetSpec.quality_package`).
    quality: QualityConfig = field(default_factory=QualityConfig)

    def __post_init__(self) -> None:
        self.datasets = [DatasetSpec(**spec) if isinstance(spec, dict) else spec for spec in self.datasets]
        if isinstance(self.quality, dict):
            self.quality = QualityConfig(**self.quality)
        self.custom_gripper_profiles = [
            GripperProfile(**profile) if isinstance(profile, dict) else profile
            for profile in self.custom_gripper_profiles
        ]
        if self.drop_n_first_frames < 0 or self.drop_n_last_frames < 0:
            raise ValueError("drop_n_first_frames and drop_n_last_frames must be >= 0")
        if self.fps_align_target_fps is not None and self.fps_align_target_fps <= 0:
            raise ValueError(f"fps_align_target_fps must be > 0, got {self.fps_align_target_fps}")
        if self.normalization not in NORMALIZATION_MODES:
            raise ValueError(
                f"normalization must be one of {NORMALIZATION_MODES}, got {self.normalization!r}"
            )
        if not 0.0 <= self.gripper_range_tolerance <= 1e-5:
            raise ValueError("gripper_range_tolerance must be in [0, 1e-5]")
        for index, spec in enumerate(self.datasets):
            _validate_spec(index, spec)
        weights = [spec.weight is None for spec in self.datasets]
        if any(weights) and not all(weights):
            raise ValueError("Either every dataset sets `weight` or none does")
        if self.datasets and not 0 <= self.inference_dataset_index < len(self.datasets):
            raise ValueError(
                f"inference_dataset_index={self.inference_dataset_index} is outside {len(self.datasets)} datasets"
            )
        if self.normalization == "global" and len({(s.source, s.adapter) for s in self.datasets}) > 1:
            raise ValueError("normalization='global' requires one source schema; use 'per_embodiment'")
        if self.normalization == "per_embodiment":
            schemas: dict[str, set] = {}
            for spec in self.datasets:
                schemas.setdefault(spec.embodiment, set()).add((spec.source, spec.adapter))
            mixed = sorted(name for name, found in schemas.items() if len(found) > 1)
            if mixed:
                raise ValueError(f"Embodiments {mixed} pool datasets of different source schemas")
        ids = [profile.id for profile in self.custom_gripper_profiles]
        if len(ids) != len(set(ids)):
            raise ValueError("custom_gripper_profiles contains duplicate ids")
        shadowing = sorted(set(ids) & set(BUILTIN_GRIPPER_ENDPOINTS))
        if shadowing:
            raise ValueError(f"custom_gripper_profiles cannot redefine builtin ids {shadowing}")
        for profile in self.custom_gripper_profiles:
            _validate_profile(profile)

    def validate_for_model(
        self, camera_keys: list[str], gripper_indices: list[int], gripper_target: str = "probability"
    ) -> None:
        """Checks that need the model configuration (camera keys, gripper channels and their target type)."""
        for index, spec in enumerate(self.datasets):
            unknown = sorted(set(spec.camera_map) - set(camera_keys))
            if unknown:
                raise ValueError(f"datasets[{index}].camera_map maps unknown canonical cameras {unknown}")
        if gripper_indices and gripper_target == "probability" and self.datasets:
            resolve_gripper_profiles(self, gripper_indices)


def _validate_spec(index: int, spec: DatasetSpec) -> None:
    where = f"datasets[{index}]"
    if not spec.path:
        raise ValueError(f"{where}.path must be set")
    if spec.source not in SOURCES:
        raise ValueError(f"{where}.source must be one of {SOURCES}, got {spec.source!r}")
    if not spec.embodiment:
        raise ValueError(f"{where}.embodiment must be set")
    if spec.source == "robomind":
        if spec.adapter not in ROBOMIND_ADAPTERS:
            raise ValueError(f"{where}.adapter must be one of {ROBOMIND_ADAPTERS} for RoboMIND")
    elif spec.adapter is not None:
        raise ValueError(f"{where}.adapter is only used by RoboMIND datasets")
    if spec.weight is not None and (not math.isfinite(spec.weight) or spec.weight < 0):
        raise ValueError(f"{where}.weight must be finite and >= 0")
    if spec.source_fps is not None and spec.source_fps <= 0:
        raise ValueError(f"{where}.source_fps must be > 0")
    if spec.valid_action_dim not in (SINGLE_ARM_DIM, DUAL_ARM_DIM):
        raise ValueError(f"{where}.valid_action_dim must be {SINGLE_ARM_DIM} or {DUAL_ARM_DIM}")
    for key, candidates in spec.camera_map.items():
        if not candidates or any(not candidate for candidate in candidates):
            raise ValueError(f"{where}.camera_map[{key!r}] must list at least one source camera")
    if spec.episodes is not None and (
        not spec.episodes or min(spec.episodes) < 0 or len(set(spec.episodes)) != len(spec.episodes)
    ):
        raise ValueError(f"{where}.episodes must list distinct non-negative episode indices")


def _validate_profile(profile: GripperProfile) -> None:
    where = f"gripper profile {profile.id!r}"
    if not profile.id:
        raise ValueError("gripper profile id must be set")
    if not profile.indices or len(set(profile.indices)) != len(profile.indices) or min(profile.indices) < 0:
        raise ValueError(f"{where}: indices must be unique non-negative integers")
    if len(profile.open_values) != len(profile.indices) or len(profile.closed_values) != len(profile.indices):
        raise ValueError(f"{where}: endpoint lists must match indices")
    for opened, closed in zip(profile.open_values, profile.closed_values, strict=True):
        if not (math.isfinite(opened) and math.isfinite(closed)) or opened == closed:
            raise ValueError(f"{where}: endpoints must be finite and distinct")
    if profile.source not in SOURCES:
        raise ValueError(f"{where}: source must be one of {SOURCES}")
    if profile.source == "robomind" and profile.adapter is None:
        raise ValueError(f"{where}: RoboMIND profiles must name their adapter")
    if profile.out_of_range not in ("error", "clip"):
        raise ValueError(f"{where}: out_of_range must be 'error' or 'clip'")


def resolve_gripper_profiles(
    data: ConcertinaDataConfig, gripper_indices: list[int]
) -> tuple[list[dict], list[str]]:
    """Resolve the calibration profile of every dataset.

    Returns the used profiles (as dicts, in first-use order) and the profile id of each `dataset_index`.
    """
    custom = {profile.id: asdict(profile) for profile in data.custom_gripper_profiles}
    resolved: dict[str, dict] = {}
    dataset_profiles: list[str] = []
    for index, spec in enumerate(data.datasets):
        profile_id = spec.gripper_profile or spec.source
        if profile_id in custom:
            profile = custom[profile_id]
        elif profile_id in BUILTIN_GRIPPER_ENDPOINTS:
            opened, closed, provenance = BUILTIN_GRIPPER_ENDPOINTS[profile_id]
            if not set(gripper_indices) <= set(CANONICAL_GRIPPER_INDICES):
                raise ValueError(
                    f"builtin gripper profile {profile_id!r} covers canonical indices {CANONICAL_GRIPPER_INDICES}, "
                    f"got gripper_indices={gripper_indices}"
                )
            profile = {
                "id": profile_id,
                "indices": list(gripper_indices),
                "open_values": [opened] * len(gripper_indices),
                "closed_values": [closed] * len(gripper_indices),
                "source": profile_id,
                "adapter": None,
                "out_of_range": "error",
                "provenance": provenance,
            }
        else:
            raise ValueError(f"datasets[{index}] references unknown gripper profile {profile_id!r}")
        if profile["source"] != spec.source:
            raise ValueError(
                f"gripper profile {profile_id!r} describes source {profile['source']!r}, "
                f"but datasets[{index}] is {spec.source!r}"
            )
        if profile["adapter"] is not None and profile["adapter"] != spec.adapter:
            raise ValueError(
                f"gripper profile {profile_id!r} describes adapter {profile['adapter']!r}, "
                f"but datasets[{index}] uses {spec.adapter!r}"
            )
        if not set(profile["indices"]) <= set(gripper_indices):
            raise ValueError(f"gripper profile {profile_id!r} indices must be a subset of {gripper_indices}")
        resolved[profile_id] = profile
        dataset_profiles.append(profile_id)
    unused = sorted(set(custom) - set(resolved))
    if unused:
        raise ValueError(f"custom_gripper_profiles {unused} are not used by any dataset")
    return list(resolved.values()), dataset_profiles
