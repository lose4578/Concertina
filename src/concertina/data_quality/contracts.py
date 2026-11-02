"""Source contracts: what each converted dataset family stores and which invariants hold.

The contracts deliberately describe source semantics separately from cleaning
rules.  A field being *comparable* with a later observation is a provenance
claim; it must not be inferred from equal tensor dimensions alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, prod
from typing import Literal

RotationRepresentation = Literal["none", "quaternion", "euler", "rotation_vector"]
RawQuaternionNormPolicy = Literal["unit_required", "normalizable_nonzero"]
AbsNextRelation = Literal[
    "previous_action_to_current_state",
    "current_action_to_next_state",
    "none",
]
AbsNextPolicy = Literal["disabled", "review", "mask_field"]


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """A serialisable, immutable half-open feature slice."""

    start: int
    stop: int

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise TypeError("slice start must be an integer")
        if isinstance(self.stop, bool) or not isinstance(self.stop, int):
            raise TypeError("slice stop must be an integer")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError(f"invalid half-open slice [{self.start}, {self.stop})")

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class SourceComponentSpec:
    """One exact raw source field consumed by a logical feature layout.

    LeRobot frame tables may store one training feature across several columns
    and may give each row a rank greater than one (for example AgiBot's
    ``[2, 3]`` dual-arm positions).  ``expected_shape`` is the complete
    per-row source shape.  Flattening is an audit-only view used to run the
    existing numeric checks; it does not assert that the raw dataset contains a
    materialized vector with the logical layout name.
    """

    column: str
    expected_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("source component column must be non-empty")
        if not self.expected_shape:
            raise ValueError("source component expected_shape must be non-empty")
        for dimension in self.expected_shape:
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError("source component dimensions must be integers")
            if dimension <= 0:
                raise ValueError("source component dimensions must be positive")

    @property
    def flat_dim(self) -> int:
        return int(prod(self.expected_shape))


@dataclass(frozen=True, slots=True)
class FeatureLayout:
    """Layout and semantics of one vector-valued source column.

    ``position_slices`` and ``rotation_slices`` contain one slice per arm.
    Gripper indices are separate because command grippers must not be compared
    with measured gripper widths merely because both occupy one scalar slot.
    """

    column: str
    expected_dim: int
    position_slices: tuple[SliceSpec, ...] = ()
    rotation_slices: tuple[SliceSpec, ...] = ()
    gripper_indices: tuple[int, ...] = ()
    rotation_representation: RotationRepresentation = "none"
    rotation_order: str | None = None
    action_semantics: str | None = None
    source_components: tuple[SourceComponentSpec, ...] = ()

    def __post_init__(self) -> None:
        if not self.column:
            raise ValueError("feature layout column must be non-empty")
        if isinstance(self.expected_dim, bool) or not isinstance(self.expected_dim, int):
            raise TypeError("expected_dim must be an integer")
        if self.expected_dim <= 0:
            raise ValueError("expected_dim must be positive")

        if self.source_components:
            component_columns = tuple(component.column for component in self.source_components)
            if len(set(component_columns)) != len(component_columns):
                raise ValueError(f"duplicate source components in logical layout {self.column!r}")
            flattened_dim = sum(component.flat_dim for component in self.source_components)
            if flattened_dim != self.expected_dim:
                raise ValueError(
                    f"logical layout {self.column!r} expected_dim={self.expected_dim} does not match "
                    f"the source-component flattened dim {flattened_dim}"
                )

        for spec in (*self.position_slices, *self.rotation_slices):
            if spec.stop > self.expected_dim:
                raise ValueError(
                    f"slice [{spec.start}, {spec.stop}) exceeds {self.column} dim {self.expected_dim}"
                )
        for index in self.gripper_indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("gripper indices must be integers")
            if not 0 <= index < self.expected_dim:
                raise ValueError(f"gripper index {index} exceeds {self.column} dim {self.expected_dim}")

        if self.rotation_slices and self.rotation_representation == "none":
            raise ValueError("rotation slices require a rotation representation")
        if not self.rotation_slices and self.rotation_representation != "none":
            raise ValueError("rotation representation requires at least one rotation slice")
        if self.rotation_representation != "none" and not self.rotation_order:
            raise ValueError("rotation_order is required for represented rotations")
        if self.rotation_representation == "none" and self.rotation_order is not None:
            raise ValueError("rotation_order must be None when no rotation is represented")

    @property
    def position_python_slices(self) -> tuple[slice, ...]:
        return tuple(spec.as_slice() for spec in self.position_slices)

    @property
    def rotation_python_slices(self) -> tuple[slice, ...]:
        return tuple(spec.as_slice() for spec in self.rotation_slices)

    @property
    def components(self) -> tuple[SourceComponentSpec, ...]:
        """The explicit components, or the layout column itself as its single component."""

        if self.source_components:
            return self.source_components
        return (SourceComponentSpec(self.column, (self.expected_dim,)),)

    @property
    def source_columns(self) -> tuple[str, ...]:
        return tuple(component.column for component in self.components)

    @property
    def is_composite(self) -> bool:
        return bool(self.source_components)


@dataclass(frozen=True, slots=True)
class ThresholdProfile:
    """Source-aware candidate thresholds, never universal deletion limits.

    Values here seed candidate generation and dry-run reports.  Only explicit
    contract violations may be promoted to automatic quarantine without the
    review and coverage gates in the run configuration.
    """

    profile_version: str
    static_translation_m: float | None = None
    static_rotation_rad: float | None = None
    static_gripper_delta: float | None = None
    long_pause_seconds: float = 5.0
    review_translation_jump_m: float | None = None
    review_rotation_jump_rad: float | None = None
    abs_next_candidate_translation_m: float | None = None
    abs_next_candidate_rotation_rad: float | None = None
    abs_next_candidate_gripper_delta: float | None = None

    def __post_init__(self) -> None:
        if not self.profile_version:
            raise ValueError("threshold profile_version must be non-empty")
        positive_or_none = (
            "static_translation_m",
            "static_rotation_rad",
            "static_gripper_delta",
            "review_translation_jump_m",
            "review_rotation_jump_rad",
            "abs_next_candidate_translation_m",
            "abs_next_candidate_rotation_rad",
            "abs_next_candidate_gripper_delta",
        )
        for name in positive_or_none:
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when set")
        if self.long_pause_seconds <= 0:
            raise ValueError("long_pause_seconds must be positive")


@dataclass(frozen=True, slots=True)
class SourceContract:
    """Immutable provenance contract for one converted dataset family."""

    name: str
    contract_version: str
    aliases: tuple[str, ...]
    required_projection_columns: tuple[str, ...]
    optional_projection_columns: tuple[str, ...]
    video_keys: tuple[str, ...]
    state_layout: FeatureLayout
    action_layouts: tuple[FeatureLayout, ...]
    canonical_action_column: str
    abs_next_relation: AbsNextRelation
    abs_next_policy: AbsNextPolicy
    abs_next_provenance: str
    abs_next_provenance_deterministic: bool
    abs_next_mask_field_authority: str | None
    compare_gripper_in_abs_next: bool
    default_fps: float
    threshold_profile: ThresholdProfile
    abs_next_mask_source_field: str | None = None
    raw_quaternion_norm_policy: RawQuaternionNormPolicy = "unit_required"
    raw_quaternion_norm_epsilon: float = 1e-12
    raw_quaternion_norm_provenance: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.contract_version:
            raise ValueError("contract name and version must be non-empty")
        if self.default_fps <= 0:
            raise ValueError("default_fps must be positive")
        if self.raw_quaternion_norm_policy not in {"unit_required", "normalizable_nonzero"}:
            raise ValueError(f"unsupported raw quaternion norm policy: {self.raw_quaternion_norm_policy!r}")
        if (
            isinstance(self.raw_quaternion_norm_epsilon, bool)
            or not isinstance(self.raw_quaternion_norm_epsilon, (int, float))
            or not isfinite(self.raw_quaternion_norm_epsilon)
            or self.raw_quaternion_norm_epsilon <= 0
        ):
            raise ValueError("raw_quaternion_norm_epsilon must be finite and positive")
        quaternion_layouts = tuple(
            layout
            for layout in (self.state_layout, *self.action_layouts)
            if layout.rotation_representation == "quaternion"
        )
        if self.raw_quaternion_norm_policy == "normalizable_nonzero":
            if not quaternion_layouts:
                raise ValueError("normalizable_nonzero requires at least one quaternion layout")
            if (
                not isinstance(self.raw_quaternion_norm_provenance, str)
                or not self.raw_quaternion_norm_provenance
            ):
                raise ValueError(
                    "normalizable_nonzero requires explicit raw quaternion normalization provenance"
                )
        elif self.raw_quaternion_norm_provenance is not None and not quaternion_layouts:
            raise ValueError("raw quaternion norm provenance requires a quaternion layout")
        required = self.required_projection_columns
        optional = self.optional_projection_columns
        if len(set(required)) != len(required):
            raise ValueError(f"duplicate required columns in {self.name} contract")
        if len(set(optional)) != len(optional):
            raise ValueError(f"duplicate optional columns in {self.name} contract")
        overlap = set(required).intersection(optional)
        if overlap:
            raise ValueError(f"columns cannot be both required and optional: {sorted(overlap)}")
        if len(set(self.video_keys)) != len(self.video_keys):
            raise ValueError(f"duplicate video keys in {self.name} contract")
        if set(self.video_keys).intersection(self.all_projection_columns):
            raise ValueError("video keys are v3 media features, not parquet projection columns")
        missing_state_components = sorted(set(self.state_layout.source_columns).difference(required))
        if missing_state_components:
            raise ValueError(f"state layout source components must be required: {missing_state_components}")
        action_columns = tuple(layout.column for layout in self.action_layouts)
        if len(set(action_columns)) != len(action_columns):
            raise ValueError(f"duplicate action layouts in {self.name} contract")
        if self.canonical_action_column not in action_columns:
            raise ValueError("canonical action column needs a matching action layout")
        for layout in self.action_layouts:
            missing_action_components = sorted(
                set(layout.source_columns).difference(self.all_projection_columns)
            )
            if missing_action_components:
                raise ValueError(
                    f"action layout {layout.column!r} source components must be projected: "
                    f"{missing_action_components}"
                )
        missing_canonical_components = sorted(
            set(self.canonical_action_layout.source_columns).difference(required)
        )
        if missing_canonical_components:
            raise ValueError(
                f"canonical action layout source components must be required: {missing_canonical_components}"
            )
        if self.abs_next_policy not in {"disabled", "review", "mask_field"}:
            raise ValueError(f"unsupported absolute-next policy: {self.abs_next_policy!r}")
        if not self.abs_next_provenance:
            raise ValueError("absolute-next provenance must be non-empty")
        if not isinstance(self.abs_next_provenance_deterministic, bool):
            raise TypeError("abs_next_provenance_deterministic must be a boolean")
        if self.abs_next_relation == "none" and self.abs_next_policy != "disabled":
            raise ValueError("contracts without an absolute-next relation must disable that check")
        if self.abs_next_relation != "none" and self.abs_next_policy == "disabled":
            raise ValueError(
                "contracts with an absolute-next relation must declare review or mask_field policy"
            )
        if self.abs_next_policy == "mask_field" and not self.abs_next_provenance_deterministic:
            raise ValueError("mask_field requires deterministic converter provenance")
        if self.abs_next_policy == "mask_field" and not self.abs_next_mask_field_authority:
            raise ValueError(
                "mask_field requires independent authority for the field being masked; "
                "a deterministic action/state pairing alone does not identify the corrupt side"
            )
        if self.abs_next_policy == "mask_field":
            if not isinstance(self.abs_next_mask_source_field, str) or not self.abs_next_mask_source_field:
                raise ValueError("mask_field requires an exact abs_next_mask_source_field")
            if self.abs_next_mask_source_field not in {
                self.canonical_action_column,
                self.state_layout.column,
            }:
                raise ValueError("abs_next_mask_source_field must exactly match one compared source field")
        elif self.abs_next_mask_field_authority is not None or self.abs_next_mask_source_field is not None:
            raise ValueError("abs_next mask authority and source field are only valid with mask_field policy")
        if self.compare_gripper_in_abs_next:
            action_grippers = self.canonical_action_layout.gripper_indices
            state_grippers = self.state_layout.gripper_indices
            if not action_grippers or len(action_grippers) != len(state_grippers):
                raise ValueError("absolute-next gripper comparison requires dimension-aligned indices")
            if self.threshold_profile.abs_next_candidate_gripper_delta is None:
                raise ValueError("absolute-next gripper comparison requires an explicit candidate threshold")

    @property
    def all_projection_columns(self) -> tuple[str, ...]:
        return self.required_projection_columns + self.optional_projection_columns

    @property
    def canonical_action_layout(self) -> FeatureLayout:
        return self.layout_for(self.canonical_action_column)

    @property
    def source_layout_columns(self) -> tuple[str, ...]:
        """Return exact raw source fields participating in numeric layouts."""

        return tuple(
            dict.fromkeys(
                column
                for layout in (self.state_layout, *self.action_layouts)
                for column in layout.source_columns
            )
        )

    def layout_for(self, column: str) -> FeatureLayout:
        if column == self.state_layout.column:
            return self.state_layout
        for layout in self.action_layouts:
            if layout.column == column:
                return layout
        raise KeyError(f"{column!r} has no vector layout in contract {self.name!r}")


_COMMON_COLUMNS = ("index", "episode_index", "frame_index", "timestamp", "task_index")

DROID_CONTRACT = SourceContract(
    name="droid",
    contract_version="droid-1.0.1-lerobot-source-quality-v1",
    aliases=("droid_1.0.1", "droid-1.0.1", "oxe_droid"),
    required_projection_columns=(
        *_COMMON_COLUMNS,
        "observation.state.cartesian_position",
        "observation.state.gripper_position",
        "action.cartesian_position",
        "action.gripper_position",
    ),
    optional_projection_columns=(),
    video_keys=(
        "observation.images.wrist_left",
        "observation.images.exterior_1_left",
        "observation.images.exterior_2_left",
    ),
    state_layout=FeatureLayout(
        column="droid.observation.pose_gripper.audit_view",
        expected_dim=7,
        position_slices=(SliceSpec(0, 3),),
        rotation_slices=(SliceSpec(3, 6),),
        gripper_indices=(6,),
        rotation_representation="euler",
        rotation_order="xyz",
        source_components=(
            SourceComponentSpec("observation.state.cartesian_position", (6,)),
            SourceComponentSpec("observation.state.gripper_position", (1,)),
        ),
    ),
    action_layouts=(
        FeatureLayout(
            column="droid.action.pose_gripper.audit_view",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            action_semantics="end_effector_and_gripper_command_consumed_by_canonical_adapter",
            source_components=(
                SourceComponentSpec("action.cartesian_position", (6,)),
                SourceComponentSpec("action.gripper_position", (1,)),
            ),
        ),
    ),
    canonical_action_column="droid.action.pose_gripper.audit_view",
    abs_next_relation="none",
    abs_next_policy="disabled",
    abs_next_provenance=(
        "DROID conversion exposes commanded action and observed robot state at the same dataset row; "
        "the canonical adapter does not establish either an action-to-next-state equality or an "
        "authoritative side for repair"
    ),
    abs_next_provenance_deterministic=False,
    abs_next_mask_field_authority=None,
    compare_gripper_in_abs_next=False,
    default_fps=15.0,
    threshold_profile=ThresholdProfile(
        profile_version="pretrain-source-quality-2026-08-10-v1",
        static_translation_m=0.0005,
        static_rotation_rad=0.005,
        static_gripper_delta=0.0001,
        long_pause_seconds=5.0,
        review_translation_jump_m=0.05,
        review_rotation_jump_rad=0.5,
    ),
)


AGIBOT_CONTRACT = SourceContract(
    name="agibot",
    contract_version="agibotworld-alpha-lerobot-source-quality-v1",
    aliases=("agibotworld", "agibotworld-alpha", "agibotworld-alpha-lerobot"),
    required_projection_columns=(
        *_COMMON_COLUMNS,
        "observation.states.end.position",
        "observation.states.end.orientation",
        "observation.states.effector.position",
        "actions.end.position",
        "actions.end.orientation",
        "actions.effector.position",
    ),
    optional_projection_columns=(),
    video_keys=(
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ),
    state_layout=FeatureLayout(
        column="agibot.observation.pose_gripper.audit_view",
        expected_dim=16,
        position_slices=(SliceSpec(0, 3), SliceSpec(3, 6)),
        rotation_slices=(SliceSpec(6, 10), SliceSpec(10, 14)),
        gripper_indices=(14, 15),
        rotation_representation="quaternion",
        rotation_order="xyzw",
        source_components=(
            SourceComponentSpec("observation.states.end.position", (2, 3)),
            SourceComponentSpec("observation.states.end.orientation", (2, 4)),
            SourceComponentSpec("observation.states.effector.position", (2,)),
        ),
    ),
    action_layouts=(
        FeatureLayout(
            column="agibot.action.pose_gripper.audit_view",
            expected_dim=16,
            position_slices=(SliceSpec(0, 3), SliceSpec(3, 6)),
            rotation_slices=(SliceSpec(6, 10), SliceSpec(10, 14)),
            gripper_indices=(14, 15),
            rotation_representation="quaternion",
            rotation_order="xyzw",
            action_semantics="dual_arm_end_effector_and_gripper_command_consumed_by_canonical_adapter",
            source_components=(
                SourceComponentSpec("actions.end.position", (2, 3)),
                SourceComponentSpec("actions.end.orientation", (2, 4)),
                SourceComponentSpec("actions.effector.position", (2,)),
            ),
        ),
    ),
    canonical_action_column="agibot.action.pose_gripper.audit_view",
    abs_next_relation="none",
    abs_next_policy="disabled",
    abs_next_provenance=(
        "AgiBotWorld action and observation components are separately recorded source fields; "
        "the canonical adapter only reshapes them and supplies no authoritative absolute-next invariant"
    ),
    abs_next_provenance_deterministic=False,
    abs_next_mask_field_authority=None,
    compare_gripper_in_abs_next=False,
    default_fps=30.0,
    threshold_profile=ThresholdProfile(
        profile_version="pretrain-source-quality-2026-08-10-v1",
        static_translation_m=0.0005,
        static_rotation_rad=0.005,
        static_gripper_delta=0.0001,
        long_pause_seconds=5.0,
        review_translation_jump_m=0.05,
        review_rotation_jump_rad=0.5,
    ),
)


def _thresholds() -> ThresholdProfile:
    return ThresholdProfile(
        profile_version="pretrain-source-quality-2026-08-10-v1",
        static_translation_m=0.0005,
        static_rotation_rad=0.005,
        static_gripper_delta=0.0001,
        long_pause_seconds=5.0,
        review_translation_jump_m=0.05,
        review_rotation_jump_rad=0.5,
    )


def _contract(
    *,
    name: str,
    aliases: tuple[str, ...],
    version: str,
    state_layout: FeatureLayout,
    action_layout: FeatureLayout,
    video_keys: tuple[str, ...],
    raw_quaternion_norm_policy: RawQuaternionNormPolicy = "unit_required",
    raw_quaternion_norm_epsilon: float = 1e-12,
    raw_quaternion_norm_provenance: str | None = None,
) -> SourceContract:
    required_components = tuple(dict.fromkeys((*state_layout.source_columns, *action_layout.source_columns)))
    return SourceContract(
        name=name,
        contract_version=version,
        aliases=aliases,
        required_projection_columns=(*_COMMON_COLUMNS, *required_components),
        optional_projection_columns=(),
        video_keys=video_keys,
        state_layout=state_layout,
        action_layouts=(action_layout,),
        canonical_action_column=action_layout.column,
        abs_next_relation="none",
        abs_next_policy="disabled",
        abs_next_provenance=(
            "RoboMIND action and observation components are separately recorded source fields; "
            "the canonical adapter only reshapes them and supplies no authoritative "
            "absolute-next invariant"
        ),
        abs_next_provenance_deterministic=False,
        abs_next_mask_field_authority=None,
        compare_gripper_in_abs_next=False,
        default_fps=30.0,
        threshold_profile=_thresholds(),
        raw_quaternion_norm_policy=raw_quaternion_norm_policy,
        raw_quaternion_norm_epsilon=raw_quaternion_norm_epsilon,
        raw_quaternion_norm_provenance=raw_quaternion_norm_provenance,
    )


ROBOMIND_AGILEX_CONTRACT = _contract(
    name="robomind_agilex",
    aliases=("robomind-agilex", "agilex_3rgb", "robomind_agilex_3rgb"),
    version="robomind-agilex-3rgb-lerobot-source-quality-v2",
    state_layout=FeatureLayout(
        column="robomind_agilex.observation.pose_gripper.audit_view",
        expected_dim=16,
        position_slices=(SliceSpec(0, 3), SliceSpec(7, 10)),
        rotation_slices=(SliceSpec(3, 7), SliceSpec(10, 14)),
        gripper_indices=(14, 15),
        rotation_representation="quaternion",
        rotation_order="xyzw",
        source_components=(
            SourceComponentSpec("observation.states.end_effector_left", (7,)),
            SourceComponentSpec("observation.states.end_effector_right", (7,)),
            SourceComponentSpec("observation.states.gripper_left", (1,)),
            SourceComponentSpec("observation.states.gripper_right", (1,)),
        ),
    ),
    action_layout=FeatureLayout(
        column="robomind_agilex.action.pose_gripper.audit_view",
        expected_dim=16,
        position_slices=(SliceSpec(0, 3), SliceSpec(7, 10)),
        rotation_slices=(SliceSpec(3, 7), SliceSpec(10, 14)),
        gripper_indices=(14, 15),
        rotation_representation="quaternion",
        rotation_order="xyzw",
        action_semantics="dual_arm_end_effector_and_gripper_command_consumed_by_canonical_adapter",
        source_components=(
            SourceComponentSpec("actions.end_effector_left", (7,)),
            SourceComponentSpec("actions.end_effector_right", (7,)),
            SourceComponentSpec("actions.gripper_left", (1,)),
            SourceComponentSpec("actions.gripper_right", (1,)),
        ),
    ),
    video_keys=(
        "observation.images.camera_front",
        "observation.images.camera_left_wrist",
        "observation.images.camera_right_wrist",
    ),
    raw_quaternion_norm_policy="normalizable_nonzero",
    raw_quaternion_norm_epsilon=1e-6,
    raw_quaternion_norm_provenance=(
        "the RoboMIND Agilex canonical adapter normalizes left/right end-effector quaternions "
        "(eps=1e-6) before the rotate6d conversion; a finite non-zero raw scale is "
        "representation-equivalent"
    ),
)


def _single_arm_contract(
    *,
    name: str,
    aliases: tuple[str, ...],
    version: str,
    video_keys: tuple[str, ...],
) -> SourceContract:
    return _contract(
        name=name,
        aliases=aliases,
        version=version,
        state_layout=FeatureLayout(
            column=f"{name}.observation.pose_gripper.audit_view",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            source_components=(
                SourceComponentSpec("observation.states.end_effector", (6,)),
                SourceComponentSpec("observation.states.gripper", (1,)),
            ),
        ),
        action_layout=FeatureLayout(
            column=f"{name}.action.pose_gripper.audit_view",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            action_semantics="single_arm_end_effector_and_gripper_command_consumed_by_canonical_adapter",
            source_components=(
                SourceComponentSpec("actions.end_effector", (6,)),
                SourceComponentSpec("actions.gripper", (1,)),
            ),
        ),
        video_keys=video_keys,
    )


ROBOMIND_FRANKA_CONTRACT = _single_arm_contract(
    name="robomind_franka",
    aliases=("robomind-franka", "franka_1rgb", "robomind_franka_1rgb"),
    version="robomind-franka-1rgb-lerobot-source-quality-v1",
    video_keys=("observation.images.camera_top",),
)

ROBOMIND_FR3_CONTRACT = _contract(
    name="robomind_fr3",
    aliases=("robomind-fr3", "franka_fr3_dual", "robomind_franka_fr3_dual"),
    version="robomind-fr3-dual-lerobot-source-quality-v1",
    state_layout=FeatureLayout(
        column="robomind_fr3.observation.pose_gripper.audit_view",
        expected_dim=14,
        position_slices=(SliceSpec(0, 3), SliceSpec(6, 9)),
        rotation_slices=(SliceSpec(3, 6), SliceSpec(9, 12)),
        gripper_indices=(12, 13),
        rotation_representation="euler",
        rotation_order="xyz",
        source_components=(
            SourceComponentSpec("observation.states.end_effector", (12,)),
            SourceComponentSpec("observation.states.gripper", (2,)),
        ),
    ),
    action_layout=FeatureLayout(
        column="robomind_fr3.action.pose_gripper.audit_view",
        expected_dim=14,
        position_slices=(SliceSpec(0, 3), SliceSpec(6, 9)),
        rotation_slices=(SliceSpec(3, 6), SliceSpec(9, 12)),
        gripper_indices=(12, 13),
        rotation_representation="euler",
        rotation_order="xyz",
        action_semantics="dual_arm_end_effector_and_gripper_command_consumed_by_canonical_adapter",
        source_components=(
            SourceComponentSpec("actions.end_effector", (12,)),
            SourceComponentSpec("actions.gripper", (2,)),
        ),
    ),
    video_keys=(
        "observation.images.camera_front",
        "observation.images.camera_top",
        "observation.images.camera_left",
        "observation.images.camera_right",
    ),
)

ROBOMIND_UR_CONTRACT = _single_arm_contract(
    name="robomind_ur",
    aliases=("robomind-ur", "ur_1rgb", "robomind_ur_1rgb"),
    version="robomind-ur-1rgb-lerobot-source-quality-v1",
    video_keys=("observation.images.camera_top",),
)


CALVIN_CONTRACT = SourceContract(
    name="calvin",
    contract_version="calvin-abcd-lerobot-source-quality-v2",
    aliases=("calvin_abc_d", "calvin_abc_d-lerobot", "calvin-abcd"),
    required_projection_columns=(
        "index",
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "observation.state",
        "action.absolute",
    ),
    optional_projection_columns=(
        "observation.environment_state",
        "action",
        "action.relative",
    ),
    video_keys=(
        "observation.images.top",
        "observation.images.wrist",
    ),
    state_layout=FeatureLayout(
        column="observation.state",
        expected_dim=15,
        position_slices=(SliceSpec(0, 3),),
        rotation_slices=(SliceSpec(3, 6),),
        # Index 6 is measured width.  Index 14 is the discrete gripper state;
        # neither is asserted equal to action.absolute[6].
        gripper_indices=(6, 14),
        rotation_representation="euler",
        rotation_order="xyz",
    ),
    action_layouts=(
        FeatureLayout(
            column="action.absolute",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            action_semantics="absolute_next_end_effector_command",
        ),
        FeatureLayout(
            column="action.relative",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            action_semantics="clipped_normalized_relative_command",
        ),
        FeatureLayout(
            column="action",
            expected_dim=7,
            position_slices=(SliceSpec(0, 3),),
            rotation_slices=(SliceSpec(3, 6),),
            gripper_indices=(6,),
            rotation_representation="euler",
            rotation_order="xyz",
            action_semantics="training_action_alias",
        ),
    ),
    canonical_action_column="action.absolute",
    abs_next_relation="current_action_to_next_state",
    abs_next_policy="review",
    abs_next_provenance=(
        "CALVIN action.absolute is an end-effector command; residuals to the observed next state "
        "include simulator/control response and are candidate evidence, not converter-invariant corruption"
    ),
    abs_next_provenance_deterministic=False,
    abs_next_mask_field_authority=None,
    # action.absolute[6] is a command; observation.state[6] is measured width
    # and observation.state[14] is a discrete state.
    compare_gripper_in_abs_next=False,
    default_fps=30.0,
    threshold_profile=ThresholdProfile(
        profile_version="source-quality-2026-08-10-v2",
        static_translation_m=0.0005,
        static_rotation_rad=0.005,
        long_pause_seconds=5.0,
        abs_next_candidate_translation_m=0.005,
        abs_next_candidate_rotation_rad=0.02,
    ),
)


CONTRACTS: dict[str, SourceContract] = {
    contract.name: contract
    for contract in (
        DROID_CONTRACT,
        AGIBOT_CONTRACT,
        ROBOMIND_AGILEX_CONTRACT,
        ROBOMIND_FRANKA_CONTRACT,
        ROBOMIND_FR3_CONTRACT,
        ROBOMIND_UR_CONTRACT,
        CALVIN_CONTRACT,
    )
}
_ALIASES = {alias.lower(): contract.name for contract in CONTRACTS.values() for alias in contract.aliases}


def get_source_contract(name: str) -> SourceContract:
    """Contract by canonical name or documented alias."""
    key = name.strip().lower()
    canonical = key if key in CONTRACTS else _ALIASES.get(key)
    if canonical is None:
        raise ValueError(f"Unknown source contract {name!r}; expected one of {sorted(CONTRACTS)}")
    return CONTRACTS[canonical]
