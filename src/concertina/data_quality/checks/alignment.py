"""Action/state alignment checks for explicit source timing contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .base import QualityCheckResult, QualityFinding, json_safe, true_runs
from .trajectory import quaternion_geodesic_angle, wrapped_euler_delta

AlignmentContract = Literal["action_t_to_state_t_plus_1", "action_t_minus_1_to_state_t"]
RotationRepresentation = Literal["none", "euler", "quaternion"]
AbsoluteCandidatePolicy = Literal["review", "mask_field"]


@dataclass(frozen=True, slots=True)
class AbsoluteAlignmentConfig:
    """Indices and thresholds for a declared action/state source-pair relation."""

    contract: AlignmentContract
    action_field: str = "action"
    state_field: str = "state"
    action_source_field: str | None = None
    state_source_field: str | None = None
    action_translation_indices: tuple[int, ...] = ()
    state_translation_indices: tuple[int, ...] = ()
    translation_threshold: float | None = None
    rotation_representation: RotationRepresentation = "none"
    action_rotation_indices: tuple[int, ...] = ()
    state_rotation_indices: tuple[int, ...] = ()
    rotation_threshold: float | None = None
    quaternion_norm_epsilon: float | None = None
    candidate_policy: AbsoluteCandidatePolicy = "review"
    provenance: str = "unspecified source relation"
    provenance_deterministic: bool = False
    mask_field_authority: str | None = None
    mask_source_field: str | None = None

    def __post_init__(self) -> None:
        if self.contract not in {"action_t_to_state_t_plus_1", "action_t_minus_1_to_state_t"}:
            raise ValueError(f"Unsupported alignment contract: {self.contract!r}")
        if not isinstance(self.action_field, str) or not self.action_field.strip():
            raise ValueError("action_field must be a non-empty string")
        if not isinstance(self.state_field, str) or not self.state_field.strip():
            raise ValueError("state_field must be a non-empty string")
        for name, value in (
            ("action_source_field", self.action_source_field),
            ("state_source_field", self.state_source_field),
        ):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string when set")
        if len(self.action_translation_indices) != len(self.state_translation_indices):
            raise ValueError("Action/state translation index counts must match")
        if len(self.action_rotation_indices) != len(self.state_rotation_indices):
            raise ValueError("Action/state rotation index counts must match")
        if self.translation_threshold is not None and self.translation_threshold < 0:
            raise ValueError("translation_threshold must be non-negative")
        if self.rotation_threshold is not None and self.rotation_threshold < 0:
            raise ValueError("rotation_threshold must be non-negative")
        if self.rotation_representation == "quaternion":
            if len(self.action_rotation_indices) != 4:
                raise ValueError("Quaternion alignment requires four action/state rotation indices")
            if self.quaternion_norm_epsilon is None or self.quaternion_norm_epsilon < 0:
                raise ValueError("Quaternion alignment requires a non-negative norm epsilon")
        if self.rotation_representation == "euler" and not self.action_rotation_indices:
            raise ValueError("Euler alignment requires rotation indices")
        if self.rotation_representation == "none" and self.rotation_threshold is not None:
            raise ValueError("rotation_threshold requires an Euler or quaternion representation")
        if self.translation_threshold is not None and not self.action_translation_indices:
            raise ValueError("translation_threshold requires translation indices")
        if self.candidate_policy not in {"review", "mask_field"}:
            raise ValueError(f"unsupported absolute candidate policy: {self.candidate_policy!r}")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("absolute candidate provenance must be a non-empty string")
        if not isinstance(self.provenance_deterministic, bool):
            raise TypeError("provenance_deterministic must be a boolean")
        if self.candidate_policy == "mask_field" and not self.provenance_deterministic:
            raise ValueError("mask_field requires deterministic converter provenance")
        if self.candidate_policy == "mask_field" and not self.mask_field_authority:
            raise ValueError(
                "mask_field requires independent authority for the field being masked; "
                "a deterministic action/state pairing alone does not identify the corrupt side"
            )
        if self.candidate_policy == "mask_field":
            if not isinstance(self.mask_source_field, str) or not self.mask_source_field.strip():
                raise ValueError("mask_field requires an exact mask_source_field")
            if self.mask_source_field not in {
                self.resolved_action_source_field,
                self.resolved_state_source_field,
            }:
                raise ValueError("mask_source_field must exactly match one compared source field")
        elif self.mask_field_authority is not None or self.mask_source_field is not None:
            raise ValueError(
                "mask_field_authority and mask_source_field are only valid with mask_field policy"
            )

    @property
    def resolved_action_source_field(self) -> str:
        return self.action_source_field or self.action_field

    @property
    def resolved_state_source_field(self) -> str:
        return self.state_source_field or self.state_field


@dataclass(frozen=True, slots=True)
class AlignmentResiduals:
    """Pair-aligned residual vectors and the resulting review mask."""

    contract: AlignmentContract
    action_indices: np.ndarray
    state_indices: np.ndarray
    translation_error: np.ndarray
    rotation_error: np.ndarray
    valid_mask: np.ndarray
    candidate_mask: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "action_indices": json_safe(self.action_indices),
            "state_indices": json_safe(self.state_indices),
            "translation_error": json_safe(self.translation_error),
            "rotation_error": json_safe(self.rotation_error),
            "valid_mask": json_safe(self.valid_mask),
            "candidate_mask": json_safe(self.candidate_mask),
        }


@dataclass(frozen=True, slots=True)
class LagEstimate:
    """Cross-correlation curve and its best state-lag estimate."""

    lag_steps: int
    lag_seconds: float
    correlation: float
    candidate_lag_steps: np.ndarray
    correlations: np.ndarray
    overlaps: np.ndarray
    sample_period_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_steps": self.lag_steps,
            "lag_seconds": self.lag_seconds,
            "correlation": json_safe(self.correlation),
            "candidate_lag_steps": json_safe(self.candidate_lag_steps),
            "correlations": json_safe(self.correlations),
            "overlaps": json_safe(self.overlaps),
            "sample_period_seconds": self.sample_period_seconds,
            "lag_sign_convention": "positive means state lags action",
        }


@dataclass(frozen=True, slots=True)
class DirectionalAgreement:
    """Active-motion sign agreement after applying an integer lag."""

    lag_steps: int
    per_dimension: np.ndarray
    active_counts: np.ndarray
    matching_counts: np.ndarray
    aggregate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "lag_steps": self.lag_steps,
            "per_dimension": json_safe(self.per_dimension),
            "active_counts": json_safe(self.active_counts),
            "matching_counts": json_safe(self.matching_counts),
            "aggregate": json_safe(self.aggregate),
        }


@dataclass(frozen=True, slots=True)
class AlignmentCheckConfig:
    """Optional absolute residual, lag, and directional-agreement checks."""

    absolute: AbsoluteAlignmentConfig | None = None
    motion_action_indices: tuple[int, ...] = ()
    motion_state_indices: tuple[int, ...] = ()
    motion_action_field: str = "action"
    motion_state_field: str = "state"
    lag_max_seconds: float | None = None
    lag_minimum_overlap: int = 3
    lag_review_threshold_seconds: float | None = None
    activity_threshold: float | tuple[float, ...] | None = None
    directional_agreement_review_below: float | None = None

    def __post_init__(self) -> None:
        if len(self.motion_action_indices) != len(self.motion_state_indices):
            raise ValueError("Action/state motion index counts must match")
        if not isinstance(self.motion_action_field, str) or not self.motion_action_field.strip():
            raise ValueError("motion_action_field must be a non-empty string")
        if not isinstance(self.motion_state_field, str) or not self.motion_state_field.strip():
            raise ValueError("motion_state_field must be a non-empty string")
        if self.lag_max_seconds is not None and self.lag_max_seconds < 0:
            raise ValueError("lag_max_seconds must be non-negative")
        if self.lag_minimum_overlap < 2:
            raise ValueError("lag_minimum_overlap must be at least two")
        if self.lag_review_threshold_seconds is not None and self.lag_review_threshold_seconds < 0:
            raise ValueError("lag_review_threshold_seconds must be non-negative")
        if self.directional_agreement_review_below is not None and not (
            0 <= self.directional_agreement_review_below <= 1
        ):
            raise ValueError("directional_agreement_review_below must be in [0, 1]")
        if self.lag_max_seconds is not None and not self.motion_action_indices:
            raise ValueError("Lag checks require configured motion indices")
        if self.activity_threshold is not None and self.lag_max_seconds is None:
            raise ValueError("Directional agreement requires lag estimation")


def _as_2d(values: np.ndarray | Sequence[Sequence[float]], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 1-D or 2-D time series, got shape={array.shape}")
    return array


def _check_indices(indices: tuple[int, ...], width: int, name: str) -> None:
    if any(index < 0 or index >= width for index in indices):
        raise ValueError(f"{name} contains an out-of-range feature index for width={width}")


def absolute_state_residual_mask(
    actions: np.ndarray | Sequence[Sequence[float]],
    states: np.ndarray | Sequence[Sequence[float]],
    *,
    config: AbsoluteAlignmentConfig,
) -> AlignmentResiduals:
    """Compare absolute actions with the state selected by a timing contract.

    Both supported contracts pair ``action[t]`` with ``state[t + 1]``.  They
    preserve different provenance wording/timing semantics in the returned
    object so callers cannot silently conflate dataset conventions.
    """

    action_array = _as_2d(actions, "actions")
    state_array = _as_2d(states, "states")
    _check_indices(config.action_translation_indices, action_array.shape[1], "action_translation_indices")
    _check_indices(config.state_translation_indices, state_array.shape[1], "state_translation_indices")
    _check_indices(config.action_rotation_indices, action_array.shape[1], "action_rotation_indices")
    _check_indices(config.state_rotation_indices, state_array.shape[1], "state_rotation_indices")

    pair_count = min(action_array.shape[0], max(state_array.shape[0] - 1, 0))
    action_indices = np.arange(pair_count, dtype=np.int64)
    state_indices = action_indices + 1
    action_rows = action_array[action_indices]
    state_rows = state_array[state_indices]
    translation_error = np.full(pair_count, np.nan, dtype=np.float64)
    rotation_error = np.full(pair_count, np.nan, dtype=np.float64)
    valid = np.isfinite(action_rows).all(axis=1) & np.isfinite(state_rows).all(axis=1)
    candidate = np.zeros(pair_count, dtype=bool)

    if config.action_translation_indices:
        delta = (
            action_rows[:, config.action_translation_indices]
            - state_rows[:, config.state_translation_indices]
        )
        translation_error = np.linalg.norm(delta, axis=1)
        if config.translation_threshold is not None:
            candidate |= valid & (translation_error > config.translation_threshold)

    if config.rotation_representation == "euler":
        delta = wrapped_euler_delta(
            action_rows[:, config.action_rotation_indices],
            state_rows[:, config.state_rotation_indices],
        )
        rotation_error = np.linalg.norm(delta, axis=1)
    elif config.rotation_representation == "quaternion":
        norm_epsilon = config.quaternion_norm_epsilon
        if norm_epsilon is None:
            raise ValueError("Quaternion alignment requires a norm epsilon")
        rotation_error = quaternion_geodesic_angle(
            action_rows[:, config.action_rotation_indices],
            state_rows[:, config.state_rotation_indices],
            norm_epsilon=norm_epsilon,
        )
    if config.rotation_threshold is not None:
        candidate |= valid & np.isfinite(rotation_error) & (rotation_error > config.rotation_threshold)
    return AlignmentResiduals(
        contract=config.contract,
        action_indices=action_indices,
        state_indices=state_indices,
        translation_error=translation_error,
        rotation_error=rotation_error,
        valid_mask=valid,
        candidate_mask=candidate,
    )


def _lag_aligned(action: np.ndarray, state: np.ndarray, lag_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Align arrays so positive lag means the state follows the action."""

    if lag_steps > 0:
        return action[:-lag_steps], state[lag_steps:]
    if lag_steps < 0:
        return action[-lag_steps:], state[:lag_steps]
    return action, state


def _uniform_resample(signal: np.ndarray, timestamps: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.interp(grid, timestamps, signal[:, dimension]) for dimension in range(signal.shape[1])]
    )


def estimate_lag_cross_correlation(
    action_signal: np.ndarray | Sequence[Sequence[float]],
    state_signal: np.ndarray | Sequence[Sequence[float]],
    timestamps: np.ndarray | Sequence[float],
    *,
    max_lag_seconds: float,
    minimum_overlap: int,
) -> LagEstimate:
    """Estimate action/state lag after resampling irregular data to its median period."""

    action = _as_2d(action_signal, "action_signal")
    state = _as_2d(state_signal, "state_signal")
    time_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if action.shape != state.shape or action.shape[0] != time_array.size:
        raise ValueError("action_signal, state_signal, and timestamps must be frame-aligned")
    if action.shape[1] == 0:
        raise ValueError("Signals must have at least one dimension")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("Lag estimation requires finite signals")
    if not np.isfinite(time_array).all() or np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")
    if max_lag_seconds < 0 or minimum_overlap < 2:
        raise ValueError("max_lag_seconds must be non-negative and minimum_overlap at least two")
    if time_array.size < minimum_overlap:
        raise ValueError("Not enough samples for the requested lag overlap")

    sample_period = float(np.median(np.diff(time_array)))
    grid_count = int(np.floor((time_array[-1] - time_array[0]) / sample_period)) + 1
    if grid_count < minimum_overlap:
        raise ValueError("Uniform resampling leaves fewer samples than minimum_overlap")
    grid = time_array[0] + np.arange(grid_count, dtype=np.float64) * sample_period
    action_uniform = _uniform_resample(action, time_array, grid)
    state_uniform = _uniform_resample(state, time_array, grid)
    max_steps = min(int(np.floor(max_lag_seconds / sample_period)), grid_count - minimum_overlap)
    lag_steps = np.arange(-max_steps, max_steps + 1, dtype=np.int64)
    correlations = np.full(lag_steps.size, np.nan, dtype=np.float64)
    overlaps = np.zeros(lag_steps.size, dtype=np.int64)
    for output_index, lag in enumerate(lag_steps):
        action_aligned, state_aligned = _lag_aligned(action_uniform, state_uniform, int(lag))
        overlaps[output_index] = action_aligned.shape[0]
        dimension_correlations: list[float] = []
        for dimension in range(action_aligned.shape[1]):
            left = action_aligned[:, dimension] - np.mean(action_aligned[:, dimension])
            right = state_aligned[:, dimension] - np.mean(state_aligned[:, dimension])
            denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denominator > 0:
                dimension_correlations.append(float(np.dot(left, right) / denominator))
        if dimension_correlations:
            correlations[output_index] = float(np.mean(dimension_correlations))
    finite_indices = np.flatnonzero(np.isfinite(correlations))
    if finite_indices.size == 0:
        raise ValueError("Lag correlation is undefined for constant or degenerate signals")
    best_index = max(
        finite_indices.tolist(),
        key=lambda index: (correlations[index], -abs(int(lag_steps[index])), -int(lag_steps[index])),
    )
    best_lag = int(lag_steps[best_index])
    return LagEstimate(
        lag_steps=best_lag,
        lag_seconds=best_lag * sample_period,
        correlation=float(correlations[best_index]),
        candidate_lag_steps=lag_steps,
        correlations=correlations,
        overlaps=overlaps,
        sample_period_seconds=sample_period,
    )


def active_motion_directional_agreement(
    action_derivative: np.ndarray | Sequence[Sequence[float]],
    state_derivative: np.ndarray | Sequence[Sequence[float]],
    *,
    lag_steps: int,
    activity_threshold: float | Sequence[float],
) -> DirectionalAgreement:
    """Measure sign agreement only where the action derivative is active."""

    action = _as_2d(action_derivative, "action_derivative")
    state = _as_2d(state_derivative, "state_derivative")
    if action.shape != state.shape:
        raise ValueError("Action/state derivatives must have equal shapes")
    thresholds = np.asarray(activity_threshold, dtype=np.float64)
    if thresholds.ndim == 0:
        thresholds = np.full(action.shape[1], float(thresholds))
    thresholds = thresholds.reshape(-1)
    if thresholds.size != action.shape[1] or np.any(thresholds < 0) or not np.isfinite(thresholds).all():
        raise ValueError("activity_threshold must be finite, non-negative, and dimension-aligned")
    action_aligned, state_aligned = _lag_aligned(action, state, lag_steps)
    finite = np.isfinite(action_aligned) & np.isfinite(state_aligned)
    active = finite & (np.abs(action_aligned) >= thresholds[None, :])
    matching = active & (np.sign(action_aligned) == np.sign(state_aligned))
    active_counts = active.sum(axis=0).astype(np.int64)
    matching_counts = matching.sum(axis=0).astype(np.int64)
    per_dimension = np.full(action.shape[1], np.nan, dtype=np.float64)
    np.divide(matching_counts, active_counts, out=per_dimension, where=active_counts > 0)
    total_active = int(active_counts.sum())
    aggregate = float(matching_counts.sum() / total_active) if total_active else float("nan")
    return DirectionalAgreement(lag_steps, per_dimension, active_counts, matching_counts, aggregate)


def check_action_state_alignment(
    actions: np.ndarray | Sequence[Sequence[float]],
    states: np.ndarray | Sequence[Sequence[float]],
    timestamps: np.ndarray | Sequence[float],
    *,
    episode_id: str | int | None = None,
    config: AlignmentCheckConfig | None = None,
) -> QualityCheckResult:
    """Run configured residual and lag checks for one episode."""

    config = AlignmentCheckConfig() if config is None else config
    action_array = _as_2d(actions, "actions")
    state_array = _as_2d(states, "states")
    time_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    result = QualityCheckResult()
    if config.absolute is not None:
        residuals = absolute_state_residual_mask(action_array, state_array, config=config.absolute)
        candidate_indices = residuals.action_indices[residuals.candidate_mask]
        result.metrics["absolute"] = {
            "contract": residuals.contract,
            "candidate_policy": config.absolute.candidate_policy,
            "provenance": config.absolute.provenance,
            "provenance_deterministic": config.absolute.provenance_deterministic,
            "mask_field_authority": config.absolute.mask_field_authority,
            "mask_source_field": config.absolute.mask_source_field,
            "source_pair": {
                "action_field": config.absolute.action_field,
                "state_field": config.absolute.state_field,
                "action_source_field": config.absolute.resolved_action_source_field,
                "state_source_field": config.absolute.resolved_state_source_field,
            },
            "pair_count": residuals.action_indices.size,
            "valid_count": int(residuals.valid_mask.sum()),
            "candidate_count": int(residuals.candidate_mask.sum()),
            "candidate_action_indices": candidate_indices,
            "candidate_state_indices": residuals.state_indices[residuals.candidate_mask],
            "translation_error_max": (
                float(np.nanmax(residuals.translation_error))
                if np.isfinite(residuals.translation_error).any()
                else None
            ),
            "rotation_error_max": (
                float(np.nanmax(residuals.rotation_error))
                if np.isfinite(residuals.rotation_error).any()
                else None
            ),
        }
        for start, end in true_runs(residuals.candidate_mask):
            action_start = int(residuals.action_indices[start])
            action_end = int(residuals.action_indices[end - 1]) + 1
            state_start = int(residuals.state_indices[start])
            state_end = int(residuals.state_indices[end - 1]) + 1
            field_intervals = {
                config.absolute.resolved_action_source_field: {
                    "start_index": action_start,
                    "end_index": action_end,
                },
                config.absolute.resolved_state_source_field: {
                    "start_index": state_start,
                    "end_index": state_end,
                },
            }
            result.findings.append(
                QualityFinding(
                    rule_id="ALG.ABSOLUTE_RESIDUAL",
                    severity="medium",
                    disposition=("mask" if config.absolute.candidate_policy == "mask_field" else "review"),
                    message="Absolute action/state residual exceeds a source-configured threshold",
                    episode_id=episode_id,
                    start_index=action_start,
                    end_index=action_end,
                    field=f"{config.absolute.action_field}<->{config.absolute.state_field}",
                    deterministic=config.absolute.provenance_deterministic,
                    metrics={
                        "contract": residuals.contract,
                        "candidate_policy": config.absolute.candidate_policy,
                        "provenance": config.absolute.provenance,
                        "provenance_deterministic": config.absolute.provenance_deterministic,
                        "mask_field_authority": config.absolute.mask_field_authority,
                        "mask_source_field": config.absolute.mask_source_field,
                        "source_pair": {
                            "action_field": config.absolute.action_field,
                            "state_field": config.absolute.state_field,
                            "action_source_field": config.absolute.resolved_action_source_field,
                            "state_source_field": config.absolute.resolved_state_source_field,
                            "action_start_index": action_start,
                            "action_end_index": action_end,
                            "state_start_index": state_start,
                            "state_end_index": state_end,
                            "field_intervals": field_intervals,
                        },
                        "translation_error_max": (
                            float(np.nanmax(residuals.translation_error[start:end]))
                            if np.isfinite(residuals.translation_error[start:end]).any()
                            else None
                        ),
                        "rotation_error_max": (
                            float(np.nanmax(residuals.rotation_error[start:end]))
                            if np.isfinite(residuals.rotation_error[start:end]).any()
                            else None
                        ),
                    },
                )
            )

    if config.lag_max_seconds is not None:
        if action_array.shape[0] != state_array.shape[0] or time_array.size != action_array.shape[0]:
            result.metrics["lag.skipped_reason"] = "lag inputs are not frame-aligned"
            return result
        _check_indices(config.motion_action_indices, action_array.shape[1], "motion_action_indices")
        _check_indices(config.motion_state_indices, state_array.shape[1], "motion_state_indices")
        dt = np.diff(time_array)
        if not np.isfinite(time_array).all() or np.any(dt <= 0):
            result.metrics["lag.skipped_reason"] = "timestamps are not finite and strictly increasing"
            return result
        action_motion = np.diff(action_array[:, config.motion_action_indices], axis=0) / dt[:, None]
        state_motion = np.diff(state_array[:, config.motion_state_indices], axis=0) / dt[:, None]
        motion_times = (time_array[:-1] + time_array[1:]) / 2.0
        try:
            lag = estimate_lag_cross_correlation(
                action_motion,
                state_motion,
                motion_times,
                max_lag_seconds=config.lag_max_seconds,
                minimum_overlap=config.lag_minimum_overlap,
            )
        except ValueError as error:
            result.metrics["lag.skipped_reason"] = str(error)
            return result
        result.metrics["lag"] = lag.to_dict()
        source_pair = {
            "action_field": config.motion_action_field,
            "state_field": config.motion_state_field,
            "field_intervals": {
                config.motion_action_field: {
                    "start_index": 0,
                    "end_index": action_array.shape[0],
                },
                config.motion_state_field: {
                    "start_index": 0,
                    "end_index": state_array.shape[0],
                },
            },
        }
        if (
            config.lag_review_threshold_seconds is not None
            and abs(lag.lag_seconds) > config.lag_review_threshold_seconds
            and action_array.shape[0] > 0
        ):
            result.findings.append(
                QualityFinding(
                    rule_id="ALG.LAG",
                    severity="medium",
                    disposition="review",
                    message="Best action/state lag exceeds the configured review threshold",
                    episode_id=episode_id,
                    start_index=0,
                    end_index=action_array.shape[0],
                    field=f"{config.motion_action_field}<->{config.motion_state_field}",
                    deterministic=False,
                    metrics={**lag.to_dict(), "source_pair": source_pair},
                )
            )
        if config.activity_threshold is not None:
            agreement = active_motion_directional_agreement(
                action_motion,
                state_motion,
                lag_steps=lag.lag_steps,
                activity_threshold=config.activity_threshold,
            )
            result.metrics["directional_agreement"] = agreement.to_dict()
            if (
                config.directional_agreement_review_below is not None
                and np.isfinite(agreement.aggregate)
                and agreement.aggregate < config.directional_agreement_review_below
                and action_array.shape[0] > 0
            ):
                result.findings.append(
                    QualityFinding(
                        rule_id="ALG.DIRECTIONAL_AGREEMENT",
                        severity="medium",
                        disposition="review",
                        message="Active-motion directional agreement is below the configured threshold",
                        episode_id=episode_id,
                        start_index=0,
                        end_index=action_array.shape[0],
                        field=f"{config.motion_action_field}<->{config.motion_state_field}",
                        deterministic=False,
                        metrics={**agreement.to_dict(), "source_pair": source_pair},
                    )
                )
    return result
