"""Pure NumPy trajectory checks with time-aware motion statistics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import QualityCheckResult, QualityFinding, TimeInterval, json_safe, true_runs


@dataclass(frozen=True, slots=True)
class DerivativeSeries:
    """First through third finite differences and their physical timestamps."""

    velocity: np.ndarray
    velocity_timestamps: np.ndarray
    acceleration: np.ndarray
    acceleration_timestamps: np.ndarray
    jerk: np.ndarray
    jerk_timestamps: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "velocity": json_safe(self.velocity),
            "velocity_timestamps": json_safe(self.velocity_timestamps),
            "acceleration": json_safe(self.acceleration),
            "acceleration_timestamps": json_safe(self.acceleration_timestamps),
            "jerk": json_safe(self.jerk),
            "jerk_timestamps": json_safe(self.jerk_timestamps),
        }


@dataclass(frozen=True, slots=True)
class RobustReference:
    """Median/MAD reference for one named stratum."""

    center: float
    mad: float


@dataclass(frozen=True, slots=True)
class RobustScoreResult:
    """Per-value robust scores and optional candidate mask."""

    scores: np.ndarray
    candidate_mask: np.ndarray
    centers: Mapping[str, float]
    mads: Mapping[str, float]
    counts: Mapping[str, int]
    threshold: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scores": json_safe(self.scores),
            "candidate_mask": json_safe(self.candidate_mask),
            "centers": json_safe(self.centers),
            "mads": json_safe(self.mads),
            "counts": json_safe(self.counts),
            "threshold": json_safe(self.threshold),
        }


@dataclass(frozen=True, slots=True)
class TrajectoryCheckConfig:
    """Explicit source thresholds for :func:`check_trajectory`.

    ``near_static_*`` values are speeds in physical units per second.  A rule
    is disabled when its threshold is ``None``.  ``robust_z_threshold`` creates
    review signals from either explicit frozen references or episode-local
    statistics; it never authorizes automatic deletion. Distinct signals are
    projected back to source-frame spans and must meet
    ``minimum_corroborating_signals`` before a robust review finding is emitted.
    """

    translation_indices: tuple[int, ...] = ()
    euler_indices: tuple[int, ...] = ()
    quaternion_indices: tuple[int, ...] = ()
    rotation_vector_indices: tuple[int, ...] = ()
    quaternion_order: str = "xyzw"
    quaternion_norm_epsilon: float | None = None
    quaternion_unit_tolerance: float | None = None
    near_static_translation_speed: float | None = None
    near_static_rotation_speed: float | None = None
    near_static_min_duration_seconds: float | None = None
    robust_z_threshold: float | None = None
    robust_scale_floor: float | None = None
    minimum_corroborating_signals: int = 1
    review_translation_jump_m: float | None = None
    review_rotation_jump_rad: float | None = None
    check_velocity: bool = True
    check_acceleration: bool = True
    check_jerk: bool = True
    robust_references: Mapping[str, Mapping[str, RobustReference]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quaternion_order not in {"xyzw", "wxyz"}:
            raise ValueError("quaternion_order must be 'xyzw' or 'wxyz'")
        if self.quaternion_indices and len(self.quaternion_indices) != 4:
            raise ValueError("quaternion_indices must identify exactly four components")
        if self.rotation_vector_indices and len(self.rotation_vector_indices) != 3:
            raise ValueError("rotation_vector_indices must identify exactly three components")
        for name in (
            "quaternion_norm_epsilon",
            "quaternion_unit_tolerance",
            "near_static_translation_speed",
            "near_static_rotation_speed",
            "near_static_min_duration_seconds",
            "robust_z_threshold",
            "robust_scale_floor",
            "review_translation_jump_m",
            "review_rotation_jump_rad",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.minimum_corroborating_signals < 1:
            raise ValueError("minimum_corroborating_signals must be >= 1")
        for name in ("review_translation_jump_m", "review_rotation_jump_rad"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive when set")
        if (
            self.near_static_translation_speed is not None or self.near_static_rotation_speed is not None
        ) and (self.near_static_min_duration_seconds is None or self.near_static_min_duration_seconds <= 0):
            raise ValueError("near-static checks require a positive minimum duration")


def canonicalize_quaternions(
    quaternions: np.ndarray | Sequence[Sequence[float]],
    *,
    norm_epsilon: float,
    order: str = "xyzw",
) -> np.ndarray:
    """Normalize quaternion signs so equivalent ``q``/``-q`` map identically.

    Rows whose norm is not finite or does not exceed ``norm_epsilon`` are
    returned as ``NaN``; callers can then emit deterministic contract findings.
    """

    if norm_epsilon < 0:
        raise ValueError("norm_epsilon must be non-negative")
    if order not in {"xyzw", "wxyz"}:
        raise ValueError("order must be 'xyzw' or 'wxyz'")
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dimension 4, got shape={values.shape}")

    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    valid = np.isfinite(values).all(axis=-1, keepdims=True) & np.isfinite(norms) & (norms > norm_epsilon)
    normalized = np.full(values.shape, np.nan, dtype=np.float64)
    np.divide(values, norms, out=normalized, where=valid)

    flattened = normalized.reshape(-1, 4)
    valid_rows = valid.reshape(-1)
    scalar_index = 3 if order == "xyzw" else 0
    for row_index in np.flatnonzero(valid_rows):
        row = flattened[row_index]
        sign = 1.0
        if row[scalar_index] < 0:
            sign = -1.0
        elif row[scalar_index] == 0:
            nonzero = np.flatnonzero(row != 0)
            if nonzero.size and row[nonzero[0]] < 0:
                sign = -1.0
        flattened[row_index] *= sign
    return normalized


def quaternion_geodesic_angle(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
    *,
    norm_epsilon: float,
) -> np.ndarray:
    """Return the shortest SO(3) angle in radians, treating ``q`` and ``-q`` equally."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape[-1:] != (4,) or right_array.shape[-1:] != (4,):
        raise ValueError("Quaternion inputs must have last dimension four")
    left_array, right_array = np.broadcast_arrays(left_array, right_array)
    left_norm = np.linalg.norm(left_array, axis=-1, keepdims=True)
    right_norm = np.linalg.norm(right_array, axis=-1, keepdims=True)
    valid = (
        np.isfinite(left_array).all(axis=-1)
        & np.isfinite(right_array).all(axis=-1)
        & (left_norm[..., 0] > norm_epsilon)
        & (right_norm[..., 0] > norm_epsilon)
    )
    dot = np.full(valid.shape, np.nan, dtype=np.float64)
    normalized_dot = np.full(valid.shape, np.nan, dtype=np.float64)
    denominator = left_norm[..., 0] * right_norm[..., 0]
    np.divide(
        np.sum(left_array * right_array, axis=-1),
        denominator,
        out=normalized_dot,
        where=valid,
    )
    dot[valid] = np.clip(np.abs(normalized_dot[valid]), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def rotation_vector_to_quaternion(
    rotation_vectors: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Convert axis-angle rotation vectors to unit quaternions in ``xyzw`` order.

    A rotation vector's direction is its axis and its norm is the angle in
    radians.  Conversion before differencing is important near equivalent
    ``2*pi`` representations, where Euclidean component subtraction can report
    a large jump for the same physical rotation.
    """

    values = np.asarray(rotation_vectors, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(f"Expected rotation-vector last dimension 3, got shape={values.shape}")
    flattened = values.reshape(-1, 3)
    quaternions = np.full((flattened.shape[0], 4), np.nan, dtype=np.float64)
    finite = np.isfinite(flattened).all(axis=1)
    if finite.any():
        vectors = flattened[finite]
        angles = np.linalg.norm(vectors, axis=1)
        half_angles = angles / 2.0
        scale = np.empty_like(angles)
        small = angles < 1e-8
        # sin(theta / 2) / theta, expanded around zero to avoid 0 / 0.
        scale[small] = 0.5 - (angles[small] ** 2) / 48.0
        scale[~small] = np.sin(half_angles[~small]) / angles[~small]
        quaternions[finite, :3] = vectors * scale[:, None]
        quaternions[finite, 3] = np.cos(half_angles)
    return quaternions.reshape((*values.shape[:-1], 4))


def rotation_vector_geodesic_angle(
    left: np.ndarray | Sequence[float],
    right: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Return the shortest SO(3) angle between two rotation vectors."""

    return quaternion_geodesic_angle(
        rotation_vector_to_quaternion(left),
        rotation_vector_to_quaternion(right),
        norm_epsilon=1e-12,
    )


def wrapped_euler_delta(
    current: np.ndarray | Sequence[float],
    previous: np.ndarray | Sequence[float],
    *,
    period: float | np.ndarray = 2.0 * np.pi,
) -> np.ndarray:
    """Compute shortest signed Euler-coordinate deltas under a chosen period."""

    current_array = np.asarray(current, dtype=np.float64)
    previous_array = np.asarray(previous, dtype=np.float64)
    period_array = np.asarray(period, dtype=np.float64)
    if np.any(period_array <= 0) or not np.isfinite(period_array).all():
        raise ValueError("period must contain finite positive values")
    delta = current_array - previous_array
    return (delta + period_array / 2.0) % period_array - period_array / 2.0


def _differentiate(values: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.shape[0] < 2:
        return np.empty((0, *values.shape[1:]), dtype=np.float64), np.empty(0, dtype=np.float64)
    dt = np.diff(timestamps)
    derivative = np.diff(values, axis=0) / dt.reshape((-1,) + (1,) * (values.ndim - 1))
    return derivative, (timestamps[:-1] + timestamps[1:]) / 2.0


def compute_dt_derivatives(
    values: np.ndarray | Sequence[Sequence[float]],
    timestamps: np.ndarray | Sequence[float],
    *,
    wrap_period: float | np.ndarray | None = None,
) -> DerivativeSeries:
    """Compute velocity, acceleration, and jerk on possibly irregular timestamps."""

    value_array = np.asarray(values, dtype=np.float64)
    time_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if value_array.ndim == 1:
        value_array = value_array[:, None]
    if value_array.ndim < 2:
        raise ValueError("values must have a leading sample dimension")
    if value_array.shape[0] != time_array.size:
        raise ValueError("values and timestamps must have equal leading lengths")
    if not np.isfinite(time_array).all() or np.any(np.diff(time_array) <= 0):
        raise ValueError("timestamps must be finite and strictly increasing")

    if value_array.shape[0] < 2:
        velocity = np.empty((0, *value_array.shape[1:]), dtype=np.float64)
        velocity_times = np.empty(0, dtype=np.float64)
    else:
        dt = np.diff(time_array)
        delta = (
            np.diff(value_array, axis=0)
            if wrap_period is None
            else wrapped_euler_delta(value_array[1:], value_array[:-1], period=wrap_period)
        )
        velocity = delta / dt.reshape((-1,) + (1,) * (value_array.ndim - 1))
        velocity_times = (time_array[:-1] + time_array[1:]) / 2.0
    acceleration, acceleration_times = _differentiate(velocity, velocity_times)
    jerk, jerk_times = _differentiate(acceleration, acceleration_times)
    return DerivativeSeries(
        velocity=velocity,
        velocity_timestamps=velocity_times,
        acceleration=acceleration,
        acceleration_timestamps=acceleration_times,
        jerk=jerk,
        jerk_timestamps=jerk_times,
    )


def find_near_static_runs(
    transition_motion: np.ndarray | Sequence[float],
    frame_timestamps: np.ndarray | Sequence[float],
    *,
    motion_threshold: float,
    min_duration_seconds: float,
) -> list[TimeInterval]:
    """Find static transition runs by elapsed seconds rather than frame count.

    ``transition_motion[i]`` describes the movement from frame ``i`` to
    ``i + 1``; therefore ``frame_timestamps`` must contain one more value.
    Returned indices are transition indices, while time boundaries span the
    corresponding frames.
    """

    if motion_threshold < 0 or min_duration_seconds <= 0:
        raise ValueError("motion_threshold must be non-negative and duration must be positive")
    motion = np.asarray(transition_motion, dtype=np.float64).reshape(-1)
    timestamps = np.asarray(frame_timestamps, dtype=np.float64).reshape(-1)
    if timestamps.size != motion.size + 1:
        raise ValueError("frame_timestamps must have one more element than transition_motion")
    if not np.isfinite(timestamps).all() or np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame_timestamps must be finite and strictly increasing")
    mask = np.isfinite(motion) & (motion <= motion_threshold)
    intervals: list[TimeInterval] = []
    for start, end in true_runs(mask):
        start_time = float(timestamps[start])
        end_time = float(timestamps[end])
        if end_time - start_time >= min_duration_seconds:
            intervals.append(TimeInterval(start, end, start_time, end_time))
    return intervals


def robust_mad_scores(
    values: np.ndarray | Sequence[float],
    *,
    strata: np.ndarray | Sequence[Any] | None = None,
    candidate_threshold: float | None = None,
    references: Mapping[str, RobustReference] | None = None,
    scale_floor: float | None = None,
    normal_consistency: float = 1.4826,
) -> RobustScoreResult:
    """Score scalar metrics against per-stratum median/MAD references.

    A zero-MAD stratum yields score zero at the center and infinity away from
    the center unless a caller-provided ``scale_floor`` is used.  The function
    only returns candidates and never assigns a destructive disposition.
    """

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if strata is None:
        labels = np.full(array.size, "__all__", dtype=object)
    else:
        raw_labels = np.asarray(strata, dtype=object).reshape(-1)
        if raw_labels.size != array.size:
            raise ValueError("strata and values must have equal lengths")
        labels = np.asarray([str(label) for label in raw_labels], dtype=object)
    if candidate_threshold is not None and candidate_threshold < 0:
        raise ValueError("candidate_threshold must be non-negative")
    if scale_floor is not None and scale_floor < 0:
        raise ValueError("scale_floor must be non-negative")
    if normal_consistency <= 0:
        raise ValueError("normal_consistency must be positive")

    scores = np.full(array.shape, np.nan, dtype=np.float64)
    centers: dict[str, float] = {}
    mads: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in sorted(set(labels.tolist())):
        selected = labels == label
        finite = selected & np.isfinite(array)
        counts[label] = int(finite.sum())
        reference = None if references is None else references.get(label)
        if reference is not None:
            center, mad = float(reference.center), float(reference.mad)
        elif finite.any():
            center = float(np.median(array[finite]))
            mad = float(np.median(np.abs(array[finite] - center)))
        else:
            center = float("nan")
            mad = float("nan")
        centers[label] = center
        mads[label] = mad
        if not finite.any() or not np.isfinite(center) or not np.isfinite(mad):
            continue
        scale = normal_consistency * mad
        if scale_floor is not None:
            scale = max(scale, scale_floor)
        residual = np.abs(array[finite] - center)
        if scale > 0:
            scores[finite] = residual / scale
        else:
            scores[finite] = np.where(residual == 0, 0.0, np.inf)
    candidates = np.zeros(array.shape, dtype=bool)
    if candidate_threshold is not None:
        candidates = np.isfinite(scores) & (scores >= candidate_threshold) | np.isposinf(scores)
    return RobustScoreResult(scores, candidates, centers, mads, counts, candidate_threshold)


def _deterministic_finding(
    rule_id: str,
    message: str,
    *,
    episode_id: str | int | None,
    field: str,
    start: int | None = None,
    end: int | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> QualityFinding:
    return QualityFinding(
        rule_id=rule_id,
        severity="critical",
        disposition="quarantine",
        message=message,
        episode_id=episode_id,
        start_index=start,
        end_index=end,
        field=field,
        deterministic=True,
        metrics={} if metrics is None else metrics,
    )


def _candidate_findings(
    mask: np.ndarray,
    *,
    rule_id: str,
    message: str,
    episode_id: str | int | None,
    field: str,
    scores: np.ndarray | None = None,
    frame_span_extra: int = 1,
    extra_metrics: Mapping[str, Any] | None = None,
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for start, end in true_runs(mask):
        metrics: dict[str, Any] = {
            **({} if extra_metrics is None else dict(extra_metrics)),
            "candidate_count": end - start,
        }
        if scores is not None:
            metrics["max_score"] = float(np.nanmax(scores[start:end]))
        findings.append(
            QualityFinding(
                rule_id=rule_id,
                severity="medium",
                disposition="review",
                message=message,
                episode_id=episode_id,
                start_index=start,
                end_index=end + frame_span_extra,
                field=field,
                deterministic=False,
                metrics=metrics,
            )
        )
    return findings


@dataclass(frozen=True, slots=True)
class _TrajectoryReviewSignal:
    """One independent review signal before source-frame corroboration."""

    name: str
    candidate_mask: np.ndarray
    normalized_scores: np.ndarray
    frame_span_extra: int
    active_flat_indices: tuple[int, ...]
    reference_scope: str


def _project_signal_to_source_frames(
    signal: _TrajectoryReviewSignal,
    *,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Project derivative-index evidence onto its episode-local source frames."""

    mask = np.asarray(signal.candidate_mask, dtype=bool).reshape(-1)
    scores = np.asarray(signal.normalized_scores, dtype=np.float64).reshape(-1)
    if scores.shape != mask.shape:
        raise ValueError(f"signal {signal.name!r} scores and mask must have equal shapes")
    projected_mask = np.zeros(frame_count, dtype=bool)
    projected_scores = np.full(frame_count, np.nan, dtype=np.float64)
    for candidate_index in np.flatnonzero(mask):
        start = int(candidate_index)
        end = min(start + signal.frame_span_extra + 1, frame_count)
        if end <= start:
            continue
        projected_mask[start:end] = True
        score = scores[candidate_index]
        if np.isnan(score):
            continue
        existing = projected_scores[start:end]
        replace_mask = np.isnan(existing) | (score > existing)
        existing[replace_mask] = score
    return projected_mask, projected_scores


def _corroborated_review_findings(
    signals: Sequence[_TrajectoryReviewSignal],
    *,
    frame_count: int,
    minimum_signals: int,
    episode_id: str | int | None,
    field: str,
) -> tuple[dict[str, Any], list[QualityFinding]]:
    """Gate distinct signals and merge only frames with identical evidence."""

    if minimum_signals < 1:
        raise ValueError("minimum_signals must be >= 1")
    signal_names = [signal.name for signal in signals]
    if len(set(signal_names)) != len(signal_names):
        raise ValueError("duplicate signal kinds cannot count toward corroboration")
    projected = {
        signal.name: (*_project_signal_to_source_frames(signal, frame_count=frame_count), signal)
        for signal in signals
    }
    signal_count = np.zeros(frame_count, dtype=np.int16)
    for projected_mask, _projected_scores, _signal in projected.values():
        signal_count += projected_mask.astype(np.int16)
    candidate_union = signal_count > 0
    corroborated = signal_count >= minimum_signals
    suppressed = candidate_union & ~corroborated

    # A merely contiguous corroborated mask is not enough to define an exact
    # evidence interval.  If the active signals change within that mask, their
    # component indices may change too.  Split on the complete active signal
    # set so a finding never authorizes a component on frames where that
    # component had no evidence.
    evidence_runs: list[tuple[int, int, tuple[str, ...]]] = []
    run_start: int | None = None
    run_signal_names: tuple[str, ...] | None = None
    for frame_index in range(frame_count + 1):
        active_names = (
            tuple(
                sorted(
                    name
                    for name, (projected_mask, _projected_scores, _signal) in projected.items()
                    if projected_mask[frame_index]
                )
            )
            if frame_index < frame_count and corroborated[frame_index]
            else None
        )
        if active_names == run_signal_names:
            continue
        if run_start is not None and run_signal_names is not None:
            evidence_runs.append((run_start, frame_index, run_signal_names))
        run_start = frame_index if active_names is not None else None
        run_signal_names = active_names

    contiguous_corroborated_runs = true_runs(corroborated)
    metrics: dict[str, Any] = {
        "minimum_corroborating_signals": minimum_signals,
        "distinct_signal_names": sorted(projected),
        "raw_candidate_count_by_signal": {
            signal.name: int(np.asarray(signal.candidate_mask, dtype=bool).sum()) for signal in signals
        },
        "projected_source_frame_count_by_signal": {
            name: int(projected_mask.sum())
            for name, (projected_mask, _projected_scores, _signal) in projected.items()
        },
        "corroborated_source_frame_count": int(corroborated.sum()),
        "corroborated_interval_count": len(evidence_runs),
        "corroborated_contiguous_interval_count": len(contiguous_corroborated_runs),
        "suppressed_uncorroborated_source_frame_count": int(suppressed.sum()),
        "suppressed_uncorroborated_interval_count": len(true_runs(suppressed)),
        "suppressed_uncorroborated_source_frame_count_by_signal": {
            name: int((projected_mask & suppressed).sum())
            for name, (projected_mask, _projected_scores, _signal) in projected.items()
        },
    }
    if suppressed.any():
        metrics["suppressed_uncorroborated_reason"] = (
            "source-frame evidence within the episode did not reach the configured number of distinct signals"
        )

    findings: list[QualityFinding] = []
    for start, end, run_signal_names in evidence_runs:
        active = {name: projected[name] for name in run_signal_names}
        per_signal_max_score: dict[str, float] = {}
        for name, (projected_mask, projected_scores, _signal) in active.items():
            selected = projected_scores[start:end][projected_mask[start:end]]
            finite_or_infinite = selected[~np.isnan(selected)]
            if finite_or_infinite.size:
                per_signal_max_score[name] = float(np.max(finite_or_infinite))
        active_indices = sorted(
            {
                index
                for _projected_mask, _projected_scores, signal in active.values()
                for index in signal.active_flat_indices
            }
        )
        signal_names = sorted(active)
        max_score = max(per_signal_max_score.values(), default=1.0)
        findings.append(
            QualityFinding(
                rule_id="TRJ.ROBUST_CANDIDATE",
                severity="medium",
                disposition="review",
                message=("Corroborated source-frame trajectory review signals: " + ", ".join(signal_names)),
                episode_id=episode_id,
                start_index=start,
                end_index=end,
                field=field,
                deterministic=False,
                metrics={
                    "active_flat_indices": active_indices,
                    "candidate_count": end - start,
                    "signal_names": signal_names,
                    "signal_count": len(signal_names),
                    "minimum_corroborating_signals": minimum_signals,
                    "minimum_simultaneous_signal_count": int(signal_count[start:end].min()),
                    "maximum_simultaneous_signal_count": int(signal_count[start:end].max()),
                    "per_signal_max_score": per_signal_max_score,
                    "signal_reference_scopes": {
                        name: signal.reference_scope for name, (_mask, _scores, signal) in active.items()
                    },
                    "max_score": max_score,
                },
            )
        )
    return metrics, findings


def check_trajectory(
    values: np.ndarray | Sequence[Sequence[float]],
    timestamps: np.ndarray | Sequence[float],
    *,
    episode_id: str | int | None = None,
    field: str = "trajectory",
    strata: np.ndarray | Sequence[Any] | None = None,
    config: TrajectoryCheckConfig | None = None,
) -> QualityCheckResult:
    """Check one episode's trajectory using an explicit source configuration."""

    config = TrajectoryCheckConfig() if config is None else config
    result = QualityCheckResult()
    array = np.asarray(values, dtype=np.float64)
    time_array = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if array.ndim != 2:
        result.findings.append(
            _deterministic_finding(
                "TRJ.SHAPE",
                "Trajectory must be a two-dimensional [time, feature] array",
                episode_id=episode_id,
                field=field,
                metrics={"shape": array.shape},
            )
        )
        return result
    if array.shape[0] != time_array.size:
        result.findings.append(
            _deterministic_finding(
                "TRJ.LENGTH",
                "Trajectory and timestamps have different lengths",
                episode_id=episode_id,
                field=field,
                metrics={"trajectory_length": array.shape[0], "timestamp_length": time_array.size},
            )
        )
        return result
    if array.shape[0] == 0:
        result.findings.append(
            _deterministic_finding(
                "TRJ.EMPTY",
                "Trajectory contains no frames",
                episode_id=episode_id,
                field=field,
            )
        )
        result.metrics["frame_count"] = 0
        return result
    if not np.isfinite(time_array).all() or np.any(np.diff(time_array) <= 0):
        result.findings.append(
            _deterministic_finding(
                "TRJ.TIMESTAMP",
                "Trajectory timestamps must be finite and strictly increasing",
                episode_id=episode_id,
                field="timestamp",
            )
        )
        return result
    finite_frames = np.isfinite(array).all(axis=1)
    for start, end in true_runs(~finite_frames):
        active_nonfinite_indices = np.flatnonzero(~np.isfinite(array[start:end]).all(axis=0)).tolist()
        result.findings.append(
            _deterministic_finding(
                "TRJ.NONFINITE",
                "Trajectory contains non-finite values",
                episode_id=episode_id,
                field=field,
                start=start,
                end=end,
                metrics={"active_flat_indices": active_nonfinite_indices},
            )
        )

    all_indices = (
        config.translation_indices
        + config.euler_indices
        + config.quaternion_indices
        + config.rotation_vector_indices
    )
    if any(index < 0 or index >= array.shape[1] for index in all_indices):
        result.findings.append(
            _deterministic_finding(
                "TRJ.INDEX_CONTRACT",
                "Configured trajectory feature index is out of bounds",
                episode_id=episode_id,
                field=field,
                metrics={"feature_count": array.shape[1], "indices": all_indices},
            )
        )
        return result

    result.metrics.update(
        {
            "frame_count": array.shape[0],
            "duration_seconds": float(time_array[-1] - time_array[0]) if time_array.size else 0.0,
            "finite_frame_count": int(finite_frames.sum()),
        }
    )
    if config.quaternion_indices:
        quaternions = array[:, config.quaternion_indices]
        norms = np.linalg.norm(quaternions, axis=1)
        result.metrics["quaternion.norm_min"] = float(np.nanmin(norms))
        result.metrics["quaternion.norm_max"] = float(np.nanmax(norms))
        if config.quaternion_norm_epsilon is not None:
            invalid = ~np.isfinite(norms) | (norms <= config.quaternion_norm_epsilon)
            for start, end in true_runs(invalid):
                result.findings.append(
                    _deterministic_finding(
                        "ROT.INVALID_QUATERNION",
                        "Quaternion norm violates the explicit representation contract",
                        episode_id=episode_id,
                        field=field,
                        start=start,
                        end=end,
                        metrics={
                            "active_flat_indices": list(config.quaternion_indices),
                            "norms": norms[start:end],
                        },
                    )
                )
        if config.quaternion_unit_tolerance is not None:
            off_unit = np.isfinite(norms) & (np.abs(norms - 1.0) > config.quaternion_unit_tolerance)
            result.findings.extend(
                _candidate_findings(
                    off_unit,
                    rule_id="ROT.NONUNIT_QUATERNION",
                    message="Quaternion norm is outside the configured review tolerance",
                    episode_id=episode_id,
                    field=field,
                    frame_span_extra=0,
                    extra_metrics={"active_flat_indices": list(config.quaternion_indices)},
                )
            )

    transition_dt = np.diff(time_array)
    translation_step: np.ndarray | None = None
    rotation_step: np.ndarray | None = None
    translation_speed: np.ndarray | None = None
    rotation_speed: np.ndarray | None = None
    translation_series: DerivativeSeries | None = None
    if config.translation_indices and array.shape[0] >= 2:
        translation = array[:, config.translation_indices]
        translation_step = np.linalg.norm(np.diff(translation, axis=0), axis=1)
        translation_series = compute_dt_derivatives(translation, time_array)
        translation_speed = np.linalg.norm(translation_series.velocity, axis=1)
        result.metrics["translation_speed.max"] = (
            float(np.nanmax(translation_speed)) if np.isfinite(translation_speed).any() else None
        )
        result.metrics["translation_speed.median"] = (
            float(np.nanmedian(translation_speed)) if np.isfinite(translation_speed).any() else None
        )
    if config.euler_indices and array.shape[0] >= 2:
        euler = array[:, config.euler_indices]
        rotation_step = np.linalg.norm(wrapped_euler_delta(euler[1:], euler[:-1]), axis=1)
        series = compute_dt_derivatives(euler, time_array, wrap_period=2.0 * np.pi)
        rotation_speed = np.linalg.norm(series.velocity, axis=1)
    elif config.quaternion_indices and array.shape[0] >= 2:
        rotation_step = quaternion_geodesic_angle(
            array[:-1, config.quaternion_indices],
            array[1:, config.quaternion_indices],
            norm_epsilon=config.quaternion_norm_epsilon or 0.0,
        )
        rotation_speed = rotation_step / transition_dt
    elif config.rotation_vector_indices and array.shape[0] >= 2:
        rotation_step = rotation_vector_geodesic_angle(
            array[:-1, config.rotation_vector_indices],
            array[1:, config.rotation_vector_indices],
        )
        rotation_speed = rotation_step / transition_dt
    if rotation_speed is not None and rotation_speed.size:
        result.metrics["rotation_speed.max"] = (
            float(np.nanmax(rotation_speed)) if np.isfinite(rotation_speed).any() else None
        )
        result.metrics["rotation_speed.median"] = (
            float(np.nanmedian(rotation_speed)) if np.isfinite(rotation_speed).any() else None
        )

    if config.near_static_translation_speed is not None or config.near_static_rotation_speed is not None:
        static_mask = np.ones(max(array.shape[0] - 1, 0), dtype=bool)
        near_static_active_indices: set[int] = set()
        if config.near_static_translation_speed is not None:
            if translation_speed is None:
                raise ValueError("near_static_translation_speed requires translation_indices")
            static_mask &= np.isfinite(translation_speed) & (
                translation_speed <= config.near_static_translation_speed
            )
            near_static_active_indices.update(config.translation_indices)
        if config.near_static_rotation_speed is not None:
            if rotation_speed is None:
                raise ValueError(
                    "near_static_rotation_speed requires Euler, quaternion, or rotation-vector indices"
                )
            static_mask &= np.isfinite(rotation_speed) & (rotation_speed <= config.near_static_rotation_speed)
            near_static_active_indices.update(
                config.euler_indices or config.quaternion_indices or config.rotation_vector_indices
            )
        min_static_duration = config.near_static_min_duration_seconds
        if min_static_duration is None:
            raise ValueError("near-static checks require a minimum duration")
        static_runs = find_near_static_runs(
            np.where(static_mask, 0.0, 1.0),
            time_array,
            motion_threshold=0.0,
            min_duration_seconds=min_static_duration,
        )
        result.metrics["near_static.transition_count"] = int(static_mask.sum())
        result.metrics["near_static.run_count"] = len(static_runs)
        result.metrics["near_static.duration_seconds"] = float(
            sum(interval.duration_seconds or 0.0 for interval in static_runs)
        )
        for interval in static_runs:
            result.findings.append(
                QualityFinding(
                    rule_id="TRJ.NEAR_STATIC_RUN",
                    severity="low",
                    disposition="review",
                    message="Trajectory contains a source-threshold near-static run",
                    episode_id=episode_id,
                    start_index=interval.start_index,
                    end_index=interval.end_index + 1,
                    field=field,
                    deterministic=False,
                    metrics={
                        **interval.to_dict(),
                        "active_flat_indices": sorted(near_static_active_indices),
                    },
                )
            )

    review_signals: list[_TrajectoryReviewSignal] = []
    if config.robust_z_threshold is not None:
        rotation_indices = config.euler_indices or config.quaternion_indices or config.rotation_vector_indices
        metric_arrays: dict[str, tuple[np.ndarray, int, tuple[int, ...]]] = {}
        if config.check_velocity and translation_speed is not None:
            metric_arrays["translation_speed"] = (translation_speed, 1, config.translation_indices)
        if config.check_velocity and rotation_speed is not None:
            metric_arrays["rotation_speed"] = (rotation_speed, 1, rotation_indices)
        if translation_series is not None:
            if config.check_acceleration and translation_series.acceleration.size:
                metric_arrays["translation_acceleration"] = (
                    np.linalg.norm(translation_series.acceleration, axis=1),
                    2,
                    config.translation_indices,
                )
            if config.check_jerk and translation_series.jerk.size:
                metric_arrays["translation_jerk"] = (
                    np.linalg.norm(translation_series.jerk, axis=1),
                    3,
                    config.translation_indices,
                )
        raw_strata = None if strata is None else np.asarray(strata, dtype=object).reshape(-1)
        if raw_strata is not None and raw_strata.size != array.shape[0]:
            raise ValueError("strata must be frame-aligned")
        for metric_name, (metric_values, frame_span, active_indices) in metric_arrays.items():
            metric_strata = None if raw_strata is None else raw_strata[frame_span:]
            references = config.robust_references.get(metric_name)
            reference_scope = (
                "frozen_external"
                if references
                else ("episode_local" if metric_strata is None else "episode_local_strata")
            )
            scored = robust_mad_scores(
                metric_values,
                strata=metric_strata,
                candidate_threshold=config.robust_z_threshold,
                references=references,
                scale_floor=config.robust_scale_floor,
            )
            result.metrics[f"robust.{metric_name}"] = {
                "centers": scored.centers,
                "mads": scored.mads,
                "counts": scored.counts,
                "candidate_count": int(scored.candidate_mask.sum()),
                "threshold": scored.threshold,
                "reference_scope": reference_scope,
            }
            normalization = config.robust_z_threshold if config.robust_z_threshold > 0 else 1.0
            review_signals.append(
                _TrajectoryReviewSignal(
                    name=f"{metric_name}.robust_{reference_scope}",
                    candidate_mask=scored.candidate_mask,
                    normalized_scores=scored.scores / normalization,
                    frame_span_extra=frame_span,
                    active_flat_indices=active_indices,
                    reference_scope=reference_scope,
                )
            )

    if config.review_translation_jump_m is not None and translation_step is not None:
        threshold = config.review_translation_jump_m
        candidate = np.isfinite(translation_step) & (translation_step >= threshold)
        result.metrics["physical.translation_step"] = {
            "candidate_count": int(candidate.sum()),
            "max_m": float(np.nanmax(translation_step)) if np.isfinite(translation_step).any() else None,
            "threshold_m": threshold,
            "reference_scope": "contract_physical_step",
        }
        review_signals.append(
            _TrajectoryReviewSignal(
                name="translation_step.physical_threshold",
                candidate_mask=candidate,
                normalized_scores=translation_step / threshold,
                frame_span_extra=1,
                active_flat_indices=config.translation_indices,
                reference_scope="contract_physical_step",
            )
        )
    if config.review_rotation_jump_rad is not None and rotation_step is not None:
        threshold = config.review_rotation_jump_rad
        candidate = np.isfinite(rotation_step) & (rotation_step >= threshold)
        result.metrics["physical.rotation_step"] = {
            "candidate_count": int(candidate.sum()),
            "max_rad": float(np.nanmax(rotation_step)) if np.isfinite(rotation_step).any() else None,
            "threshold_rad": threshold,
            "reference_scope": "contract_physical_step",
        }
        review_signals.append(
            _TrajectoryReviewSignal(
                name="rotation_step.physical_threshold",
                candidate_mask=candidate,
                normalized_scores=rotation_step / threshold,
                frame_span_extra=1,
                active_flat_indices=(
                    config.euler_indices or config.quaternion_indices or config.rotation_vector_indices
                ),
                reference_scope="contract_physical_step",
            )
        )

    if review_signals:
        corroboration_metrics, corroborated_findings = _corroborated_review_findings(
            review_signals,
            frame_count=array.shape[0],
            minimum_signals=config.minimum_corroborating_signals,
            episode_id=episode_id,
            field=field,
        )
        result.metrics["corroboration"] = corroboration_metrics
        result.findings.extend(corroborated_findings)
    return result
