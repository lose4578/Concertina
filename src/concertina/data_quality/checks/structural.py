"""Deterministic episode schema, index, timestamp, and task checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import QualityCheckResult, QualityFinding, true_runs


@dataclass(frozen=True, slots=True)
class StructuralCheckConfig:
    """Source-contract settings for :func:`check_episode_structure`."""

    required_fields: tuple[str, ...] = ()
    expected_trailing_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    check_nonfinite: bool = True
    require_frame_indices: bool = False
    require_integer_indices: bool = True
    require_contiguous_indices: bool = True
    require_zero_based_indices: bool = False
    require_timestamps: bool = False
    require_strict_timestamps: bool = True
    require_tasks: bool = False
    require_constant_task: bool = True
    allow_empty_task: bool = False
    allow_empty_episode: bool = False


def _quarantine(
    rule_id: str,
    message: str,
    *,
    episode_id: str | int | None,
    field: str | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    metrics: Mapping[str, Any] | None = None,
    severity: str = "critical",
) -> QualityFinding:
    return QualityFinding(
        rule_id=rule_id,
        severity=severity,  # type: ignore[arg-type]
        disposition="quarantine",
        message=message,
        episode_id=episode_id,
        start_index=start_index,
        end_index=end_index,
        field=field,
        deterministic=True,
        metrics={} if metrics is None else metrics,
    )


def _metadata_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


def check_episode_structure(
    arrays: Mapping[str, np.ndarray | Sequence[Any]],
    *,
    episode_id: str | int | None = None,
    frame_indices: np.ndarray | Sequence[int | float] | None = None,
    timestamps: np.ndarray | Sequence[float] | None = None,
    tasks: np.ndarray | Sequence[str] | str | None = None,
    config: StructuralCheckConfig | None = None,
) -> QualityCheckResult:
    """Check one episode against an explicit source contract.

    All emitted findings are deterministic because this function only checks
    representation invariants.  Distributional or physical plausibility checks
    belong in the trajectory/alignment modules.
    """

    config = StructuralCheckConfig() if config is None else config
    result = QualityCheckResult()
    normalized: dict[str, np.ndarray] = {}

    missing = sorted(set(config.required_fields).difference(arrays))
    for name in missing:
        result.findings.append(
            _quarantine(
                "STR.MISSING_FIELD",
                f"Required field {name!r} is absent",
                episode_id=episode_id,
                field=name,
            )
        )

    for name, values in arrays.items():
        array = np.asarray(values)
        normalized[name] = array
        if array.ndim == 0:
            result.findings.append(
                _quarantine(
                    "STR.RANK",
                    "Episode arrays must have a leading frame dimension",
                    episode_id=episode_id,
                    field=name,
                    metrics={"shape": array.shape},
                )
            )
            continue

        expected_shape = config.expected_trailing_shapes.get(name)
        if expected_shape is not None and tuple(array.shape[1:]) != tuple(expected_shape):
            result.findings.append(
                _quarantine(
                    "STR.SHAPE",
                    f"Trailing shape does not match the source contract for {name!r}",
                    episode_id=episode_id,
                    field=name,
                    metrics={"actual": array.shape[1:], "expected": expected_shape},
                )
            )

        if config.check_nonfinite and np.issubdtype(array.dtype, np.number):
            finite = np.isfinite(array)
            finite_per_frame = finite.reshape(array.shape[0], -1).all(axis=1)
            bad_count = int((~finite).sum())
            for start, end in true_runs(~finite_per_frame):
                result.findings.append(
                    _quarantine(
                        "STR.NONFINITE",
                        f"Non-finite values found in {name!r}",
                        episode_id=episode_id,
                        field=name,
                        start_index=start,
                        end_index=end,
                        metrics={"nonfinite_values": int((~finite[start:end]).sum())},
                    )
                )
            result.metrics[f"field.{name}.nonfinite_count"] = bad_count

    lengths = {name: int(value.shape[0]) for name, value in normalized.items() if value.ndim > 0}
    unique_lengths = sorted(set(lengths.values()))
    row_count = unique_lengths[0] if len(unique_lengths) == 1 else None
    result.metrics.update(
        {
            "episode_id": episode_id,
            "field_count": len(arrays),
            "field_lengths": lengths,
            "row_count": row_count,
        }
    )
    if len(unique_lengths) > 1:
        result.findings.append(
            _quarantine(
                "STR.LENGTH_MISMATCH",
                "Episode fields have inconsistent leading lengths",
                episode_id=episode_id,
                metrics={"field_lengths": lengths},
            )
        )
    if (row_count == 0 or not lengths) and not config.allow_empty_episode:
        result.findings.append(
            _quarantine("STR.EMPTY_EPISODE", "Episode contains no frames", episode_id=episode_id)
        )

    if frame_indices is None:
        if config.require_frame_indices:
            result.findings.append(
                _quarantine("IDX.MISSING_FRAME_INDEX", "Frame indices are required", episode_id=episode_id)
            )
    else:
        indices = _metadata_array(frame_indices)
        result.metrics["frame_index_count"] = int(indices.size)
        if row_count is not None and indices.size != row_count:
            result.findings.append(
                _quarantine(
                    "IDX.FRAME_LENGTH",
                    "Frame-index count does not match episode length",
                    episode_id=episode_id,
                    field="frame_index",
                    metrics={"actual": indices.size, "expected": row_count},
                )
            )
        if not np.issubdtype(indices.dtype, np.number) or not np.isfinite(indices).all():
            result.findings.append(
                _quarantine(
                    "IDX.INVALID_FRAME_INDEX",
                    "Frame indices must be finite numeric values",
                    episode_id=episode_id,
                    field="frame_index",
                )
            )
        else:
            if config.require_integer_indices and not np.equal(indices, np.floor(indices)).all():
                result.findings.append(
                    _quarantine(
                        "IDX.NONINTEGER_FRAME_INDEX",
                        "Frame indices must be integers",
                        episode_id=episode_id,
                        field="frame_index",
                    )
                )
            differences = np.diff(indices.astype(np.float64))
            bad = differences != 1 if config.require_contiguous_indices else differences <= 0
            for start, end in true_runs(bad):
                result.findings.append(
                    _quarantine(
                        "IDX.FRAME_SEQUENCE",
                        "Frame indices violate the configured sequence contract",
                        episode_id=episode_id,
                        field="frame_index",
                        start_index=start,
                        end_index=end + 1,
                        metrics={"differences": differences[start:end]},
                    )
                )
            if config.require_zero_based_indices:
                expected = np.arange(indices.size, dtype=np.float64)
                numeric_indices = indices.astype(np.float64)
                if not np.array_equal(numeric_indices, expected):
                    result.findings.append(
                        _quarantine(
                            "IDX.FRAME_COORDINATE",
                            "Frame indices must exactly equal the zero-based episode-local coordinate",
                            episode_id=episode_id,
                            field="frame_index",
                            metrics={
                                "actual_first": float(numeric_indices[0]) if numeric_indices.size else None,
                                "actual_last": float(numeric_indices[-1]) if numeric_indices.size else None,
                                "expected_first": 0 if expected.size else None,
                                "expected_last": int(expected[-1]) if expected.size else None,
                            },
                        )
                    )

    if timestamps is None:
        if config.require_timestamps:
            result.findings.append(
                _quarantine("IDX.MISSING_TIMESTAMP", "Timestamps are required", episode_id=episode_id)
            )
    else:
        raw_timestamps = _metadata_array(timestamps)
        time_array: np.ndarray | None
        try:
            time_array = raw_timestamps.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            time_array = None
            result.findings.append(
                _quarantine(
                    "IDX.INVALID_TIMESTAMP_TYPE",
                    "Timestamps must be convertible to finite numeric values",
                    episode_id=episode_id,
                    field="timestamp",
                    metrics={"dtype": str(raw_timestamps.dtype)},
                )
            )
            result.metrics["timestamp_count"] = int(raw_timestamps.size)
        if time_array is not None:
            result.metrics["timestamp_count"] = int(time_array.size)
            if row_count is not None and time_array.size != row_count:
                result.findings.append(
                    _quarantine(
                        "IDX.TIMESTAMP_LENGTH",
                        "Timestamp count does not match episode length",
                        episode_id=episode_id,
                        field="timestamp",
                        metrics={"actual": time_array.size, "expected": row_count},
                    )
                )
            finite = np.isfinite(time_array)
            if not finite.all():
                for start, end in true_runs(~finite):
                    result.findings.append(
                        _quarantine(
                            "IDX.NONFINITE_TIMESTAMP",
                            "Timestamp is not finite",
                            episode_id=episode_id,
                            field="timestamp",
                            start_index=start,
                            end_index=end,
                        )
                    )
            if finite.all() and time_array.size > 1:
                differences = np.diff(time_array)
                bad = differences <= 0 if config.require_strict_timestamps else differences < 0
                for start, end in true_runs(bad):
                    result.findings.append(
                        _quarantine(
                            "IDX.TIMESTAMP_ORDER",
                            "Timestamps violate the configured monotonicity contract",
                            episode_id=episode_id,
                            field="timestamp",
                            start_index=start,
                            end_index=end + 1,
                            metrics={"differences": differences[start:end]},
                        )
                    )
                positive = differences[differences > 0]
                if positive.size:
                    result.metrics["timestamp.dt_median"] = float(np.median(positive))
                    result.metrics["timestamp.dt_min"] = float(positive.min())
                    result.metrics["timestamp.dt_max"] = float(positive.max())

    if tasks is None:
        if config.require_tasks:
            result.findings.append(
                _quarantine("TASK.MISSING", "Task metadata is required", episode_id=episode_id)
            )
    else:
        task_array = np.asarray([tasks] if isinstance(tasks, str) else tasks, dtype=object).reshape(-1)
        task_strings = np.asarray([str(value).strip() for value in task_array], dtype=object)
        result.metrics["task_count"] = int(task_strings.size)
        result.metrics["task_distinct"] = sorted(set(task_strings.tolist()))
        if task_strings.size not in {1, row_count} and row_count is not None:
            result.findings.append(
                _quarantine(
                    "TASK.LENGTH",
                    "Task metadata must be scalar-per-episode or frame-aligned",
                    episode_id=episode_id,
                    field="task",
                    metrics={"actual": task_strings.size, "episode_length": row_count},
                )
            )
        if not config.allow_empty_task and any(not value for value in task_strings):
            result.findings.append(
                _quarantine(
                    "TASK.EMPTY",
                    "Task metadata contains an empty value",
                    episode_id=episode_id,
                    field="task",
                )
            )
        if config.require_constant_task and len(set(task_strings.tolist())) > 1:
            result.findings.append(
                _quarantine(
                    "TASK.INCONSISTENT",
                    "Task metadata changes within an episode",
                    episode_id=episode_id,
                    field="task",
                    metrics={"distinct_tasks": sorted(set(task_strings.tolist()))},
                )
            )

    result.metrics["deterministic_finding_count"] = len(result.findings)
    return result
