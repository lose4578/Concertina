"""Shared, JSON-serializable result types for data-quality checks.

The checks deliberately distinguish deterministic contract violations from
statistical candidates.  Only deterministic findings may request quarantine;
distributional checks must remain review, mask, or metric-only suggestions.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Literal

import numpy as np

type Severity = Literal["critical", "high", "medium", "low", "info"]
type Disposition = Literal["quarantine", "review", "mask", "weight", "metric"]
type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]

_SEVERITIES = frozenset({"critical", "high", "medium", "low", "info"})
_DISPOSITIONS = frozenset({"quarantine", "review", "mask", "weight", "metric"})


def json_safe(value: Any) -> JsonValue:
    """Convert common NumPy/dataclass payloads to strict JSON-safe values.

    Non-finite scalar values are represented by descriptive strings instead of
    emitting non-standard JSON ``NaN``/``Infinity`` literals.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return json_safe(to_dict())
    raise TypeError(f"Value of type {type(value).__name__!r} is not JSON serializable")


def true_runs(mask: np.ndarray | Sequence[bool]) -> list[tuple[int, int]]:
    """Return half-open intervals for contiguous true values in a 1-D mask."""

    array = np.asarray(mask, dtype=bool)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1-D mask, got shape={array.shape}")
    if array.size == 0:
        return []
    padded = np.pad(array.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """A half-open sample interval with optional physical-time boundaries."""

    start_index: int
    end_index: int
    start_time: float | None = None
    end_time: float | None = None

    def __post_init__(self) -> None:
        if self.start_index < 0 or self.end_index <= self.start_index:
            raise ValueError(f"Invalid half-open interval [{self.start_index}, {self.end_index})")
        if self.start_time is not None and self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("end_time must not precede start_time")

    @property
    def duration_seconds(self) -> float | None:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "start_index": self.start_index,
            "end_index": self.end_index,
            "start_time": json_safe(self.start_time),
            "end_time": json_safe(self.end_time),
            "duration_seconds": json_safe(self.duration_seconds),
        }


@dataclass(frozen=True, slots=True)
class QualityFinding:
    """One interval-level quality finding suitable for a sidecar table."""

    rule_id: str
    severity: Severity
    disposition: Disposition
    message: str
    episode_id: str | int | None = None
    start_index: int | None = None
    end_index: int | None = None
    field: str | None = None
    deterministic: bool = False
    metrics: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.rule_id:
            raise ValueError("rule_id must be non-empty")
        if self.severity not in _SEVERITIES:
            raise ValueError(f"Unsupported severity: {self.severity!r}")
        if self.disposition not in _DISPOSITIONS:
            raise ValueError(f"Unsupported disposition: {self.disposition!r}")
        if self.disposition == "quarantine" and not self.deterministic:
            raise ValueError("Only deterministic contract violations may request quarantine")
        if (self.start_index is None) != (self.end_index is None):
            raise ValueError("start_index and end_index must either both be set or both be omitted")
        if self.start_index is not None and (
            self.start_index < 0 or self.end_index is None or self.end_index <= self.start_index
        ):
            raise ValueError("Finding indices must describe a non-empty half-open interval")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "rule_id": self.rule_id,
            "severity": self.severity,
            "disposition": self.disposition,
            "message": self.message,
            "episode_id": json_safe(self.episode_id),
            "start_index": self.start_index,
            "end_index": self.end_index,
            "field": self.field,
            "deterministic": self.deterministic,
            "metrics": json_safe(self.metrics),
        }


@dataclass(slots=True)
class QualityCheckResult:
    """Metrics and interval findings emitted by one or more checks."""

    metrics: dict[str, Any] = dataclass_field(default_factory=dict)
    findings: list[QualityFinding] = dataclass_field(default_factory=list)

    @property
    def deterministic_failure(self) -> bool:
        return any(item.deterministic and item.disposition == "quarantine" for item in self.findings)

    @property
    def valid_for_automatic_use(self) -> bool:
        return not self.deterministic_failure

    def extend(self, other: QualityCheckResult, *, metric_prefix: str | None = None) -> None:
        prefix = f"{metric_prefix}." if metric_prefix else ""
        for key, value in other.metrics.items():
            target_key = f"{prefix}{key}"
            if target_key in self.metrics:
                raise ValueError(f"Duplicate metric key while merging results: {target_key!r}")
            self.metrics[target_key] = value
        self.findings.extend(other.findings)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "valid_for_automatic_use": self.valid_for_automatic_use,
            "deterministic_failure": self.deterministic_failure,
            "metrics": json_safe(self.metrics),
            "findings": [item.to_dict() for item in self.findings],
        }
