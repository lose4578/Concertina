"""Episode scan of a local LeRobot v3 dataset: structural, trajectory, pause and alignment evidence.

Each data file is read once (only the columns of the source contract) and split into its episodes. Every
episode gets one ``episode_metrics`` row, its ``interval_findings`` and a ``keep``/``quarantine`` decision;
only deterministic contract violations (schema, indices, non-finite values, metadata disagreement) quarantine
an episode. Statistical candidates, pauses and residuals are review evidence: they change nothing by
themselves.

Findings name exact source columns (``field_intervals``) even when a check runs on a composite audit view (for
example DROID's pose and gripper columns concatenated), so a later decision can mask exactly the right column.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from concertina.data_quality.checks import (
    AbsoluteAlignmentConfig,
    AlignmentCheckConfig as AlignmentRuleConfig,
    QualityCheckResult,
    QualityFinding,
    StructuralCheckConfig as StructuralRuleConfig,
    TrajectoryCheckConfig as TrajectoryRuleConfig,
    check_action_state_alignment,
    check_episode_structure,
    check_trajectory,
    quaternion_geodesic_angle,
    robust_mad_scores,
    rotation_vector_geodesic_angle,
    true_runs,
    wrapped_euler_delta,
)
from concertina.data_quality.config import ScanConfig
from concertina.data_quality.contracts import FeatureLayout, SourceContract

logger = logging.getLogger(__name__)

PAUSE_RULE = "TRJ.NEAR_STATIC_RUN"
CANONICAL_ACTION_ORIGIN = "canonical_action"
_STRUCTURED_INSTRUCTION = re.compile(
    r"^\s*\[\s*(?P<embodiment>[^\]]+?)\s*\]\s*(?P<semantic_task>[^:]+?)\s*:\s*(?P<instruction>.*)$"
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


# ── dataset metadata ────────────────────────────────────────────────────


@dataclass(frozen=True)
class EpisodeRecord:
    """One row of ``meta/episodes``."""

    episode_index: int
    length: int
    task: str
    data_path: str
    dataset_from_index: int | None

    @property
    def semantic_task(self) -> str:
        match = _STRUCTURED_INSTRUCTION.match(self.task)
        return match.group("semantic_task").strip() if match else self.task

    @property
    def embodiment_name(self) -> str:
        match = _STRUCTURED_INSTRUCTION.match(self.task)
        return match.group("embodiment").strip() if match else "default"


def read_info(root: Path) -> dict[str, Any]:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    if not str(info.get("codebase_version", "")).startswith("v3."):
        raise ValueError(
            f"{root} is not a LeRobot v3 dataset (codebase_version={info.get('codebase_version')!r})"
        )
    return info


def read_episode_table(root: Path, columns: Sequence[str] | None = None) -> list[dict[str, Any]]:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(root / "meta" / "episodes"), format="parquet")
    names = None if columns is None else [name for name in columns if name in dataset.schema.names]
    return dataset.to_table(columns=names).to_pylist()


def load_episode_records(root: Path, info: Mapping[str, Any]) -> dict[int, EpisodeRecord]:
    template = str(info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"))
    columns = (
        "episode_index",
        "length",
        "tasks",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
    )
    records: dict[int, EpisodeRecord] = {}
    for row in read_episode_table(root, columns):
        episode = int(row["episode_index"])
        if episode in records:
            raise ValueError(f"{root}: episode {episode} appears twice in meta/episodes")
        tasks = row.get("tasks") or []
        records[episode] = EpisodeRecord(
            episode_index=episode,
            length=int(row["length"]),
            task=str(tasks[0]) if tasks else "",
            data_path=template.format(
                chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
            ),
            dataset_from_index=(
                None if row.get("dataset_from_index") is None else int(row["dataset_from_index"])
            ),
        )
    return records


# ── source vectors ──────────────────────────────────────────────────────


def arrow_column_to_numpy(column: Any) -> np.ndarray:
    """A projected Arrow column as a NumPy array, restoring nested fixed-size list shapes."""
    import pyarrow as pa

    array = column.combine_chunks()
    if pa.types.is_fixed_size_list(array.type):
        dimensions: list[int] = []
        values = array
        while pa.types.is_fixed_size_list(values.type):
            dimensions.append(int(values.type.list_size))
            values = values.values
        try:
            return np.asarray(values.to_numpy(zero_copy_only=False)).reshape(len(array), *dimensions)
        except (TypeError, ValueError, NotImplementedError):
            return np.asarray(array.to_pylist())
    if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
        rows = array.to_pylist()
        try:
            return np.asarray(rows)
        except ValueError:
            return np.asarray(rows, dtype=object)
    if pa.types.is_string(array.type) or pa.types.is_large_string(array.type):
        return np.asarray(array.to_pylist(), dtype=object)
    try:
        return np.asarray(array.to_numpy(zero_copy_only=False))
    except (TypeError, ValueError, NotImplementedError):
        return np.asarray(array.to_pylist(), dtype=object)


def _normalize_component_storage(
    values: np.ndarray, *, expected_shape: tuple[int, ...], episode_length: int
) -> tuple[np.ndarray, bool]:
    """A feature of shape ``(1,)`` may be stored as one primitive per frame; view it as ``(L, 1)``."""
    array = np.asarray(values)
    numeric = np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)
    if expected_shape == (1,) and array.shape == (episode_length,) and numeric:
        return array.reshape(episode_length, 1), True
    return array, False


def parse_contract_vector(
    arrays: Mapping[str, np.ndarray], layout: FeatureLayout, *, episode_length: int
) -> tuple[np.ndarray | None, str | None]:
    """The audit view of a layout (its source components flattened and concatenated) or why it failed."""
    flattened: list[np.ndarray] = []
    for component in layout.components:
        if component.column not in arrays:
            return None, f"source component {component.column!r} is absent"
        raw, _ = _normalize_component_storage(
            arrays[component.column], expected_shape=component.expected_shape, episode_length=episode_length
        )
        try:
            values = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError):
            return None, f"source component {component.column!r} is not numeric"
        expected = (episode_length, *component.expected_shape)
        if values.shape != expected:
            return (
                None,
                f"source component {component.column!r} expected shape={expected}, got shape={values.shape}",
            )
        flattened.append(values.reshape(episode_length, component.flat_dim))
    return np.concatenate(flattened, axis=1), None


def _provenance(layout: FeatureLayout, role: str) -> dict[str, Any]:
    return {
        "field": layout.column,
        "role": role,
        "is_composite": layout.is_composite,
        "source_fields": list(layout.source_columns),
    }


def _layout_source_fields(layout: FeatureLayout, flat_indices: Sequence[int]) -> tuple[str, ...]:
    """Exact source columns touched by flattened audit-view indices."""
    requested = {int(index) for index in flat_indices}
    if any(index < 0 or index >= layout.expected_dim for index in requested):
        raise ValueError(f"flat component index is outside layout {layout.column!r}")
    selected: list[str] = []
    offset = 0
    for component in layout.components:
        stop = offset + component.flat_dim
        if any(offset <= index < stop for index in requested):
            selected.append(component.column)
        offset = stop
    return tuple(selected)


def _field_intervals(
    layout: FeatureLayout, flat_indices: Sequence[int], start: int, end: int
) -> dict[str, dict[str, int]]:
    return {
        field: {"start_index": int(start), "end_index": int(end)}
        for field in _layout_source_fields(layout, flat_indices)
    }


def _annotate(result: QualityCheckResult, layout: FeatureLayout, role: str) -> QualityCheckResult:
    """Attach exact source-column provenance to a check result run on an audit view."""
    provenance = _provenance(layout, role)
    result.metrics["source_provenance"] = provenance
    annotated = []
    for finding in result.findings:
        metrics = {**dict(finding.metrics), "source_provenance": provenance}
        if finding.start_index is not None:
            active = metrics.get("active_flat_indices")
            if not isinstance(active, Sequence) or isinstance(active, (str, bytes)):
                if layout.is_composite:
                    raise ValueError(f"composite finding {finding.rule_id!r} lacks active_flat_indices")
                active = tuple(range(layout.expected_dim))
            metrics["active_source_fields"] = list(_layout_source_fields(layout, active))
            metrics["field_intervals"] = _field_intervals(
                layout, active, int(finding.start_index), int(finding.end_index)
            )
        annotated.append(replace(finding, metrics=metrics))
    result.findings = annotated
    return result


# ── per-episode checks ──────────────────────────────────────────────────


def _generic_action_jumps(
    values: np.ndarray, timestamps: np.ndarray, *, layout: FeatureLayout, episode: int, threshold: float
) -> QualityCheckResult:
    """Review jumps of action channels without pose or gripper semantics (for example joint commands)."""
    result = QualityCheckResult()
    mapped = {
        i for spec in (*layout.position_slices, *layout.rotation_slices) for i in range(spec.start, spec.stop)
    }
    mapped.update(layout.gripper_indices)
    generic = tuple(index for index in range(layout.expected_dim) if index not in mapped)
    result.metrics.update({"generic_indices": generic, "transition_count": max(len(values) - 1, 0)})
    if len(values) < 2 or not generic:
        result.metrics["skipped_reason"] = (
            "fewer than two source rows"
            if len(values) < 2
            else "all channels have declared pose or gripper semantics"
        )
        return result
    speed = np.linalg.norm(np.diff(values[:, generic], axis=0), axis=1) / np.diff(timestamps)
    scored = robust_mad_scores(speed, candidate_threshold=threshold, scale_floor=1e-12)
    finite = np.isfinite(speed).any()
    result.metrics.update(
        {
            "speed_l2_max": float(np.nanmax(speed)) if finite else None,
            "speed_l2_median": float(np.nanmedian(speed)) if finite else None,
            "candidate_count": int(scored.candidate_mask.sum()),
            "robust_center": scored.centers.get("__all__"),
            "robust_mad": scored.mads.get("__all__"),
            "robust_threshold": threshold,
        }
    )
    for start, end in true_runs(scored.candidate_mask):
        result.findings.append(
            QualityFinding(
                rule_id="TRJ.GENERIC_ACTION_CANDIDATE",
                severity="medium",
                disposition="review",
                message="Unmapped action channels exceed their episode-local robust review threshold",
                episode_id=episode,
                start_index=start,
                end_index=end + 1,
                field=layout.column,
                metrics={
                    "source_provenance": _provenance(layout, "action"),
                    "active_flat_indices": list(generic),
                    "active_source_fields": list(_layout_source_fields(layout, generic)),
                    "candidate_count": end - start,
                    "field_intervals": _field_intervals(layout, generic, start, end + 1),
                    "max_score": float(np.nanmax(scored.scores[start:end])) / threshold,
                    "raw_robust_score_max": float(np.nanmax(scored.scores[start:end])),
                    "speed_l2_max": float(np.nanmax(speed[start:end])),
                    "robust_threshold": threshold,
                },
            )
        )
    return result


def _trajectory_rule_config(
    layout: FeatureLayout, *, contract: SourceContract, arm: int, config: ScanConfig
) -> TrajectoryRuleConfig:
    position = layout.position_slices[arm] if arm < len(layout.position_slices) else None
    rotation = layout.rotation_slices[arm] if arm < len(layout.rotation_slices) else None
    position_indices = tuple(range(position.start, position.stop)) if position else ()
    rotation_indices = tuple(range(rotation.start, rotation.stop)) if rotation else ()
    representation = layout.rotation_representation
    quaternion = representation == "quaternion"
    return TrajectoryRuleConfig(
        translation_indices=position_indices,
        euler_indices=rotation_indices if representation == "euler" else (),
        quaternion_indices=rotation_indices if quaternion else (),
        rotation_vector_indices=rotation_indices if representation == "rotation_vector" else (),
        quaternion_order=layout.rotation_order if quaternion else "xyzw",
        quaternion_norm_epsilon=contract.raw_quaternion_norm_epsilon if quaternion else None,
        quaternion_unit_tolerance=(
            1e-3 if quaternion and contract.raw_quaternion_norm_policy == "unit_required" else None
        ),
        robust_z_threshold=config.trajectory.jump_mad_multiplier,
        robust_scale_floor=1e-12,
        minimum_corroborating_signals=config.trajectory.minimum_corroborating_signals,
        review_translation_jump_m=contract.threshold_profile.review_translation_jump_m,
        review_rotation_jump_rad=contract.threshold_profile.review_rotation_jump_rad,
        check_velocity=config.trajectory.check_velocity,
        check_acceleration=config.trajectory.check_acceleration,
        check_jerk=config.trajectory.check_jerk,
    )


def _max_position_delta(values: np.ndarray, layout: FeatureLayout) -> np.ndarray | None:
    if not layout.position_slices or len(values) < 2:
        return None
    per_arm = [
        np.linalg.norm(np.diff(values[:, s.start : s.stop], axis=0), axis=1) for s in layout.position_slices
    ]
    return np.max(np.column_stack(per_arm), axis=1)


def _max_rotation_delta(
    values: np.ndarray, layout: FeatureLayout, *, norm_epsilon: float
) -> np.ndarray | None:
    if not layout.rotation_slices or len(values) < 2:
        return None
    per_arm = []
    for spec in layout.rotation_slices:
        left, right = values[:-1, spec.start : spec.stop], values[1:, spec.start : spec.stop]
        if layout.rotation_representation == "quaternion":
            per_arm.append(quaternion_geodesic_angle(left, right, norm_epsilon=norm_epsilon))
        elif layout.rotation_representation == "euler":
            per_arm.append(np.linalg.norm(wrapped_euler_delta(left, right), axis=1))
        elif layout.rotation_representation == "rotation_vector":
            per_arm.append(rotation_vector_geodesic_angle(left, right))
        else:
            per_arm.append(np.linalg.norm(right - left, axis=1))
    return np.max(np.column_stack(per_arm), axis=1)


def _max_gripper_delta(values: np.ndarray, layout: FeatureLayout) -> np.ndarray | None:
    if not layout.gripper_indices or len(values) < 2:
        return None
    return np.max(np.abs(np.diff(values[:, layout.gripper_indices], axis=0)), axis=1)


def pause_runs(
    action: np.ndarray,
    state: np.ndarray,
    timestamps: np.ndarray,
    *,
    contract: SourceContract,
    episode: int,
    pause_thresholds: Sequence[float],
    check_exact_duplicates: bool,
) -> QualityCheckResult:
    """Near-static runs of the canonical action, corroborated by the observed state.

    A transition is static when the translation, rotation and gripper steps of both the action and the state
    stay within the contract's thresholds. Runs at least ``min(pause_thresholds)`` seconds long become review
    findings led by the canonical action column; they are the candidates of the pause-cleaning policy.
    """
    result = QualityCheckResult()
    profile = contract.threshold_profile
    action_layout, state_layout = contract.canonical_action_layout, contract.state_layout
    action_field = action_layout.column
    result.metrics.update(
        {
            "source_field": action_field,
            "state_field": state_layout.column,
            "candidate_origin": CANONICAL_ACTION_ORIGIN,
            "candidate_authority": contract.canonical_action_column,
        }
    )
    transitions = max(len(action) - 1, 0)
    if transitions == 0:
        result.metrics["transition_count"] = 0
        return result
    static = np.ones(transitions, dtype=bool)
    signal_count = 0
    active: dict[str, set[int]] = {"action": set(), "state": set()}
    for name, values, layout in (("action", action, action_layout), ("state", state, state_layout)):
        translation = _max_position_delta(values, layout)
        if translation is not None and profile.static_translation_m is not None:
            static &= np.isfinite(translation) & (translation <= profile.static_translation_m)
            result.metrics[f"{name}.translation_step.max"] = (
                float(np.nanmax(translation)) if np.isfinite(translation).any() else None
            )
            active[name].update(i for s in layout.position_slices for i in range(s.start, s.stop))
            signal_count += 1
        rotation = _max_rotation_delta(values, layout, norm_epsilon=contract.raw_quaternion_norm_epsilon)
        if rotation is not None and profile.static_rotation_rad is not None:
            static &= np.isfinite(rotation) & (rotation <= profile.static_rotation_rad)
            result.metrics[f"{name}.rotation_step.max"] = (
                float(np.nanmax(rotation)) if np.isfinite(rotation).any() else None
            )
            active[name].update(i for s in layout.rotation_slices for i in range(s.start, s.stop))
            signal_count += 1
        gripper = _max_gripper_delta(values, layout)
        if gripper is not None and profile.static_gripper_delta is not None:
            static &= np.isfinite(gripper) & (gripper <= profile.static_gripper_delta)
            active[name].update(layout.gripper_indices)
            signal_count += 1

    duplicate = np.all(action[1:] == action[:-1], axis=1) & np.all(state[1:] == state[:-1], axis=1)
    runs = true_runs(static) if signal_count else []
    result.metrics.update(
        {
            "signal_count": signal_count,
            "near_static_transition_count": int(static.sum()) if signal_count else 0,
            "exact_duplicate_transition_count": int(duplicate.sum()) if check_exact_duplicates else None,
        }
    )
    if not signal_count:
        result.metrics["skipped_reason"] = "contract has no static-motion threshold"
        return result

    durations = {str(value): 0.0 for value in pause_thresholds}
    frame_counts = {str(value): 0 for value in pause_thresholds}
    minimum = min(pause_thresholds)
    action_indices, state_indices = sorted(active["action"]), sorted(active["state"])
    for start, end in runs:
        frame_end = end + 1
        duration = float(timestamps[frame_end - 1] - timestamps[start])
        for threshold in pause_thresholds:
            if duration >= threshold:
                durations[str(threshold)] += duration
                frame_counts[str(threshold)] += frame_end - start
        if duration < minimum:
            continue
        result.findings.append(
            QualityFinding(
                rule_id=PAUSE_RULE,
                severity="medium" if duration >= profile.long_pause_seconds else "low",
                disposition="review",
                message="Action/state trajectory contains a source-threshold near-static run",
                episode_id=episode,
                start_index=start,
                end_index=frame_end,
                field=action_field,
                metrics={
                    "source_field": action_field,
                    "state_field": state_layout.column,
                    "source_pair": {
                        "action_fields": list(_layout_source_fields(action_layout, action_indices)),
                        "state_fields": list(_layout_source_fields(state_layout, state_indices)),
                        "field_intervals": {
                            **_field_intervals(action_layout, action_indices, start, frame_end),
                            **_field_intervals(state_layout, state_indices, start, frame_end),
                        },
                    },
                    "active_flat_indices": {"action": action_indices, "state": state_indices},
                    "candidate_origin": CANONICAL_ACTION_ORIGIN,
                    "candidate_authority": contract.canonical_action_column,
                    "duration_seconds": duration,
                    "minimum_review_seconds": minimum,
                    "long_pause_seconds": profile.long_pause_seconds,
                    "exact_duplicate_transitions": (
                        int(duplicate[start : frame_end - 1].sum()) if check_exact_duplicates else None
                    ),
                },
            )
        )
    result.metrics["near_static_duration_by_threshold_s"] = durations
    result.metrics["near_static_frames_by_threshold_s"] = frame_counts
    return result


def _directional_gate(aggregate: Any, *, fail_below: float, pass_at_or_above: float) -> dict[str, Any]:
    numeric = not isinstance(aggregate, bool) and isinstance(aggregate, (int, float, np.integer, np.floating))
    if not numeric or not np.isfinite(float(aggregate)):
        value, classification = None, "unavailable"
    else:
        value = float(aggregate)
        classification = (
            "fail" if value < fail_below else "pass" if value >= pass_at_or_above else "ambiguous"
        )
    return {
        "aggregate": value,
        "classification": classification,
        "fail_below": fail_below,
        "pass_at_or_above": pass_at_or_above,
    }


def _alignment(
    action: np.ndarray,
    state: np.ndarray,
    timestamps: np.ndarray,
    *,
    contract: SourceContract,
    config: ScanConfig,
    episode: int,
) -> QualityCheckResult:
    """Residual of the canonical action against the provenance-aligned state, and action/state lag."""
    result = QualityCheckResult()
    action_layout, state_layout = contract.canonical_action_layout, contract.state_layout
    action_column, state_column = contract.canonical_action_column, state_layout.column
    pair = {"action_field": action_column, "state_field": state_column}
    result.metrics["source_pair"] = pair
    profile = contract.threshold_profile
    absolute = config.alignment.check_absolute_next and contract.abs_next_relation != "none"
    if not absolute:
        result.metrics["absolute.skipped_reason"] = (
            "alignment.check_absolute_next is disabled"
            if not config.alignment.check_absolute_next
            else "source contract has no absolute action/state equality"
        )
    translation_threshold = (
        config.alignment.abs_next_candidate_translation_m
        if config.alignment.abs_next_candidate_translation_m is not None
        else profile.abs_next_candidate_translation_m
    )
    rotation_threshold = (
        config.alignment.abs_next_candidate_rotation_rad
        if config.alignment.abs_next_candidate_rotation_rad is not None
        else profile.abs_next_candidate_rotation_rad
    )
    relation = (
        "action_t_to_state_t_plus_1"
        if contract.abs_next_relation == "current_action_to_next_state"
        else "action_t_minus_1_to_state_t"
    )
    arms = max(len(action_layout.position_slices), len(action_layout.rotation_slices), 1) if absolute else 0
    for arm in range(arms):

        def pick(slices, index=arm):
            return slices[index] if index < len(slices) else None

        action_position, state_position = (
            pick(action_layout.position_slices),
            pick(state_layout.position_slices),
        )
        action_rotation, state_rotation = (
            pick(action_layout.rotation_slices),
            pick(state_layout.rotation_slices),
        )
        representation = action_layout.rotation_representation
        rule = AbsoluteAlignmentConfig(
            contract=relation,
            action_field=f"{action_column}.arm{arm}",
            state_field=f"{state_column}.arm{arm}",
            action_source_field=action_column,
            state_source_field=state_column,
            action_translation_indices=(
                tuple(range(action_position.start, action_position.stop)) if action_position else ()
            ),
            state_translation_indices=tuple(range(state_position.start, state_position.stop))
            if state_position
            else (),
            translation_threshold=translation_threshold if action_position and state_position else None,
            rotation_representation=representation if representation in {"euler", "quaternion"} else "none",
            action_rotation_indices=(
                tuple(range(action_rotation.start, action_rotation.stop)) if action_rotation else ()
            ),
            state_rotation_indices=tuple(range(state_rotation.start, state_rotation.stop))
            if state_rotation
            else (),
            rotation_threshold=rotation_threshold if action_rotation and state_rotation else None,
            quaternion_norm_epsilon=contract.raw_quaternion_norm_epsilon
            if representation == "quaternion"
            else None,
            candidate_policy="mask_field" if contract.abs_next_policy == "mask_field" else "review",
            provenance=contract.abs_next_provenance,
            provenance_deterministic=contract.abs_next_provenance_deterministic,
            mask_field_authority=contract.abs_next_mask_field_authority,
            mask_source_field=contract.abs_next_mask_source_field,
        )
        arm_result = check_action_state_alignment(
            action, state, timestamps, episode_id=episode, config=AlignmentRuleConfig(absolute=rule)
        )
        result.extend(arm_result, metric_prefix=f"arm{arm}")

    pair_count = min(len(action), max(len(state) - 1, 0))
    if (
        absolute
        and contract.compare_gripper_in_abs_next
        and action_layout.gripper_indices
        and len(action_layout.gripper_indices) == len(state_layout.gripper_indices)
        and pair_count > 0
        and profile.abs_next_candidate_gripper_delta is not None
    ):
        error = np.max(
            np.abs(
                action[:pair_count, action_layout.gripper_indices]
                - state[1 : pair_count + 1, state_layout.gripper_indices]
            ),
            axis=1,
        )
        threshold = float(profile.abs_next_candidate_gripper_delta)
        candidate = np.isfinite(error) & (error > threshold)
        result.metrics["gripper"] = {
            "pair_count": pair_count,
            "candidate_count": int(candidate.sum()),
            "error_max": float(np.nanmax(error)) if np.isfinite(error).any() else None,
            "threshold": threshold,
        }
        for start, end in true_runs(candidate):
            result.findings.append(
                QualityFinding(
                    rule_id="ALG.ABSOLUTE_GRIPPER_RESIDUAL",
                    severity="medium",
                    disposition="mask" if contract.abs_next_policy == "mask_field" else "review",
                    message="Absolute gripper action differs from the provenance-aligned next state",
                    episode_id=episode,
                    start_index=int(start),
                    end_index=int(end),
                    field=f"{action_column}<->{state_column}",
                    deterministic=contract.abs_next_provenance_deterministic,
                    metrics={
                        "gripper_error_max": float(np.nanmax(error[start:end])),
                        "gripper_threshold": threshold,
                        "candidate_policy": contract.abs_next_policy,
                        "mask_field_authority": contract.abs_next_mask_field_authority,
                        "mask_source_field": contract.abs_next_mask_source_field,
                        "source_pair": {
                            **pair,
                            "field_intervals": {
                                action_column: {"start_index": int(start), "end_index": int(end)},
                                state_column: {"start_index": int(start) + 1, "end_index": int(end) + 1},
                            },
                        },
                    },
                )
            )

    if config.alignment.check_lag:
        action_position = tuple(i for s in action_layout.position_slices for i in range(s.start, s.stop))
        state_position = tuple(i for s in state_layout.position_slices for i in range(s.start, s.stop))
        if action_position and len(action_position) == len(state_position):
            static_step = profile.static_translation_m or 1e-6
            lag = check_action_state_alignment(
                action,
                state,
                timestamps,
                episode_id=episode,
                config=AlignmentRuleConfig(
                    motion_action_indices=action_position,
                    motion_state_indices=state_position,
                    motion_action_field=action_column,
                    motion_state_field=state_column,
                    lag_max_seconds=config.alignment.lag_window_seconds,
                    lag_minimum_overlap=max(3, min(10, len(action) // 3)),
                    lag_review_threshold_seconds=1.0 / contract.default_fps,
                    activity_threshold=static_step * contract.default_fps,
                    directional_agreement_review_below=config.alignment.directional_fail_below,
                ),
            )
            directional = lag.metrics.get("directional_agreement")
            lag.metrics["directional_agreement_gate"] = _directional_gate(
                directional.get("aggregate") if isinstance(directional, Mapping) else None,
                fail_below=config.alignment.directional_fail_below,
                pass_at_or_above=config.alignment.directional_pass_at_or_above,
            )
            lag.findings = [
                replace(
                    finding,
                    field=f"{action_column}<->{state_column}",
                    metrics={
                        **dict(finding.metrics),
                        "source_pair": {
                            **pair,
                            "field_intervals": {
                                column: {"start_index": finding.start_index, "end_index": finding.end_index}
                                for column in (action_column, state_column)
                            },
                        },
                    },
                )
                for finding in lag.findings
            ]
            result.extend(lag, metric_prefix="lag")
        else:
            result.metrics["lag.skipped_reason"] = "action/state position layouts are not comparable"
    return result


def _same_row_copy(action: np.ndarray, state: np.ndarray) -> dict[str, Any]:
    """Diagnostic profile of literal same-row action/state copies (never a decision)."""
    if action.shape != state.shape:
        return {"skipped_reason": f"audit views differ in shape: action={action.shape}, state={state.shape}"}
    exact = np.all(action == state, axis=1) if len(action) else np.zeros(0, dtype=bool)
    return {
        "row_count": int(len(action)),
        "exact_copy_row_count": int(exact.sum()),
        "entire_episode_exact_copy": bool(len(action) and exact.all()),
    }


def _structural(
    arrays: Mapping[str, np.ndarray], *, contract: SourceContract, episode: int, config: ScanConfig
) -> QualityCheckResult:
    result = QualityCheckResult()
    length = len(np.asarray(arrays["episode_index"]).reshape(-1))
    coordinate_valid = False
    frame_indices = arrays.get("frame_index")
    if frame_indices is not None:
        raw = np.asarray(frame_indices)
        if raw.ndim == 1 and raw.size == length:
            try:
                numeric = raw.astype(np.float64, copy=False)
                coordinate_valid = bool(
                    np.isfinite(numeric).all()
                    and np.array_equal(numeric, np.arange(length, dtype=np.float64))
                )
            except (TypeError, ValueError):
                coordinate_valid = False
    if not coordinate_valid:
        result.findings.append(
            QualityFinding(
                rule_id="IDX.FRAME_COORDINATE",
                severity="critical",
                disposition="quarantine",
                message="frame_index must be the zero-based episode-local coordinate 0..L-1",
                episode_id=episode,
                start_index=0,
                end_index=length,
                field="frame_index",
                deterministic=True,
                metrics={"episode_length": length},
            )
        )
    result.metrics["frame_coordinate_valid"] = coordinate_valid
    if not config.structural.enabled:
        return result
    layouts = (contract.state_layout, *contract.action_layouts)
    expected_shapes = {
        component.column: component.expected_shape
        for layout in layouts
        for component in layout.components
        if component.column in arrays
    }
    normalized = dict(arrays)
    for field, shape in expected_shapes.items():
        values, changed = _normalize_component_storage(
            arrays[field], expected_shape=shape, episode_length=length
        )
        if changed:
            normalized[field] = values
    structural = config.structural
    checked = check_episode_structure(
        normalized,
        episode_id=episode,
        frame_indices=normalized.get("frame_index"),
        timestamps=normalized.get("timestamp"),
        tasks=normalized.get("task_index"),
        config=StructuralRuleConfig(
            required_fields=contract.required_projection_columns
            if structural.verify_required_columns
            else (),
            expected_trailing_shapes=expected_shapes if structural.verify_feature_dimensions else {},
            check_nonfinite=structural.verify_nonfinite,
            require_frame_indices=structural.verify_frame_indices,
            require_integer_indices=True,
            require_contiguous_indices=structural.verify_episode_contiguity,
            require_zero_based_indices=structural.verify_episode_contiguity,
            require_timestamps=structural.verify_timestamps,
            require_strict_timestamps=True,
            require_tasks=True,
            require_constant_task=True,
            allow_empty_task=False,
        ),
    )
    result.extend(checked, metric_prefix="configured")
    return result


def _finding_score(metrics: Mapping[str, Any]) -> float:
    if "max_score" in metrics:
        return float(metrics["max_score"])
    if "duration_seconds" in metrics:
        return float(metrics["duration_seconds"]) / max(
            float(metrics.get("long_pause_seconds") or 1.0), 1e-12
        )
    ratios = [
        float(metrics[value]) / float(metrics[threshold])
        for value, threshold in (
            ("translation_error_max", "translation_threshold"),
            ("rotation_error_max", "rotation_threshold"),
        )
        if metrics.get(value) is not None and metrics.get(threshold) not in (None, 0)
    ]
    return max(ratios, default=1.0)


def _exact_field_intervals(row: dict[str, Any]) -> None:
    """Make sure every interval finding names the exact source columns it covers."""
    start, end = row.get("start_index"), row.get("end_index")
    if start is None:
        return
    metrics = dict(row.get("metrics") or {})
    pair = metrics.get("source_pair")
    if isinstance(pair, Mapping) and isinstance(pair.get("field_intervals"), Mapping):
        return
    if isinstance(metrics.get("field_intervals"), Mapping) and metrics["field_intervals"]:
        return
    field = row.get("field")
    if isinstance(field, str) and "<->" in field:
        raise ValueError(f"paired finding {field!r} lacks exact per-field intervals")
    provenance = metrics.get("source_provenance")
    if isinstance(provenance, Mapping) and provenance.get("is_composite"):
        raise ValueError(f"composite finding {field!r} lacks exact component field_intervals")
    fields = list(provenance.get("source_fields", ())) if isinstance(provenance, Mapping) else []
    if not fields:
        source = metrics.get("source_field") or field
        fields = [source] if isinstance(source, str) and source else []
    if fields:
        metrics["field_intervals"] = {name: {"start_index": start, "end_index": end} for name in fields}
        row["metrics"] = metrics


def finding_id(dataset: str, finding: Mapping[str, Any]) -> str:
    """Stable content identity of a finding (also the ``candidate_id`` of a human decision)."""
    identity = {
        "dataset": dataset,
        "episode_id": finding["episode_id"],
        "rule_id": finding["rule_id"],
        "start_index": finding["start_index"],
        "end_index": finding["end_index"],
        "field": finding["field"],
        "message": finding["message"],
    }
    return canonical_hash(identity)[:24]


def _finding_row(
    finding: QualityFinding, *, dataset: str, episode: EpisodeRecord | None, source_path: str
) -> dict:
    row = dict(finding.to_dict())
    _exact_field_intervals(row)
    row["finding_id"] = finding_id(dataset, row)
    row["score"] = _finding_score(row.get("metrics") or {})
    start, end = row.pop("start_index"), row.pop("end_index")
    row.pop("episode_id")
    row.update(
        {
            "episode_index": int(finding.episode_id) if finding.episode_id is not None else None,
            "start_frame": start,
            "end_frame": end,
            "source_path": source_path,
            "task": episode.task if episode else None,
        }
    )
    return row


def _field_mask_rows(
    findings: Sequence[Mapping[str, Any]], *, episode: int, length: int, fields: Sequence[str]
):
    """Union of deterministic ``mask`` findings per exact source column (``mask_field`` contracts)."""
    allowed = set(fields)
    intervals: dict[str, list[tuple[int, int, str]]] = {}
    for finding in findings:
        if finding.get("disposition") != "mask":
            continue
        metrics = finding.get("metrics") or {}
        if finding.get("deterministic") is not True or not metrics.get("mask_field_authority"):
            raise ValueError(
                f"mask finding {finding.get('rule_id')!r} lacks deterministic, independent authority"
            )
        source_field = metrics.get("mask_source_field")
        if source_field not in allowed:
            raise ValueError(f"mask finding names no exact contract column: {source_field!r}")
        pair = metrics.get("source_pair")
        spans = pair.get("field_intervals") if isinstance(pair, Mapping) else metrics.get("field_intervals")
        span = (spans or {}).get(source_field)
        if not isinstance(span, Mapping):
            raise ValueError(f"mask finding has no interval for {source_field!r}")
        start, end = int(span["start_index"]), int(span["end_index"])
        if not 0 <= start < end <= length:
            raise ValueError(f"field mask [{start}, {end}) is outside episode {episode} of {length} frames")
        intervals.setdefault(source_field, []).append((start, end, str(finding["rule_id"])))
    rows = []
    for field in sorted(intervals):
        spans = intervals[field]
        boundaries = sorted({value for start, end, _ in spans for value in (start, end)})
        merged: list[list[Any]] = []
        for start, end in pairwise(boundaries):
            rules = {rule for left, right, rule in spans if left < end and right > start}
            if not rules:
                continue
            if merged and merged[-1][1] == start and merged[-1][2] == rules:
                merged[-1][1] = end
            else:
                merged.append([start, end, rules])
        rows.extend(
            {
                "episode_index": episode,
                "start_frame": start,
                "end_frame": end,
                "field": field,
                "decision": "mask_field",
                "rules": sorted(rules),
                "origin": "source_contract",
            }
            for start, end, rules in merged
        )
    return rows


def scan_episode(
    arrays: Mapping[str, np.ndarray],
    *,
    episode: EpisodeRecord | None,
    contract: SourceContract,
    config: ScanConfig,
    dataset: str,
    source_path: str,
) -> dict[str, list[dict[str, Any]]]:
    """Metrics, findings, decision and contract field masks of one episode's rows."""
    episode_index = int(np.asarray(arrays["episode_index"]).reshape(-1)[0])
    length = len(np.asarray(arrays["episode_index"]).reshape(-1))
    combined = QualityCheckResult()
    combined.extend(
        _structural(arrays, contract=contract, episode=episode_index, config=config),
        metric_prefix="structural",
    )
    if episode is None:
        combined.findings.append(
            QualityFinding(
                rule_id="META.MISSING_EPISODE",
                severity="critical",
                disposition="quarantine",
                message="Frame table episode has no matching metadata row",
                episode_id=episode_index,
                deterministic=True,
            )
        )
    elif episode.length != length:
        combined.findings.append(
            QualityFinding(
                rule_id="META.LENGTH_MISMATCH",
                severity="critical",
                disposition="quarantine",
                message="Frame-table length disagrees with episode metadata",
                episode_id=episode_index,
                deterministic=True,
                metrics={"actual": length, "expected": episode.length},
            )
        )

    timestamps = None
    timestamp_skip = None
    if "timestamp" not in arrays:
        timestamp_skip = "source field is absent"
    else:
        try:
            candidate = np.asarray(arrays["timestamp"], dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            timestamp_skip = "source field is not numeric"
        else:
            if candidate.size != length:
                timestamp_skip = f"expected {length} rows, got {candidate.size}"
            elif not np.isfinite(candidate).all() or np.any(np.diff(candidate) <= 0):
                timestamp_skip = "timestamps are not finite and strictly increasing"
            else:
                timestamps = candidate

    state, state_skip = parse_contract_vector(arrays, contract.state_layout, episode_length=length)
    parsed: list[tuple[FeatureLayout, np.ndarray]] = []
    skipped: dict[str, str] = {}
    for layout in contract.action_layouts:
        if not any(column in arrays for column in layout.source_columns):
            continue
        values, reason = parse_contract_vector(arrays, layout, episode_length=length)
        if values is None:
            skipped[layout.column] = reason or "source field is not parseable"
        else:
            parsed.append((layout, values))
    canonical = next((v for layout, v in parsed if layout.column == contract.canonical_action_column), None)
    combined.metrics.update(
        {
            "motion.action_fields_parseable": [layout.column for layout, _ in parsed],
            "motion.action_fields_skipped": skipped,
            "motion.state_field_skipped_reason": state_skip,
            "motion.timestamp_skipped_reason": timestamp_skip,
        }
    )

    if config.trajectory.enabled and timestamps is not None:
        targets = [(contract.state_layout, state, "observation_state")] if state is not None else []
        targets += [(layout, values, "action") for layout, values in parsed]
        for layout, values, role in targets:
            for arm in range(max(len(layout.position_slices), len(layout.rotation_slices), 1)):
                trajectory = check_trajectory(
                    values,
                    timestamps,
                    episode_id=episode_index,
                    field=f"{layout.column}.arm{arm}",
                    config=_trajectory_rule_config(layout, contract=contract, arm=arm, config=config),
                )
                _annotate(trajectory, layout, role)
                combined.extend(trajectory, metric_prefix=f"trajectory.{layout.column}.arm{arm}")
            if role == "action" and config.trajectory.check_velocity:
                generic = _generic_action_jumps(
                    values,
                    timestamps,
                    layout=layout,
                    episode=episode_index,
                    threshold=config.trajectory.jump_mad_multiplier,
                )
                combined.extend(generic, metric_prefix=f"trajectory.{layout.column}.generic")
        if canonical is not None and state is not None:
            pauses = pause_runs(
                canonical,
                state,
                timestamps,
                contract=contract,
                episode=episode_index,
                pause_thresholds=config.trajectory.pause_duration_seconds,
                check_exact_duplicates=config.trajectory.check_exact_duplicates,
            )
            combined.extend(pauses, metric_prefix=f"stop.{contract.canonical_action_column}")

    if canonical is not None and state is not None:
        combined.metrics["alignment.same_row_copy"] = _same_row_copy(canonical, state)
        if config.alignment.enabled and timestamps is not None:
            alignment = _alignment(
                canonical, state, timestamps, contract=contract, config=config, episode=episode_index
            )
            combined.extend(alignment, metric_prefix="alignment")

    decision = "quarantine" if combined.deterministic_failure else "keep"
    findings = [
        _finding_row(f, dataset=dataset, episode=episode, source_path=source_path) for f in combined.findings
    ]
    masks = _field_mask_rows(
        findings, episode=episode_index, length=length, fields=contract.all_projection_columns
    )
    reasons = sorted(
        {row["rule_id"] for row in findings if row["deterministic"] and row["disposition"] == "quarantine"}
    )
    return {
        "episode_metrics": [
            {
                "episode_index": episode_index,
                "frame_count": length,
                "task": episode.task if episode else None,
                "semantic_task": episode.semantic_task if episode else None,
                "embodiment_name": episode.embodiment_name if episode else None,
                "source_path": source_path,
                "finding_count": len(findings),
                "metrics": combined.to_dict()["metrics"],
            }
        ],
        "interval_findings": findings,
        "episode_decisions": [
            {
                "episode_index": episode_index,
                "decision": decision,
                "reasons": reasons,
                "origin": "source_contract",
            }
        ],
        "field_masks": masks,
    }


# ── file and dataset scan ───────────────────────────────────────────────


def contiguous_runs(episode_ids: np.ndarray) -> list[tuple[int, int, int]]:
    values = np.asarray(episode_ids).reshape(-1)
    if values.size == 0:
        return []
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [values.size]))
    return [(int(values[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


def scan_data_file(
    root: Path,
    relative_path: str,
    episodes: Mapping[int, EpisodeRecord],
    config: ScanConfig,
    dataset: str,
) -> dict[str, Any]:
    """Scan every episode stored in one data file."""
    import pyarrow.parquet as pq

    contract = config.source_contract
    parquet = pq.ParquetFile(root / relative_path)
    available = set(parquet.schema_arrow.names)
    columns = [name for name in contract.all_projection_columns if name in available]
    table = parquet.read(columns=columns, use_threads=False)
    arrays = {name: arrow_column_to_numpy(table[name]) for name in columns}
    if "episode_index" not in arrays:
        raise ValueError(f"{relative_path} has no episode_index column")
    output: dict[str, Any] = {
        "source_path": relative_path,
        "row_count": int(len(arrays["episode_index"])),
        "missing_required_columns": sorted(set(contract.required_projection_columns) - available),
        "episode_metrics": [],
        "interval_findings": [],
        "episode_decisions": [],
        "field_masks": [],
    }
    seen: set[int] = set()
    for episode, start, end in contiguous_runs(arrays["episode_index"]):
        result = scan_episode(
            {name: values[start:end] for name, values in arrays.items()},
            episode=episodes.get(episode),
            contract=contract,
            config=config,
            dataset=dataset,
            source_path=relative_path,
        )
        if episode in seen:
            # Rows of one episode in several runs cannot be repaired here: quarantine the episode.
            for row in (*result["episode_decisions"], *output["episode_decisions"]):
                if row["episode_index"] == episode:
                    row["decision"] = "quarantine"
                    row["reasons"] = sorted({*row["reasons"], "IDX.NON_CONTIGUOUS_EPISODE_ROWS"})
        seen.add(episode)
        for name, rows in result.items():
            output[name].extend(rows)
    output["episode_indices"] = sorted(seen)
    return output


def _limit_worker_threads() -> None:
    try:
        import pyarrow

        pyarrow.set_cpu_count(1)
    except ImportError:  # pragma: no cover - pyarrow is a dependency
        pass
    try:
        import cv2

        cv2.setNumThreads(1)
    except ImportError:  # pragma: no cover - opencv ships with lerobot
        pass


def run_parallel(
    function: Callable, jobs: Sequence[tuple], workers: int, *, kwargs: Mapping[str, Any] | None = None
) -> list[Any]:
    """``function(*job, **kwargs)`` for every job, in order; ``workers > 1`` uses a spawn process pool."""
    kwargs = dict(kwargs or {})
    if workers <= 1 or len(jobs) <= 1:
        return [function(*job, **kwargs) for job in jobs]
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    import multiprocessing

    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_limit_worker_threads,
    ) as pool:
        futures = [pool.submit(function, *job, **kwargs) for job in jobs]
        return [future.result() for future in futures]
