"""Pause-cleaning policy: turn scan evidence into frame exclusions, conservatively.

The scanner reports facts; this policy is the versioned authority that derives decisions from them. Incomplete
auxiliary, semantic or visual evidence can only keep data or leave it for review, never authorize deletion.

1. AgiBot: the ordered ``action_config[start_frame, end_frame)`` phases of an episode are its authoritative
   useful frames; everything outside their union is excluded. An episode without phases is kept whole.
2. Canonical-action pause candidates (``TRJ.NEAR_STATIC_RUN``), restricted to the useful frames, at least
   ``minimum_exclusion_seconds`` long. Frames are vetoed (kept) where the auxiliary joint action moves, where
   cameras show motion or their evidence is missing/uncertain (``require_visual_evidence``), or inside a
   protected AgiBot pour phase. Remaining pieces at least ``minimum_exclusion_seconds`` long are excluded; a
   piece covering its whole episode quarantines the episode.
3. Contracts whose canonical action target is known to be unreliable (RoboMIND UR) block the dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concertina.data_quality.scan import CANONICAL_ACTION_ORIGIN, PAUSE_RULE, arrow_column_to_numpy

POLICY_VERSION = "pause-cleaning-v5"

# Phase texts of the AgiBot pour task: a dwell while pouring is the task, not a pause.
AGIBOT_POUR_TEXTS = frozenset(
    {
        "Pour the water from the held golden kettle into the nearby cup.",
        "Pour the water from the held white kettle into the nearby cup.",
        "Pour the water from the held tea-colored kettle into the nearby cup.",
        "Pour the water from the held gray water bottle into the nearby cup.",
    }
)

# Joint-space action columns, recorded separately from the canonical end-effector action.
AUXILIARY_ACTION_FIELDS: dict[str, tuple[str, ...]] = {
    "agibot": ("actions.joint.position",),
    "droid": ("action.joint_position",),
    "robomind_agilex": ("actions.joint_position_left", "actions.joint_position_right"),
    "robomind_franka": ("actions.joint_position",),
    "robomind_fr3": ("actions.joint_position",),
    "robomind_ur": ("actions.joint_position",),
}

# RoboMIND UR has long intervals where the end-effector action is frozen while the joint action and the video
# move: its canonical action target needs an authoritative reconstruction before it can be cleaned or trained
# on.
BLOCKED_TARGET_CONTRACTS = frozenset({"robomind_ur"})

Interval = tuple[int, int]


class CleaningPolicyError(ValueError):
    """Evidence is malformed or cannot be attributed."""


@dataclass(frozen=True)
class PausePolicyConfig:
    minimum_exclusion_seconds: float = 3.0
    auxiliary_joint_delta: float = 1e-3
    auxiliary_motion_minimum_seconds: float = 0.4
    auxiliary_motion_gap_seconds: float = 0.15
    require_visual_evidence: bool = True
    use_agibot_useful_intervals: bool = True
    protect_agibot_pour_phase: bool = True

    def __post_init__(self) -> None:
        for name in (
            "minimum_exclusion_seconds",
            "auxiliary_joint_delta",
            "auxiliary_motion_minimum_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise CleaningPolicyError(f"{name} must be finite and positive")
        if not math.isfinite(self.auxiliary_motion_gap_seconds) or self.auxiliary_motion_gap_seconds < 0:
            raise CleaningPolicyError("auxiliary_motion_gap_seconds must be finite and non-negative")

    @property
    def policy_id(self) -> str:
        payload = {"version": POLICY_VERSION, "config": asdict(self)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PolicyResult:
    decisions: tuple[dict[str, Any], ...]
    blockers: tuple[str, ...]

    @property
    def unresolved_review_count(self) -> int:
        return sum(decision["status"] == "review" for decision in self.decisions)

    def applied(self) -> list[dict[str, Any]]:
        return [decision for decision in self.decisions if decision["status"] == "applied"]


def _decision(
    config: PausePolicyConfig,
    *,
    status: str,
    action: str,
    episode: int | None,
    start: int | None,
    end: int | None,
    finding: str | None,
    rule: str,
    reasons: Sequence[str],
) -> dict[str, Any]:
    content = {
        "policy_id": config.policy_id,
        "status": status,
        "action": action,
        "episode_index": episode,
        "start_frame": start,
        "end_frame": end,
        "finding_id": finding,
        "rule": rule,
        "reasons": sorted(set(reasons)),
    }
    return {
        "decision_id": hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:24],
        **content,
    }


def normalize_intervals(
    values: Sequence[Sequence[int] | Mapping[str, Any]], *, start: int, end: int
) -> list[Interval]:
    """Intervals clipped to ``[start, end)``, sorted and merged (touching intervals merge)."""
    intervals = []
    for value in values:
        left, right = (
            (value.get("start_frame"), value.get("end_frame")) if isinstance(value, Mapping) else value
        )
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, int)
            or not isinstance(right, int)
        ):
            raise CleaningPolicyError("interval bounds must be integers")
        left, right = max(start, left), min(end, right)
        if left < right:
            intervals.append((left, right))
    merged: list[Interval] = []
    for left, right in sorted(intervals):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    return merged


def subtract(start: int, end: int, cuts: Sequence[Interval]) -> list[Interval]:
    """``[start, end)`` minus sorted, merged ``cuts``."""
    retained = []
    cursor = start
    for left, right in cuts:
        if cursor < left:
            retained.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        retained.append((cursor, end))
    return retained


def intersect(values: Sequence[Interval], allowed: Sequence[Interval]) -> list[Interval]:
    pieces = [
        (max(left, a), min(right, b))
        for left, right in values
        for a, b in allowed
        if max(left, a) < min(right, b)
    ]
    if not pieces:
        return []
    return normalize_intervals(pieces, start=min(p[0] for p in pieces), end=max(p[1] for p in pieces))


def _useful_intervals(
    phases: Sequence[Mapping[str, Any]], *, episode: int, frame_count: int
) -> list[Interval]:
    intervals = []
    previous_end = 0
    for ordinal, phase in enumerate(phases):
        start, end = phase.get("start_frame"), phase.get("end_frame")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or not previous_end <= start < end <= frame_count
        ):
            raise CleaningPolicyError(
                f"episode {episode} action_config[{ordinal}] is not an ordered in-bounds interval"
            )
        intervals.append((start, end))
        previous_end = end
    return normalize_intervals(intervals, start=0, end=frame_count)


def _pour_protection(phases: Sequence[Mapping[str, Any]] | None, *, start: int, end: int) -> list[Interval]:
    if phases is None or len(phases) < 2 or phases[1].get("action_text") not in AGIBOT_POUR_TEXTS:
        return []
    left, right = phases[1].get("start_frame"), phases[1].get("end_frame")
    if (
        isinstance(left, int)
        and isinstance(right, int)
        and not isinstance(left, bool)
        and not isinstance(right, bool)
    ):
        return normalize_intervals([(left, right)], start=start, end=end)
    return []


def _visual_intervals(
    finding: Mapping[str, Any], *, start: int, end: int, required: bool
) -> tuple[list[Interval], list[Interval], list[str]]:
    """(protected, review, reasons) frames of a candidate from its camera evidence."""
    evidence = finding.get("visual_motion_evidence")
    if not isinstance(evidence, Mapping):
        return [], ([(start, end)] if required else []), (["visual_evidence_missing"] if required else [])
    protected = normalize_intervals(evidence.get("protected_motion_intervals", ()), start=start, end=end)
    review = normalize_intervals(evidence.get("review_intervals", ()), start=start, end=end)
    static = normalize_intervals(evidence.get("static_support_intervals", ()), start=start, end=end)
    if required:
        accounted = normalize_intervals((*static, *protected, *review), start=start, end=end)
        review = normalize_intervals((*review, *subtract(start, end, accounted)), start=start, end=end)
    return protected, review, []


def _auxiliary_intervals(record: Mapping[str, Any], *, start: int, end: int) -> list[Interval]:
    if record.get("status") != "complete":
        return [(start, end)]
    return normalize_intervals(record.get("motion_intervals", ()), start=start, end=end)


def compile_pause_policy(
    *,
    contract_name: str,
    fps: float,
    findings: Sequence[Mapping[str, Any]],
    episode_lengths: Mapping[int, int],
    auxiliary_motion: Mapping[str, Mapping[str, Any]] | None,
    phases_by_episode: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
    config: PausePolicyConfig = PausePolicyConfig(),
) -> PolicyResult:
    """Derived decisions (``applied`` exclusions/quarantines, ``vetoed_keep``, ``review``, ``blocked``)."""
    if not math.isfinite(fps) or fps <= 0:
        raise CleaningPolicyError("fps must be finite and positive")
    blockers = []
    if contract_name in BLOCKED_TARGET_CONTRACTS:
        blockers.append(
            "target_representation_blocked:canonical_eef_action_requires_authoritative_reconstruction"
        )
    decisions: list[dict[str, Any]] = []

    def keep(episode, start, end, finding, rule, reasons):
        decisions.append(
            _decision(
                config,
                status="vetoed_keep",
                action="keep",
                episode=episode,
                start=start,
                end=end,
                finding=finding,
                rule=rule,
                reasons=reasons,
            )
        )

    useful: dict[int, list[Interval]] = {}
    agibot = contract_name == "agibot" and config.use_agibot_useful_intervals
    if agibot and (episode_lengths or phases_by_episode):
        if phases_by_episode is None or set(phases_by_episode) != set(episode_lengths):
            raise CleaningPolicyError(
                "AgiBot action_config phases must cover exactly the episodes of the scan"
            )
        for episode, frame_count in sorted(episode_lengths.items()):
            phases = phases_by_episode[episode]
            if not phases:
                keep(
                    episode,
                    0,
                    frame_count,
                    None,
                    "AGIBOT.AUTHORITATIVE_USEFUL_INTERVAL_MISSING",
                    ["authoritative_agibot_useful_interval_missing"],
                )
                continue
            useful[episode] = _useful_intervals(phases, episode=episode, frame_count=frame_count)
            for left, right in subtract(0, frame_count, useful[episode]):
                if blockers:
                    keep(
                        episode,
                        left,
                        right,
                        None,
                        "DATASET.BLOCKED_NO_MUTATION",
                        [*blockers, "dataset_application_blocked"],
                    )
                else:
                    decisions.append(
                        _decision(
                            config,
                            status="applied",
                            action="exclude_frame_interval",
                            episode=episode,
                            start=left,
                            end=right,
                            finding=None,
                            rule="AGIBOT.OUTSIDE_AUTHORITATIVE_USEFUL_INTERVAL",
                            reasons=["outside_authoritative_agibot_useful_interval"],
                        )
                    )

    excludes: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    pauses = [row for row in findings if row.get("rule_id") == PAUSE_RULE]
    for finding in sorted(pauses, key=lambda r: (r["episode_index"], r["start_frame"], r["finding_id"])):
        finding_id, episode = finding["finding_id"], finding["episode_index"]
        start, end = finding["start_frame"], finding["end_frame"]
        if episode not in episode_lengths or not 0 <= start < end <= episode_lengths[episode]:
            raise CleaningPolicyError(f"pause finding {finding_id} has an invalid interval")
        analysis = [(start, end)]
        if agibot:
            analysis = intersect(analysis, useful.get(episode, ()))
            if not analysis:
                continue
        metrics = finding.get("metrics") or {}
        if metrics.get("candidate_origin") != CANONICAL_ACTION_ORIGIN:
            for left, right in analysis:
                decisions.append(
                    _decision(
                        config,
                        status="review",
                        action="review",
                        episode=episode,
                        start=left,
                        end=right,
                        finding=finding_id,
                        rule="PAUSE.AMBIGUOUS",
                        reasons=["not_canonical_action_candidate"],
                    )
                )
            continue
        duration = float(metrics.get("duration_seconds", (end - start - 1) / fps))
        if duration + 1e-5 < config.minimum_exclusion_seconds:
            for left, right in analysis:
                keep(
                    episode,
                    left,
                    right,
                    finding_id,
                    "PAUSE.MINIMUM_DURATION",
                    ["below_automatic_exclusion_duration"],
                )
            continue

        veto: list[Interval] = []
        reasons: list[str] = []
        record = (auxiliary_motion or {}).get(finding_id)
        if record is None:
            veto.append((start, end))
            reasons.append("auxiliary_motion_evidence_missing")
        else:
            moving = _auxiliary_intervals(record, start=start, end=end)
            veto.extend(moving)
            if moving:
                reasons.append(
                    "auxiliary_action_motion"
                    if record.get("status") == "complete"
                    else "auxiliary_motion_evidence_incomplete"
                )
        protected, review, visual_reasons = _visual_intervals(
            finding, start=start, end=end, required=config.require_visual_evidence
        )
        veto.extend(protected)
        veto.extend(review)
        reasons += (["visual_motion_veto"] if protected else []) + (
            ["visual_evidence_uncertain"] if review else []
        )
        reasons += visual_reasons
        if contract_name == "agibot" and config.protect_agibot_pour_phase:
            pour = _pour_protection((phases_by_episode or {}).get(episode), start=start, end=end)
            veto.extend(pour)
            if pour:
                reasons.append("approved_semantic_dwell")
        veto = intersect(normalize_intervals(veto, start=start, end=end), analysis)

        for analysis_start, analysis_end in analysis:
            pieces = subtract(analysis_start, analysis_end, intersect(veto, [(analysis_start, analysis_end)]))
            for left, right in pieces:
                if (right - left - 1) / fps + 1e-5 < config.minimum_exclusion_seconds:
                    continue
                if blockers:
                    keep(
                        episode,
                        left,
                        right,
                        finding_id,
                        "DATASET.BLOCKED_NO_MUTATION",
                        [*blockers, "dataset_application_blocked"],
                    )
                else:
                    excludes[episode].append((left, right, finding_id))
        for left, right in veto:
            keep(episode, left, right, finding_id, "PAUSE.PROTECTION_VETO", reasons)

    for episode, values in sorted(excludes.items()):
        for left, right in normalize_intervals(
            [(a, b) for a, b, _ in values], start=0, end=episode_lengths[episode]
        ):
            ids = sorted({fid for a, b, fid in values if a < right and b > left})
            whole = left == 0 and right == episode_lengths[episode]
            decisions.append(
                _decision(
                    config,
                    status="applied",
                    action="quarantine_episode" if whole else "exclude_frame_interval",
                    episode=episode,
                    start=None if whole else left,
                    end=None if whole else right,
                    finding=ids[0] if len(ids) == 1 else None,
                    rule="PAUSE.CANONICAL_STATIC_COMPOSITE",
                    reasons=[
                        "canonical_action_static",
                        "auxiliary_motion_absent",
                        "visual_static_support",
                        "semantic_protection_absent",
                    ],
                )
            )
    if blockers:
        decisions.append(
            _decision(
                config,
                status="blocked",
                action="block_dataset",
                episode=None,
                start=None,
                end=None,
                finding=None,
                rule="DATASET.TARGET_OR_CATALOG_BLOCK",
                reasons=blockers,
            )
        )
    decisions.sort(
        key=lambda row: (
            -1 if row["episode_index"] is None else row["episode_index"],
            -1 if row["start_frame"] is None else row["start_frame"],
            row["action"],
            row["decision_id"],
        )
    )
    return PolicyResult(decisions=tuple(decisions), blockers=tuple(blockers))


def load_agibot_phases(root: Path) -> dict[int, tuple[dict[str, Any], ...]]:
    """``action_config`` phases of every episode (empty when the metadata has no such column)."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(root / "meta" / "episodes"), format="parquet")
    if "action_config" not in dataset.schema.names:
        return {}
    phases: dict[int, tuple[dict[str, Any], ...]] = {}
    for row in dataset.to_table(columns=["episode_index", "length", "action_config"]).to_pylist():
        episode, length = int(row["episode_index"]), int(row["length"])
        if episode in phases:
            raise CleaningPolicyError(f"episode {episode} appears twice in meta/episodes")
        values = tuple(dict(phase) for phase in row.get("action_config") or ())
        _useful_intervals(values, episode=episode, frame_count=length)
        phases[episode] = values
    return phases


def _numeric_matrix(column: Any) -> np.ndarray:
    values = arrow_column_to_numpy(column)
    if values.dtype == object:
        raise CleaningPolicyError("auxiliary action columns must hold fixed-size numeric vectors")
    return np.asarray(values, dtype=np.float64).reshape(len(values), -1)


def _close_short_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    output = np.asarray(mask, dtype=bool).copy()
    if maximum_gap <= 0 or not output.any():
        return output
    edges = np.diff(np.pad((~output).astype(np.int8), (1, 1)))
    for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
        if left > 0 and right < len(output) and right - left <= maximum_gap:
            output[left:right] = True
    return output


def _transition_frame_intervals(mask: np.ndarray, *, frame_start: int) -> list[list[int]]:
    if not len(mask) or not mask.any():
        return []
    edges = np.diff(np.pad(mask.astype(np.int8), (1, 1)))
    # Transition k moves frame k to k + 1 and protects both: [k, k + 2).
    return [
        [frame_start + int(left), frame_start + int(right) + 1]
        for left, right in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True)
    ]


def extract_auxiliary_motion(
    *,
    root: Path,
    contract_name: str,
    fps: float,
    findings: Sequence[Mapping[str, Any]],
    config: PausePolicyConfig = PausePolicyConfig(),
) -> dict[str, dict[str, Any]]:
    """Joint-action motion inside every pause candidate long enough to be excluded, per finding id."""
    import pyarrow.parquet as pq

    fields = AUXILIARY_ACTION_FIELDS.get(contract_name)
    if fields is None:
        return {}
    by_file: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for finding in findings:
        metrics = finding.get("metrics") or {}
        duration = metrics.get("duration_seconds")
        if (
            finding.get("rule_id") != PAUSE_RULE
            or metrics.get("candidate_origin") != CANONICAL_ACTION_ORIGIN
            or not isinstance(duration, (int, float))
            or float(duration) + 1e-5 < config.minimum_exclusion_seconds
        ):
            continue
        by_file[finding["source_path"]].append(finding)
    maximum_gap = int(math.floor(config.auxiliary_motion_gap_seconds * fps + 1e-9))
    minimum_transitions = max(1, int(math.ceil(config.auxiliary_motion_minimum_seconds * fps - 1e-9)))
    records: dict[str, dict[str, Any]] = {}
    for relative, file_findings in sorted(by_file.items()):
        path = root / relative
        missing = [field for field in fields if field not in pq.read_schema(path).names]
        if missing:
            for finding in file_findings:
                records[finding["finding_id"]] = {
                    "status": "incomplete",
                    "reason": "auxiliary_action_fields_missing",
                    "missing_fields": missing,
                }
            continue
        table = pq.read_table(path, columns=["episode_index", "frame_index", *fields])
        episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        values = np.concatenate([_numeric_matrix(table[field]) for field in fields], axis=1)
        for finding in file_findings:
            rows = np.flatnonzero(episodes == finding["episode_index"])
            start, end = finding["start_frame"], finding["end_frame"]
            if not len(rows) or rows[-1] - rows[0] + 1 != len(rows):
                records[finding["finding_id"]] = {
                    "status": "incomplete",
                    "reason": "episode_rows_not_contiguous",
                }
                continue
            if not np.array_equal(frames[rows], np.arange(len(rows))) or not 0 <= start < end <= len(rows):
                raise CleaningPolicyError(
                    f"episode {finding['episode_index']} frames do not match the finding"
                )
            window = values[rows[0] + start : rows[0] + end]
            if not np.isfinite(window).all():
                records[finding["finding_id"]] = {
                    "status": "incomplete",
                    "reason": "auxiliary_action_nonfinite",
                }
                continue
            delta = np.linalg.norm(np.diff(window, axis=0), axis=1)
            motion = _close_short_gaps(delta > config.auxiliary_joint_delta, maximum_gap)
            # Sparse controller impulses do not veto a pause unless the activity reaches the persistence
            # floor.
            persistent = int(motion.sum()) >= minimum_transitions
            records[finding["finding_id"]] = {
                "status": "complete",
                "source_fields": list(fields),
                "motion_intervals": _transition_frame_intervals(motion, frame_start=start)
                if persistent
                else [],
                "active_transition_count": int(motion.sum()),
                "maximum_joint_delta": float(delta.max(initial=0.0)),
            }
    return records
