"""Camera evidence for canonical-action pause candidates.

Only pause findings of the canonical action are examined. Per packed MP4 and camera, the candidate windows are
merged, decoded once at ``sample_fps`` and judged twice: FFmpeg ``freezedetect`` and two optical-flow
estimators (DIS and Farneback) must both see motion for a time run to count as motion (external cameras first
remove the median whole-frame flow, ego cameras keep it since camera motion is robot motion). Per candidate
frame, motion on an ego camera or on ``external_corroborating_cameras`` external cameras protects the frame;
disagreement, single-camera motion, missing cameras and decode failures leave it for review; the rest is
static support. The evidence can only protect frames or leave them for review: it never authorizes deletion by
itself.
"""

from __future__ import annotations

import math
import re
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concertina.data_quality.config import VisualMotionConfig
from concertina.data_quality.scan import CANONICAL_ACTION_ORIGIN, PAUSE_RULE

FFMPEG_UNAVAILABLE = "ffmpeg_unavailable"
_FREEZE_START = re.compile(r"freeze_start:\s*(-?[0-9]+(?:\.[0-9]+)?)")
_FREEZE_END = re.compile(r"freeze_end:\s*(-?[0-9]+(?:\.[0-9]+)?)")
Run = tuple[float, float]


@dataclass(frozen=True)
class Candidate:
    finding_id: str
    episode_index: int
    start_frame: int
    end_frame: int
    duration_seconds: float


@dataclass(frozen=True)
class CandidateSpan:
    """A candidate's frames in the time base of one packed MP4."""

    finding_id: str
    episode_index: int
    start_frame: int
    end_frame: int
    episode_from_timestamp: float
    start_seconds: float
    end_seconds: float
    status: str = "ready"
    error: str | None = None


@dataclass(frozen=True)
class VideoGroup:
    camera: str
    relative_path: str | None
    spans: tuple[CandidateSpan, ...]


def select_candidates(
    findings: Sequence[Mapping[str, Any]], *, canonical_action_column: str, minimum_seconds: float
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Canonical-action pause findings at least ``minimum_seconds`` long; the others with their reasons."""
    selected: list[Candidate] = []
    skipped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for finding in findings:
        if finding.get("rule_id") != PAUSE_RULE:
            continue
        metrics = finding.get("metrics") or {}
        reasons = []
        if finding["finding_id"] in seen:
            reasons.append("duplicate_finding_id")
        if finding.get("field") != canonical_action_column:
            reasons.append("field_is_not_canonical_action")
        if metrics.get("candidate_origin") != CANONICAL_ACTION_ORIGIN:
            reasons.append("candidate_origin_is_not_canonical_action")
        duration = metrics.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            reasons.append("invalid_candidate_duration")
        # Timestamps are often float32: a nominal 3 s run may serialize as 2.99999...
        elif duration + 1e-5 < minimum_seconds:
            reasons.append("below_visual_scan_minimum_candidate_seconds")
        if reasons:
            skipped.append(
                {
                    "finding_id": finding["finding_id"],
                    "episode_index": finding["episode_index"],
                    "start_frame": finding["start_frame"],
                    "end_frame": finding["end_frame"],
                    "reasons": sorted(set(reasons)),
                }
            )
            continue
        seen.add(finding["finding_id"])
        selected.append(
            Candidate(
                finding_id=finding["finding_id"],
                episode_index=int(finding["episode_index"]),
                start_frame=int(finding["start_frame"]),
                end_frame=int(finding["end_frame"]),
                duration_seconds=float(duration),
            )
        )
    selected.sort(key=lambda item: (item.episode_index, item.start_frame, item.end_frame, item.finding_id))
    return selected, skipped


def plan_groups(
    candidates: Sequence[Candidate],
    *,
    episode_rows: Mapping[int, Mapping[str, Any]],
    info: Mapping[str, Any],
    cameras: Sequence[str],
) -> list[VideoGroup]:
    """Group candidate spans by packed MP4 file and camera, so each file is decoded once."""
    fps = float(info["fps"])
    template = info.get("video_path")
    grouped: dict[tuple[str | None, str], list[CandidateSpan]] = defaultdict(list)
    for candidate in candidates:
        row = episode_rows.get(candidate.episode_index)
        for camera in cameras:
            keys = [f"videos/{camera}/{name}" for name in ("chunk_index", "file_index", "from_timestamp")]
            status, error, path, origin = "ready", None, None, 0.0
            if row is None:
                status, error = (
                    "missing_episode_metadata",
                    f"no metadata row for episode {candidate.episode_index}",
                )
            elif not isinstance(template, str) or any(row.get(key) is None for key in keys):
                status, error = (
                    "missing_camera_mapping",
                    f"episode metadata has no mapping for camera {camera!r}",
                )
            else:
                origin = float(row[keys[2]])
                path = template.format(
                    video_key=camera, chunk_index=int(row[keys[0]]), file_index=int(row[keys[1]])
                )
            grouped[(path, camera)].append(
                CandidateSpan(
                    finding_id=candidate.finding_id,
                    episode_index=candidate.episode_index,
                    start_frame=candidate.start_frame,
                    end_frame=candidate.end_frame,
                    episode_from_timestamp=origin,
                    start_seconds=origin + candidate.start_frame / fps,
                    end_seconds=origin + candidate.end_frame / fps,
                    status=status,
                    error=error,
                )
            )
    return [
        VideoGroup(
            camera=camera,
            relative_path=path,
            spans=tuple(sorted(spans, key=lambda s: (s.start_seconds, s.end_seconds, s.finding_id))),
        )
        for (path, camera), spans in sorted(grouped.items(), key=lambda item: (str(item[0][0]), item[0][1]))
    ]


def merge_runs(runs: Sequence[Run], *, gap: float = 0.0, minimum_duration: float = 0.0) -> list[Run]:
    merged: list[list[float]] = []
    for start, end in sorted(runs):
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged if end - start + 1e-9 >= minimum_duration]


def intersect_runs(left: Sequence[Run], right: Sequence[Run]) -> list[Run]:
    output = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start, end = max(left_start, right_start), min(left_end, right_end)
            if end > start:
                output.append((start, end))
    return merge_runs(output)


def subtract_runs(source: Sequence[Run], removed: Sequence[Run]) -> list[Run]:
    output: list[Run] = []
    for start, end in source:
        pieces = [(start, end)]
        for cut_start, cut_end in removed:
            next_pieces = []
            for piece_start, piece_end in pieces:
                if cut_end <= piece_start or cut_start >= piece_end:
                    next_pieces.append((piece_start, piece_end))
                    continue
                if cut_start > piece_start:
                    next_pieces.append((piece_start, min(cut_start, piece_end)))
                if cut_end < piece_end:
                    next_pieces.append((max(cut_end, piece_start), piece_end))
            pieces = next_pieces
        output.extend(pieces)
    return merge_runs(output)


def union_windows(
    spans: Sequence[CandidateSpan], config: VisualMotionConfig
) -> list[tuple[float, float, list[str]]]:
    windows: list[list[Any]] = []
    for span in spans:
        start = max(0.0, span.start_seconds - config.decode_padding_seconds)
        end = span.end_seconds + config.decode_padding_seconds
        if windows and start <= windows[-1][1] + config.union_gap_seconds:
            windows[-1][1] = max(windows[-1][1], end)
            windows[-1][2].append(span.finding_id)
        else:
            windows.append([start, end, [span.finding_id]])
    return [(float(start), float(end), ids) for start, end, ids in windows]


def _resize_gray(frame: np.ndarray, cv2: Any, max_dimension: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(
            frame, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA
        )
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)


def decode_window(
    container: Any, stream: Any, *, start: float, end: float, sample_fps: float, cv2: Any, max_dimension: int
) -> list[tuple[float, np.ndarray]]:
    """Gray frames of ``[start, end]`` sampled at ``sample_fps``."""
    container.seek(
        int(max(start, 0.0) / float(stream.time_base)), stream=stream, any_frame=False, backward=True
    )
    interval = 1.0 / sample_fps
    next_sample = start
    frames = []
    for frame in container.decode(stream):
        if frame.pts is None:
            continue
        timestamp = float(frame.pts * stream.time_base)
        if timestamp + 1e-9 < start:
            continue
        if timestamp > end + interval:
            break
        if timestamp + interval * 0.2 < next_sample:
            continue
        frames.append((timestamp, _resize_gray(frame.to_ndarray(format="rgb24"), cv2, max_dimension)))
        next_sample = max(next_sample + interval, timestamp + interval)
    return frames


def flow_motion_runs(
    frames: Sequence[tuple[float, np.ndarray]],
    *,
    cv2: Any,
    config: VisualMotionConfig,
    compensate_global: bool,
) -> tuple[list[Run], dict[str, Any]]:
    """Time runs where DIS and Farneback flow both show local motion (empty unless it is sustained)."""
    if len(frames) < 2:
        raise ValueError("visual motion decode produced fewer than two sampled frames")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    active: list[Run] = []
    for (left_time, left), (right_time, right) in zip(frames[:-1], frames[1:], strict=True):
        flows = [
            dis.calc(left, right, None),
            cv2.calcOpticalFlowFarneback(left, right, None, 0.5, 3, 15, 3, 5, 1.2, 0),
        ]
        moving = True
        for flow in flows:
            if compensate_global:
                flow = flow - np.median(flow.reshape(-1, 2), axis=0)
            magnitude = np.linalg.norm(flow, axis=2)
            moving &= bool(
                float(np.quantile(magnitude, config.flow_quantile)) >= config.flow_threshold_px
                and float(np.mean(magnitude >= config.flow_active_pixel_threshold_px))
                >= config.flow_active_area_fraction
            )
        if moving:
            active.append((left_time, right_time))
    merged = merge_runs(active, gap=config.motion_gap_seconds)
    longest = max((end - start for start, end in merged), default=0.0)
    total = sum(end - start for start, end in merged)
    qualified = longest >= config.minimum_motion_run_seconds or total >= config.minimum_total_motion_seconds
    summary = {
        "pairs": len(frames) - 1,
        "longest_run_s": longest,
        "total_active_s": total,
        "qualified": qualified,
    }
    return (merged if qualified else []), summary


def ffmpeg_motion_runs(
    video_path: Path, *, start: float, end: float, config: VisualMotionConfig, ffmpeg: str
) -> tuple[list[Run], list[Run]]:
    """Motion runs (the window minus ``freezedetect`` freezes) and the freezes themselves."""
    duration = end - start
    command = [
        ffmpeg,
        "-hide_banner",
        "-nostats",
        "-ss",
        f"{start:.9f}",
        "-t",
        f"{duration:.9f}",
        "-i",
        str(video_path),
        "-vf",
        f"freezedetect=noise={config.ffmpeg_noise_db:g}dB:d={config.ffmpeg_min_duration_seconds:g}",
        "-an",
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, capture_output=True, check=False, text=True, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg freezedetect failed ({completed.returncode}): {completed.stderr[-2000:]}")
    events = []
    for line in (completed.stderr or "").splitlines():
        if match := _FREEZE_START.search(line):
            events.append(("start", float(match.group(1))))
        if match := _FREEZE_END.search(line):
            events.append(("end", float(match.group(1))))
    freezes: list[Run] = []
    pending: float | None = None
    for event, value in events:
        relative = min(max(value, 0.0), duration)
        if event == "start":
            pending = relative
        elif pending is not None and relative > pending:
            freezes.append((start + pending, start + relative))
            pending = None
    if pending is not None:
        freezes.append((start + pending, end))
    freezes = merge_runs(freezes)
    motion = merge_runs(
        subtract_runs([(start, end)], freezes),
        gap=config.motion_gap_seconds,
        minimum_duration=config.minimum_motion_run_seconds,
    )
    return motion, freezes


def frame_runs(runs: Sequence[Run], span: CandidateSpan, *, fps: float) -> list[list[int]]:
    """Time runs as episode frame intervals clipped to the candidate."""
    projected = []
    for start, end in runs:
        left, right = max(start, span.start_seconds), min(end, span.end_seconds)
        if right > left:
            projected.append(
                (
                    max(span.start_frame, int(math.floor((left - span.episode_from_timestamp) * fps))),
                    min(span.end_frame, int(math.ceil((right - span.episode_from_timestamp) * fps))),
                )
            )
    merged: list[list[int]] = []
    for start, end in sorted(projected):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def camera_role(camera: str, patterns: Sequence[str]) -> str:
    return "ego" if any(pattern.casefold() in camera.casefold() for pattern in patterns) else "external"


def _failed(span: CandidateSpan, status: str, error: str | None) -> dict[str, Any]:
    return {
        "finding_id": span.finding_id,
        "status": status,
        "error": error,
        "motion_frame_intervals": [],
        "gray_frame_intervals": [[span.start_frame, span.end_frame]],
    }


def analyze_group(root: Path, group: VideoGroup, *, fps: float, config: VisualMotionConfig) -> dict[str, Any]:
    """Per-candidate motion evidence of one camera file (worker entry point)."""
    base = {"camera": group.camera, "relative_path": group.relative_path}
    if group.relative_path is None:
        return {**base, "evidence": [_failed(span, span.status, span.error) for span in group.spans]}
    path = root / group.relative_path
    if not path.is_file():
        error = f"video file does not exist: {group.relative_path}"
        return {**base, "evidence": [_failed(span, "missing_video", error) for span in group.spans]}
    ffmpeg = shutil.which(config.ffmpeg_binary)
    if ffmpeg is None:
        error = f"FFmpeg executable {config.ffmpeg_binary!r} is not on PATH"
        return {**base, "evidence": [_failed(span, FFMPEG_UNAVAILABLE, error) for span in group.spans]}
    evidence: dict[str, dict[str, Any]] = {}
    try:
        import av
        import cv2

        cv2.setNumThreads(1)
        spans = {span.finding_id: span for span in group.spans}
        with av.open(str(path), mode="r") as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            for start, end, ids in union_windows(group.spans, config):
                try:
                    frames = decode_window(
                        container,
                        stream,
                        start=start,
                        end=end,
                        sample_fps=config.sample_fps,
                        cv2=cv2,
                        max_dimension=config.flow_max_dimension,
                    )
                    flow, summary = flow_motion_runs(
                        frames,
                        cv2=cv2,
                        config=config,
                        compensate_global=camera_role(group.camera, config.ego_camera_patterns) == "external",
                    )
                    freeze_motion, _ = ffmpeg_motion_runs(
                        path, start=start, end=end, config=config, ffmpeg=ffmpeg
                    )
                    confirmed = intersect_runs(flow, freeze_motion)
                    gray = merge_runs(
                        [*subtract_runs(flow, freeze_motion), *subtract_runs(freeze_motion, flow)]
                    )
                    for finding in ids:
                        span = spans[finding]
                        evidence[finding] = {
                            "finding_id": finding,
                            "status": "complete",
                            "motion_frame_intervals": frame_runs(confirmed, span, fps=fps),
                            "gray_frame_intervals": frame_runs(gray, span, fps=fps),
                            "flow_motion_frame_intervals": frame_runs(flow, span, fps=fps),
                            "ffmpeg_motion_frame_intervals": frame_runs(freeze_motion, span, fps=fps),
                            "sampled_frame_count": len(frames),
                            "flow": summary,
                        }
                except Exception as exc:  # noqa: BLE001 - decode failures become review evidence
                    for finding in ids:
                        evidence[finding] = _failed(
                            spans[finding], "decode_failure", f"{type(exc).__name__}: {exc}"
                        )
    except Exception as exc:  # noqa: BLE001 - a file that cannot be opened becomes review evidence
        for span in group.spans:
            evidence[span.finding_id] = _failed(span, "decode_failure", f"{type(exc).__name__}: {exc}")
    return {**base, "evidence": [evidence[span.finding_id] for span in group.spans]}


def _covers(intervals: Sequence[Sequence[int]], start: int, end: int) -> bool:
    return any(int(left) < end and int(right) > start for left, right in intervals)


def aggregate_evidence(
    candidates: Sequence[Candidate],
    camera_results: Sequence[Mapping[str, Any]],
    *,
    cameras: Sequence[str],
    config: VisualMotionConfig,
) -> dict[str, dict[str, Any]]:
    """Per finding: ``protected_motion_intervals``, ``review_intervals`` and ``static_support_intervals``."""
    by_finding: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in camera_results:
        for row in result["evidence"]:
            by_finding[row["finding_id"]].append(
                {
                    **row,
                    "camera": result["camera"],
                    "role": camera_role(result["camera"], config.ego_camera_patterns),
                }
            )
    output = {}
    for candidate in candidates:
        rows = sorted(by_finding.get(candidate.finding_id, []), key=lambda row: row["camera"])
        missing = sorted(set(cameras) - {row["camera"] for row in rows})
        boundaries = {candidate.start_frame, candidate.end_frame}
        for row in rows:
            for name in ("motion_frame_intervals", "gray_frame_intervals"):
                for start, end in row.get(name, []):
                    boundaries.update((max(candidate.start_frame, start), min(candidate.end_frame, end)))
        segments: list[tuple[int, int, str, list[str]]] = []
        ordered = sorted(boundaries)
        for start, end in zip(ordered[:-1], ordered[1:], strict=True):
            if end <= start:
                continue
            motion = [row for row in rows if _covers(row.get("motion_frame_intervals", []), start, end)]
            gray = [row for row in rows if _covers(row.get("gray_frame_intervals", []), start, end)]
            failed = [row for row in rows if row["status"] != "complete"]
            external = sum(row["role"] == "external" for row in motion)
            if motion and (
                any(row["role"] == "ego" for row in motion)
                or external >= config.external_corroborating_cameras
            ):
                category, reasons = "motion_veto", [row["camera"] for row in motion]
            elif motion:
                category, reasons = (
                    "review",
                    ["single_external_camera_motion", *(row["camera"] for row in motion)],
                )
            elif gray or failed or missing:
                category = "review"
                reasons = [
                    *(["algorithm_disagreement"] if gray else []),
                    *(["camera_failure"] if failed else []),
                    *(f"missing_camera:{camera}" for camera in missing),
                ]
            else:
                category, reasons = "static_support", []
            if segments and segments[-1][2:] == (category, reasons) and segments[-1][1] == start:
                segments[-1] = (segments[-1][0], end, category, reasons)
            else:
                segments.append((start, end, category, reasons))
        protected = [
            {"start_frame": s, "end_frame": e, "cameras": r} for s, e, c, r in segments if c == "motion_veto"
        ]
        review = [
            {"start_frame": s, "end_frame": e, "reasons": r} for s, e, c, r in segments if c == "review"
        ]
        static = [{"start_frame": s, "end_frame": e} for s, e, c, _ in segments if c == "static_support"]
        output[candidate.finding_id] = {
            "finding_id": candidate.finding_id,
            "episode_index": candidate.episode_index,
            "start_frame": candidate.start_frame,
            "end_frame": candidate.end_frame,
            "status": "complete",
            "disposition": "motion_veto" if protected else "review" if review else "static_support",
            "protected_motion_intervals": protected,
            "review_intervals": review,
            "static_support_intervals": static,
            "camera_evidence": rows,
        }
    return output


def skipped_evidence(skipped: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Ineligible pause findings stay entirely under review."""
    return {
        item["finding_id"]: {
            **item,
            "status": "not_eligible",
            "disposition": "review",
            "protected_motion_intervals": [],
            "review_intervals": [
                {
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                    "reasons": item["reasons"],
                }
            ],
            "static_support_intervals": [],
            "camera_evidence": [],
        }
        for item in skipped
    }


def tool_versions(config: VisualMotionConfig) -> dict[str, Any]:
    import cv2

    ffmpeg = shutil.which(config.ffmpeg_binary)
    version = None
    if ffmpeg is not None:
        completed = subprocess.run(
            [ffmpeg, "-version"], capture_output=True, check=False, text=True, timeout=10
        )
        version = (
            (completed.stdout or completed.stderr).splitlines()[0].strip()
            if completed.returncode == 0
            else None
        )
    return {"opencv": cv2.__version__, "ffmpeg": version}
