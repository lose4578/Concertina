"""Bounded video health probe: decode a few consecutive-frame bursts from a sample of the MP4 files.

Per camera, a deterministic fraction of the packed MP4 files referenced by the episode metadata is opened;
each probe decodes ``sample_count`` bursts of ``minimum_consecutive_frames`` truly consecutive frames (one
seek and a sequential decode per burst). Findings: decode failures and incomplete bursts, duration, frame size
or frame rate that disagree with ``meta/info.json`` (severity critical/high: they block the release when the
video stage is required), and bursts whose every frame is nearly black or white (medium: review only).
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concertina.data_quality.config import VideoConfig

BLOCKING_SEVERITIES = frozenset({"critical", "high"})
_FRAME_RATE_ABSOLUTE_TOLERANCE = 0.05
_FRAME_RATE_RELATIVE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class VideoFile:
    relative_path: str
    camera: str

    @property
    def file_id(self) -> str:
        return hashlib.sha256(self.relative_path.encode("utf-8")).hexdigest()[:20]


def referenced_video_files(rows: Iterable[Mapping[str, Any]], info: Mapping[str, Any]) -> list[VideoFile]:
    """Packed MP4 files referenced by ``meta/episodes`` rows, per camera."""
    template = info.get("video_path")
    cameras = sorted(key for key, feature in info["features"].items() if feature.get("dtype") == "video")
    if not cameras:
        return []
    if not isinstance(template, str):
        raise ValueError("meta/info.json declares video features but no video_path")
    files = set()
    for row in rows:
        for camera in cameras:
            chunk, index = row.get(f"videos/{camera}/chunk_index"), row.get(f"videos/{camera}/file_index")
            if chunk is None or index is None:
                continue
            files.add(
                VideoFile(
                    template.format(video_key=camera, chunk_index=int(chunk), file_index=int(index)), camera
                )
            )
    return sorted(files, key=lambda item: item.relative_path)


def sample_video_files(files: Sequence[VideoFile], sample_fraction: float) -> list[VideoFile]:
    """A stable quota per camera (ordered by path hash), interleaved across cameras."""
    by_camera: dict[str, list[VideoFile]] = defaultdict(list)
    for item in files:
        by_camera[item.camera].append(item)
    selected = {
        camera: sorted(items, key=lambda item: (item.file_id, item.relative_path))[
            : max(1, math.ceil(len(items) * sample_fraction))
        ]
        for camera, items in by_camera.items()
    }
    output = []
    for position in range(max((len(items) for items in selected.values()), default=0)):
        for camera in sorted(selected):
            if position < len(selected[camera]):
                output.append(selected[camera][position])
    return output


def _timestamp(frame: Any, stream: Any) -> float | None:
    return None if frame.pts is None else float(frame.pts * stream.time_base)


def _decode_burst(
    container: Any, stream: Any, start: float, *, frame_count: int
) -> list[tuple[np.ndarray, float]]:
    """``frame_count`` consecutive frames from the one nearest ``start`` (one seek, one sequential decode)."""
    target = max(float(start), 0.0)
    container.seek(int(target / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
    previous: tuple[np.ndarray, float] | None = None
    decoded: list[tuple[np.ndarray, float]] = []
    for position, frame in enumerate(container.decode(stream)):
        if position >= max(10_000, frame_count * 32):
            break
        current = _timestamp(frame, stream)
        if current is None:
            if decoded:
                break
            previous = None
            continue
        candidate = (frame.to_ndarray(format="rgb24"), current)
        if not decoded:
            if current < target:
                previous = candidate
                continue
            if previous is not None and abs(previous[1] - target) <= abs(current - target):
                decoded.append(previous)
                if len(decoded) < frame_count:
                    decoded.append(candidate)
            else:
                decoded.append(candidate)
        else:
            if current <= decoded[-1][1]:
                break
            decoded.append(candidate)
        if len(decoded) >= frame_count:
            break
    return decoded[:frame_count]


def _image_metrics(frame: np.ndarray, *, black: int, white: int) -> dict[str, Any]:
    import cv2

    gray = np.mean(frame.astype(np.float32) / 255.0, axis=2)
    return {
        "height": int(frame.shape[0]),
        "width": int(frame.shape[1]),
        "black_fraction": float(np.mean(frame <= black)),
        "white_fraction": float(np.mean(frame >= white)),
        "blur_laplacian_variance": float(cv2.Laplacian((gray * 255).astype(np.uint8), cv2.CV_64F).var()),
    }


def probe_video_file(root: Path, item: VideoFile, config: VideoConfig) -> dict[str, Any]:
    """Decode bounded bursts of one MP4 file; errors are returned as data."""
    import av

    burst_size = config.minimum_consecutive_frames
    output: dict[str, Any] = {
        "relative_path": item.relative_path,
        "camera": item.camera,
        "decode_ok": False,
        "samples": [],
        "error": None,
    }
    try:
        with av.open(str(root / item.relative_path), mode="r") as container:
            if not container.streams.video:
                raise ValueError("container has no video stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            if stream.duration is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = float(container.duration / 1_000_000)
            else:
                duration = None
            rate = float(stream.average_rate) if stream.average_rate is not None else None
            output.update(
                {
                    "codec": stream.codec_context.name,
                    "width": int(stream.codec_context.width),
                    "height": int(stream.codec_context.height),
                    "average_rate": rate,
                    "duration_s": duration,
                }
            )
            if duration is None or not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"video duration must be finite and positive, got {duration!r}")
            frame_rate = rate
            if frame_rate is None or not math.isfinite(frame_rate) or frame_rate <= 0:
                frame_rate = int(stream.frames) / duration if stream.frames else None
            interval = 1.0 / frame_rate if frame_rate else 0.0
            for index, requested in enumerate(
                np.linspace(0.0, max(duration - burst_size * interval, 0.0), config.sample_count_per_file)
            ):
                burst = _decode_burst(container, stream, float(requested), frame_count=burst_size)
                output["samples"].append(
                    {
                        "sample_index": index,
                        "requested_s": float(requested),
                        "decoded": len(burst) == burst_size,
                        "frames": [
                            {
                                "actual_s": actual,
                                **_image_metrics(
                                    frame, black=config.black_pixel_value, white=config.white_pixel_value
                                ),
                            }
                            for frame, actual in burst
                        ],
                    }
                )
            complete = sum(sample["decoded"] for sample in output["samples"])
            if complete != config.sample_count_per_file:
                raise ValueError(
                    f"decoded {complete}/{config.sample_count_per_file} consecutive-frame bursts"
                )
            output["decode_ok"] = True
    except Exception as exc:  # noqa: BLE001 - probe failures are evidence, not crashes
        output["error"] = f"{type(exc).__name__}: {exc}"
    return output


def probe_findings(
    probe: Mapping[str, Any], *, expected: Mapping[str, Any] | None, config: VideoConfig
) -> list[dict]:
    """Findings of one probe result against the declared camera geometry and frame rate."""
    base = {"camera": probe["camera"], "source_path": probe["relative_path"]}
    findings: list[dict[str, Any]] = []
    if not probe.get("decode_ok"):
        findings.append(
            {
                **base,
                "rule_id": "VID.DECODE_FAILURE",
                "severity": "critical",
                "message": str(probe.get("error") or "video could not be decoded"),
            }
        )
    if expected is not None and probe.get("width") is not None:
        if (probe.get("height"), probe.get("width")) != (expected["height"], expected["width"]):
            findings.append(
                {
                    **base,
                    "rule_id": "VID.FRAME_SHAPE_MISMATCH",
                    "severity": "high",
                    "message": "Encoded video dimensions disagree with meta/info.json",
                    "expected": [expected["height"], expected["width"]],
                    "actual": [probe.get("height"), probe.get("width")],
                }
            )
        rate = probe.get("average_rate")
        tolerance = max(_FRAME_RATE_ABSOLUTE_TOLERANCE, _FRAME_RATE_RELATIVE_TOLERANCE * expected["fps"])
        if (
            not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or abs(rate - expected["fps"]) > tolerance
        ):
            findings.append(
                {
                    **base,
                    "rule_id": "VID.FRAME_RATE_MISMATCH",
                    "severity": "high",
                    "message": "Encoded video frame rate disagrees with meta/info.json fps",
                    "expected": expected["fps"],
                    "actual": rate,
                }
            )
    for sample in probe.get("samples", []):
        frames = sample["frames"]
        if not sample["decoded"]:
            findings.append(
                {
                    **base,
                    "rule_id": "VID.SAMPLE_DECODE_FAILURE",
                    "severity": "high",
                    "message": "A consecutive-frame burst could not be decoded",
                    "sample_index": sample["sample_index"],
                    "requested_s": sample["requested_s"],
                    "decoded_frame_count": len(frames),
                }
            )
            continue
        for rule_id, metric in (
            ("VID.BLACK_SAMPLE", "black_fraction"),
            ("VID.WHITE_SAMPLE", "white_fraction"),
        ):
            fractions = [frame[metric] for frame in frames]
            if min(fractions) >= config.monochrome_pixel_fraction:
                findings.append(
                    {
                        **base,
                        "rule_id": rule_id,
                        "severity": "medium",
                        "message": "Every frame of a consecutively decoded burst is nearly monochrome",
                        "sample_index": sample["sample_index"],
                        "requested_s": sample["requested_s"],
                        "actual_s": frames[0]["actual_s"],
                        "frame_fractions": fractions,
                        "threshold": config.monochrome_pixel_fraction,
                    }
                )
    return findings


def camera_expectations(info: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        camera: {
            "height": int(feature["shape"][0]),
            "width": int(feature["shape"][1]),
            "fps": float(info["fps"]),
        }
        for camera, feature in info["features"].items()
        if feature.get("dtype") == "video" and len(feature.get("shape", ())) == 3
    }


def probe_and_judge(
    root: Path, item: VideoFile, config: VideoConfig, expected: Mapping[str, Any] | None
) -> dict:
    """Worker entry point: probe one file and attach its findings."""
    probe = probe_video_file(root, item, config)
    probe["findings"] = probe_findings(probe, expected=expected, config=config)
    return probe
