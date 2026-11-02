"""Write the cleaned copy of a LeRobot v3 dataset described by a quality package.

Quarantined episodes are dropped and every excluded source-frame interval splits its episode: each retained
interval becomes one output episode. Output episodes, frame indices and global indices are dense; timestamps
restart at zero per output episode and the video ``from_timestamp``/``to_timestamp`` of each output episode
point at its frames inside the unchanged packed MP4 files, which are linked (symlink, hardlink) or copied.
Data and episode-metadata files keep their source paths, ``action_config`` phases are clipped to each segment,
and the statistics of numeric features are recomputed per output episode and pooled into ``meta/stats.json``.

Field masks cannot be applied by rewriting rows: a package with field masks is refused (train on the source
dataset with the package instead, see `concertina.data.quality`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from concertina.data.quality import QualityPackage, complement_intervals, union_intervals
from concertina.data_quality.scan import arrow_column_to_numpy, contiguous_runs, read_info, run_parallel

logger = logging.getLogger(__name__)

MEDIA_MODES = ("symlink", "hardlink", "copy")
CLEANING_DIR = "meta/cleaning"
_ROBOMIND_FRAME_FILENAME = re.compile(r"^(?P<prefix>[^/\\]*?)(?P<index>[0-9]+)(?P<suffix>\.[^./\\]+)$")
_NON_NUMERIC = frozenset({"video", "image", "string", "language"})


@dataclass(frozen=True)
class Segment:
    """One output episode: source frames ``[source_start, source_end)`` of a source episode."""

    source_episode: int
    source_start: int
    source_end: int
    source_length: int
    episode: int
    from_index: int

    @property
    def length(self) -> int:
        return self.source_end - self.source_start


def plan_segments(package: QualityPackage) -> list[Segment]:
    """Output episodes of the package, in source episode order."""
    lengths = {
        int(row["episode_index"]): int(row["frame_count"]) for row in package.records("episode_metrics")
    }
    quarantined = {
        int(row["episode_index"])
        for row in package.records("episode_decisions")
        if row["decision"] == "quarantine"
    }
    excluded: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for row in package.records("frame_decisions"):
        excluded[int(row["episode_index"])].append((int(row["start_frame"]), int(row["end_frame"])))
    segments: list[Segment] = []
    cursor = 0
    for episode in sorted(lengths):
        if episode in quarantined:
            continue
        for start, end in complement_intervals(
            union_intervals(excluded.get(episode, ())), 0, lengths[episode]
        ):
            segments.append(Segment(episode, start, end, lengths[episode], len(segments), cursor))
            cursor += end - start
    return segments


def _robomind_frame_parts(value: Any, where: str) -> tuple[str, int, int, str]:
    if not isinstance(value, str) or (match := _ROBOMIND_FRAME_FILENAME.fullmatch(value)) is None:
        raise ValueError(f"{where} is not an image filename with a numeric frame suffix: {value!r}")
    if not match.group("prefix"):
        raise ValueError(f"{where} has an unsupported frame filename: {value!r}")
    digits = match.group("index")
    return match.group("prefix"), int(digits), len(digits), match.group("suffix")


def segment_action_config(action_config: Any, *, start: int, end: int, source_length: int) -> Any:
    """Clip and rebase ``action_config`` phases to the segment ``[start, end)``.

    Two layouts exist: a list of ``{start_frame, end_frame, ...}`` half-open phases (AgiBot) and a RoboMIND
    mapping whose ``steps`` name their inclusive first/last frames by image filename
    (``camera_top_0081.jpg``).
    """
    if not 0 <= start < end <= source_length:
        raise ValueError(f"segment [{start}, {end}) is outside an episode of {source_length} frames")
    if action_config is None:
        return None
    if isinstance(action_config, Mapping):
        output = json.loads(json.dumps(dict(action_config)))
        if "steps" not in output:
            raise ValueError("unsupported action_config mapping: it has no steps")
        if output["steps"] is None:
            return output
        clipped = []
        for ordinal, raw in enumerate(output["steps"]):
            step = dict(raw)
            where = f"action_config.steps[{ordinal}]"
            first = _robomind_frame_parts(step.get("start_frame"), f"{where}.start_frame")
            last = _robomind_frame_parts(step.get("end_frame"), f"{where}.end_frame")
            if (first[0], first[2], first[3]) != (last[0], last[2], last[3]):
                raise ValueError(f"{where} frame filenames use different naming schemes")
            if first[1] >= source_length or last[1] >= source_length:
                raise ValueError(f"{where} references frames outside the episode")
            # Reversed annotations (for example N -> 0) have no recoverable interval: they are dropped.
            if first[1] > last[1]:
                continue
            left, right = max(start, first[1]), min(end, last[1] + 1)
            if left >= right:
                continue
            prefix, _, width, suffix = first
            step["start_frame"] = f"{prefix}{left - start:0{width}d}{suffix}"
            step["end_frame"] = f"{prefix}{right - start - 1:0{width}d}{suffix}"
            clipped.append(step)
        output["steps"] = clipped
        return output
    output = []
    previous_end = 0
    for ordinal, raw in enumerate(action_config):
        phase_start, phase_end = raw.get("start_frame"), raw.get("end_frame")
        if (
            isinstance(phase_start, bool)
            or not isinstance(phase_start, int)
            or isinstance(phase_end, bool)
            or not isinstance(phase_end, int)
            or not previous_end <= phase_start < phase_end <= source_length
        ):
            raise ValueError(f"action_config[{ordinal}] is not an ordered in-bounds half-open interval")
        previous_end = phase_end
        left, right = max(start, phase_start), min(end, phase_end)
        if left < right:
            output.append({**raw, "start_frame": left - start, "end_frame": right - start})
    return output


def _replace_column(table, name: str, values) -> Any:
    import pyarrow as pa

    index = table.schema.get_field_index(name)
    if index < 0:
        return table
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def _numeric_features(info: Mapping[str, Any]) -> dict[str, dict]:
    return {key: ft for key, ft in info["features"].items() if ft["dtype"] not in _NON_NUMERIC}


def rewrite_data_file(
    source: Path, destination: Path, segments: Sequence[Segment], *, fps: float, features: Mapping[str, dict]
) -> dict[int, dict[str, dict[str, list]]]:
    """Write the rows of ``segments`` (all stored in ``source``); returns per-episode numeric stats."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import compute_episode_stats

    table = pq.read_table(source)
    missing = {"episode_index", "frame_index", "index", "timestamp", "task_index"} - set(table.column_names)
    if missing:
        raise ValueError(f"{source} lacks the columns {sorted(missing)}")
    pieces, stats = [], {}
    runs: dict[int, tuple[int, int]] = {}
    for episode, start, end in contiguous_runs(table["episode_index"].to_numpy()):
        if episode in runs:
            raise ValueError(f"the rows of episode {episode} in {source} are not contiguous")
        runs[episode] = (start, end)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    for segment in segments:
        start, end = runs.get(segment.source_episode, (0, 0))
        if not np.array_equal(frame_index[start:end], np.arange(segment.source_length)):
            raise ValueError(f"episode {segment.source_episode} of {source} does not have dense frame_index")
        piece = table.slice(start + segment.source_start, segment.length)
        count = segment.length
        piece = _replace_column(piece, "episode_index", np.full(count, segment.episode))
        piece = _replace_column(piece, "frame_index", np.arange(count))
        piece = _replace_column(piece, "index", np.arange(segment.from_index, segment.from_index + count))
        piece = _replace_column(piece, "timestamp", np.arange(count, dtype=np.float64) / fps)
        pieces.append(piece)
        data = {key: arrow_column_to_numpy(piece[key]) for key in features if key in piece.column_names}
        stats[segment.episode] = {
            key: {name: np.asarray(value).tolist() for name, value in feature_stats.items()}
            for key, feature_stats in compute_episode_stats(data, dict(features)).items()
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.concat_tables(pieces), destination, compression="snappy", use_dictionary=True)
    return stats


def _set_value(row, name: str, value: Any):
    import pyarrow as pa

    index = row.schema.get_field_index(name)
    if index < 0:
        return row
    field = row.schema.field(index)
    return row.set_column(index, field, pa.array([value], type=field.type))


def _link(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        os.symlink(source.resolve(), destination)
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _check_package_matches(
    package: QualityPackage, info: Mapping[str, Any], episode_rows: Sequence[dict]
) -> None:
    lengths = {
        int(row["episode_index"]): int(row["frame_count"]) for row in package.records("episode_metrics")
    }
    source = {int(row["episode_index"]): int(row["length"]) for row in episode_rows}
    if lengths != source or int(info["total_frames"]) != sum(source.values()):
        raise ValueError("The quality package was compiled from another dataset (episodes or lengths differ)")


def materialize(
    package: QualityPackage | str | Path,
    *,
    source_root: str | Path,
    output_root: str | Path,
    media_mode: str = "symlink",
    workers: int = 1,
    rotation_stats_source: str | None = None,
) -> dict[str, Any]:
    """Write the cleaned dataset to ``output_root`` (which must not exist); returns a summary."""
    import pyarrow.parquet as pq
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import cast_stats_to_numpy, write_stats

    package = package if isinstance(package, QualityPackage) else QualityPackage.load(package)
    if media_mode not in MEDIA_MODES:
        raise ValueError(f"media_mode must be one of {MEDIA_MODES}, got {media_mode!r}")
    if not package.release_ready:
        raise ValueError(f"Quality package {package.root} is not release-ready: {package.release_blockers}")
    if package.records("field_masks"):
        raise ValueError(
            "The package masks source fields, which rewriting rows cannot express; train on the source "
            "dataset with the quality package instead"
        )
    source = Path(source_root).expanduser().resolve()
    output = Path(output_root).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"{output} already exists")
    if output.resolve() == source or output.resolve().is_relative_to(source):
        raise ValueError("The output must not be inside the source dataset")
    info = read_info(source)
    fps = float(info["fps"])
    meta_files = sorted((source / "meta" / "episodes").glob("*/*.parquet"))
    meta_tables = {path: pq.read_table(path) for path in meta_files}
    episode_rows = [row for table in meta_tables.values() for row in table.to_pylist()]
    _check_package_matches(package, info, episode_rows)
    order = [int(row["episode_index"]) for row in episode_rows]
    if order != sorted(order):
        raise ValueError("meta/episodes rows are not in episode order across files")
    segments = plan_segments(package)
    if not segments:
        raise ValueError("The package retains no frames")
    by_episode: dict[int, list[Segment]] = defaultdict(list)
    for segment in segments:
        by_episode[segment.source_episode].append(segment)
    template = str(info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"))
    by_file: dict[str, list[Segment]] = defaultdict(list)
    for row in episode_rows:
        path = template.format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        by_file[path].extend(by_episode.get(int(row["episode_index"]), ()))

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        numeric = _numeric_features(info)
        jobs = [
            (source / path, staging / path, file_segments)
            for path, file_segments in sorted(by_file.items())
            if file_segments
        ]
        stats: dict[int, dict] = {}
        for result in run_parallel(
            rewrite_data_file, jobs, workers, kwargs={"fps": fps, "features": numeric}
        ):
            stats.update(result)

        video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
        videos: set[str] = set()
        rows_by_file: dict[Path, list] = defaultdict(list)
        for path, table in meta_tables.items():
            for position, row in enumerate(table.to_pylist()):
                for segment in by_episode.get(int(row["episode_index"]), ()):
                    out = table.slice(position, 1)
                    for name, value in (
                        ("episode_index", segment.episode),
                        ("length", segment.length),
                        ("dataset_from_index", segment.from_index),
                        ("dataset_to_index", segment.from_index + segment.length),
                    ):
                        out = _set_value(out, name, value)
                    for key in video_keys:
                        origin = float(row[f"videos/{key}/from_timestamp"])
                        out = _set_value(
                            out, f"videos/{key}/from_timestamp", origin + segment.source_start / fps
                        )
                        out = _set_value(out, f"videos/{key}/to_timestamp", origin + segment.source_end / fps)
                        videos.add(
                            str(info["video_path"]).format(
                                video_key=key,
                                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                                file_index=int(row[f"videos/{key}/file_index"]),
                            )
                        )
                    if "action_config" in table.column_names:
                        out = _set_value(
                            out,
                            "action_config",
                            segment_action_config(
                                row["action_config"],
                                start=segment.source_start,
                                end=segment.source_end,
                                source_length=segment.source_length,
                            ),
                        )
                    for feature, feature_stats in stats[segment.episode].items():
                        for name, value in feature_stats.items():
                            out = _set_value(out, f"stats/{feature}/{name}", value)
                    rows_by_file[path].append(out)
        import pyarrow as pa

        for path, rows in rows_by_file.items():
            destination = staging / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.concat_tables(rows), destination, compression="snappy", use_dictionary=True)
        for relative in sorted(videos):
            _link(source / relative, staging / relative, media_mode)

        (staging / "meta").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "meta" / "tasks.parquet", staging / "meta" / "tasks.parquet")
        info_out = {
            **info,
            "total_episodes": len(segments),
            "total_frames": sum(segment.length for segment in segments),
            "splits": {"train": f"0:{len(segments)}"},
        }
        (staging / "meta" / "info.json").write_text(json.dumps(info_out, indent=4) + "\n", encoding="utf-8")
        source_stats_path = source / "meta" / "stats.json"
        pooled = (
            json.loads(source_stats_path.read_text(encoding="utf-8")) if source_stats_path.is_file() else {}
        )
        pooled = cast_stats_to_numpy(pooled)
        pooled.update(aggregate_stats([cast_stats_to_numpy(value) for value in stats.values()]))
        write_stats(pooled, staging)

        cleaning = staging / CLEANING_DIR
        cleaning.mkdir(parents=True)
        with (cleaning / "segments.jsonl").open("w", encoding="utf-8") as stream:
            for segment in segments:
                stream.write(json.dumps(asdict(segment), sort_keys=True) + "\n")
        (cleaning / "quality_package.json").write_text(
            json.dumps(package.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        omitted = sorted(
            path.relative_to(source).as_posix()
            for path in (source / "meta").iterdir()
            if path.name not in ("info.json", "stats.json", "tasks.parquet", "episodes")
        )
        if rotation_stats_source is not None:
            from concertina.data.stats import ROTATION_STATS_FILENAME
            from concertina.data.tools.rotation_stats import compute_rotation_stats, write_rotation_stats

            write_rotation_stats(
                compute_rotation_stats(staging, rotation_stats_source),
                staging / "meta" / ROTATION_STATS_FILENAME,
                rotation_stats_source,
            )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    summary = {
        "output": str(output),
        "source_episodes": len(episode_rows),
        "output_episodes": len(segments),
        "source_frames": int(info["total_frames"]),
        "output_frames": sum(segment.length for segment in segments),
        "linked_videos": len(videos),
        "media_mode": media_mode,
        "omitted_meta_files": omitted,
    }
    if omitted:
        logger.warning("Not copied (recompute them for the cleaned dataset if needed): %s", omitted)
    return summary
