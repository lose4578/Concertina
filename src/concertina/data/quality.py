"""Data-quality packages at training time: sample filtering and action-target masking.

A quality package (written by `concertina-data-quality`) records decisions about one LeRobot dataset in source
coordinates: quarantined episodes, excluded source-frame intervals and masks of exact source columns, all
independent of any action horizon. Training projects them through the frame offsets of its sample reader:

* a frame is a valid sample iff its episode is kept, it lies inside the drop window, every observation row it
  reads is valid and at least one of its future action targets (offset >= 0; history targets only provide
  context) is valid. The sampler draws only valid samples because
  each dataset's valid intervals are restricted by `QualityIndex.valid_intervals` (same format as
  `concertina.data.sampler.episode_valid_intervals`, identical output for a package without decisions);
* the invalid action targets (history included) of a valid sample are marked as padding by
  `QualityIndex.mark_invalid_targets`, so the model ignores them like any padded step.

Field masks bind to the source columns the reader consumes: a mask of a consumed action column invalidates
targets, a mask of a consumed observation column invalidates the samples that read the row, and masks of other
columns of the dataset (an alternative action representation, an unused camera) are inert. A mask of a column
the dataset does not have fails: the package then describes another dataset.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

PACKAGE_SCHEMA = "concertina.data-quality-package.v1"
PACKAGE_MANIFEST = "quality_package.json"
PACKAGE_TABLES = (
    "episode_metrics",
    "interval_findings",
    "episode_decisions",
    "frame_decisions",
    "field_masks",
    "group_distribution",
)
EPISODE_DECISIONS = ("keep", "quarantine")
EXCLUDE_FRAMES = "exclude_source_frames"
MASK_FIELD = "mask_field"
ACTION_PAD_KEY = "action_is_pad"

Interval = tuple[int, int]


@dataclass
class QualityConfig:
    """Training-time options of the data-quality filter (packages are set per dataset)."""

    # Refuse packages compiled from a partial scan or with unresolved release blockers.
    require_release_ready: bool = True


# ── interval algebra (half-open integer intervals) ──────────────────────


def union_intervals(intervals: Iterable[Sequence[int]]) -> list[Interval]:
    """Sorted union; overlapping and touching intervals merge, empty ones vanish."""
    merged: list[list[int]] = []
    for start, end in sorted((int(start), int(end)) for start, end in intervals):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def complement_intervals(intervals: Iterable[Sequence[int]], start: int, end: int) -> list[Interval]:
    """``[start, end)`` minus the union of ``intervals``."""
    output: list[Interval] = []
    cursor = start
    for left, right in union_intervals(intervals):
        left, right = max(start, left), min(end, right)
        if left >= right:
            continue
        if cursor < left:
            output.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        output.append((cursor, end))
    return output


def intersect_intervals(left: Iterable[Sequence[int]], right: Iterable[Sequence[int]]) -> list[Interval]:
    """Intersection of two interval unions, in linear time."""
    first, second = union_intervals(left), union_intervals(right)
    output: list[Interval] = []
    i = j = 0
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if start < end:
            output.append((start, end))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return output


def _contains(intervals: Sequence[Interval], starts: Sequence[int], value: int) -> bool:
    position = bisect.bisect_right(starts, value) - 1
    return position >= 0 and value < intervals[position][1]


# ── package files ───────────────────────────────────────────────────────


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(record)
    return records


def _relative_table_path(value: Any, where: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: table path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{where}: table path {value!r} must stay inside the package directory")
    return path


@dataclass(frozen=True)
class QualityPackage:
    """A compiled package: ``quality_package.json`` plus one JSONL file per table."""

    root: Path
    manifest: dict[str, Any]

    @classmethod
    def write(
        cls,
        root: str | Path,
        *,
        dataset: Mapping[str, Any],
        tables: Mapping[str, Sequence[Mapping[str, Any]]],
        release_blockers: Sequence[str] = (),
        **extra: Any,
    ) -> QualityPackage:
        """Write a package directory; ``extra`` keys (JSON values) are stored in the manifest."""
        missing = sorted(set(PACKAGE_TABLES) - set(tables))
        if missing:
            raise ValueError(f"A quality package needs the tables {missing}")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        entries = {}
        for name, rows in tables.items():
            path = root / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            entries[name] = {"path": path.name, "records": len(rows), "sha256": file_sha256(path)}
        blockers = sorted(set(release_blockers))
        manifest = {
            "schema_version": PACKAGE_SCHEMA,
            "dataset": dict(dataset),
            "release_ready": not blockers,
            "release_blockers": blockers,
            "tables": entries,
            **extra,
        }
        (root / PACKAGE_MANIFEST).write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        return cls(root=root, manifest=manifest)

    @classmethod
    def load(cls, path: str | Path) -> QualityPackage:
        requested = Path(path).expanduser()
        manifest_path = requested / PACKAGE_MANIFEST if requested.is_dir() else requested
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Quality package manifest {manifest_path} not found")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != PACKAGE_SCHEMA:
            found = manifest.get("schema_version") if isinstance(manifest, dict) else None
            raise ValueError(
                f"{manifest_path}: unsupported quality package schema {found!r}, expected {PACKAGE_SCHEMA!r}"
            )
        tables = manifest.get("tables")
        if not isinstance(tables, dict) or set(PACKAGE_TABLES) - set(tables):
            missing = sorted(set(PACKAGE_TABLES) - set(tables or {}))
            raise ValueError(f"{manifest_path}: the package lacks the tables {missing}")
        for name, entry in tables.items():
            where = f"{manifest_path} table {name!r}"
            if not isinstance(entry, dict) or not isinstance(entry.get("records"), int):
                raise ValueError(f"{where}: entries need a path, a record count and a sha256")
            _relative_table_path(entry.get("path"), where)
        if not isinstance(manifest.get("release_ready"), bool):
            raise ValueError(f"{manifest_path}: release_ready must be a boolean")
        return cls(root=manifest_path.parent, manifest=manifest)

    @property
    def release_ready(self) -> bool:
        return bool(self.manifest["release_ready"])

    @property
    def release_blockers(self) -> list[str]:
        return [str(value) for value in self.manifest.get("release_blockers", [])]

    @property
    def dataset(self) -> dict[str, Any]:
        return dict(self.manifest.get("dataset", {}))

    def records(self, table: str) -> list[dict[str, Any]]:
        """Rows of ``table``, checked against the record count and SHA-256 of the manifest."""
        entry = self.manifest["tables"].get(table)
        if entry is None:
            raise KeyError(f"Quality package {self.root} has no table {table!r}")
        path = self.root / _relative_table_path(entry["path"], table)
        if file_sha256(path) != entry.get("sha256"):
            raise ValueError(f"Quality table {path} does not match the SHA-256 recorded in the manifest")
        rows = read_jsonl(path)
        if len(rows) != entry["records"]:
            raise ValueError(f"Quality table {path} has {len(rows)} rows, the manifest {entry['records']}")
        return rows


# ── runtime index ───────────────────────────────────────────────────────


def _int(record: Mapping[str, Any], key: str, where: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where}: {key} must be an integer, got {value!r}")
    return int(value)


class QualityIndex:
    """Quality decisions of one dataset, bound to the source columns and frame offsets of its sample reader.

    Args:
        package: The dataset's quality package.
        episode_lengths: ``episode_index -> length`` of every episode of the dataset (its metadata); must
            equal the episodes the package was compiled from.
        features: Every feature key of the dataset.
        action_fields: Source columns the reader turns into action targets.
        observation_fields: Source columns (state, cameras) the reader turns into model inputs.
        action_offsets: Frame offsets of the action targets of a sample (``action_delta_indices``).
        observation_offsets: Frame offsets of the observation rows of a sample
            (``observation_delta_indices``).
    """

    def __init__(
        self,
        package: QualityPackage,
        *,
        episode_lengths: Mapping[int, int],
        features: Iterable[str],
        action_fields: Iterable[str],
        observation_fields: Iterable[str],
        action_offsets: Sequence[int],
        observation_offsets: Sequence[int] = (0,),
    ) -> None:
        self.package = package
        self.action_offsets = tuple(int(value) for value in action_offsets)
        # A sample must keep a valid future target: an action-history window alone leaves nothing to predict.
        self.eligible_offsets = tuple(d for d in self.action_offsets if d >= 0) or self.action_offsets
        self._eligible = np.asarray([d in self.eligible_offsets for d in self.action_offsets])
        self.observation_offsets = tuple(int(value) for value in observation_offsets)
        if not self.action_offsets or not self.observation_offsets:
            raise ValueError("action_offsets and observation_offsets must not be empty")
        self.action_fields = frozenset(action_fields)
        self.observation_fields = frozenset(observation_fields)
        if not self.action_fields:
            raise ValueError("action_fields must name the action columns the reader consumes")
        overlap = sorted(self.action_fields & self.observation_fields)
        if overlap:
            raise ValueError(f"Columns {overlap} are bound both as action and as observation")
        known = frozenset(features)
        unknown = sorted((self.action_fields | self.observation_fields) - known)
        if unknown:
            raise ValueError(f"Bound columns {unknown} are not features of the dataset")
        where = str(package.root)

        self.episode_lengths: dict[int, int] = {}
        for row in package.records("episode_metrics"):
            episode = _int(row, "episode_index", f"{where} episode_metrics")
            if episode in self.episode_lengths:
                raise ValueError(f"{where}: episode {episode} appears twice in episode_metrics")
            self.episode_lengths[episode] = _int(row, "frame_count", f"{where} episode_metrics")
        expected = {int(episode): int(length) for episode, length in episode_lengths.items()}
        if self.episode_lengths != expected:
            missing = sorted(set(expected) - set(self.episode_lengths))
            extra = sorted(set(self.episode_lengths) - set(expected))
            changed = sorted(
                episode
                for episode in set(expected) & set(self.episode_lengths)
                if expected[episode] != self.episode_lengths[episode]
            )
            raise ValueError(
                f"Quality package {where} was compiled from another dataset: episodes missing from the "
                f"package {missing[:5]}, unknown to the dataset {extra[:5]}, with another length "
                f"{changed[:5]}"
            )

        def interval(row: Mapping[str, Any], table: str) -> tuple[int, Interval]:
            episode = _int(row, "episode_index", f"{where} {table}")
            start, end = (
                _int(row, "start_frame", f"{where} {table}"),
                _int(row, "end_frame", f"{where} {table}"),
            )
            if episode not in self.episode_lengths:
                raise ValueError(f"{where} {table}: episode {episode} is not in episode_metrics")
            if not 0 <= start < end <= self.episode_lengths[episode]:
                raise ValueError(
                    f"{where} {table}: [{start}, {end}) is outside episode {episode} of "
                    f"{self.episode_lengths[episode]} frames"
                )
            return episode, (start, end)

        self.quarantined: set[int] = set()
        for row in package.records("episode_decisions"):
            episode = _int(row, "episode_index", f"{where} episode_decisions")
            if row.get("decision") not in EPISODE_DECISIONS:
                raise ValueError(f"{where} episode_decisions: unsupported decision {row.get('decision')!r}")
            if row["decision"] == "quarantine":
                self.quarantined.add(episode)

        excluded: dict[int, list[Interval]] = defaultdict(list)
        for row in package.records("frame_decisions"):
            if row.get("decision") != EXCLUDE_FRAMES:
                raise ValueError(f"{where} frame_decisions: unsupported decision {row.get('decision')!r}")
            episode, span = interval(row, "frame_decisions")
            excluded[episode].append(span)

        action_masks: dict[int, list[Interval]] = defaultdict(list)
        observation_masks: dict[int, list[Interval]] = defaultdict(list)
        self.inert_mask_count = 0
        for row in package.records("field_masks"):
            if row.get("decision") != MASK_FIELD:
                raise ValueError(f"{where} field_masks: unsupported decision {row.get('decision')!r}")
            episode, span = interval(row, "field_masks")
            field = row.get("field")
            if field in self.action_fields:
                action_masks[episode].append(span)
            elif field in self.observation_fields:
                observation_masks[episode].append(span)
            elif field in known:
                self.inert_mask_count += 1
            else:
                raise ValueError(f"{where} field_masks: {field!r} is not a column of the dataset")

        self.excluded = {episode: union_intervals(spans) for episode, spans in excluded.items()}
        self.invalid_targets = {
            episode: union_intervals([*self.excluded.get(episode, ()), *action_masks.get(episode, ())])
            for episode in set(self.excluded) | set(action_masks)
        }
        self.invalid_observations = {
            episode: union_intervals([*self.excluded.get(episode, ()), *observation_masks.get(episode, ())])
            for episode in set(self.excluded) | set(observation_masks)
        }
        self._target_starts = {e: [s for s, _ in spans] for e, spans in self.invalid_targets.items()}
        self._observation_starts = {
            e: [s for s, _ in spans] for e, spans in self.invalid_observations.items()
        }

    def summary(self) -> dict[str, int]:
        return {
            "episodes": len(self.episode_lengths),
            "quarantined_episodes": len(self.quarantined),
            "excluded_frames": sum(e - s for spans in self.excluded.values() for s, e in spans),
            "invalid_target_frames": sum(e - s for spans in self.invalid_targets.values() for s, e in spans),
            "invalid_observation_frames": sum(
                e - s for spans in self.invalid_observations.values() for s, e in spans
            ),
            "inert_field_masks": self.inert_mask_count,
        }

    def episode_sample_frames(
        self, episode: int, *, drop_n_first_frames: int = 0, drop_n_last_frames: int = 0
    ) -> list[Interval]:
        """Episode-local frames that are valid samples (see the module docstring)."""
        if episode in self.quarantined:
            return []
        length = self.episode_lengths[episode]
        allowed = (drop_n_first_frames, length - drop_n_last_frames)
        if allowed[0] >= allowed[1]:
            return []
        valid_rows = complement_intervals(self.invalid_observations.get(episode, ()), 0, length)
        samples = [allowed]
        for offset in self.observation_offsets:
            shifted = [(max(allowed[0], s - offset), min(allowed[1], e - offset)) for s, e in valid_rows]
            samples = intersect_intervals(samples, shifted)
            if not samples:
                return []
        valid_targets = complement_intervals(self.invalid_targets.get(episode, ()), 0, length)
        reachable = [
            (max(allowed[0], s - offset), min(allowed[1], e - offset))
            for offset in self.eligible_offsets
            for s, e in valid_targets
        ]
        return intersect_intervals(samples, reachable)

    def valid_intervals(
        self,
        episode_indices: Sequence[int],
        dataset_from_indices: Sequence[int],
        dataset_to_indices: Sequence[int],
        *,
        drop_n_first_frames: int = 0,
        drop_n_last_frames: int = 0,
        offset: int = 0,
    ) -> list[Interval]:
        """Valid samples as ``[start, end)`` intervals of reader rows shifted by ``offset``.

        ``episode_indices[i]`` is the episode stored at rows ``[dataset_from_indices[i],
        dataset_to_indices[i])`` of the reader. Each episode contributes its valid pieces in order; an episode
        without decisions contributes exactly the window of `concertina.data.sampler.episode_valid_intervals`.
        """
        if drop_n_first_frames < 0 or drop_n_last_frames < 0:
            raise ValueError("drop_n_first_frames and drop_n_last_frames must be >= 0")
        if not len(episode_indices) == len(dataset_from_indices) == len(dataset_to_indices):
            raise ValueError("episode_indices, dataset_from_indices and dataset_to_indices must align")
        intervals: list[Interval] = []
        for episode, start, end in zip(
            episode_indices, dataset_from_indices, dataset_to_indices, strict=True
        ):
            episode, start, end = int(episode), int(start), int(end)
            if self.episode_lengths.get(episode) != end - start:
                raise ValueError(
                    f"Reader rows [{start}, {end}) of episode {episode} disagree with its quality-package "
                    f"length {self.episode_lengths.get(episode)}"
                )
            frames = self.episode_sample_frames(
                episode, drop_n_first_frames=drop_n_first_frames, drop_n_last_frames=drop_n_last_frames
            )
            intervals.extend((start + offset + s, start + offset + e) for s, e in frames)
        return intervals

    def target_mask(self, episode: int, frame: int) -> np.ndarray:
        """Validity of each action target of the sample at ``frame`` (in ``action_offsets`` order).

        Targets outside the episode are invalid, as they are reader padding.
        """
        if episode in self.quarantined:
            return np.zeros(len(self.action_offsets), dtype=bool)
        length = self.episode_lengths[episode]
        spans = self.invalid_targets.get(episode, ())
        starts = self._target_starts.get(episode, ())
        return np.array(
            [
                0 <= frame + d < length and not _contains(spans, starts, frame + d)
                for d in self.action_offsets
            ],
            dtype=bool,
        )

    def observation_valid(self, episode: int, frame: int) -> bool:
        length = self.episode_lengths[episode]
        spans = self.invalid_observations.get(episode, ())
        starts = self._observation_starts.get(episode, ())
        return all(
            0 <= frame + d < length and not _contains(spans, starts, frame + d)
            for d in self.observation_offsets
        )

    def mark_invalid_targets(self, item: dict[str, Any]) -> dict[str, Any]:
        """Add the invalid action targets of a raw reader sample to its ``action_is_pad``.

        Raises if the sample is not valid: the quality-restricted sampler never draws such samples.
        """
        episode, frame = int(item["episode_index"]), int(item["frame_index"])
        valid = self.target_mask(episode, frame)
        eligible = valid[self._eligible].any()
        if episode in self.quarantined or not eligible or not self.observation_valid(episode, frame):
            raise ValueError(
                f"Frame {frame} of episode {episode} is not a valid sample of quality package "
                f"{self.package.root}"
            )
        if valid.all():
            return item
        invalid = torch.from_numpy(~valid)
        padding = item.get(ACTION_PAD_KEY)
        if padding is not None:
            padding = torch.as_tensor(padding, dtype=torch.bool)
            if padding.shape != invalid.shape:
                raise ValueError(
                    f"{ACTION_PAD_KEY} has shape {tuple(padding.shape)}, the action offsets "
                    f"{tuple(invalid.shape)}"
                )
            invalid = invalid | padding
        item[ACTION_PAD_KEY] = invalid
        return item


def load_quality_index(
    path: str | Path,
    *,
    meta,
    action_fields: Iterable[str],
    observation_fields: Iterable[str],
    action_offsets: Sequence[int],
    observation_offsets: Sequence[int] | None,
    config: QualityConfig | None = None,
) -> QualityIndex:
    """`QualityIndex` of a LeRobot dataset from its metadata (`LeRobotDatasetMetadata`)."""
    config = config or QualityConfig()
    package = QualityPackage.load(path)
    if config.require_release_ready and not package.release_ready:
        raise ValueError(
            f"Quality package {package.root} is not release-ready (blockers: {package.release_blockers}); "
            "resolve them or set data.quality.require_release_ready=false"
        )
    episodes = meta.episodes.with_format("numpy")
    lengths = dict(
        zip(
            np.asarray(episodes["episode_index"]).tolist(),
            np.asarray(episodes["length"]).tolist(),
            strict=True,
        )
    )
    index = QualityIndex(
        package,
        episode_lengths=lengths,
        features=meta.features,
        action_fields=action_fields,
        observation_fields=observation_fields,
        action_offsets=action_offsets,
        observation_offsets=observation_offsets or (0,),
    )
    logger.info("quality package %s: %s", package.root, index.summary())
    return index
