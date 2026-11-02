"""`concertina-normalize-episode-metadata`: check, and with ``--write`` repair, the episode layout that Concertina
training needs.

The upstream reader turns a frame's ``index`` and its episode's ``dataset_from_index``/``dataset_to_index`` into row
positions of the frame table, so they must be dense: frame ``index`` equals its row position, episodes are listed
in ``episode_index`` order 0..N-1, and their from/to indices are the cumulative episode lengths. In addition, all
``meta/episodes`` shards must share one Arrow schema (an all-null column in one shard breaks loading).

``--write`` rewrites the episode metadata shards with one schema and dense from/to indices, and rewrites the
``index`` column of the frame files when it is not the row position. Frames must already be grouped by episode in
``episode_index`` order with the declared lengths; anything else cannot be repaired here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EPISODE_SHARD_ROWS = 1000
_FIRST_COLUMNS = (
    "episode_index",
    "tasks",
    "length",
    "data/chunk_index",
    "data/file_index",
    "dataset_from_index",
    "dataset_to_index",
)
_SHARD_COLUMNS = ("meta/episodes/chunk_index", "meta/episodes/file_index")


@dataclass
class Layout:
    """What the dataset declares and what its frame table contains."""

    episode_shards: list[Path]
    episode_rows: list[dict]
    schemas_differ: bool
    frame_files: list[Path]
    frame_episode_runs: np.ndarray
    frame_run_lengths: np.ndarray
    index_dense: bool
    info_total_frames: int
    info_total_episodes: int
    problems: list[str] = field(default_factory=list)


def _episode_runs(values: np.ndarray, previous: int | None) -> tuple[list[int], list[int], bool]:
    """Run-length encode an episode column (``continues`` tells whether the first run extends ``previous``)."""
    if len(values) == 0:
        return [], [], False
    boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    lengths = np.diff(np.concatenate([starts, [len(values)]]))
    return values[starts].tolist(), lengths.tolist(), previous is not None and int(values[0]) == previous


def inspect_layout(root: Path) -> Layout:
    episode_shards = sorted((root / "meta" / "episodes").glob("*/*.parquet"))
    frame_files = sorted((root / "data").glob("*/*.parquet"))
    if not episode_shards or not frame_files:
        raise FileNotFoundError(f"{root} has no episode metadata or frame files")
    schemas = {tuple((f.name, str(f.type)) for f in pq.read_schema(path)) for path in episode_shards}
    rows = [row for path in episode_shards for row in pq.read_table(path).to_pylist()]

    runs: list[int] = []
    run_lengths: list[int] = []
    position = 0
    index_dense = True
    for path in frame_files:
        table = pq.read_table(path, columns=["index", "episode_index"])
        index = table.column("index").to_numpy()
        episodes = table.column("episode_index").to_numpy()
        index_dense &= bool(np.array_equal(index, np.arange(position, position + len(index))))
        values, lengths, continues = _episode_runs(episodes, runs[-1] if runs else None)
        if continues:
            run_lengths[-1] += lengths.pop(0)
            values.pop(0)
        runs += values
        run_lengths += lengths
        position += len(index)

    info = json.loads((root / "meta" / "info.json").read_text())
    layout = Layout(
        episode_shards=episode_shards,
        episode_rows=rows,
        schemas_differ=len(schemas) > 1,
        frame_files=frame_files,
        frame_episode_runs=np.asarray(runs, dtype=np.int64),
        frame_run_lengths=np.asarray(run_lengths, dtype=np.int64),
        index_dense=index_dense,
        info_total_frames=int(info["total_frames"]),
        info_total_episodes=int(info["total_episodes"]),
    )
    episode_index = np.asarray([row["episode_index"] for row in rows], dtype=np.int64)
    length = np.asarray([row["length"] for row in rows], dtype=np.int64)
    ends = np.cumsum(length)
    if layout.schemas_differ:
        layout.problems.append(
            f"{len(schemas)} different schemas across {len(episode_shards)} episode shards"
        )
    if not np.array_equal(episode_index, np.arange(len(rows))):
        layout.problems.append("episode metadata rows are not in episode_index order 0..N-1")
    from_index = np.asarray([row["dataset_from_index"] for row in rows], dtype=np.int64)
    to_index = np.asarray([row["dataset_to_index"] for row in rows], dtype=np.int64)
    if not (np.array_equal(from_index, ends - length) and np.array_equal(to_index, ends)):
        layout.problems.append("dataset_from_index/dataset_to_index are not the cumulative episode lengths")
    if not index_dense:
        layout.problems.append("the frame `index` column is not the row position")
    if layout.info_total_frames != position or layout.info_total_episodes != len(rows):
        layout.problems.append(
            f"info.json declares {layout.info_total_frames} frames / {layout.info_total_episodes} episodes, "
            f"found {position} frames / {len(rows)} episode rows"
        )
    return layout


def _frames_match_episodes(layout: Layout, rows: list[dict]) -> bool:
    expected_runs = np.arange(len(rows))
    expected_lengths = np.asarray([row["length"] for row in rows], dtype=np.int64)
    return np.array_equal(layout.frame_episode_runs, expected_runs) and np.array_equal(
        layout.frame_run_lengths, expected_lengths
    )


def _write_episode_shards(root: Path, rows: list[dict]) -> None:
    columns = [name for name in _FIRST_COLUMNS if any(name in row for row in rows)]
    for row in rows:
        columns += [name for name in row if name not in columns and name not in _SHARD_COLUMNS]
    columns += list(_SHARD_COLUMNS)
    normalized = []
    for position, row in enumerate(rows):
        entry = {name: row.get(name) for name in columns}
        entry["meta/episodes/chunk_index"] = 0
        entry["meta/episodes/file_index"] = position // EPISODE_SHARD_ROWS
        normalized.append(entry)
    schema = pa.Table.from_pylist(normalized).schema
    episodes_dir = root / "meta" / "episodes"
    staging = root / "meta" / f".episodes.staging.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        for start in range(0, len(normalized), EPISODE_SHARD_ROWS):
            path = staging / "chunk-000" / f"file-{start // EPISODE_SHARD_ROWS:03d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist(normalized[start : start + EPISODE_SHARD_ROWS], schema=schema), path
            )
        for old in episodes_dir.glob("*/*.parquet"):
            old.unlink()
        for new in sorted(staging.glob("*/*.parquet")):
            target = episodes_dir / new.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(new), str(target))
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _rewrite_frame_index(frame_files: list[Path]) -> None:
    position = 0
    for path in frame_files:
        table = pq.read_table(path)
        column = table.schema.get_field_index("index")
        index = pa.array(np.arange(position, position + table.num_rows), type=table.schema.field(column).type)
        staging = path.with_name(f".{path.name}.staging")
        pq.write_table(table.set_column(column, table.schema.field(column), index), staging)
        os.replace(staging, path)
        position += table.num_rows


def repair(root: Path, layout: Layout) -> list[str]:
    """Apply the repairs ``layout`` needs; returns what was changed."""
    rows = sorted(layout.episode_rows, key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in rows] != list(range(len(rows))):
        raise ValueError(f"{root}: episode_index values are not a permutation of 0..N-1; cannot repair")
    if not _frames_match_episodes(layout, rows):
        raise ValueError(
            f"{root}: frames are not grouped by episode in episode_index order with the declared lengths; "
            "the frame files must be rewritten, which this tool does not do"
        )
    changes = []
    ends = np.cumsum([row["length"] for row in rows])
    dense = all(
        row["dataset_from_index"] == end - row["length"] and row["dataset_to_index"] == end
        for row, end in zip(rows, ends.tolist(), strict=True)
    )
    if layout.schemas_differ or not dense or rows != layout.episode_rows:
        for row, end in zip(rows, ends.tolist(), strict=True):
            row["dataset_from_index"], row["dataset_to_index"] = end - row["length"], end
        _write_episode_shards(root, rows)
        changes.append(f"rewrote {len(rows)} episode metadata rows")
    if not layout.index_dense:
        _rewrite_frame_index(layout.frame_files)
        changes.append(f"rewrote the index column of {len(layout.frame_files)} frame files")
    return changes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset directory")
    parser.add_argument("--write", action="store_true", help="repair in place (default: only report)")
    args = parser.parse_args(argv)
    layout = inspect_layout(args.dataset)
    for problem in layout.problems:
        print(f"{args.dataset}: {problem}", file=sys.stderr)
    if not layout.problems:
        print(f"{args.dataset}: episode layout is dense", file=sys.stderr)
        return
    if not args.write:
        sys.exit(1)
    for change in repair(args.dataset, layout):
        print(f"{args.dataset}: {change}", file=sys.stderr)
    remaining = inspect_layout(args.dataset).problems
    if remaining:
        raise SystemExit(f"{args.dataset}: still not dense after repair: {remaining}")


if __name__ == "__main__":
    main()
