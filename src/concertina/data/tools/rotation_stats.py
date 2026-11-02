"""`concertina-compute-rotation-stats`: statistics of a dataset's action/state with rotations in rotate6d.

Writes ``meta/rotation_stats.json`` (``{"modes": {"rotate6d": {feature: stats}}}``), which Concertina training loads
to normalize the canonical action/state. Per source, the statistics cover the columns its canonical conversion reads:

* droid / calvin / robomind: canonical ``action`` and ``observation.state`` (single-arm sources 10D,
  dual-arm 20D; CALVIN also ``action.absolute`` and ``action.relative``);
* libero: ``action`` (10D) and ``observation.state`` (11D: both finger joints);
* agibot: the end-effector position, orientation (rotate6d, 12D) and effector position columns.

Statistics are computed per data file (float64 running statistics with histogram quantiles) and pooled with count
weights.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from lerobot.datasets.compute_stats import RunningQuantileStats
from lerobot.datasets.utils import serialize_dict
from lerobot.utils.constants import ACTION, OBS_STATE
from scipy.spatial.transform import Rotation as ScipyRotation

from concertina.data.embodiments import CalvinAdapter
from concertina.data.rotation import axis_angle_to_rotate6d_np, matrix_to_rotate6d, quat_to_rotate6d_np
from concertina.data.stats import (
    AGIBOT_FIELDS,
    ROTATION_MODE,
    ROTATION_STATS_FILENAME,
    aggregate_feature_stats,
)
from concertina.policy.config_data import SOURCES

FileStats = dict[str, dict[str, np.ndarray]]


def parse_nested_column(column: pd.Series) -> np.ndarray:
    """A parquet column of per-frame vectors (possibly nested per arm) as an ``(N, D)`` array."""
    first = column.iloc[0]
    if isinstance(first, np.ndarray):
        if first.dtype == object and len(first) > 0 and isinstance(first[0], np.ndarray):
            return np.stack([np.concatenate([arr.flatten() for arr in row]) for row in column.values])
        stacked = np.stack(column.values)
        return stacked.reshape(stacked.shape[0], -1) if stacked.ndim == 3 else stacked
    if isinstance(first, dict):
        keys = list(first.keys())
        return np.stack([np.array([row[k] for k in keys]) for row in column])
    if isinstance(first, (list, tuple)):
        return np.stack([np.array(row) for row in column])
    return column.values.reshape(-1, 1)


def _matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1) if arr.ndim > 2 else arr


def _gripper(values: np.ndarray, index: int, count: int) -> np.ndarray:
    arr = _matrix(values)
    if arr.shape[0] != count:
        raise ValueError(f"gripper rows {arr.shape[0]} do not match {count}")
    if arr.shape[1] == 0:
        return np.zeros((count, 1), dtype=np.float64)
    index = min(max(index, 0), arr.shape[1] - 1)
    return arr[:, index : index + 1]


def _euler_block(euler: np.ndarray) -> np.ndarray:
    matrices = ScipyRotation.from_euler("xyz", euler).as_matrix()
    return np.stack([matrix_to_rotate6d(matrix) for matrix in matrices], axis=0)


def _droid(df: pd.DataFrame, key: str) -> np.ndarray:
    prefix = ACTION if key == ACTION else OBS_STATE
    cartesian = parse_nested_column(df[f"{prefix}.cartesian_position"])
    gripper = np.asarray(df[f"{prefix}.gripper_position"].values, dtype=np.float64).reshape(-1, 1)
    return np.concatenate([cartesian[..., :3], _euler_block(cartesian[..., 3:6]), gripper], axis=-1)


def _robomind(df: pd.DataFrame, key: str) -> np.ndarray:
    prefix = "actions" if key == ACTION else "observation.states"
    left, right = f"{prefix}.end_effector_left", f"{prefix}.end_effector_right"
    if left in df.columns and right in df.columns:
        poses = [_matrix(parse_nested_column(df[name])) for name in (left, right)]
        parts = []
        for pose, side in zip(poses, ("left", "right"), strict=True):
            grip_key = f"{prefix}.gripper_{side}"
            grip = (
                _gripper(parse_nested_column(df[grip_key]), 0, len(pose))
                if grip_key in df.columns
                else np.zeros((len(pose), 1), dtype=np.float64)
            )
            parts += [pose[:, :3], quat_to_rotate6d_np(pose[:, 3:7]), grip]
        return np.concatenate(parts, axis=-1)
    pose = _matrix(parse_nested_column(df[f"{prefix}.end_effector"]))
    grip = _matrix(parse_nested_column(df[f"{prefix}.gripper"]))
    if pose.shape[-1] == 6:
        return np.concatenate(
            [pose[:, :3], _euler_block(pose[:, 3:6]), _gripper(grip, 0, len(pose))], axis=-1
        )
    if pose.shape[-1] == 12:
        return np.concatenate(
            [
                pose[:, :3],
                _euler_block(pose[:, 3:6]),
                _gripper(grip, 0, len(pose)),
                pose[:, 6:9],
                _euler_block(pose[:, 9:12]),
                _gripper(grip, 1, len(pose)),
            ],
            axis=-1,
        )
    raise ValueError(f"unknown RoboMIND end_effector width {pose.shape[-1]}")


def _calvin(df: pd.DataFrame, key: str) -> np.ndarray:
    column = key
    if key == ACTION and ACTION not in df.columns and "action.absolute" in df.columns:
        column = "action.absolute"
    raw = _matrix(parse_nested_column(df[column]))
    if raw.shape[-1] in (10, 20):
        return raw[:, :10]
    tensor = torch.from_numpy(raw.astype(np.float32))
    adapter = CalvinAdapter(column)
    canonical = adapter.state({OBS_STATE: tensor}) if key == OBS_STATE else adapter.action({column: tensor})
    return canonical[:, :10].numpy()


def _libero(df: pd.DataFrame, key: str) -> np.ndarray:
    data = parse_nested_column(df[key])
    return np.concatenate([data[..., :3], axis_angle_to_rotate6d_np(data[..., 3:6]), data[..., 6:]], axis=-1)


def _agibot(df: pd.DataFrame, key: str) -> np.ndarray:
    data = parse_nested_column(df[key])
    if not key.endswith("orientation"):
        return data
    arms = data.shape[-1] // 4
    return np.concatenate([quat_to_rotate6d_np(data[..., 4 * i : 4 * (i + 1)]) for i in range(arms)], axis=-1)


_CONVERTERS = {
    "droid": _droid,
    "robomind": _robomind,
    "calvin": _calvin,
    "libero": _libero,
    "agibot": _agibot,
}


def stat_keys(source: str, columns: set[str]) -> list[str]:
    if source == "agibot":
        return [*AGIBOT_FIELDS[ACTION], *AGIBOT_FIELDS[OBS_STATE]]
    keys = [ACTION, OBS_STATE]
    if source == "calvin":
        keys += [key for key in ("action.absolute", "action.relative") if key in columns]
    return keys


def _source_columns(source: str, keys: list[str], columns: set[str]) -> list[str]:
    if source in ("libero", "agibot"):
        return keys
    if source == "droid":
        return [f"{p}.{n}" for p in (ACTION, OBS_STATE) for n in ("cartesian_position", "gripper_position")]
    return sorted(c for c in columns if c.startswith(("action", "observation.state")))


def file_statistics(path: Path, source: str, keys: list[str], columns: list[str]) -> FileStats:
    df = pd.read_parquet(path, columns=columns)
    if df.empty:
        return {}
    result: FileStats = {}
    for key in keys:
        running = RunningQuantileStats()
        running.update(_CONVERTERS[source](df, key))
        if running._count >= 2:
            result[key] = running.get_statistics()
    return result


def compute_rotation_stats(
    dataset: Path, source: str, num_workers: int = 1
) -> dict[str, dict[str, np.ndarray]]:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")
    files = sorted((dataset / "data").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {dataset / 'data'}")
    columns = set().union(*(pq.read_schema(path).names for path in files))
    keys = stat_keys(source, columns)
    needed = [c for c in _source_columns(source, keys, columns) if c in columns]
    args = [(path, source, keys, needed) for path in files]
    if num_workers > 1:
        # Spawned workers: callers may already hold threads or a CUDA context (training start-up).
        with ProcessPoolExecutor(num_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            per_file = list(pool.map(file_statistics, *zip(*args, strict=True)))
    else:
        per_file = [file_statistics(*arg) for arg in args]
    stats = {}
    for key in keys:
        found = [file_stats[key] for file_stats in per_file if key in file_stats]
        if not found:
            raise ValueError(f"{dataset}: no statistics for {key!r}")
        stats[key] = aggregate_feature_stats(found)
    return stats


def write_rotation_stats(stats: dict, output: Path, source: str) -> None:
    payload = {
        "version": "1.2",
        "computed_at": datetime.now().isoformat(),
        "dataset_type": source,
        "modes": {ROTATION_MODE: serialize_dict(stats)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=Path, help="LeRobot dataset directory")
    parser.add_argument("--source", required=True, choices=SOURCES)
    parser.add_argument("--output", type=Path, help=f"default: <dataset>/meta/{ROTATION_STATS_FILENAME}")
    parser.add_argument("--num-workers", type=int, default=1, help="parallel processes over data files")
    args = parser.parse_args(argv)
    output = args.output or args.dataset / "meta" / ROTATION_STATS_FILENAME
    stats = compute_rotation_stats(args.dataset, args.source, args.num_workers)
    write_rotation_stats(stats, output, args.source)
    print(f"wrote {output}", file=sys.stderr)


if __name__ == "__main__":
    main()
