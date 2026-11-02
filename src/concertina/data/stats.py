# Copyright 2026 The Concertina Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions adapted from LeRobot (lerobot/datasets/compute_stats.py, `aggregate_feature_stats`),
# Copyright 2024 The HuggingFace Inc. team, Apache License 2.0. Modified: pools offline rotation-statistics files
# and averages quantiles with count weights. See NOTICE.
"""Normalization statistics of the canonical action/state, built from `rotation_stats.json` files.

`concertina-compute-rotation-stats` writes, per dataset, statistics of the source columns with rotations already in
rotate6d. Here they are loaded, pooled over datasets (count-weighted) and merged into the canonical 20D layout; a
dataset without the file gets them computed by the same algorithm into the user cache. The result travels to the
processor through ``dataset.meta.stats``:

* ``normalization="global"``: ``stats["action"]`` and ``stats["observation.state"]``;
* ``normalization="per_embodiment"``: ``stats[embodiment_stats_key(embodiment, feature)]`` per embodiment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from lerobot.utils.constants import ACTION, IMAGENET_STATS, OBS_STATE

from concertina.data.cache import cache_dir, file_lock
from concertina.policy.config_data import DatasetSpec

logger = logging.getLogger(__name__)

ROTATION_MODE = "rotate6d"
ROTATION_STATS_FILENAME = "rotation_stats.json"
EMBODIMENT_STATS_PREFIX = "concertina.embodiment."
_AGGREGATED_STATS = ("mean", "std", "min", "max", "count")
_CANONICAL_FEATURES = (ACTION, OBS_STATE)

RawStats = dict[str, dict[str, np.ndarray]]

# Source columns whose statistics are concatenated into the canonical vectors.
AGIBOT_FIELDS = {
    ACTION: ("actions.end.position", "actions.end.orientation", "actions.effector.position"),
    OBS_STATE: (
        "observation.states.end.position",
        "observation.states.end.orientation",
        "observation.states.effector.position",
    ),
}


def embodiment_stats_key(embodiment: str, feature: str) -> str:
    return f"{EMBODIMENT_STATS_PREFIX}{embodiment}/{feature}"


def rotation_stats_file(spec: DatasetSpec, dataset_root: Path) -> Path:
    """The spec's statistics file: its explicit path, the dataset's own file, or one computed into the cache."""
    if spec.rotation_stats_path:
        return Path(spec.rotation_stats_path)
    own = dataset_root / "meta" / ROTATION_STATS_FILENAME
    return own if own.is_file() else cached_rotation_stats(dataset_root, spec.source)


def dataset_fingerprint(dataset_root: Path, source: str) -> str:
    """Identity of the frames statistics are computed from: the episode metadata and the data file sizes."""
    digest = hashlib.sha256(f"{ROTATION_MODE}\n{source}\n".encode())
    metadata = [
        dataset_root / "meta" / "info.json",
        *sorted((dataset_root / "meta" / "episodes").glob("**/*.parquet")),
    ]
    for path in metadata:
        digest.update(f"{path.relative_to(dataset_root).as_posix()}\n".encode())
        digest.update(path.read_bytes())
    for path in sorted((dataset_root / "data").glob("**/*.parquet")):
        digest.update(f"{path.relative_to(dataset_root).as_posix()}:{path.stat().st_size}\n".encode())
    return digest.hexdigest()[:32]


def cached_rotation_stats(dataset_root: Path, source: str) -> Path:
    """Statistics of a dataset without ``meta/rotation_stats.json``, computed once into the user cache.

    They are computed like `concertina-compute-rotation-stats` and written to
    ``$CONCERTINA_CACHE_DIR/rotation_stats/<dataset>-<fingerprint>.json``; the dataset itself is not modified.
    Processes of one machine (the ranks of a job) compute them once, under a file lock.
    """
    root = dataset_root.resolve()
    path = cache_dir("rotation_stats") / f"{root.name}-{dataset_fingerprint(root, source)}.json"
    if path.is_file():
        logger.info("Using cached rotation statistics %s for %s", path, root)
        return path
    with file_lock(path.with_name(path.name + ".lock")):
        if path.is_file():
            logger.info("Using cached rotation statistics %s for %s", path, root)
            return path
        from concertina.data.tools.rotation_stats import compute_rotation_stats, write_rotation_stats

        # One process by default (LIBERO: ~10 s): worker processes would re-import the caller's main module.
        workers = int(os.environ.get("CONCERTINA_STATS_WORKERS") or 1)
        logger.warning(
            "%s has no meta/%s: computing %s rotation statistics into %s (worker processes: %d). Store them with the "
            "dataset beforehand with `concertina-compute-rotation-stats`.",
            root,
            ROTATION_STATS_FILENAME,
            source,
            path,
            workers,
        )
        stats = compute_rotation_stats(root, source, num_workers=workers)
        staging = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        write_rotation_stats(stats, staging, source)
        os.replace(staging, path)
    return path


def load_rotation_stats(path: Path, mode: str = ROTATION_MODE) -> RawStats:
    if not path.is_file():
        raise FileNotFoundError(
            f"Rotation statistics {path} not found; generate them with `concertina-compute-rotation-stats`."
        )
    payload = json.loads(path.read_text())
    modes = payload.get("modes", {})
    if mode not in modes:
        raise ValueError(
            f"{path} has no {mode!r} statistics (available: {sorted(modes)}); regenerate them with "
            "`concertina-compute-rotation-stats`."
        )
    return {
        feature: {name: np.asarray(value) for name, value in stats.items()}
        for feature, stats in modes[mode].items()
    }


def aggregate_feature_stats(stats_list: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Count-weighted pooling of per-dataset statistics of one feature (quantiles are count-weighted means)."""
    means = np.stack([s["mean"] for s in stats_list])
    variances = np.stack([s["std"] ** 2 for s in stats_list])
    counts = np.stack([s["count"] for s in stats_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta_means = means - total_mean
    total_variance = ((variances + delta_means**2) * counts).sum(axis=0) / total_count
    aggregated = {
        "min": np.min(np.stack([s["min"] for s in stats_list]), axis=0),
        "max": np.max(np.stack([s["max"] for s in stats_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    for key in stats_list[0]:
        if key.startswith("q") and key[1:].isdigit() and all(key in s for s in stats_list):
            aggregated[key] = (np.stack([s[key] for s in stats_list]) * counts).sum(axis=0) / total_count
    return aggregated


def required_features(source: str, action_key: str) -> tuple[str, ...]:
    if source == "agibot":
        return (*AGIBOT_FIELDS[ACTION], *AGIBOT_FIELDS[OBS_STATE])
    return (action_key, OBS_STATE)


def _aggregatable(raw: RawStats, features: Sequence[str], where: str) -> RawStats:
    """The pooled statistics of ``features``; every feature needs one positive integer frame count."""
    selected: RawStats = {}
    counts = set()
    for feature in features:
        stats = raw.get(feature)
        missing = sorted(set(_AGGREGATED_STATS) - set(stats or {}))
        if missing:
            raise ValueError(
                f"{where}: {feature!r} lacks {missing} needed to pool statistics across datasets"
            )
        selected[feature] = {name: np.asarray(stats[name]) for name in _AGGREGATED_STATS}
        for name in ("mean", "std", "min", "max"):
            if not np.all(np.isfinite(selected[feature][name])):
                raise ValueError(f"{where}: {feature!r} {name} is not finite")
        count = selected[feature]["count"]
        if count.shape != (1,) or count[0] <= 0 or not float(count[0]).is_integer():
            raise ValueError(f"{where}: {feature!r} count must be one positive integer, got {count}")
        counts.add(int(count[0]))
    if len(counts) != 1:
        raise ValueError(f"{where}: features disagree on the frame count {sorted(counts)}")
    return selected


def pool_raw_stats(raw_list: Sequence[RawStats], features: Sequence[str], where: Sequence[str]) -> RawStats:
    if len(raw_list) == 1:
        return raw_list[0]
    selected = [_aggregatable(raw, features, label) for raw, label in zip(raw_list, where, strict=True)]
    return {feature: aggregate_feature_stats([s[feature] for s in selected]) for feature in features}


def _pad_single_arm(stats: dict, zero_stats: set[str]) -> dict:
    padded = {}
    for name, values in stats.items():
        arr = np.asarray(values)
        if arr.ndim == 1 and len(arr) == 10:
            fill = 0.0 if name in zero_stats else 1.0
            padded[name] = np.concatenate([arr, np.full(10, fill, dtype=arr.dtype)])
        else:
            padded[name] = values
    return padded


def _merge_agibot(raw: RawStats) -> RawStats:
    merged: RawStats = {}
    for output, (pos_key, ori_key, grip_key) in AGIBOT_FIELDS.items():
        if not all(key in raw for key in (pos_key, ori_key, grip_key)):
            continue
        fields = {}
        for name in ("mean", "std", "min", "max"):
            if not all(name in raw[key] for key in (pos_key, ori_key, grip_key)):
                continue
            pos, ori, grip = (np.asarray(raw[key][name]) for key in (pos_key, ori_key, grip_key))
            if len(ori) != 12:
                raise ValueError(f"AgiBot {ori_key} {name} must be 12D rotate6d (2 arms), got {len(ori)}")
            fields[name] = np.concatenate([pos[:3], ori[:6], grip[:1], pos[3:6], ori[6:], grip[1:2]])
        if fields:
            merged[output] = fields
    return merged


def _merge_libero(raw: RawStats) -> RawStats:
    merged: RawStats = {}
    for key in _CANONICAL_FEATURES:
        stats = raw.get(key)
        if stats is None or "mean" not in stats:
            continue
        dim = len(np.asarray(stats["mean"]))
        if key == OBS_STATE and dim == 11:
            # The state carries two finger joints; the canonical gripper keeps the first one.
            stats = {
                name: np.asarray(v)[:-1] if np.asarray(v).ndim >= 1 and len(np.asarray(v)) > 1 else v
                for name, v in stats.items()
            }
        elif dim != 10:
            raise ValueError(f"LIBERO {key} statistics must be 10D (11D state), got {dim}D")
        merged[key] = _pad_single_arm(stats, {"mean", "min"})
    return merged


def _merge_droid(raw: RawStats) -> RawStats:
    merged: RawStats = {}
    for key in _CANONICAL_FEATURES:
        stats = raw.get(key)
        if stats is None or "mean" not in stats:
            continue
        dim = len(np.asarray(stats["mean"]))
        if dim != 10:
            logger.warning("DROID %s statistics are %dD, expected 10D", key, dim)
        merged[key] = _pad_single_arm(stats, {"mean", "min"})
    return merged


def _merge_calvin(raw: RawStats) -> RawStats:
    merged: RawStats = {}
    for key in _CANONICAL_FEATURES:
        stats = raw.get(key)
        if not isinstance(stats, dict) or "mean" not in stats:
            continue
        dim = len(np.asarray(stats["mean"]))
        if dim == 10:
            merged[key] = _pad_single_arm(stats, {"mean", "min", "q01", "q10", "q50"})
        elif dim == 20:
            merged[key] = dict(stats)
        else:
            raise ValueError(f"CALVIN {key} statistics must be 10D or 20D canonical, got {dim}D")
    return merged


def _merge_robomind(raw: RawStats) -> RawStats:
    zero_stats = {"mean", "min", "q01", "q10", "q50", "q90", "q99"}
    merged: RawStats = {}
    for key in _CANONICAL_FEATURES:
        stats = raw.get(key)
        if not isinstance(stats, dict) or "mean" not in stats:
            continue
        mean = np.asarray(stats["mean"])
        if mean.ndim == 1 and len(mean) == 20:
            merged[key] = dict(stats)
            continue
        out = {}
        for name, values in stats.items():
            arr = np.asarray(values)
            if arr.ndim != 1:
                out[name] = values
            elif len(arr) >= 20:
                out[name] = arr[:20]
            else:
                fill = 0.0 if name in zero_stats else 1.0
                out[name] = np.concatenate([arr, np.full(20 - len(arr), fill, dtype=np.float64)])
        merged[key] = out
    return merged


_MERGERS = {
    "agibot": _merge_agibot,
    "libero": _merge_libero,
    "droid": _merge_droid,
    "calvin": _merge_calvin,
    "robomind": _merge_robomind,
}


def merge_canonical_stats(raw: RawStats, source: str, action_key: str = ACTION) -> RawStats:
    """Source statistics -> statistics of the canonical 20D ``action`` and ``observation.state``.

    ``action_key`` selects the source action column (CALVIN trains on ``action.absolute``).
    """
    if action_key != ACTION:
        if action_key not in raw:
            raise KeyError(f"Rotation statistics lack {action_key!r} (available: {sorted(raw)})")
        raw = {**raw, ACTION: raw[action_key]}
    merged = _MERGERS[source](raw)
    missing = [key for key in _CANONICAL_FEATURES if key not in merged]
    if missing:
        raise ValueError(f"{source} rotation statistics do not provide {missing}")
    return merged


def _to_tensors(stats: RawStats) -> dict[str, dict[str, torch.Tensor]]:
    return {
        feature: {
            name: torch.as_tensor(np.asarray(value), dtype=torch.float32) for name, value in values.items()
        }
        for feature, values in stats.items()
    }


def _pooled_canonical_stats(
    members: Sequence[tuple[DatasetSpec, Path, str]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Canonical statistics of datasets sharing one source schema; files used by several datasets count once."""
    spec, _, action_key = members[0]
    files: dict[Path, None] = {}
    for member_spec, root, _ in members:
        files[rotation_stats_file(member_spec, root).resolve()] = None
    raw_list = [load_rotation_stats(path) for path in files]
    raw = pool_raw_stats(raw_list, required_features(spec.source, action_key), [str(path) for path in files])
    return _to_tensors(merge_canonical_stats(raw, spec.source, action_key))


def build_normalization_stats(
    specs: Sequence[DatasetSpec],
    roots: Sequence[Path],
    action_keys: Sequence[str],
    *,
    normalization: str,
    camera_keys: Sequence[str] = (),
    use_imagenet_stats: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """``dataset.meta.stats`` of a mixture (``action_keys`` are the source action columns of each dataset)."""
    stats: dict[str, dict[str, torch.Tensor]] = {}
    members = list(zip(specs, roots, action_keys, strict=True))
    if normalization == "global":
        stats.update(_pooled_canonical_stats(members))
    elif normalization == "per_embodiment":
        groups: dict[str, list] = {}
        for member in members:
            groups.setdefault(member[0].embodiment, []).append(member)
        for embodiment, group in groups.items():
            for feature, values in _pooled_canonical_stats(group).items():
                stats[embodiment_stats_key(embodiment, feature)] = values
    if use_imagenet_stats:
        for key in camera_keys:
            stats[key] = {
                name: torch.tensor(value, dtype=torch.float32) for name, value in IMAGENET_STATS.items()
            }
    return stats


def embodiment_stats(
    dataset_stats: dict[str, dict[str, torch.Tensor]], embodiments: Sequence[str]
) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
    """Per-embodiment canonical statistics packed by :func:`build_normalization_stats`."""
    result = {}
    for embodiment in dict.fromkeys(embodiments):
        result[embodiment] = {}
        for feature in _CANONICAL_FEATURES:
            key = embodiment_stats_key(embodiment, feature)
            if key not in dataset_stats:
                raise KeyError(f"dataset statistics lack {key!r}; was the mixture built with per_embodiment?")
            result[embodiment][feature] = dataset_stats[key]
    return result
