"""Mixture of local LeRobot datasets converted to the canonical Concertina sample layout.

Each `DatasetSpec` becomes one upstream `LeRobotDataset` (its list position is the sample's `dataset_index`).
Samples are converted per spec by a `SampleConverter` (canonical 20D action/state, camera mapping, optional resize
and frame-rate alignment).
Normalization statistics are packed into ``meta.stats``, the only channel from the dataset to the processor.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import logging
import os
import shutil
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STATE

from concertina.data.cache import cache_dir
from concertina.data.embodiments import CANONICAL_DIM, SampleConverter, make_source_adapter
from concertina.data.quality import QualityConfig, QualityIndex, load_quality_index
from concertina.data.stats import build_normalization_stats
from concertina.policy.config_data import SINGLE_ARM_DIM, DatasetSpec

logger = logging.getLogger(__name__)

PASSTHROUGH_KEYS = ("index", "episode_index", "frame_index", "task_index", "timestamp")
_CAMERA_DTYPES = ("video", "image")
_VALIDATION_ROWS = 1 << 20


def resolve_path(path: str, root: Path | None) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    if root is None:
        raise ValueError(f"Relative dataset path {path!r} needs `dataset.root`")
    return root / resolved


def _column(dataset, name: str) -> np.ndarray:
    return np.asarray(dataset.with_format("numpy")[name])


def resolve_camera_sources(
    spec: DatasetSpec, camera_keys: Sequence[str], available: Sequence[str], where: str
) -> dict[str, str | None]:
    """Canonical camera -> first available source candidate (unlisted cameras map to themselves).

    Only single-arm datasets may lack a wrist camera; it is then filled with zeros and flagged as missing.
    """
    sources: dict[str, str | None] = {}
    for camera in camera_keys:
        candidates = spec.camera_map.get(camera, [camera])
        source = next((key for key in candidates if key in available), None)
        if source is None and not (spec.valid_action_dim == SINGLE_ARM_DIM and "wrist" in camera):
            raise ValueError(
                f"{where}: no source camera for {camera!r} (candidates {candidates}, dataset cameras {list(available)})"
            )
        sources[camera] = source
    return sources


def check_local_files(
    meta: LeRobotDatasetMetadata, root: Path, episodes: np.ndarray, video_keys: Sequence[str]
) -> None:
    """Every data file and every video file of the given cameras must exist locally for the selected episodes.

    The upstream reader would otherwise try to download the dataset from the Hub.
    """

    def column(name: str):
        return meta.episodes.with_format("arrow")[name].take(episodes)

    paths = set()
    pairs = zip(column("data/chunk_index").to_numpy(), column("data/file_index").to_numpy(), strict=True)
    paths.update(meta.data_path.format(chunk_index=c, file_index=f) for c, f in set(pairs))
    for key in video_keys:
        chunks = column(f"videos/{key}/chunk_index")
        if chunks.null_count:
            raise ValueError(f"{root}: camera {key!r} is missing in {chunks.null_count} episodes")
        pairs = zip(chunks.to_numpy(), column(f"videos/{key}/file_index").to_numpy(), strict=True)
        paths.update(
            meta.video_path.format(video_key=key, chunk_index=c, file_index=f) for c, f in set(pairs)
        )
    missing = sorted(path for path in paths if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"{root} is incomplete: {len(missing)} files are missing, e.g. {missing[:3]}")


def dataset_view(root: Path, video_keys: Sequence[str]) -> Path:
    """``root``, or a view of it whose metadata declares only the video features ``video_keys`` if it has others.

    The upstream reader checks and indexes every declared video feature, so a dataset with an unused camera whose
    videos are absent or unreferenced (null in the episode metadata) cannot be read directly. The view, cached
    under ``$CONCERTINA_CACHE_DIR`` (default ``~/.cache/concertina``), holds a rewritten ``meta/info.json`` and
    links every other file to ``root``: the dataset is not modified and the frames read are the same.
    """
    info = json.loads((root / "meta" / "info.json").read_text())
    features = info["features"]
    if all(ft["dtype"] != "video" or key in video_keys for key, ft in features.items()):
        return root
    info["features"] = {
        key: ft for key, ft in features.items() if ft["dtype"] != "video" or key in video_keys
    }
    payload = json.dumps(info, indent=4)
    source = root.resolve()
    digest = hashlib.sha256(f"{source}\n{payload}".encode()).hexdigest()[:16]
    view = cache_dir("dataset_views") / f"{source.name}-{digest}"
    if view.is_dir():
        return view
    staging = view.with_name(f"{view.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    (staging / "meta").mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name != "meta":
            (staging / entry.name).symlink_to(entry)
    for entry in (source / "meta").iterdir():
        if entry.name != "info.json":
            (staging / "meta" / entry.name).symlink_to(entry)
    (staging / "meta" / "info.json").write_text(payload)
    try:
        staging.rename(view)
    except OSError:
        # Another process (e.g. another rank) created the same view first.
        shutil.rmtree(staging)
    return view


def validate_dense_layout(
    dataset: LeRobotDataset, root: Path, selected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Check that episode metadata and the frame table agree on dense row positions.

    The upstream reader turns a frame's ``index`` and its episode's ``dataset_from_index``/``dataset_to_index``
    into positions of the full frame table; any disagreement makes it read the wrong rows silently. ``selected``
    are the (sorted) episodes the reader loaded. Returns their ``(from, to)`` bounds in the loaded rows.
    """
    episodes = dataset.meta.episodes
    episode_index = _column(episodes, "episode_index")
    length = _column(episodes, "length").astype(np.int64)
    starts, ends = _column(episodes, "dataset_from_index"), _column(episodes, "dataset_to_index")
    dense_ends = np.cumsum(length)
    problems = []
    if not np.array_equal(episode_index, np.arange(len(episodes))):
        problems.append("episode_index is not 0..N-1 in row order")
    if not (np.array_equal(starts, dense_ends - length) and np.array_equal(ends, dense_ends)):
        problems.append("dataset_from_index/dataset_to_index differ from the cumulative episode lengths")
    total = int(dense_ends[-1]) if len(dense_ends) else 0
    if total != dataset.meta.total_frames:
        problems.append(f"episode lengths sum to {total}, info.json has {dataset.meta.total_frames} frames")
    selected_ends = np.cumsum(length[selected])
    selected_starts = selected_ends - length[selected]
    if len(dataset.hf_dataset) != selected_ends[-1]:
        problems.append(
            f"the episodes hold {selected_ends[-1]} frames, the frame table {len(dataset.hf_dataset)}"
        )
    if not problems:
        table = dataset.hf_dataset.data.table.select(["index", "episode_index"])
        position = 0
        for batch in table.to_batches(max_chunksize=_VALIDATION_ROWS):
            rows = np.arange(position, position + batch.num_rows)
            episode = np.searchsorted(selected_ends, rows, side="right")
            if not np.array_equal(batch.column(1).to_numpy(), selected[episode]):
                problems.append(
                    f"frame rows are not grouped by episode in metadata order (from row {position})"
                )
                break
            expected_index = (
                dense_ends[selected[episode]] - length[selected[episode]] + rows - selected_starts[episode]
            )
            if not np.array_equal(batch.column(0).to_numpy(), expected_index):
                problems.append(f"the frame `index` column is not the frame position (from row {position})")
                break
            position += batch.num_rows
    if problems:
        raise ValueError(
            f"Episode metadata of {root} is not dense: {'; '.join(problems)}. "
            f"Check or repair it with `concertina-normalize-episode-metadata {root}`."
        )
    return selected_starts, selected_ends


class ConcertinaMixtureMeta:
    """The metadata surface `lerobot_train.train()` and `make_policy` read from the training dataset."""

    def __init__(
        self,
        *,
        features: dict[str, dict],
        camera_keys: Sequence[str],
        fps: float,
        stats: dict[str, dict[str, torch.Tensor]],
        repo_id: str,
        total_episodes: int,
        total_frames: int,
    ) -> None:
        self.features = features
        self.camera_keys = list(camera_keys)
        self.video_keys = list(camera_keys)
        self.image_keys: list[str] = []
        self.depth_keys: list[str] = []
        self.has_language_columns = False
        self.fps = fps
        self.stats = stats
        self.repo_id = repo_id
        self.robot_type = None
        self.total_episodes = total_episodes
        self.total_frames = total_frames
        self.episodes = None


class ConcertinaMixtureDataset(torch.utils.data.Dataset):
    """Concatenation of the spec datasets; item ``i`` is frame ``i - dataset_offsets[k]`` of dataset ``k``.

    Args:
        specs: The datasets, in `dataset_index` order.
        root: Base directory of relative spec paths.
        camera_keys: Canonical camera keys, in model order.
        action_delta_indices / observation_delta_indices: Frame offsets of the action window and observations, in
            frames of each dataset's own frame rate. Negative action offsets add action history: the window is
            packed as [history, chunk], padded where it leaves the episode.
        image_hw: Model image size, used by ``resize_images`` and for missing-camera placeholders.
        resize_images: Resize all cameras to ``image_hw`` inside the dataset.
        fps_align_target_fps: Resample actions of datasets recorded at another rate to this rate.
        normalization: "none", "global" or "per_embodiment" (see `concertina.data.stats`).
        quality_config: Options of the specs' quality packages.
    """

    def __init__(
        self,
        specs: Sequence[DatasetSpec],
        *,
        root: Path | None,
        camera_keys: Sequence[str],
        action_delta_indices: Sequence[int],
        observation_delta_indices: Sequence[int] | None,
        image_hw: tuple[int, int],
        resize_images: bool = False,
        fps_align_target_fps: float | None = None,
        normalization: str = "none",
        use_imagenet_stats: bool = True,
        image_transforms: Callable | None = None,
        video_backend: str | None = None,
        tolerance_s: float = 1e-4,
        repo_id: str = "concertina/mixture",
        quality_config: QualityConfig | None = None,
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError("The Concertina mixture needs at least one dataset")
        if fps_align_target_fps is not None and min(action_delta_indices) < 0:
            # Resampling treats the first window frame as the current one; history frames precede it.
            raise ValueError(
                "fps_align_target_fps does not support action windows with history (negative deltas)"
            )
        self.specs = [
            dataclasses.replace(
                spec,
                path=str(resolve_path(spec.path, root)),
                rotation_stats_path=spec.rotation_stats_path
                and str(resolve_path(spec.rotation_stats_path, root)),
                quality_package=spec.quality_package and str(resolve_path(spec.quality_package, root)),
            )
            for spec in specs
        ]
        self.camera_keys = list(camera_keys)
        self.image_transforms = image_transforms
        self.children: list[LeRobotDataset] = []
        self.converters: list[SampleConverter] = []
        self.quality: list[QualityIndex | None] = []
        # Per dataset: the loaded episodes (sorted) and their (from, to) bounds in the loaded rows.
        self.episode_ids: list[np.ndarray] = []
        self.episode_bounds: list[tuple[np.ndarray, np.ndarray]] = []
        camera_shapes: dict[str, set[tuple[int, int]]] = {key: set() for key in self.camera_keys}
        action_keys = []
        fps_values = []
        for dataset_index, spec in enumerate(self.specs):
            where = f"datasets[{dataset_index}] ({spec.path})"
            path = Path(spec.path)
            if not (path / "meta" / "info.json").is_file():
                raise FileNotFoundError(f"{where}: no LeRobot dataset (meta/info.json missing)")
            meta = LeRobotDatasetMetadata(str(path), root=path)
            features = meta.features
            adapter = make_source_adapter(spec.source, spec.adapter, features)
            columns = (*adapter.action_keys, *adapter.state_keys)
            missing_columns = [key for key in columns if key not in features]
            if missing_columns:
                raise ValueError(f"{where}: {spec.source} datasets need the columns {missing_columns}")
            available = [key for key, ft in features.items() if ft["dtype"] in _CAMERA_DTYPES]
            camera_sources = resolve_camera_sources(spec, self.camera_keys, available, where)
            used_cameras = sorted({key for key in camera_sources.values() if key is not None})
            selected = np.arange(meta.total_episodes) if spec.episodes is None else np.sort(spec.episodes)
            if selected[-1] >= meta.total_episodes:
                raise ValueError(
                    f"{where}: episodes exceed the {meta.total_episodes} episodes of the dataset"
                )
            check_local_files(meta, path, selected, used_cameras)
            for camera, source in camera_sources.items():
                camera_shapes[camera].add(tuple(features[source]["shape"][:2]) if source else tuple(image_hw))

            fps = meta.fps
            delta_timestamps = {key: [d / fps for d in action_delta_indices] for key in adapter.action_keys}
            if observation_delta_indices is not None:
                for key in (*adapter.state_keys, *used_cameras):
                    delta_timestamps[key] = [d / fps for d in observation_delta_indices]
            child = LeRobotDataset(
                str(path),
                root=dataset_view(path, used_cameras),
                episodes=None if spec.episodes is None else selected.tolist(),
                delta_timestamps=delta_timestamps,
                image_transforms=image_transforms,
                tolerance_s=tolerance_s,
                video_backend=video_backend,
                return_uint8=False,
            )
            self.episode_ids.append(selected)
            self.episode_bounds.append(validate_dense_layout(child, path, selected))
            self.quality.append(
                None
                if spec.quality_package is None
                else load_quality_index(
                    spec.quality_package,
                    meta=meta,
                    action_fields=adapter.action_keys,
                    observation_fields=(*adapter.state_keys, *used_cameras),
                    action_offsets=action_delta_indices,
                    observation_offsets=observation_delta_indices,
                    config=quality_config,
                )
            )

            source_fps = spec.source_fps if spec.source_fps is not None else float(fps)
            self.converters.append(
                SampleConverter(
                    adapter=adapter,
                    camera_sources=camera_sources,
                    resize_hw=tuple(image_hw) if resize_images else None,
                    placeholder_hw=tuple(image_hw),
                    source_fps=source_fps if fps_align_target_fps is not None else None,
                    target_fps=fps_align_target_fps,
                )
            )
            self.children.append(child)
            # Statistics files name the canonical action `action`, except CALVIN's two source action columns.
            action_keys.append(adapter.action_keys[0] if spec.source == "calvin" else ACTION)
            fps_values.append(fps)

        if not resize_images:
            mixed = {key: sorted(shapes) for key, shapes in camera_shapes.items() if len(shapes) > 1}
            if mixed:
                raise ValueError(f"Camera sizes differ across datasets {mixed}; enable `resize_images`")
        lengths = [child.num_frames for child in self.children]
        self.dataset_offsets = [int(v) for v in np.concatenate([[0], np.cumsum(lengths)[:-1]])]
        self._num_frames = int(sum(lengths))

        camera_features = {}
        for camera in self.camera_keys:
            height, width = tuple(image_hw) if resize_images else next(iter(camera_shapes[camera]))
            camera_features[camera] = {
                "dtype": "video",
                "shape": (height, width, 3),
                "names": ["height", "width", "channels"],
            }
        features = {
            ACTION: {
                "dtype": "float32",
                "shape": (CANONICAL_DIM,),
                "names": [f"a{i}" for i in range(CANONICAL_DIM)],
            },
            OBS_STATE: {
                "dtype": "float32",
                "shape": (CANONICAL_DIM,),
                "names": [f"s{i}" for i in range(CANONICAL_DIM)],
            },
            **camera_features,
        }
        stats = build_normalization_stats(
            self.specs,
            [Path(spec.path) for spec in self.specs],
            action_keys,
            normalization=normalization,
            camera_keys=self.camera_keys,
            use_imagenet_stats=use_imagenet_stats,
        )
        self.meta = ConcertinaMixtureMeta(
            features=features,
            camera_keys=self.camera_keys,
            fps=fps_align_target_fps or fps_values[0],
            stats=stats,
            repo_id=repo_id,
            total_episodes=self.num_episodes,
            total_frames=self._num_frames,
        )

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @property
    def num_episodes(self) -> int:
        return sum(child.num_episodes for child in self.children)

    def __len__(self) -> int:
        return self._num_frames

    def locate(self, index: int) -> tuple[int, int]:
        """``(dataset_index, frame index within that dataset)`` of a mixture index."""
        if not 0 <= index < self._num_frames:
            raise IndexError(f"index {index} out of range for {self._num_frames} frames")
        dataset_index = bisect.bisect_right(self.dataset_offsets, index) - 1
        return dataset_index, index - self.dataset_offsets[dataset_index]

    def __getitem__(self, index: int) -> dict:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: Sequence[int]) -> list[dict]:
        groups: dict[int, list[tuple[int, int]]] = {}
        for position, index in enumerate(indices):
            dataset_index, local = self.locate(int(index))
            groups.setdefault(dataset_index, []).append((position, local))
        samples: list[dict] = [{} for _ in indices]
        for dataset_index, entries in groups.items():
            raw_items = self.children[dataset_index].__getitems__([local for _, local in entries])
            for (position, _), raw in zip(entries, raw_items, strict=True):
                samples[position] = self._convert(dataset_index, raw)
        return samples

    def _convert(self, dataset_index: int, raw: dict) -> dict:
        quality = self.quality[dataset_index]
        if quality is not None:
            raw = quality.mark_invalid_targets(raw)
        sample = self.converters[dataset_index](raw)
        sample["dataset_index"] = torch.tensor(dataset_index, dtype=torch.long)
        for key in PASSTHROUGH_KEYS:
            if key in raw:
                sample[key] = raw[key]
        sample["task"] = raw["task"]
        return sample
