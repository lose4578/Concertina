import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.feature_utils import dataset_to_policy_features
from torch.utils.data import DataLoader, default_collate

from concertina.data.embodiments import (
    interpolate_padding,
    interpolate_sequence,
    make_source_adapter,
    resize_image,
)
from concertina.data.mixture import ConcertinaMixtureDataset
from concertina.data.tools import episode_metadata
from concertina.policy.config_data import ConcertinaDataConfig, DatasetSpec

HEAD, WRIST_LEFT, WRIST_RIGHT = (
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
CAMERAS = [HEAD, WRIST_LEFT, WRIST_RIGHT]
CHUNK = 6
IMAGE_HW = (24, 32)

SPECS = {
    "libero": DatasetSpec(
        path="libero",
        source="libero",
        embodiment="libero",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.image"], WRIST_LEFT: ["observation.images.image2"]},
    ),
    "droid": DatasetSpec(
        path="droid",
        source="droid",
        embodiment="droid",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.exterior_1_left"]},
    ),
    "agibot": DatasetSpec(
        path="agibot",
        source="agibot",
        embodiment="agibot",
        camera_map={
            WRIST_LEFT: ["observation.images.hand_left"],
            WRIST_RIGHT: ["observation.images.hand_right"],
        },
    ),
    "calvin": DatasetSpec(
        path="calvin",
        source="calvin",
        embodiment="calvin",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.top"], WRIST_LEFT: ["observation.images.wrist"]},
    ),
}


def build(root: Path, names=tuple(SPECS), *, specs=None, **kwargs) -> ConcertinaMixtureDataset:
    options = {
        "root": root,
        "camera_keys": CAMERAS,
        "action_delta_indices": list(range(CHUNK)),
        "observation_delta_indices": [0],
        "image_hw": IMAGE_HW,
        "resize_images": True,
        "normalization": "per_embodiment",
        **kwargs,
    }
    return ConcertinaMixtureDataset(specs or [SPECS[name] for name in names], **options)


@pytest.fixture(scope="module")
def mixture(dataset_dir) -> ConcertinaMixtureDataset:
    return build(dataset_dir)


def _reference(path: Path, spec: DatasetSpec) -> tuple[LeRobotDataset, object]:
    meta_features = LeRobotDataset(str(path), root=path).meta.features
    adapter = make_source_adapter(spec.source, spec.adapter, meta_features)
    fps = LeRobotDataset(str(path), root=path).fps
    deltas = {key: [i / fps for i in range(CHUNK)] for key in adapter.action_keys}
    deltas.update({key: [0.0] for key in adapter.state_keys})
    return LeRobotDataset(str(path), root=path, delta_timestamps=deltas), adapter


def test_meta_describes_the_canonical_features(mixture, dataset_dir):
    features = dataset_to_policy_features(mixture.meta.features)
    assert features["action"].type is FeatureType.ACTION and features["action"].shape == (20,)
    assert features["observation.state"].type is FeatureType.STATE and features[
        "observation.state"
    ].shape == (20,)
    for camera in CAMERAS:
        assert features[camera].type is FeatureType.VISUAL and features[camera].shape == (3, *IMAGE_HW)
    assert mixture.meta.camera_keys == CAMERAS and mixture.meta.depth_keys == []
    assert not mixture.meta.has_language_columns
    lengths = [child.num_frames for child in mixture.children]
    assert len(mixture) == mixture.num_frames == sum(lengths)
    assert mixture.dataset_offsets == list(np.cumsum([0, *lengths[:-1]]))
    assert mixture.num_episodes == sum(child.num_episodes for child in mixture.children)


def test_samples_are_the_adapter_conversion_of_upstream_items(mixture, dataset_dir):
    for dataset_index, (name, spec) in enumerate(SPECS.items()):
        reference, adapter = _reference(dataset_dir / name, spec)
        bounds = mixture.episode_bounds[dataset_index]
        # First frame, a frame whose action window crosses the episode end, and the last frame.
        locals_ = [0, int(bounds[1][0]) - 2, len(reference) - 1]
        samples = mixture.__getitems__([mixture.dataset_offsets[dataset_index] + i for i in locals_])
        for local, sample in zip(locals_, samples, strict=True):
            raw = reference[local]
            torch.testing.assert_close(sample["action"], adapter.action(raw), rtol=0, atol=0)
            torch.testing.assert_close(sample["observation.state"], adapter.state(raw), rtol=0, atol=0)
            assert sample["action_is_pad"].tolist() == raw[f"{adapter.action_keys[0]}_is_pad"].tolist()
            assert (
                sample["dataset_index"].item() == dataset_index
                and sample["dataset_index"].dtype == torch.long
            )
            assert sample["index"].item() == local and sample["task"] == raw["task"]
            for key in ("episode_index", "frame_index", "task_index", "timestamp"):
                assert sample[key].item() == raw[key].item()
            for camera in CAMERAS:
                source = spec.camera_map.get(camera, [camera])[0]
                if source in raw:
                    torch.testing.assert_close(
                        sample[camera], resize_image(raw[source], IMAGE_HW), rtol=0, atol=0
                    )
                    assert not sample[f"{camera}_is_missing"]
                else:
                    assert sample[f"{camera}_is_missing"] and torch.count_nonzero(sample[camera]) == 0
        assert samples[1]["action_is_pad"].any() and not samples[0]["action_is_pad"].any()


def test_samples_collate_and_keep_request_order(mixture):
    indices = [len(mixture) - 1, 3, 3, mixture.dataset_offsets[2] + 1]
    samples = mixture.__getitems__(indices)
    assert [s["dataset_index"].item() for s in samples] == [len(SPECS) - 1, 0, 0, 2]
    torch.testing.assert_close(samples[1]["action"], samples[2]["action"])
    batch = default_collate(samples)
    assert batch["action"].shape == (4, CHUNK, 20) and batch["observation.state"].shape == (4, 1, 20)
    assert batch[HEAD].shape == (4, 3, *IMAGE_HW) and batch[f"{HEAD}_is_missing"].shape == (4,)
    assert batch["task"] == [s["task"] for s in samples]
    torch.testing.assert_close(mixture[3]["action"], samples[1]["action"])
    with pytest.raises(IndexError):
        mixture.__getitems__([len(mixture)])


def test_only_used_cameras_are_decoded(mixture):
    droid = mixture.children[list(SPECS).index("droid")]
    assert sorted(droid.meta.video_keys) == [
        "observation.images.exterior_1_left",
        "observation.images.wrist_left",
    ]
    assert set(droid.meta.camera_keys) == set(droid.meta.video_keys)


def test_statistics_are_packed_per_embodiment(mixture):
    keys = {key for key in mixture.meta.stats if key.startswith("concertina.embodiment.")}
    assert keys == {
        f"concertina.embodiment.{e}/{f}"
        for e in ("libero", "droid", "agibot", "calvin")
        for f in ("action", "observation.state")
    }
    assert set(mixture.meta.stats[HEAD]) == {"mean", "std"}


def test_native_camera_sizes_must_agree_without_resizing(dataset_dir):
    names = ("libero", "agibot")
    dataset = build(dataset_dir, names, resize_images=False)
    assert dataset.meta.features[HEAD]["shape"] == (24, 32, 3)
    assert dataset[0][HEAD].shape == (3, 24, 32)
    with pytest.raises(ValueError, match="resize_images"):
        build(dataset_dir, names, resize_images=False, image_hw=(16, 16))
    with pytest.raises(ValueError, match="resize_images"):
        build(dataset_dir, ("libero", "calvin"), resize_images=False)


def test_frame_rate_alignment_resamples_actions(dataset_dir):
    aligned = build(dataset_dir, ("droid",), fps_align_target_fps=30.0)
    plain = build(dataset_dir, ("droid",))
    assert aligned.meta.fps == 30.0 and plain.meta.fps == 15
    for index in (0, 13):
        a, p = aligned[index], plain[index]
        torch.testing.assert_close(a["action"], interpolate_sequence(p["action"], 15.0, 30.0))
        assert a["action_is_pad"].tolist() == interpolate_padding(p["action_is_pad"], 15.0, 30.0).tolist()


def test_missing_cameras_are_only_allowed_for_single_arm_wrists(dataset_dir):
    spec = DatasetSpec(
        path="agibot", source="agibot", embodiment="agibot", camera_map={WRIST_LEFT: ["missing"]}
    )
    with pytest.raises(ValueError, match="no source camera"):
        build(dataset_dir, specs=[spec])


def test_missing_columns_and_paths_are_reported(dataset_dir, tmp_path):
    with pytest.raises(ValueError, match="columns"):
        build(
            dataset_dir,
            specs=[DatasetSpec(path="libero", source="droid", embodiment="droid", valid_action_dim=10)],
        )
    with pytest.raises(FileNotFoundError, match="meta/info.json"):
        build(tmp_path, specs=[SPECS["libero"]])
    with pytest.raises(ValueError, match="dataset.root"):
        build(None, specs=[SPECS["libero"]])


def test_absolute_paths_ignore_the_root(dataset_dir):
    spec = DatasetSpec(**{**SPECS["libero"].__dict__, "path": str(dataset_dir / "libero")})
    assert len(build(None, specs=[spec])) == len(build(dataset_dir, ("libero",)))


def test_missing_statistics_are_computed_and_explicit_files_must_exist(dataset_dir, tmp_path):
    shutil.copytree(dataset_dir / "libero", tmp_path / "libero")
    (tmp_path / "libero" / "meta" / "rotation_stats.json").unlink()
    computed = build(tmp_path, ("libero",), normalization="global").meta.stats
    stored = build(dataset_dir, ("libero",), normalization="global").meta.stats
    for feature in ("action", "observation.state"):
        for name, value in stored[feature].items():
            assert torch.equal(computed[feature][name], value), (feature, name)
    assert not (tmp_path / "libero" / "meta" / "rotation_stats.json").exists()
    missing = DatasetSpec(
        **{**SPECS["libero"].__dict__, "rotation_stats_path": str(tmp_path / "missing.json")}
    )
    with pytest.raises(FileNotFoundError, match="concertina-compute-rotation-stats"):
        build(tmp_path, specs=[missing], normalization="global")
    assert build(tmp_path, ("libero",), normalization="none").meta.stats.keys() == set(CAMERAS)


def _assert_same_samples(
    a: ConcertinaMixtureDataset, b: ConcertinaMixtureDataset, indices: list[int]
) -> None:
    for x, y in zip(a.__getitems__(indices), b.__getitems__(indices), strict=True):
        assert x.keys() == y.keys()
        for key, value in x.items():
            assert torch.equal(value, y[key]) if isinstance(value, torch.Tensor) else value == y[key], key


def test_only_the_cameras_in_use_need_local_videos(dataset_dir, tmp_path):
    shutil.copytree(dataset_dir / "droid", tmp_path / "droid")
    shutil.rmtree(
        tmp_path / "droid" / "videos" / "observation.images.exterior_2_left"
    )  # not used by the spec
    _assert_same_samples(build(dataset_dir, ("droid",)), build(tmp_path, ("droid",)), [0, 9, 20])
    shutil.rmtree(tmp_path / "droid" / "videos" / "observation.images.exterior_1_left")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        build(tmp_path, ("droid",))


def _declare_unreferenced_camera(root: Path, camera: str) -> None:
    """Declare a video feature that no episode references (null video metadata, no video files)."""
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    reference = next(key for key, ft in info["features"].items() if ft["dtype"] == "video")
    info["features"][camera] = info["features"][reference]
    info_path.write_text(json.dumps(info, indent=4))
    for shard in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(shard)
        for name in ("chunk_index", "file_index", "from_timestamp", "to_timestamp"):
            column_type = table.schema.field(f"videos/{reference}/{name}").type
            table = table.append_column(f"videos/{camera}/{name}", pa.nulls(table.num_rows, column_type))
        pq.write_table(table, shard)


def test_unreferenced_unused_cameras_are_hidden_from_the_reader(dataset_dir, tmp_path):
    shutil.copytree(dataset_dir / "libero", tmp_path / "libero")
    _declare_unreferenced_camera(tmp_path / "libero", "observation.images.front")
    info = (tmp_path / "libero" / "meta" / "info.json").read_text()
    with pytest.raises(TypeError):
        LeRobotDataset("local/libero", root=tmp_path / "libero")
    _assert_same_samples(build(dataset_dir, ("libero",)), build(tmp_path, ("libero",)), [0, 7, 15, 30])
    assert (tmp_path / "libero" / "meta" / "info.json").read_text() == info


def _rewrite_episode_metadata(root: Path, fn) -> None:
    for shard in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(shard)
        for name in ("dataset_from_index", "dataset_to_index"):
            column = table.schema.get_field_index(name)
            values = fn(table.column(name).to_numpy())
            table = table.set_column(
                column, table.schema.field(column), pa.array(values, type=table.schema.field(column).type)
            )
        pq.write_table(table, shard)


def _rewrite_frame_index(root: Path, shift: int) -> None:
    for path in sorted((root / "data").glob("*/*.parquet")):
        table = pq.read_table(path)
        column = table.schema.get_field_index("index")
        values = pa.array(table.column("index").to_numpy() + shift, type=table.schema.field(column).type)
        pq.write_table(table.set_column(column, table.schema.field(column), values), path)


@pytest.mark.parametrize("corruption", ["episode_bounds", "frame_index"])
def test_non_dense_layouts_are_rejected_and_repaired_offline(dataset_dir, tmp_path, corruption):
    root = tmp_path / "libero"
    shutil.copytree(dataset_dir / "libero", root)
    if corruption == "episode_bounds":
        _rewrite_episode_metadata(root, lambda values: values + 3)
    else:
        _rewrite_frame_index(root, 7)
    with pytest.raises(ValueError, match="concertina-normalize-episode-metadata"):
        build(tmp_path, ("libero",))
    with pytest.raises(SystemExit):
        episode_metadata.main([str(root)])
    episode_metadata.main([str(root), "--write"])
    assert episode_metadata.inspect_layout(root).problems == []
    repaired, original = build(tmp_path, ("libero",)), build(dataset_dir, ("libero",))
    for index in (0, 20, len(original) - 1):
        torch.testing.assert_close(repaired[index]["action"], original[index]["action"])
        assert repaired[index]["index"] == original[index]["index"]


def test_episode_metadata_tool_reports_dense_datasets(dataset_dir, capsys):
    episode_metadata.main([str(dataset_dir / "calvin")])
    assert "dense" in capsys.readouterr().err


def test_dataset_pickles_into_spawned_workers(dataset_dir):
    dataset = build(dataset_dir, ("libero", "agibot"))
    loader = DataLoader(
        dataset, batch_size=4, sampler=[0, 5, 45, 60], num_workers=1, multiprocessing_context="spawn"
    )
    batch = next(iter(loader))
    expected = default_collate(dataset.__getitems__([0, 5, 45, 60]))
    torch.testing.assert_close(batch["action"], expected["action"])
    assert batch["dataset_index"].tolist() == [0, 0, 1, 1]


def test_episode_subsets_read_only_the_selected_episodes(dataset_dir):
    full = build(dataset_dir, ("libero",))
    spec = DatasetSpec(**{**SPECS["libero"].__dict__, "episodes": [2, 0]})
    subset = build(dataset_dir, specs=[spec])
    lengths = [int(e - s) for s, e in zip(*full.episode_bounds[0], strict=True)]
    assert len(subset) == lengths[0] + lengths[2] and subset.num_episodes == 2
    starts, ends = subset.episode_bounds[0]
    assert starts.tolist() == [0, lengths[0]] and ends.tolist() == [lengths[0], lengths[0] + lengths[2]]
    episode_two = int(full.episode_bounds[0][0][2])
    for local, absolute in ((0, 0), (lengths[0], episode_two), (len(subset) - 1, len(full) - 1)):
        sample, reference = subset[local], full[absolute]
        assert sample["index"].item() == absolute
        torch.testing.assert_close(sample["action"], reference["action"], rtol=0, atol=0)
        assert sample["action_is_pad"].tolist() == reference["action_is_pad"].tolist()
        torch.testing.assert_close(sample[HEAD], reference[HEAD], rtol=0, atol=0)
    with pytest.raises(ValueError, match="exceed"):
        build(dataset_dir, specs=[DatasetSpec(**{**SPECS["libero"].__dict__, "episodes": [0, 3]})])
    with pytest.raises(ValueError, match="distinct"):
        ConcertinaDataConfig(datasets=[DatasetSpec(**{**SPECS["libero"].__dict__, "episodes": [1, 1]})])


def test_action_history_windows_are_packed_before_the_chunk(dataset_dir):
    history = build(dataset_dir, ("libero",), action_delta_indices=list(range(-3, CHUNK)))
    plain = build(dataset_dir, ("libero",))
    first, later = history[0], history[10]
    assert first["action"].shape == (3 + CHUNK, 20) and first["action_is_pad"].shape == (3 + CHUNK,)
    assert first["action_is_pad"][:3].all() and not first["action_is_pad"][3:].any()
    torch.testing.assert_close(first["action"][3:], plain[0]["action"])
    torch.testing.assert_close(later["action"][:3], torch.stack([plain[i]["action"][0] for i in (7, 8, 9)]))
    assert history.episode_ids[0].tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="history"):
        build(dataset_dir, ("droid",), action_delta_indices=list(range(-2, CHUNK)), fps_align_target_fps=30.0)
