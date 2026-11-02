import json
import os

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from concertina.data.tools.convert_calvin import calvin_features, convert_calvin, episode_order


def test_episode_order_deals_segments_round_robin():
    assert episode_order(5, 1) == [0, 1, 2, 3, 4]
    assert episode_order(5, 2) == [0, 2, 4, 1, 3]
    assert episode_order(3, 192) == [0, 1, 2]


def test_converts_segments_in_interleaved_order(calvin_root, sources, tmp_path):
    output = tmp_path / "calvin"
    convert_calvin(calvin_root, output, interleave=2)

    assert not (output / "images").exists()
    umask = os.umask(0)
    os.umask(umask)
    assert {path.stat().st_mode & 0o777 for path in (output / "videos").rglob("*.mp4")} == {0o666 & ~umask}
    dataset = LeRobotDataset("calvin", root=output, video_backend="pyav")
    order = [0, 2, 1, 3]
    assert dataset.num_episodes == len(order)
    assert dataset.meta.fps == 30
    assert list(dataset.meta.episodes["length"]) == [
        sources.CALVIN_SEGMENTS[i][1] - sources.CALVIN_SEGMENTS[i][0] + 1 for i in order
    ]
    assert set(calvin_features()) <= set(dataset.meta.features)
    position = 0
    for segment in order:
        start, end, task = sources.CALVIN_SEGMENTS[segment]
        for frame in range(start, end + 1):
            item, source = dataset[position], sources.calvin_frame(frame)
            assert item["task"] == task
            assert int(item["frame_index"]) == frame - start
            for key, column in (
                ("observation.state", "robot_obs"),
                ("observation.environment_state", "scene_obs"),
                ("action", "actions"),
                ("action.absolute", "actions"),
                ("action.relative", "rel_actions"),
            ):
                assert torch.equal(item[key], torch.from_numpy(source[column].astype(np.float32)))
            top = item["observation.images.top"]
            assert top.shape == (3, 200, 200)
            assert abs(top.mean().item() * 255 - 10 * frame) < 2
            assert item["observation.images.wrist"].shape == (3, 84, 84)
            position += 1


def test_parallel_parts_give_the_same_dataset(calvin_root, tmp_path):
    convert_calvin(calvin_root, tmp_path / "one", interleave=2)
    convert_calvin(calvin_root, tmp_path / "two", interleave=2, workers=2)

    assert not (tmp_path / "two.parts").exists()
    one = pq.read_table(tmp_path / "one/data/chunk-000/file-000.parquet")
    two = pq.read_table(tmp_path / "two/data/chunk-000/file-000.parquet")
    for column in one.column_names:
        assert one[column].to_pylist() == two[column].to_pylist(), column
    tasks = [
        json.dumps(row) for row in pq.read_table(tmp_path / "one/meta/episodes").column("tasks").to_pylist()
    ]
    assert tasks == [
        json.dumps(row) for row in pq.read_table(tmp_path / "two/meta/episodes").column("tasks").to_pylist()
    ]
    assert LeRobotDataset("two", root=tmp_path / "two").num_frames == one.num_rows


def test_max_episodes_missing_frames_and_existing_output(calvin_root, tmp_path):
    output = tmp_path / "calvin"
    convert_calvin(calvin_root, output, interleave=1, max_episodes=1)
    assert LeRobotDataset("calvin", root=output).num_episodes == 1
    with pytest.raises(FileExistsError):
        convert_calvin(calvin_root, output, interleave=1, max_episodes=1)

    (calvin_root / "task_X/training/episode_0000012.npz").unlink()
    with pytest.raises(FileNotFoundError, match="missing"):
        convert_calvin(calvin_root, output, overwrite=True)
