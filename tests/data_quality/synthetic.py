"""Synthetic LeRobot v3 datasets with controlled pauses, joint motion and camera content."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

VIDEO = {"dtype": "video", "shape": (48, 64, 3), "names": ["height", "width", "channels"]}

# DROID-like episodes (15 fps): (length, static action/state frames, frames where the joint action still
# moves).
DROID_EPISODES = [(90, (20, 75), None), (40, None, None), (100, (5, 90), (5, 25))]
# AgiBot-like episodes (30 fps): (length, static frames, action_config phases).
AGIBOT_EPISODES = [
    (200, (40, 150), [(10, 100, "Grasp the cup."), (100, 190, "Place the cup.")]),
    (120, (0, 110), []),
    (150, (20, 140), [(0, 150, "Wipe the table.")]),
]


def _image(t: int, moving: bool) -> np.ndarray:
    image = np.full((48, 64, 3), 40, np.uint8)
    x = (t * 3) % 50 if moving else 25
    image[10:30, x : x + 12] = 220
    return image


def _vector(shape) -> dict:
    return {"dtype": "float32", "shape": shape if isinstance(shape, tuple) else (shape,), "names": None}


def make_droid_like(root: Path) -> Path:
    features = {
        "action.cartesian_position": _vector(6),
        "action.gripper_position": _vector(1),
        "action.joint_position": _vector(7),
        "observation.state.cartesian_position": _vector(6),
        "observation.state.gripper_position": _vector(1),
        "observation.images.exterior_1_left": VIDEO,
        "observation.images.exterior_2_left": VIDEO,
        "observation.images.wrist_left": VIDEO,
    }
    dataset = LeRobotDataset.create(repo_id="local/droid_like", fps=15, features=features, root=root)
    rng = np.random.default_rng(0)
    for episode, (length, static, joint_motion) in enumerate(DROID_EPISODES):
        pose, joints = np.zeros(6), np.zeros(7)
        for t in range(length):
            moving = not (static and static[0] <= t < static[1])
            if moving:
                pose = pose + np.concatenate([rng.normal(0, 0.01, 3), rng.normal(0, 0.02, 3)])
            if moving or (joint_motion and joint_motion[0] <= t < joint_motion[1]):
                joints = joints + 0.01
            image = _image(t, moving)
            dataset.add_frame(
                {
                    "action.cartesian_position": pose.astype(np.float32),
                    "action.gripper_position": np.array([0.5], np.float32),
                    "action.joint_position": joints.astype(np.float32),
                    "observation.state.cartesian_position": pose.astype(np.float32),
                    "observation.state.gripper_position": np.array([0.5], np.float32),
                    "observation.images.exterior_1_left": image,
                    "observation.images.exterior_2_left": image,
                    "observation.images.wrist_left": image,
                    "task": f"task {episode}",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    return root


def make_agibot_like(root: Path) -> Path:
    features = {
        "actions.end.position": _vector((2, 3)),
        "actions.end.orientation": _vector((2, 4)),
        "actions.effector.position": _vector(2),
        "actions.joint.position": _vector(14),
        "observation.states.end.position": _vector((2, 3)),
        "observation.states.end.orientation": _vector((2, 4)),
        "observation.states.effector.position": _vector(2),
        "observation.images.head": VIDEO,
    }
    dataset = LeRobotDataset.create(repo_id="local/agibot_like", fps=30, features=features, root=root)
    rng = np.random.default_rng(1)
    for length, static, _ in AGIBOT_EPISODES:
        position, rotation = np.zeros((2, 3)), np.zeros((2, 3))
        for t in range(length):
            moving = not (static and static[0] <= t < static[1])
            if moving:
                position = position + rng.normal(0, 0.002, (2, 3))
                rotation = rotation + rng.normal(0, 0.01, (2, 3))
            quaternion = Rotation.from_rotvec(rotation).as_quat().astype(np.float32)
            dataset.add_frame(
                {
                    "actions.end.position": position.astype(np.float32),
                    "actions.end.orientation": quaternion,
                    "actions.effector.position": np.array([0.2, 0.8], np.float32),
                    "actions.joint.position": np.tile(position.reshape(-1), 14 // 6 + 1)[:14].astype(
                        np.float32
                    ),
                    "observation.states.end.position": position.astype(np.float32),
                    "observation.states.end.orientation": quaternion,
                    "observation.states.effector.position": np.array([0.2, 0.8], np.float32),
                    "observation.images.head": _image(t, moving),
                    "task": "clean up",
                }
            )
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    # AgiBot World ships its phase annotations as an `action_config` column of the episode metadata.
    for path in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
        table = pq.read_table(path)
        phases = [
            [
                {"start_frame": start, "end_frame": end, "action_text": text, "skill": "manipulate"}
                for start, end, text in AGIBOT_EPISODES[int(episode)][2]
            ]
            for episode in table["episode_index"].to_pylist()
        ]
        phase_type = pa.list_(
            pa.struct(
                [
                    ("start_frame", pa.int64()),
                    ("end_frame", pa.int64()),
                    ("action_text", pa.string()),
                    ("skill", pa.string()),
                ]
            )
        )
        pq.write_table(table.append_column("action_config", pa.array(phases, type=phase_type)), path)
    return root
