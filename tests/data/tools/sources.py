"""Synthetic sources in the official CALVIN layout."""

from pathlib import Path

import numpy as np

CALVIN_SEGMENTS = [
    (0, 4, "open the drawer"),
    (3, 8, "push the button"),
    (10, 13, "open the drawer"),
    (14, 16, "lift"),
]


def calvin_frame(frame: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(frame)
    return {
        "rgb_static": np.full((200, 200, 3), 10 * frame, dtype=np.uint8),
        "rgb_gripper": np.full((84, 84, 3), 200 - 10 * frame, dtype=np.uint8),
        "rgb_tactile": np.zeros((160, 120, 6), dtype=np.uint8),
        "depth_static": rng.random((200, 200), dtype=np.float32),
        "robot_obs": rng.normal(size=15),
        "scene_obs": rng.normal(size=24),
        "actions": rng.normal(size=7),
        "rel_actions": rng.uniform(-1, 1, size=7),
    }


def make_calvin_root(root: Path) -> Path:
    """``<root>/task_X/training`` with one npz per frame and the language annotations."""
    split = root / "task_X" / "training"
    (split / "lang_annotations").mkdir(parents=True)
    for frame in range(17):
        np.savez(split / f"episode_{frame:07d}.npz", **calvin_frame(frame))
    annotations = {
        "info": {"indx": [(start, end) for start, end, _ in CALVIN_SEGMENTS], "episodes": []},
        "language": {
            "ann": [task for *_, task in CALVIN_SEGMENTS],
            "task": [task.replace(" ", "_") for *_, task in CALVIN_SEGMENTS],
            "emb": np.zeros((len(CALVIN_SEGMENTS), 1, 8), dtype=np.float32),
        },
    }
    np.save(split / "lang_annotations" / "auto_lang_ann.npy", annotations, allow_pickle=True)
    return root
