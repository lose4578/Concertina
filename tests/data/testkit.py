"""Synthetic LeRobot datasets and a Concertina policy-config shim shared by the data and train tests.

Datasets are written with the upstream `LeRobotDataset.create` API (videos encoded) and get their
`meta/rotation_stats.json` from the `concertina-compute-rotation-stats` implementation. The shim is a
`ConcertinaConfig` subclass registered as policy type "concertina_shim" in a temporary package whose sibling
`modeling_` module holds a tiny recording policy and whose `processor_` module re-exports the Concertina
processors, so that the upstream factories resolve it by convention.
"""

from __future__ import annotations

import importlib
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.spatial.transform import Rotation

from concertina.data.stats import ROTATION_STATS_FILENAME
from concertina.data.tools.rotation_stats import compute_rotation_stats, write_rotation_stats

IMAGE = {"names": ["height", "width", "channels"]}


def _video(height: int, width: int) -> dict:
    # The bundled AV1 encoder rejects frames narrower than 32 pixels.
    return {"dtype": "video", "shape": (height, width, 3), **IMAGE}


def _vector(dim: int | tuple[int, ...]) -> dict:
    shape = (dim,) if isinstance(dim, int) else dim
    return {"dtype": "float32", "shape": shape, "names": None}


def _frame_image(height: int, width: int, value: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = value % 256
    image[..., 1] = np.arange(width, dtype=np.uint8)[None, :] * 4
    image[..., 2] = np.arange(height, dtype=np.uint8)[:, None] * 4
    return image


def write_dataset(
    root: Path,
    *,
    fps: int,
    features: dict[str, dict],
    lengths: list[int],
    frame: Callable[[np.random.Generator, int, int], dict],
    source: str | None,
    task: Callable[[int], str] = lambda episode: f"task {episode % 2}",
) -> Path:
    """Create a dataset whose frame ``t`` of episode ``e`` is ``frame(rng, e, t)`` plus camera images."""
    rng = np.random.default_rng(len(lengths) * 1000 + fps)
    dataset = LeRobotDataset.create(
        repo_id=f"local/{root.name}", fps=fps, features=features, root=root, use_videos=True
    )
    cameras = {key: ft["shape"] for key, ft in features.items() if ft["dtype"] == "video"}
    counter = 0
    for episode, length in enumerate(lengths):
        for t in range(length):
            values = frame(rng, episode, t)
            for key, (height, width, _) in cameras.items():
                values[key] = _frame_image(height, width, counter)
            values["task"] = task(episode)
            dataset.add_frame(values)
            counter += 1
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()
    if source is not None:
        write_rotation_stats(
            compute_rotation_stats(root, source), root / "meta" / ROTATION_STATS_FILENAME, source
        )
    return root


def _unit_quat_wxyz(rng: np.random.Generator) -> np.ndarray:
    xyzw = Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat()
    return np.concatenate([xyzw[3:], xyzw[:3]]).astype(np.float32)


def make_libero(
    root: Path, lengths=(14, 9, 20), cameras=("observation.images.image", "observation.images.image2")
):
    features = {
        "action": _vector(7),
        "observation.state": _vector(8),
        **{camera: _video(24, 32) for camera in cameras},
    }

    def frame(rng, episode, t):
        gripper = 1.0 if (t // 4) % 2 else -1.0
        return {
            "action": np.concatenate([rng.normal(0, 0.1, 6), [gripper]]).astype(np.float32),
            "observation.state": np.concatenate(
                [rng.normal(0, 0.3, 3), rng.normal(0, 0.8, 3), rng.uniform(0, 0.04, 2)]
            ).astype(np.float32),
        }

    return write_dataset(root, fps=10, features=features, lengths=list(lengths), frame=frame, source="libero")


def make_calvin(root: Path, lengths=(12, 16)):
    features = {
        "action": _vector(7),
        "action.absolute": _vector(7),
        "action.relative": _vector(7),
        "observation.state": _vector(15),
        "observation.images.top": _video(32, 32),
        "observation.images.wrist": _video(16, 32),
    }

    def frame(rng, episode, t):
        gripper = 1.0 if (t // 3) % 2 else -1.0
        absolute = np.concatenate([rng.normal(0, 0.2, 3), rng.uniform(-3, 3, 3), [gripper]]).astype(
            np.float32
        )
        return {
            "action": absolute,
            "action.absolute": absolute,
            "action.relative": np.concatenate([rng.normal(0, 0.05, 6), [gripper]]).astype(np.float32),
            "observation.state": np.concatenate(
                [
                    rng.normal(0, 0.2, 3),
                    rng.uniform(-3, 3, 3),
                    rng.uniform(0, 0.08, 1),
                    rng.normal(0, 1, 7),
                    [gripper],
                ]
            ).astype(np.float32),
        }

    return write_dataset(root, fps=10, features=features, lengths=list(lengths), frame=frame, source="calvin")


def make_droid(root: Path, lengths=(16, 11)):
    features = {
        "action.cartesian_position": _vector(6),
        "action.gripper_position": _vector(1),
        "observation.state.cartesian_position": _vector(6),
        "observation.state.gripper_position": _vector(1),
        "observation.images.exterior_1_left": _video(16, 32),
        "observation.images.exterior_2_left": _video(16, 32),
        "observation.images.wrist_left": _video(16, 32),
    }

    def frame(rng, episode, t):
        def cartesian():
            return np.concatenate([rng.normal(0, 0.3, 3), rng.uniform(-3, 3, 3)]).astype(np.float32)

        return {
            "action.cartesian_position": cartesian(),
            "action.gripper_position": np.array([rng.uniform(0, 1)], dtype=np.float32),
            "observation.state.cartesian_position": cartesian(),
            "observation.state.gripper_position": np.array([rng.uniform(0, 1)], dtype=np.float32),
        }

    return write_dataset(root, fps=15, features=features, lengths=list(lengths), frame=frame, source="droid")


def make_agibot(root: Path, lengths=(9, 14)):
    features = {
        "actions.end.position": _vector((2, 3)),
        "actions.end.orientation": _vector((2, 4)),
        "actions.effector.position": _vector(2),
        "observation.states.end.position": _vector((2, 3)),
        "observation.states.end.orientation": _vector((2, 4)),
        "observation.states.effector.position": _vector(2),
        "observation.images.head": _video(24, 32),
        "observation.images.hand_left": _video(24, 32),
        "observation.images.hand_right": _video(24, 32),
    }

    def quats(rng):
        return np.stack(
            [Rotation.random(random_state=int(rng.integers(1 << 30))).as_quat() for _ in range(2)]
        )

    def frame(rng, episode, t):
        return {
            "actions.end.position": rng.normal(0, 0.3, (2, 3)).astype(np.float32),
            "actions.end.orientation": quats(rng).astype(np.float32),
            "actions.effector.position": rng.uniform(0, 1, 2).astype(np.float32),
            "observation.states.end.position": rng.normal(0, 0.3, (2, 3)).astype(np.float32),
            "observation.states.end.orientation": quats(rng).astype(np.float32),
            "observation.states.effector.position": rng.uniform(0, 1, 2).astype(np.float32),
        }

    return write_dataset(root, fps=30, features=features, lengths=list(lengths), frame=frame, source="agibot")


SHIM_PACKAGE = "concertina_shim_policy"
SHIM_TYPE = "concertina_shim"
SHIM_MODULES = {
    "__init__.py": "",
    "configuration_concertina_shim.py": """
        from dataclasses import dataclass

        from lerobot.configs import PreTrainedConfig

        from concertina.policy.configuration_concertina import ConcertinaConfig


        @PreTrainedConfig.register_subclass("concertina_shim")
        @dataclass
        class ConcertinaShimConfig(ConcertinaConfig):
            pass
        """,
    "modeling_concertina_shim.py": """
        import torch
        from lerobot.policies import PreTrainedPolicy

        from .configuration_concertina_shim import ConcertinaShimConfig

        SEEN = []


        class ConcertinaShimPolicy(PreTrainedPolicy):
            config_class = ConcertinaShimConfig
            name = "concertina_shim"

            def __init__(self, config, dataset_stats=None, dataset_meta=None, **kwargs):
                super().__init__(config)
                self.head = torch.nn.Linear(20, 20)
                # Mirrors ConcertinaPolicy.model: a training-step counter saved with the weights.
                self.model = torch.nn.Module()
                self.model.register_buffer("training_step", torch.zeros((), dtype=torch.long))

            def get_optim_params(self):
                return self.parameters()

            def reset(self):
                pass

            def forward(self, batch, reduction="mean"):
                prediction = self.head(batch["observation.state"][:, -1])
                loss = ((prediction - batch["action"][:, 0]) ** 2).mean()
                record = (batch["dataset_index"].tolist(), batch["index"].tolist(), list(batch["task"]))
                if self.training:
                    self.model.training_step += 1
                SEEN.append((*record, loss.item(), frozenset(batch), int(self.model.training_step)))
                return loss, {}

            def predict_action_chunk(self, batch, **kwargs):
                return self.head(batch["observation.state"][:, -1]).unsqueeze(1)

            def select_action(self, batch, **kwargs):
                return self.predict_action_chunk(batch)[:, 0]
        """,
    "processor_concertina_shim.py": """
        from concertina.policy.processor_concertina import (
            make_concertina_pre_post_processors as make_concertina_shim_pre_post_processors,
            make_concertina_pre_post_processors_from_pretrained as
                make_concertina_shim_pre_post_processors_from_pretrained,
        )
        """,
}


def import_shim(directory: Path):
    """Write (once per session) and import the shim package; returns its configuration and modeling modules."""
    if SHIM_PACKAGE not in sys.modules:
        package = directory / SHIM_PACKAGE
        package.mkdir(parents=True, exist_ok=True)
        for filename, source in SHIM_MODULES.items():
            (package / filename).write_text(textwrap.dedent(source))
        sys.path.insert(0, str(directory))
    configuration = importlib.import_module(f"{SHIM_PACKAGE}.configuration_concertina_shim")
    modeling = importlib.import_module(f"{SHIM_PACKAGE}.modeling_concertina_shim")
    return configuration, modeling
