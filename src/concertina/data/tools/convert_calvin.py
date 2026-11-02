"""`concertina-convert-calvin`: convert an official CALVIN split (one ``episode_XXXXXXX.npz`` per frame) to
LeRobot v3.

Every language-annotated segment (``lang_annotations/auto_lang_ann.npy``) becomes one episode with the annotation as
its task. Columns: ``action`` and ``action.absolute`` (CALVIN ``actions``: absolute TCP pose + gripper), ``action.relative``
(``rel_actions``), ``observation.state`` (``robot_obs``), ``observation.environment_state`` (``scene_obs``) and the
``rgb_static`` / ``rgb_gripper`` cameras as H.264 videos ``observation.images.top`` / ``observation.images.wrist``.

Episode order: the annotated segments are dealt round-robin into ``--episode-interleave`` groups that are
concatenated (segment ``i`` goes to group ``i % G``). The released CALVIN dataset was written by 192 parallel
converters with this assignment, so the default ``G=192`` reproduces its episode order; ``G=1`` keeps the
annotation order. ``--workers`` only splits the ordered episodes into contiguous parts that are converted in
parallel and aggregated; it does not change the order or the content of any frame.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALVIN_FPS = 30
DEFAULT_EPISODE_INTERLEAVE = 192
TOP_CAMERA = "observation.images.top"
WRIST_CAMERA = "observation.images.wrist"
REQUIRED_KEYS = ("rgb_static", "rgb_gripper", "robot_obs", "scene_obs", "actions", "rel_actions")

POSE_AXES = [f"ee_position_{a}" for a in "xyz"] + [f"ee_orientation_r{a}" for a in "xyz"]
ACTION_AXES = [*POSE_AXES, "gripper_action"]
REL_ACTION_AXES = [f"delta_{axis}" for axis in POSE_AXES] + ["gripper_action"]
STATE_AXES = [*POSE_AXES, "gripper_width", *(f"joint_position_{i:02d}" for i in range(7)), "gripper_action"]
SCENE_AXES = ["sliding_door", "drawer", "button", "switch", "lightbulb", "green_light"] + [
    f"{block}_block_{axis}" for block in ("red", "blue", "pink") for axis in ("x", "y", "z", "rx", "ry", "rz")
]


@dataclass(frozen=True)
class Segment:
    start: int
    end: int  # inclusive, as in CALVIN's annotations
    task: str


def calvin_features() -> dict:
    def vector(axes: list[str]) -> dict:
        return {"dtype": "float32", "shape": (len(axes),), "names": {"axes": axes}}

    def camera(size: int) -> dict:
        return {"dtype": "video", "shape": (size, size, 3), "names": ["height", "width", "channels"]}

    return {
        TOP_CAMERA: camera(200),
        WRIST_CAMERA: camera(84),
        "observation.state": vector(STATE_AXES),
        "observation.environment_state": vector(SCENE_AXES),
        "action": vector(ACTION_AXES),
        "action.absolute": vector(ACTION_AXES),
        "action.relative": vector(REL_ACTION_AXES),
    }


def resolve_split_dir(calvin_root: Path, split: str) -> Path:
    """``<root>/<split>``, or ``<root>/task_*/<split>`` when ``root`` holds exactly one extracted CALVIN task."""
    if (calvin_root / split).is_dir():
        return calvin_root / split
    candidates = sorted(path / split for path in calvin_root.glob("task_*") if (path / split).is_dir())
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected {calvin_root}/{split} or exactly one {calvin_root}/task_*/{split}, found {candidates}"
        )
    return candidates[0]


def load_segments(split_dir: Path) -> list[Segment]:
    path = split_dir / "lang_annotations" / "auto_lang_ann.npy"
    if not path.is_file():
        raise FileNotFoundError(f"No CALVIN language annotations at {path}")
    annotations = np.load(path, allow_pickle=True).item()
    return [
        Segment(int(start), int(end), str(task))
        for (start, end), task in zip(
            annotations["info"]["indx"], annotations["language"]["ann"], strict=True
        )
    ]


def episode_order(num_segments: int, interleave: int) -> list[int]:
    """Segment index of each output episode."""
    return [i for group in range(interleave) for i in range(group, num_segments, interleave)]


def frame_path(split_dir: Path, frame: int) -> Path:
    return split_dir / f"episode_{frame:07d}.npz"


def parts_dir(output: Path) -> Path:
    return output.with_name(output.name + ".parts")


def convert_part(split_dir: Path, segments: list[Segment], root: Path, name: str, part: int) -> None:
    """Write ``segments`` (in order) as one LeRobot dataset at ``root``."""
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset = LeRobotDataset.create(
        repo_id=name,
        root=root,
        robot_type="panda",
        fps=CALVIN_FPS,
        features=calvin_features(),
        use_videos=True,
        image_writer_threads=8,
        rgb_encoder=RGBEncoderConfig(vcodec="h264"),
        metadata_buffer_size=100,
    )
    started = time.monotonic()
    try:
        for count, segment in enumerate(segments, start=1):
            for frame in range(segment.start, segment.end + 1):
                with np.load(frame_path(split_dir, frame)) as npz:
                    absolute = npz["actions"].astype(np.float32)
                    dataset.add_frame(
                        {
                            TOP_CAMERA: npz["rgb_static"],
                            WRIST_CAMERA: npz["rgb_gripper"],
                            "observation.state": npz["robot_obs"].astype(np.float32),
                            "observation.environment_state": npz["scene_obs"].astype(np.float32),
                            "action": absolute,
                            "action.absolute": absolute,
                            "action.relative": npz["rel_actions"].astype(np.float32),
                            "task": segment.task,
                        }
                    )
            dataset.save_episode()
            if count % 100 == 0 or count == len(segments):
                rate = count / max(time.monotonic() - started, 1e-6)
                logging.info("part %d: %d/%d episodes (%.1f episodes/s)", part, count, len(segments), rate)
    finally:
        dataset.finalize()
    # The writer leaves its emptied temporary image directories behind.
    images = root / "images"
    if images.is_dir() and not any(path.is_file() for path in images.rglob("*")):
        shutil.rmtree(images)


def convert_calvin(
    calvin_root: Path,
    output: Path,
    *,
    split: str = "training",
    interleave: int = DEFAULT_EPISODE_INTERLEAVE,
    max_episodes: int | None = None,
    workers: int = 1,
    overwrite: bool = False,
) -> None:
    if interleave < 1 or workers < 1 or (max_episodes is not None and max_episodes < 1):
        raise ValueError("--episode-interleave, --workers and --max-episodes must be positive")
    split_dir = resolve_split_dir(calvin_root, split)
    segments = load_segments(split_dir)
    selected = [segments[i] for i in episode_order(len(segments), interleave)][:max_episodes]
    missing = [
        frame_path(split_dir, frame)
        for segment in selected
        for frame in range(segment.start, segment.end + 1)
        if not frame_path(split_dir, frame).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"{len(missing)} CALVIN frame files are missing, e.g. {missing[:3]}")
    with np.load(frame_path(split_dir, selected[0].start)) as npz:
        absent = sorted(set(REQUIRED_KEYS) - set(npz.files))
    if absent:
        raise KeyError(f"CALVIN frames lack the keys {absent}")

    for path in (output, parts_dir(output)):
        if path.exists():
            if not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            shutil.rmtree(path)
    workers = min(workers, len(selected))
    print(f"converting {len(selected)} of {len(segments)} CALVIN segments from {split_dir}", file=sys.stderr)
    if workers == 1:
        convert_part(split_dir, selected, output, output.name, 0)
    else:
        convert_parts(split_dir, selected, output, workers)
    # LeRobot's writer publishes videos from private temporary files (mode 0600).
    umask = os.umask(0)
    os.umask(umask)
    for path in (output / "videos").rglob("*.mp4"):
        path.chmod(0o666 & ~umask)


def convert_parts(split_dir: Path, segments: list[Segment], output: Path, workers: int) -> None:
    """Convert contiguous parts of ``segments`` in parallel and aggregate them, in order, into ``output``."""
    from lerobot.datasets.aggregate import aggregate_datasets

    parts_root = parts_dir(output)
    bounds = np.linspace(0, len(segments), workers + 1).astype(int)
    roots = [parts_root / f"part-{part:04d}" for part in range(workers)]
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [
            pool.submit(convert_part, split_dir, segments[lo:hi], root, root.name, part)
            for part, (lo, hi, root) in enumerate(zip(bounds[:-1], bounds[1:], roots, strict=True))
        ]
        for future in futures:
            future.result()
    aggregate_datasets(
        repo_ids=[root.name for root in roots], aggr_repo_id=output.name, roots=roots, aggr_root=output
    )
    shutil.rmtree(parts_root)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--calvin-root",
        type=Path,
        required=True,
        help="extracted CALVIN task directory (e.g. task_ABC_D) or its parent",
    )
    parser.add_argument("--output", type=Path, required=True, help="output LeRobot dataset directory")
    parser.add_argument("--split", choices=["training", "validation"], default="training")
    parser.add_argument(
        "--episode-interleave",
        type=int,
        default=DEFAULT_EPISODE_INTERLEAVE,
        help="episode order (see above); 192 reproduces the released dataset, 1 keeps annotation order",
    )
    parser.add_argument(
        "--max-episodes", type=int, help="convert only the first N episodes of the order (smoke tests)"
    )
    parser.add_argument("--workers", type=int, default=1, help="parallel converter processes")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    convert_calvin(
        args.calvin_root,
        args.output,
        split=args.split,
        interleave=args.episode_interleave,
        max_episodes=args.max_episodes,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
