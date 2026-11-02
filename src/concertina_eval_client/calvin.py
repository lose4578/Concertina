# Adapted from CALVIN (MIT License), Copyright (c) 2021 Oier Mees: `evaluate_sequence` and `rollout` of
# calvin_models/calvin_agent/evaluation/evaluate_policy.py and `count_success` of calvin_agent/evaluation/utils.py.
# Modified: the policy runs in a separate server process, the sequence reset restores the gripper command,
# sequences come from a saved manifest, and per-sequence results and videos are written. See NOTICE.
"""CALVIN long-horizon evaluation client (ABC -> D protocol).

Runs CALVIN sequences against a Concertina policy server: for every sequence the environment is reset to the
sequence's initial state, and each of its five subtasks gets at most `ep_len` steps; a sequence ends at the
first failed subtask. The model is reset at every subtask start. Actions are absolute end-effector targets
``[xyz, Euler xyz, gripper]``.

Run in the CALVIN environment (Python >= 3.8, `calvin_env` / `calvin_agent` importable or found under
``--calvin-root``)::

    python -m concertina_eval_client.calvin --calvin-root $CALVIN_ROOT --dataset-path $CALVIN_DATA/task_ABC_D \
        --manifest sequences.json --port 9999 --output-dir out/worker_00 --egl-device 0

Every sequence is appended to ``results.jsonl`` as soon as it finishes (an interrupted shard resumes after
the last finished sequence) and, unless ``--no-video``, its static-camera rollout is saved as an MP4.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from concertina_eval_client.protocol import PolicyClient

logger = logging.getLogger("concertina_eval_client.calvin")

SEQUENCE_LENGTH = 5
CAMERAS = ("rgb_static", "rgb_gripper")
OBS_SPACE = {"rgb_obs": list(CAMERAS), "depth_obs": []}
TASKS_CONFIG = "callbacks/rollout/tasks/new_playtable_tasks.yaml"
ANNOTATIONS_CONFIG = "annotations/new_playtable_validation.yaml"

_EGL_PROBE = r"""
import ctypes, ctypes.util, json
lib = ctypes.CDLL(ctypes.util.find_library("EGL") or "libEGL.so.1")
lib.eglGetProcAddress.restype = ctypes.c_void_p
lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]
def proc(name, restype, *argtypes):
    address = lib.eglGetProcAddress(name.encode())
    if not address:
        raise RuntimeError("missing " + name)
    return ctypes.CFUNCTYPE(restype, *argtypes)(address)
query_devices = proc("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                     ctypes.POINTER(ctypes.c_int))
query_attrib = proc("eglQueryDeviceAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                    ctypes.POINTER(ctypes.c_longlong))
devices = (ctypes.c_void_p * 64)()
count = ctypes.c_int(0)
if not query_devices(64, devices, ctypes.byref(count)):
    raise RuntimeError("eglQueryDevicesEXT failed")
mapping = {}
for egl_id in range(count.value):
    value = ctypes.c_longlong(-1)
    if query_attrib(devices[egl_id], 0x323A, ctypes.byref(value)):  # EGL_CUDA_DEVICE_NV
        mapping[int(value.value)] = egl_id
print(json.dumps(mapping))
"""


def add_calvin_to_path(calvin_root: Path) -> None:
    """Make the packages of a CALVIN checkout importable (what CALVIN's editable installs register)."""
    root = calvin_root.expanduser().resolve()
    for package in ("calvin_models", "calvin_env", "calvin_env/tacto"):
        if str(root / package) not in sys.path:
            sys.path.insert(0, str(root / package))


def egl_device_for_gpu(gpu: int) -> int:
    """The EGL device index of the physical GPU ``gpu`` (nvidia-smi / PCI bus order)."""
    env = {key: value for key, value in os.environ.items() if key != "CUDA_VISIBLE_DEVICES"}
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    output = subprocess.run(
        [sys.executable, "-c", _EGL_PROBE], env=env, capture_output=True, text=True, timeout=60, check=True
    ).stdout
    mapping = {int(cuda): egl for cuda, egl in json.loads(output).items()}
    if gpu not in mapping:
        raise RuntimeError(f"No EGL device reports CUDA device {gpu}; EGL devices: {mapping}")
    return mapping[gpu]


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Sequence rows ``{"sequence_index", "initial_state", "sequence"}``.

    The key order of each ``initial_state`` is significant: CALVIN seeds the block placement of an initial
    condition with a hash of ``str(initial_state.values())``.
    """
    rows = json.loads(path.read_text(encoding="utf-8"))["sequences"]
    for position, row in enumerate(rows):
        if int(row["sequence_index"]) != position:
            raise ValueError(f"{path}: row {position} has sequence_index {row['sequence_index']}")
        if len(row["sequence"]) != SEQUENCE_LENGTH:
            raise ValueError(f"{path}: sequence {position} does not have {SEQUENCE_LENGTH} subtasks")
    return rows


def write_manifest(path: Path, num_sequences: int) -> None:
    """Generate CALVIN's evaluation sequences (`calvin_agent.evaluation.multistep_sequences.get_sequences`)."""
    from calvin_agent.evaluation.multistep_sequences import get_sequences

    rows = []
    for index, (initial_state, sequence) in enumerate(get_sequences(num_sequences, num_workers=1)):
        state = {
            str(key): (value.item() if isinstance(value, np.generic) else value)
            for key, value in initial_state.items()
        }
        rows.append(
            {"sequence_index": index, "initial_state": state, "sequence": [str(task) for task in sequence]}
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ",\n".join(json.dumps(row) for row in rows)
    path.write_text(f'{{"num_sequences": {len(rows)}, "sequences": [\n{lines}\n]}}\n', encoding="utf-8")


def chain_success(success_lens: Sequence[int]) -> list[float]:
    """Fraction of sequences whose first k subtasks succeeded, for k = 1..5."""
    total = len(success_lens)
    return [sum(1 for value in success_lens if value >= k) / total if total else 0.0 for k in range(1, 6)]


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sequence rows: average successful length, chain success rates, per-task success."""
    rows = sorted(rows, key=lambda row: int(row["sequence_index"]))
    success_lens = [int(row["success_len"]) for row in rows]
    succeeded, attempted = Counter(), Counter()
    for row in rows:
        done = int(row["success_len"])
        for task in row["sequence"][: done + 1]:
            attempted[task] += 1
        succeeded.update(row["sequence"][:done])
    return {
        "num_sequences": len(rows),
        "avg_seq_len": float(np.mean(success_lens)) if rows else 0.0,
        "chain_sr": {str(k): rate for k, rate in enumerate(chain_success(success_lens), start=1)},
        "tasks": {
            task: {
                "success": succeeded[task],
                "total": attempted[task],
                "success_rate": succeeded[task] / attempted[task],
            }
            for task in sorted(attempted)
        },
    }


def read_results(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def env_action(action: np.ndarray) -> list[np.ndarray]:
    """Server action ``[xyz, euler(3), gripper]`` -> CALVIN's absolute action ``[pos, orn, gripper]``.

    A flat array would be interpreted as a normalized relative action. The gripper command is +1 (open) / -1.
    """
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError(f"Expected a finite 7D CALVIN action, got {action}")
    gripper = np.float32(1.0 if action[6] > 0 else -1.0)
    return [action[:3].copy(), action[3:6].copy(), np.array([gripper], dtype=np.float32)]


def observation_payload(obs: dict[str, Any], instruction: str) -> dict[str, Any]:
    return {
        "images": {camera: np.asarray(obs["rgb_obs"][camera]) for camera in CAMERAS},
        "state": {"robot_obs": np.asarray(obs["robot_obs"], dtype=np.float32)},
        "instruction": instruction,
    }


def reset_to_initial_state(env: Any, robot_obs: np.ndarray, scene_obs: np.ndarray) -> dict[str, Any]:
    """Reset for a new sequence, restoring the initial gripper command.

    `Robot.reset` restores the arm joints and finger width but not `Robot.gripper_action`, which the
    observation exposes (`robot_obs[14]`); without this the previous sequence's last command leaks in.
    """
    command = float(robot_obs[14])
    if command not in (-1.0, 1.0):
        raise ValueError(f"The initial gripper command must be -1 or +1, got {command}")
    env.robot.gripper_action = int(command)
    obs = env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    if float(np.asarray(obs["robot_obs"])[14]) != command:
        raise RuntimeError("CALVIN reset returned an inconsistent gripper command")
    return obs


def write_video(path: Path, frames: Sequence[np.ndarray], fps: int) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial.mp4")
    imageio.mimsave(
        partial,
        [np.asarray(frame, dtype=np.uint8) for frame in frames],
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_params=["-threads", "1"],
    )
    os.replace(partial, path)


class CalvinEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        add_calvin_to_path(Path(args.calvin_root))
        # CALVIN's dependencies still use NumPy aliases removed in NumPy 1.24.
        for alias, value in (("int", int), ("float", float), ("bool", bool)):
            if alias not in np.__dict__:
                setattr(np, alias, value)

        import calvin_env.envs.play_table_env as play_table_env
        import hydra
        from omegaconf import OmegaConf

        # Avoid shelling out to git for the dataset's commit hash on every environment construction.
        play_table_env.get_git_commit_hash = lambda _path: "unknown"
        if args.renderer == "egl":
            egl_device = egl_device_for_gpu(int(args.egl_device))
            os.environ["EGL_VISIBLE_DEVICES"] = os.environ["EGL_VISIBLE_DEVICE"] = str(egl_device)
            logger.info("EGL rendering on GPU %s (EGL device %d)", args.egl_device, egl_device)
        random.seed(args.seed)
        np.random.seed(args.seed)

        conf_dir = Path(args.calvin_root).expanduser().resolve() / "calvin_models" / "conf"
        self.task_oracle = hydra.utils.instantiate(OmegaConf.load(conf_dir / TASKS_CONFIG))
        self.annotations = OmegaConf.load(conf_dir / ANNOTATIONS_CONFIG)
        validation = Path(args.dataset_path).expanduser().resolve() / "validation"
        if args.renderer == "egl":
            self.env = play_table_env.get_env(validation, obs_space=OBS_SPACE, show_gui=False)
        else:
            # `get_env` ignores a renderer argument; build the same environment with `use_egl` off.
            render_conf = OmegaConf.load(validation / ".hydra" / "merged_config.yaml")
            for camera in set(render_conf.cameras.keys()) - {name.split("_", 1)[1] for name in CAMERAS}:
                del render_conf.cameras[camera]
            render_conf.env.use_egl = False
            if not hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
                hydra.initialize(".")
            self.env = hydra.utils.instantiate(
                render_conf.env, show_gui=False, use_vr=False, use_scene_info=True
            )
        self.client = PolicyClient(args.host, args.port, timeout=args.timeout, connect_timeout=args.timeout)

    def close(self) -> None:
        self.client.close()
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def run_sequence(self, row: dict[str, Any], video_dir: Path | None) -> dict[str, Any]:
        from calvin_agent.evaluation.utils import get_env_state_for_initial_condition

        start = time.perf_counter()
        robot_obs, scene_obs = get_env_state_for_initial_condition(row["initial_state"])
        obs = reset_to_initial_state(self.env, robot_obs, scene_obs)
        frames = [np.asarray(obs["rgb_obs"]["rgb_static"]).copy()] if video_dir is not None else None
        success_len, steps = 0, 0
        for subtask in row["sequence"]:
            obs = self.env.get_obs()
            instruction = str(self.annotations[subtask][0])
            self.client.reset()
            start_info = self.env.get_info()
            succeeded = False
            for _ in range(self.args.ep_len):
                action = self.client.predict(observation_payload(obs, instruction))
                obs, _, _, info = self.env.step(env_action(action))
                steps += 1
                if frames is not None:
                    frames.append(np.asarray(obs["rgb_obs"]["rgb_static"]).copy())
                if self.task_oracle.get_task_info_for_set(start_info, info, {subtask}):
                    succeeded = True
                    break
            if not succeeded:
                break
            success_len += 1
        result = {
            "sequence_index": int(row["sequence_index"]),
            "initial_state": row["initial_state"],
            "sequence": list(row["sequence"]),
            "success_len": success_len,
            "num_steps": steps,
            "wall_s": round(time.perf_counter() - start, 3),
        }
        if frames is not None:
            video = video_dir / f"sequence_{row['sequence_index']:04d}_success_{success_len}.mp4"
            write_video(video, frames, self.args.video_fps)
            result["video"] = video.name
        return result


def parse_indices(text: str) -> list[int]:
    indices: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            indices.extend(range(int(low), int(high) + 1))
        elif part:
            indices.append(int(part))
    return indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--calvin-root", required=True, help="CALVIN repository checkout")
    parser.add_argument("--dataset-path", help="CALVIN split directory holding validation/ (e.g. task_ABC_D)")
    parser.add_argument("--manifest", required=True, help="Sequence manifest JSON")
    parser.add_argument(
        "--write-manifest", type=int, metavar="N", help="Generate N sequences into --manifest and exit"
    )
    parser.add_argument(
        "--indices", help="Sequence indices of this shard, e.g. '0-99' or '0,4,8' (default: all)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="Seconds for connecting and for each request"
    )
    parser.add_argument("--output-dir", help="Shard output directory")
    parser.add_argument("--ep-len", type=int, default=360, help="Step budget per subtask")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--renderer", choices=("egl", "tiny"), default="egl", help="PyBullet renderer")
    parser.add_argument("--egl-device", type=int, default=0, help="Physical GPU (nvidia-smi index) for EGL")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--no-video", action="store_true", help="Do not save rollout videos")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    manifest = Path(args.manifest)
    if args.write_manifest is not None:
        add_calvin_to_path(Path(args.calvin_root))
        write_manifest(manifest, args.write_manifest)
        return
    if not args.output_dir or not args.dataset_path:
        raise SystemExit("--output-dir and --dataset-path are required for evaluation")
    rows = load_manifest(manifest)
    indices = parse_indices(args.indices) if args.indices else list(range(len(rows)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    done = {int(row["sequence_index"]) for row in read_results(results_path)}
    pending = [index for index in indices if index not in done]
    logger.info("%d sequences in shard, %d already done", len(indices), len(indices) - len(pending))
    video_dir = None if args.no_video else output_dir / "videos"
    evaluator = CalvinEvaluator(args)
    try:
        with results_path.open("a", encoding="utf-8") as results:
            for count, index in enumerate(pending, start=1):
                result = evaluator.run_sequence(rows[index], video_dir)
                results.write(json.dumps(result) + "\n")
                results.flush()
                os.fsync(results.fileno())
                logger.info(
                    "sequence %d: %d/5 in %d steps (%.1fs) [%d/%d]",
                    index,
                    result["success_len"],
                    result["num_steps"],
                    result["wall_s"],
                    count,
                    len(pending),
                )
    finally:
        evaluator.close()
    shard_rows = [row for row in read_results(results_path) if int(row["sequence_index"]) in set(indices)]
    summary = summarize(shard_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("avg_seq_len %.3f over %d sequences", summary["avg_seq_len"], summary["num_sequences"])


if __name__ == "__main__":
    main()
