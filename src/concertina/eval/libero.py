"""In-process LIBERO evaluation with LeRobot's environments and rollout loop.

Each task is evaluated with LeRobot's `run_one` (the per-task loop of `lerobot-eval`: LIBERO's
init states, 10 settle steps, suite-dependent step limit, seeds ``seed + episode``) and written to
``tasks/<suite>_<task_id>.json`` as soon as it finishes, so an interrupted run resumes after the last finished
task. Parallel workers (`concertina-eval` with ``benchmark: libero``) share the run directory, each evaluating a
shard. Unlike `lerobot-eval`, every episode is rendered by default and the decoding recipe can be replaced.

    MUJOCO_GL=egl python -m concertina.eval.libero --config_path configs/eval/libero.yaml \
        --policy_path <policy dir> --output_dir <run dir>
"""

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import draccus
import torch
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
from lerobot.envs.configs import EnvConfig, LiberoEnv
from lerobot.scripts.lerobot_eval import run_one
from lerobot.utils.eval_stats import success_summary
from lerobot.utils.random_utils import set_seed

from concertina.eval.server import load_policy, resolve_policy_path
from concertina.policy.config_model import ConcertinaInferenceConfig

logger = logging.getLogger(__name__)


@dataclass
class LiberoEvalConfig:
    policy_path: str = ""
    output_dir: str = ""
    # `env.task` lists the suites (comma separated); `tasks` restricts them to "<suite>/<task_id>" entries.
    env: EnvConfig = field(default_factory=LiberoEnv)
    tasks: list[str] = field(default_factory=list)
    n_episodes: int = 50
    batch_size: int = 1
    use_async_envs: bool = False
    seed: int = 1000
    device: str = "cuda"
    # Decoding recipe replacing the checkpoint's saved one; None keeps the checkpoint recipe.
    inference: ConcertinaInferenceConfig | None = None
    # Episodes rendered per task; -1 renders all of them.
    max_episodes_rendered: int = -1
    allow_tf32: bool = True
    # Evaluate every `num_shards`-th task starting at `shard_index` (parallel workers sharing `output_dir`).
    shard_index: int = 0
    num_shards: int = 1


def _suite_tasks(env: EnvConfig) -> list[tuple[str, int]]:
    from libero.libero import benchmark

    suites = benchmark.get_benchmark_dict()
    pairs = []
    for suite in (name.strip() for name in env.task.split(",") if name.strip()):
        ids = env.task_ids if env.task_ids is not None else range(suites[suite]().n_tasks)
        pairs += [(suite, int(task_id)) for task_id in ids]
    return pairs


def _selected_tasks(config: LiberoEvalConfig) -> list[tuple[str, int]]:
    if not config.tasks:
        return _suite_tasks(config.env)
    pairs = []
    for entry in config.tasks:
        suite, _, task_id = entry.partition("/")
        pairs.append((suite, int(task_id)))
    return pairs


def _task_file(output_dir: Path, suite: str, task_id: int) -> Path:
    return output_dir / "tasks" / f"{suite}_{task_id}.json"


def merge_tasks(output_dir: Path) -> dict[str, Any]:
    """Success rates over every task listed in ``tasks.json``; all of them must have a result."""
    expected = json.loads((output_dir / "tasks.json").read_text())
    missing = [
        f"{suite}/{task_id}"
        for suite, task_id in expected
        if not _task_file(output_dir, suite, task_id).is_file()
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} LIBERO tasks have no result, e.g. {missing[:5]}")
    by_suite: dict[str, list[bool]] = {}
    for suite, task_id in expected:
        result = json.loads(_task_file(output_dir, suite, task_id).read_text())
        by_suite.setdefault(suite, []).extend(result["successes"])
    every = [success for successes in by_suite.values() for success in successes]
    return {
        "overall": success_summary(every),
        "per_suite": {suite: success_summary(successes) for suite, successes in by_suite.items()},
        "num_tasks": len(expected),
    }


def run(config: LiberoEvalConfig) -> dict[str, Any]:
    if not config.policy_path or not config.output_dir:
        raise ValueError("policy_path and output_dir are required")
    if not 0 <= config.shard_index < config.num_shards:
        raise ValueError(f"shard_index must be in [0, {config.num_shards}), got {config.shard_index}")
    output_dir = Path(config.output_dir)
    (output_dir / "tasks").mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"eval_config_shard{config.shard_index}.yaml", "w") as file:
        draccus.dump(config, file)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    all_tasks = _selected_tasks(config)
    if config.shard_index == 0:
        (output_dir / "tasks.json").write_text(json.dumps(all_tasks) + "\n")
    tasks = [
        task
        for task in all_tasks[config.shard_index :: config.num_shards]
        if not _task_file(output_dir, *task).is_file()
    ]
    logger.info(
        "%d tasks in shard %d/%d still to evaluate", len(tasks), config.shard_index, config.num_shards
    )
    rendered = config.n_episodes if config.max_episodes_rendered < 0 else config.max_episodes_rendered
    policy, preprocessor, postprocessor = load_policy(config.policy_path, config.device, config.inference)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(config.env, policy.config)

    for suite, task_id in tasks:
        path = _task_file(output_dir, suite, task_id)
        set_seed(config.seed)
        start = time.perf_counter()
        env_config = dataclasses.replace(config.env, task=suite, task_ids=[task_id])
        envs = make_env(env_config, n_envs=config.batch_size, use_async_envs=config.use_async_envs)
        env = envs[suite][task_id]
        names, descriptions = list(env.get_attr("task")), list(env.get_attr("task_description"))
        try:
            with torch.no_grad():
                _, _, metrics = run_one(
                    suite,
                    task_id,
                    env,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=config.n_episodes,
                    max_episodes_rendered=rendered,
                    videos_dir=output_dir / "videos" if rendered > 0 else None,
                    return_episode_data=False,
                    start_seed=config.seed,
                )
        finally:
            close_envs(envs)
        result = {
            "suite": suite,
            "task_id": task_id,
            "task": names[0],
            "instruction": descriptions[0],
            "successes": [bool(success) for success in metrics["successes"]],
            "max_rewards": [float(reward) for reward in metrics["max_rewards"]],
            "video_paths": [
                str(Path(video).relative_to(output_dir)) for video in metrics.get("video_paths", [])
            ],
            "wall_s": round(time.perf_counter() - start, 1),
        }
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(result, indent=2) + "\n")
        os.replace(partial, path)
        logger.info("%s/%d: %d/%d", suite, task_id, sum(result["successes"]), len(result["successes"]))

    if config.num_shards > 1:
        return {}
    summary = {"policy_path": resolve_policy_path(config.policy_path), **merge_tasks(output_dir)}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@draccus.wrap()
def main(config: LiberoEvalConfig) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True
    )
    summary = run(config)
    if summary:
        overall = summary["overall"]
        logger.info("Success rate %.1f%% over %d episodes", overall["pc_success"], overall["n_episodes"])


if __name__ == "__main__":
    main()
