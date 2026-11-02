"""`concertina-train`: the official LeRobot training loop with the Concertina mixture dataset and sampler.

Arguments are those of `lerobot-train` (for example ``--config_path=configs/train/<name>.yaml`` or
``--config_path=<checkpoint>/pretrained_model/train_config.json --resume=true``). The configuration is parsed
into an exact `TrainPipelineConfig`; the dataset, dataloader and policy factories of
`lerobot.scripts.lerobot_train` are replaced for the duration of the run (see
`concertina.train.factories.TrainingRunHooks`) and the upstream `train(cfg)` is called. Its `validate()` re-reads
``--policy.path`` and ``--config_path`` from ``sys.argv`` as usual.

A policy checkpoint given with ``--policy.path`` replaces the configuration file's ``policy:`` block, unless that
block has its own ``path``; `concertina-train` refuses such a dropped block instead of ignoring it.
"""

import json
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from huggingface_hub.errors import HFValidationError
from huggingface_hub.utils import validate_repo_id
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train

from concertina.policy.configuration_concertina import ConcertinaConfig
from concertina.train.factories import TrainingRunHooks, make_dataloaders
from concertina.train.overrides import install_overrides


@parser.wrap()
def parse_config(cfg: TrainPipelineConfig) -> TrainPipelineConfig:
    return cfg


def run(cfg: TrainPipelineConfig) -> None:
    if cfg.job.is_remote:
        raise ValueError(
            "concertina-train runs locally; a remote job would run plain lerobot-train without Concertina data"
        )
    if cfg.reward_model is not None:
        raise ValueError("concertina-train trains the Concertina policy, not a reward model")
    if cfg.policy is not None and not isinstance(cfg.policy, ConcertinaConfig):
        raise ValueError(f"concertina-train trains the Concertina policy, got {type(cfg.policy).__name__}")
    check_pretrained_path(cfg)
    hooks = TrainingRunHooks(make_policy=lerobot_train.make_policy)
    with install_overrides(
        lerobot_train,
        make_train_eval_datasets=hooks.make_train_eval_datasets,
        make_dataloaders=make_dataloaders,
        make_policy=hooks.make_policy,
    ):
        lerobot_train.train(cfg)


def check_pretrained_path(cfg: TrainPipelineConfig) -> None:
    """Fail before the datasets are built when a fresh run's ``policy.pretrained_path`` cannot name a checkpoint.

    Only the form is checked (a local directory or a valid Hub repository id), without network access. Resumed runs
    are skipped: their configuration may name a parent checkpoint that is not on this machine.
    """
    if cfg.resume or cfg.policy is None or cfg.policy.pretrained_path is None:
        return
    path = str(cfg.policy.pretrained_path)
    if Path(path).is_dir():
        return
    try:
        validate_repo_id(path)
    except HFValidationError as error:
        raise ValueError(
            f"policy.pretrained_path={path} is neither a local directory nor a Hub repository id; pass "
            "--policy.pretrained_path=<dir> with the checkpoint's pretrained_model directory"
        ) from error


def check_policy_arguments(argv: Sequence[str]) -> None:
    """Reject ``--policy.path`` together with a configuration file whose ``policy:`` block it would discard."""
    policy_path = parser.parse_arg("policy.path", argv)
    config_path = parser.parse_arg("config_path", argv)
    if policy_path is None or config_path is None:
        return
    path = Path(config_path)
    if path.suffix.lower() in (".yaml", ".yml"):
        config = yaml.safe_load(path.read_text())
    elif path.suffix.lower() == ".json":
        config = json.loads(path.read_text())
    else:
        return
    policy = config.get("policy") if isinstance(config, dict) else None
    if isinstance(policy, dict) and policy and "path" not in policy:
        raise ValueError(
            f"--policy.path={policy_path} loads the policy configuration from that checkpoint and would silently "
            f"replace the `policy:` block of {config_path}. Put `path: {policy_path}` inside that block (its other "
            "fields then override the checkpoint's configuration), or drop the block and pass overrides as "
            "--policy.<field>=<value>."
        )


def main() -> None:
    check_policy_arguments(sys.argv[1:])
    run(parse_config())


if __name__ == "__main__":
    main()
