import sys
from pathlib import Path

import pytest
import yaml
from lerobot.configs import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train

from concertina.policy.configuration_concertina import ConcertinaConfig
from concertina.train import entry


def write(path, config) -> str:
    path.write_text(yaml.safe_dump(config))
    return str(path)


def test_a_policy_path_must_not_silently_replace_the_configured_policy(tmp_path, monkeypatch):
    config = write(
        tmp_path / "train.yaml", {"steps": 2, "policy": {"type": "concertina", "dtype": "float32"}}
    )
    monkeypatch.setattr(
        sys, "argv", ["concertina-train", f"--config_path={config}", "--policy.path=/checkpoint"]
    )
    with pytest.raises(ValueError, match="would silently replace the `policy:` block"):
        entry.main()


@pytest.mark.parametrize(
    "policy",
    [
        {"path": "/checkpoint", "dtype": "float32"},  # overrides of the checkpoint configuration
        None,
    ],
)
def test_policy_paths_inside_the_file_or_without_a_policy_block_are_accepted(tmp_path, policy):
    config = {"steps": 2} if policy is None else {"steps": 2, "policy": policy}
    path = write(tmp_path / "train.yaml", config)
    entry.check_policy_arguments([f"--config_path={path}", "--policy.path=/other"])
    entry.check_policy_arguments(
        [f"--config_path={write(tmp_path / 'full.yaml', {'policy': {'type': 'concertina'}})}"]
    )


PLACEHOLDER = "/path/to/pretrain_60k/checkpoints/last/pretrained_model"


@pytest.fixture
def started(monkeypatch) -> list[TrainPipelineConfig]:
    """Configurations that reached the upstream training loop (which would build the datasets first)."""
    runs: list[TrainPipelineConfig] = []

    def train(cfg: TrainPipelineConfig) -> None:  # same signature: `concertina-train` checks it
        runs.append(cfg)

    monkeypatch.setattr(lerobot_train, "train", train)
    return runs


def config(pretrained_path, *, resume: bool = False) -> TrainPipelineConfig:
    policy = ConcertinaConfig(
        device="cpu", pretrained_path=None if pretrained_path is None else Path(pretrained_path)
    )
    return TrainPipelineConfig(dataset=DatasetConfig(repo_id="concertina/test"), policy=policy, resume=resume)


def test_an_unusable_pretrained_path_fails_before_the_datasets_are_built(started):
    with pytest.raises(ValueError, match="neither a local directory nor a Hub repository id"):
        entry.run(config(PLACEHOLDER))
    assert started == []


@pytest.mark.parametrize("kind", ["directory", "hub_id", "none", "resume"])
def test_usable_pretrained_paths_and_resumed_runs_start_training(kind, tmp_path, started):
    cfg = {
        "directory": lambda: config(tmp_path),
        "hub_id": lambda: config("Traly/concertina-pretrain-90k"),
        "none": lambda: config(None),
        "resume": lambda: config(PLACEHOLDER, resume=True),
    }[kind]()
    entry.run(cfg)
    assert started == [cfg]
