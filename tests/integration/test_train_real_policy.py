"""concertina-train through the official LeRobot loop with the real `ConcertinaPolicy`, processors and mixture
dataset.

The tiny randomly initialized Qwen3-VL backbone keeps these runs on CPU; the synthetic datasets come from the data
testkit. Unlike the shim-policy tests in `tests/train`, these catch contract breaks between the data and model lines.
"""

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from lerobot.configs import DatasetConfig, PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from safetensors.torch import load_file
from torch.utils.data import default_collate

from concertina.policy.config_data import ConcertinaDataConfig, DatasetSpec
from concertina.policy.config_model import (
    ConcertinaActionHistoryConfig,
    ConcertinaInferenceConfig,
    ConcertinaModelConfig,
)
from concertina.policy.configuration_concertina import ConcertinaConfig
from concertina.policy.modeling_concertina import ConcertinaPolicy
from concertina.train import entry
from concertina.train.factories import make_train_eval_datasets

HEAD, WRIST_LEFT, WRIST_RIGHT = (
    "observation.images.head",
    "observation.images.wrist_left",
    "observation.images.wrist_right",
)
CHUNK = 10
CALVIN = DatasetSpec(
    path="calvin",
    source="calvin",
    embodiment="calvin",
    valid_action_dim=10,
    camera_map={HEAD: ["observation.images.top"], WRIST_LEFT: ["observation.images.wrist"]},
)
AGIBOT = DatasetSpec(
    path="agibot",
    source="agibot",
    embodiment="agibot",
    camera_map={
        HEAD: ["observation.images.head"],
        WRIST_LEFT: ["observation.images.hand_left"],
        WRIST_RIGHT: ["observation.images.hand_right"],
    },
)


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory, testkit) -> Path:
    root = tmp_path_factory.mktemp("integration_datasets")
    testkit.make_calvin(root / "calvin")
    testkit.make_agibot(root / "agibot")
    return root


@pytest.fixture
def records(monkeypatch) -> list[tuple]:
    """(training_step, dataset_index, index, loss, action_is_pad) of every training forward pass."""
    seen: list[tuple] = []
    forward = ConcertinaPolicy.forward

    def recording_forward(self, batch):
        step = int(self.model.training_step)
        padding = batch["action_is_pad"].tolist()
        loss, metrics = forward(self, batch)
        seen.append((step, batch["dataset_index"].tolist(), batch["index"].tolist(), loss.item(), padding))
        return loss, metrics

    monkeypatch.setattr(ConcertinaPolicy, "forward", recording_forward)
    return seen


def make_cfg(
    backbone: str,
    root: Path,
    output_dir: Path,
    specs,
    cameras,
    normalization,
    action_history: ConcertinaActionHistoryConfig | None = None,
    dtype: torch.dtype | None = None,
    **kwargs,
):
    policy = ConcertinaConfig(
        device="cpu",
        dtype=dtype,
        model=ConcertinaModelConfig(
            backbone_path=backbone,
            training_attn_implementation="sdpa",
            gradient_checkpointing=False,
            use_liger_kernels=False,
            camera_keys=list(cameras),
            chunk_size=CHUNK,
            inference=ConcertinaInferenceConfig(n_action_steps=CHUNK),
            action_history=action_history or ConcertinaActionHistoryConfig(),
        ),
        data=ConcertinaDataConfig(
            datasets=list(specs), normalization=normalization, resize_images=True, drop_n_last_frames=2
        ),
    )
    options = {"batch_size": 2, "num_workers": 0, "log_freq": 1, "seed": 7, **kwargs}
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="concertina/integration", root=str(root)),
        policy=policy,
        output_dir=output_dir,
        **options,
    )


def run(argv: list[str], monkeypatch, cfg: TrainPipelineConfig | None = None) -> None:
    monkeypatch.setattr(sys, "argv", ["concertina-train", *argv])
    entry.run(cfg if cfg is not None else entry.parse_config())


@pytest.mark.parametrize("dtype", [None, torch.float32], ids=["bf16-storage", "fp32-storage"])
def test_training_saves_resumes_fine_tunes_and_reloads(
    dtype, tiny_backbone, dataset_root, tmp_path, records, monkeypatch
):
    output = tmp_path / "run"
    cfg = make_cfg(
        tiny_backbone,
        dataset_root,
        output,
        [CALVIN],
        [HEAD, WRIST_LEFT],
        "global",
        dtype=dtype,
        steps=4,
        save_freq=2,
    )
    run([], monkeypatch, cfg)
    assert [record[0] for record in records] == [0, 1, 2, 3]
    assert all(torch.isfinite(torch.tensor(record[3])) for record in records)
    uninterrupted = list(records)
    checkpoints = output / "checkpoints"
    final = checkpoints / "000004" / "pretrained_model"
    final_weights = load_file(final / "model.safetensors")
    stored = {tensor.dtype for tensor in final_weights.values() if tensor.is_floating_point()}
    assert stored == {dtype or torch.bfloat16}

    records.clear()
    middle = checkpoints / "000002" / "pretrained_model"
    run([f"--config_path={middle / 'train_config.json'}", "--resume=true"], monkeypatch)
    assert records == uninterrupted[2:]
    resumed_weights = load_file(final / "model.safetensors")
    assert resumed_weights.keys() == final_weights.keys()
    for key, tensor in final_weights.items():
        assert torch.equal(resumed_weights[key], tensor), key

    records.clear()
    fine_tune = [
        f"--policy.path={final}",
        "--dataset.repo_id=concertina/integration",
        f"--dataset.root={dataset_root}",
        f"--output_dir={tmp_path / 'finetune'}",
        "--steps=1",
        "--batch_size=2",
        "--num_workers=0",
        "--save_freq=1",
        "--seed=7",
    ]
    run(fine_tune, monkeypatch)
    assert [record[0] for record in records] == [0]

    policy = ConcertinaPolicy.from_pretrained(final)
    assert int(policy.model.training_step) == 4
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=final)
    dataset, _ = make_train_eval_datasets(cfg)
    batch = preprocessor(default_collate(dataset.__getitems__([0, 7])))
    action = postprocessor(policy.select_action(batch))
    assert action.shape == (2, 7) and torch.isfinite(action).all()
    assert set(action[:, -1].tolist()) <= {-1.0, 1.0}


@pytest.mark.parametrize(
    "history",
    [
        ConcertinaActionHistoryConfig(enabled=True, horizon=3, mode="prefix"),
        ConcertinaActionHistoryConfig(
            enabled=True, horizon=10, mode="native", missing_history="random_noise"
        ),
    ],
    ids=["prefix", "native-random-noise"],
)
def test_training_with_action_history_resumes_exactly(
    history, tiny_backbone, dataset_root, tmp_path, records, monkeypatch
):
    """The dataset packs [history, chunk] windows; the model splits them and conditions on the history."""
    output = tmp_path / "run"
    cfg = make_cfg(
        tiny_backbone,
        dataset_root,
        output,
        [CALVIN],
        [HEAD, WRIST_LEFT],
        "global",
        action_history=history,
        steps=4,
        save_freq=2,
        batch_size=4,
    )
    assert cfg.policy.action_delta_indices == list(range(-history.horizon, CHUNK))
    run([], monkeypatch, cfg)
    assert len(records) == 4
    assert all(torch.isfinite(torch.tensor(record[3])) for record in records)
    windows = [window for record in records for window in record[4]]
    assert {len(window) for window in windows} == {history.horizon + CHUNK}
    # Windows of early frames reach before the episode start: their history is padding.
    assert any(any(window[: history.horizon]) for window in windows)
    uninterrupted = list(records)

    records.clear()
    middle = output / "checkpoints" / "000002" / "pretrained_model"
    run([f"--config_path={middle / 'train_config.json'}", "--resume=true"], monkeypatch)
    assert records == uninterrupted[2:]


def test_training_on_a_per_embodiment_mixture(tiny_backbone, dataset_root, tmp_path, records, monkeypatch):
    """Single- and dual-arm sources together: placeholder wrist camera, per-embodiment statistics."""
    cfg = make_cfg(
        tiny_backbone,
        dataset_root,
        tmp_path / "run",
        [CALVIN, AGIBOT],
        [HEAD, WRIST_LEFT, WRIST_RIGHT],
        "per_embodiment",
        steps=3,
        save_freq=3,
        batch_size=4,
    )
    run([], monkeypatch, cfg)
    assert len(records) == 3
    assert {index for record in records for index in record[1]} == {0, 1}
    assert all(torch.isfinite(torch.tensor(record[3])) for record in records)


def export_without_backbone_weights(checkpoint: Path, backbone: str, destination: Path) -> Path:
    """A released policy: weights in `model.safetensors`, a weightless `backbone/`, empty `backbone_path`."""
    shutil.copytree(checkpoint, destination)
    shutil.copytree(
        backbone, destination / "backbone", ignore=shutil.ignore_patterns("*.safetensors*", "*.bin")
    )
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    config["model"]["backbone_path"] = ""
    config_path.write_text(json.dumps(config, indent=4))
    return destination


def test_resuming_a_fine_tune_of_a_released_policy_builds_without_backbone_weights(
    tiny_backbone, dataset_root, tmp_path, records, monkeypatch
):
    """`train()` builds the policy with `make_policy(defer_weight_load=True)` when resuming a DCP checkpoint.

    Sharded checkpoints need a multi-GPU run, so this reproduces that construction directly.
    """
    cfg = make_cfg(
        tiny_backbone,
        dataset_root,
        tmp_path / "base",
        [CALVIN],
        [HEAD, WRIST_LEFT],
        "global",
        steps=1,
        save_freq=1,
    )
    run([], monkeypatch, cfg)
    released = export_without_backbone_weights(
        tmp_path / "base" / "checkpoints" / "000001" / "pretrained_model",
        tiny_backbone,
        tmp_path / "released",
    )
    fine_tune = [
        f"--policy.path={released}",
        "--dataset.repo_id=concertina/integration",
        f"--dataset.root={dataset_root}",
        f"--output_dir={tmp_path / 'finetune'}",
        "--steps=1",
        "--batch_size=2",
        "--num_workers=0",
        "--save_freq=1",
        "--seed=7",
    ]
    run(fine_tune, monkeypatch)
    checkpoint = tmp_path / "finetune" / "checkpoints" / "000001" / "pretrained_model"

    policy_cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy_cfg.pretrained_path = released
    meta = make_train_eval_datasets(cfg)[0].meta
    policy = make_policy(policy_cfg, ds_meta=meta, defer_weight_load=True, pretrained_path=checkpoint)
    policy.load_state_dict(load_file(checkpoint / "model.safetensors"))
    expected = ConcertinaPolicy.from_pretrained(checkpoint).state_dict()
    actual = policy.state_dict()
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor), key
