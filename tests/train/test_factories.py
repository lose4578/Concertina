import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from lerobot.configs import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train
from lerobot.transforms import ImageTransformsConfig

from concertina.policy.config_data import ConcertinaDataConfig, DatasetSpec
from concertina.policy.config_model import ConcertinaModelConfig
from concertina.train import entry
from concertina.train.factories import make_dataloaders, make_mixture_sampler, make_train_eval_datasets

HEAD, WRIST_LEFT = "observation.images.head", "observation.images.wrist_left"
SPECS = [
    DatasetSpec(
        path="libero",
        source="libero",
        embodiment="libero",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.image"], WRIST_LEFT: ["observation.images.image2"]},
    ),
    DatasetSpec(
        path="calvin",
        source="calvin",
        embodiment="calvin",
        valid_action_dim=10,
        camera_map={HEAD: ["observation.images.top"], WRIST_LEFT: ["observation.images.wrist"]},
    ),
]
DROP_LAST = 2
# Keys the Concertina model reads from the preprocessed batch.
MODEL_BATCH_KEYS = {
    "index",
    "dataset_index",
    "task",
    "action",
    "action_is_pad",
    "action_gripper_binary",
    "action_valid_dim_mask",
    "state_valid_dim_mask",
    "observation.state",
    HEAD,
    WRIST_LEFT,
}


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory, testkit) -> Path:
    root = tmp_path_factory.mktemp("train_datasets")
    testkit.make_libero(root / "libero")
    testkit.make_calvin(root / "calvin")
    return root


def make_cfg(
    shim, dataset_root: Path, output_dir: Path, *, weights=None, gripper_indices=(9,), **kwargs
) -> TrainPipelineConfig:
    specs = [
        DatasetSpec(**{**spec.__dict__, "weight": w})
        for spec, w in zip(SPECS, weights or [None] * 2, strict=True)
    ]
    policy = shim[0].ConcertinaShimConfig(
        device="cpu",
        model=ConcertinaModelConfig(
            camera_keys=[HEAD, WRIST_LEFT], chunk_size=10, gripper_indices=list(gripper_indices)
        ),
        data=ConcertinaDataConfig(
            datasets=specs,
            normalization="per_embodiment",
            resize_images=True,
            drop_n_last_frames=DROP_LAST,
        ),
    )
    options = {
        "steps": 4,
        "batch_size": 4,
        "num_workers": 0,
        "save_freq": 2,
        "log_freq": 1,
        "seed": 7,
        **kwargs,
    }
    return TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="concertina/test-mixture", root=str(dataset_root)),
        policy=policy,
        output_dir=output_dir,
        **options,
    )


def test_dataset_factory_builds_the_mixture(shim, dataset_root, tmp_path):
    cfg = make_cfg(shim, dataset_root, tmp_path / "run")
    cfg.dataset.image_transforms = ImageTransformsConfig(enable=True)
    dataset, eval_dataset = make_train_eval_datasets(cfg)
    assert eval_dataset is None and dataset.meta.repo_id == "concertina/test-mixture"
    assert dataset.image_transforms is not None
    assert all(child.image_transforms is dataset.image_transforms for child in dataset.children)
    assert dataset[0]["action"].shape == (10, 20)
    assert dataset.meta.camera_keys == [HEAD, WRIST_LEFT]


@pytest.mark.parametrize(
    ("field", "value"),
    [("streaming", True), ("eval_split", 0.1), ("episodes", [0]), ("return_uint8", True)],
)
def test_dataset_factory_rejects_unsupported_dataset_options(shim, dataset_root, tmp_path, field, value):
    cfg = make_cfg(shim, dataset_root, tmp_path / "run")
    object.__setattr__(cfg.dataset, field, value)
    with pytest.raises(ValueError, match=field):
        make_train_eval_datasets(cfg)


def test_factories_require_a_concertina_policy(dataset_root, tmp_path):
    from lerobot.policies.act.configuration_act import ACTConfig

    cfg = TrainPipelineConfig(dataset=DatasetConfig(repo_id="x"), policy=ACTConfig(device="cpu"))
    with pytest.raises(ValueError, match="Concertina"):
        make_train_eval_datasets(cfg)
    with pytest.raises(ValueError, match="Concertina"):
        entry.run(cfg)


def test_sampler_epochs_are_whole_global_batches_and_honour_weights(shim, dataset_root, tmp_path):
    cfg = make_cfg(shim, dataset_root, tmp_path / "run")
    dataset, _ = make_train_eval_datasets(cfg)
    sampler = make_mixture_sampler(dataset, cfg.policy.data, batch_size=4, dp_world_size=3, seed=0)
    valid = [43 - 3 * DROP_LAST, 28 - 2 * DROP_LAST]
    assert sampler.repo_valid_lengths == valid
    assert len(sampler) % 12 == 0 and len(sampler) >= sum(valid)
    torch.testing.assert_close(
        torch.tensor(sampler.repo_probs), torch.tensor(valid, dtype=torch.float64) / sum(valid)
    )
    for index in sampler.indices(0):
        dataset_index, local = dataset.locate(int(index))
        start, end = dataset.episode_bounds[dataset_index]
        episode = int((local >= start).sum()) - 1
        assert local < end[episode] - DROP_LAST

    weighted = make_cfg(shim, dataset_root, tmp_path / "run", weights=[1.0, 3.0])
    dataset, _ = make_train_eval_datasets(weighted)
    sampler = make_mixture_sampler(dataset, weighted.policy.data, batch_size=4, dp_world_size=1, seed=0)
    torch.testing.assert_close(
        torch.tensor(sampler.repo_probs), torch.tensor([0.25, 0.75], dtype=torch.float64)
    )


def test_dataloader_yields_canonical_batches(shim, dataset_root, tmp_path):
    cfg = make_cfg(shim, dataset_root, tmp_path / "run")
    dataset, _ = make_train_eval_datasets(cfg)
    parallel_dims = SimpleNamespace(dp_world_size=1, device_type="cpu")
    loader, eval_loader = make_dataloaders(cfg, dataset, None, 0, parallel_dims)
    assert eval_loader is None and len(loader.sampler) % 4 == 0
    batch = next(iter(loader))
    assert batch["action"].shape == (4, 10, 20) and batch[HEAD].shape == (4, 3, 224, 224)
    expected = loader.sampler.indices(0, 0, 4).tolist()
    located = [dataset.locate(i) for i in expected]
    assert batch["dataset_index"].tolist() == [d for d, _ in located]
    assert batch["index"].tolist() == [i for _, i in located]
    with pytest.raises(ValueError, match="held-out"):
        make_dataloaders(cfg, dataset, dataset, 0, parallel_dims)


def _train(cfg: TrainPipelineConfig, argv: list[str], monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["concertina-train", *argv])
    entry.run(cfg)


def test_upstream_training_resumes_sample_exactly(shim, dataset_root, tmp_path, monkeypatch):
    """Train through the official loop, then resume from a checkpoint across an epoch boundary."""
    seen = shim[1].SEEN
    seen.clear()
    output = tmp_path / "run"
    cfg = make_cfg(shim, dataset_root, output, steps=20, save_freq=10)
    _train(cfg, [], monkeypatch)
    assert len(seen) == 20
    assert cfg.policy.model.rng_seed == 7
    for keys in (record[4] for record in seen):
        assert keys >= MODEL_BATCH_KEYS, sorted(MODEL_BATCH_KEYS - keys)
    uninterrupted = list(seen)
    steps_per_epoch = (
        len(
            make_mixture_sampler(
                make_train_eval_datasets(cfg)[0], cfg.policy.data, batch_size=4, dp_world_size=1, seed=7
            )
        )
        // 4
    )
    assert 10 < steps_per_epoch < 20
    assert lerobot_train.make_train_eval_datasets.__module__ == "lerobot.datasets.factory"

    checkpoint = output / "checkpoints" / "000010" / "pretrained_model"
    assert (checkpoint / "policy_preprocessor.json").is_file()
    seen.clear()
    argv = [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true"]
    monkeypatch.setattr(sys, "argv", ["concertina-train", *argv])
    entry.run(entry.parse_config())
    assert seen == uninterrupted[10:]


def test_training_without_gripper_channels(shim, dataset_root, tmp_path, monkeypatch):
    """The 60k pretraining regressed every dimension: no gripper target reaches the policy."""
    seen = shim[1].SEEN
    seen.clear()
    cfg = make_cfg(shim, dataset_root, tmp_path / "run", gripper_indices=(), steps=3, save_freq=3, seed=None)
    _train(cfg, [], monkeypatch)
    assert len(seen) == 3 and cfg.policy.model.rng_seed == 0
    for keys in (record[4] for record in seen):
        assert MODEL_BATCH_KEYS - {"action_gripper_binary"} <= keys and "action_gripper_binary" not in keys


def test_single_dataset_runs_visit_each_frame_once_per_epoch(shim, dataset_root, tmp_path):
    single = make_cfg(shim, dataset_root, tmp_path / "run")
    single.policy.data.datasets = single.policy.data.datasets[:1]
    dataset, _ = make_train_eval_datasets(single)
    sampler = make_mixture_sampler(dataset, single.policy.data, batch_size=4, dp_world_size=1, seed=5)
    valid = sampler.total_valid_frames
    assert not sampler.replacement and len(sampler) == 4 * -(-valid // 4)
    for epoch in (0, 1):
        order = sampler.indices(epoch)
        assert sorted(order[:valid].tolist()) == sampler.valid_indices.tolist()
        permutation = torch.randperm(valid, generator=torch.Generator().manual_seed(5 + epoch)).numpy()
        assert order[:valid].tolist() == sampler.valid_indices[permutation].tolist()

    mixture = make_cfg(shim, dataset_root, tmp_path / "run", weights=[1.0, 3.0])
    dataset, _ = make_train_eval_datasets(mixture)
    assert make_mixture_sampler(
        dataset, mixture.policy.data, batch_size=4, dp_world_size=1, seed=5
    ).replacement


def test_fine_tuning_restarts_the_training_step_and_resuming_keeps_it(
    shim, dataset_root, tmp_path, monkeypatch
):
    seen = shim[1].SEEN
    seen.clear()
    first = tmp_path / "first"
    _train(make_cfg(shim, dataset_root, first, steps=7, save_freq=7), [], monkeypatch)
    assert [record[-1] for record in seen] == list(range(1, 8))
    checkpoint = first / "checkpoints" / "000007" / "pretrained_model"

    seen.clear()
    argv = [
        f"--policy.path={checkpoint}",
        "--dataset.repo_id=concertina/test-mixture",
        f"--dataset.root={dataset_root}",
        f"--output_dir={tmp_path / 'finetune'}",
        "--steps=2",
        "--batch_size=4",
        "--num_workers=0",
        "--save_freq=2",
        "--seed=7",
    ]
    monkeypatch.setattr(sys, "argv", ["concertina-train", *argv])
    entry.run(entry.parse_config())
    assert [record[-1] for record in seen] == [1, 2]

    seen.clear()
    argv = [f"--config_path={checkpoint / 'train_config.json'}", "--resume=true", "--steps=9"]
    monkeypatch.setattr(sys, "argv", ["concertina-train", *argv])
    entry.run(entry.parse_config())
    assert [record[-1] for record in seen] == [8, 9]


def test_policy_hook_requires_the_dataset_hook_first():
    from concertina.train.factories import TrainingRunHooks

    hooks = TrainingRunHooks(make_policy=lambda cfg, **kwargs: None)
    with pytest.raises(RuntimeError, match="before make_train_eval_datasets"):
        hooks.make_policy(None)
