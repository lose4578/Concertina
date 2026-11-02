import ast
import dis
import inspect
import sys
from dataclasses import dataclass

import draccus
import pytest
from lerobot.configs import DatasetConfig, parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts import lerobot_train

from concertina.policy.configuration_concertina import ConcertinaConfig

OVERRIDES_CONSEQUENCE = (
    "concertina/train/overrides.py replaces or calls this function; re-audit it against the new upstream "
    "source before updating the expected signature here."
)

EXPECTED_SIGNATURES = {
    "make_train_eval_datasets": (
        "(cfg: lerobot.configs.train.TrainPipelineConfig) -> "
        "tuple[lerobot.datasets.lerobot_dataset.LeRobotDataset | "
        "lerobot.datasets.streaming_dataset.StreamingLeRobotDataset, "
        "lerobot.datasets.lerobot_dataset.LeRobotDataset | None]"
    ),
    "make_dataloaders": (
        "(cfg: lerobot.configs.train.TrainPipelineConfig, "
        "dataset: lerobot.datasets.lerobot_dataset.LeRobotDataset | "
        "lerobot.datasets.streaming_dataset.StreamingLeRobotDataset, "
        "eval_dataset: lerobot.datasets.lerobot_dataset.LeRobotDataset | None, "
        "step: int, "
        "parallel_dims: lerobot.distributed.parallel_dims.ParallelDims) -> "
        "tuple[torch.utils.data.dataloader.DataLoader, torch.utils.data.dataloader.DataLoader | None]"
    ),
    "make_policy": (
        "(cfg: 'PreTrainedConfig', ds_meta: 'LeRobotDatasetMetadata | None' = None, "
        "env_cfg: 'EnvConfig | None' = None, rename_map: 'dict[str, str] | None' = None, "
        "defer_weight_load: 'bool' = False, pretrained_path: 'str | Path | None' = None) -> 'PreTrainedPolicy'"
    ),
    "train": "(cfg: lerobot.configs.train.TrainPipelineConfig) -> None",
    "update_policy": (
        "(train_metrics: lerobot.utils.logging_utils.MetricsTracker, "
        "policy: lerobot.policies.pretrained.PreTrainedPolicy, "
        "batch: Any, "
        "optimizer: torch.optim.optimizer.Optimizer, "
        "grad_clip_norm: float, "
        "accelerator: 'Accelerator', "
        "lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None, "
        "lock: contextlib.AbstractContextManager[typing.Any] | None = None, "
        "sample_weighter: lerobot.utils.sample_weighting.SampleWeighter | None = None) -> "
        "tuple[lerobot.utils.logging_utils.MetricsTracker, dict | None]"
    ),
}

# Direct attribute reads on the dataset objects returned by make_train_eval_datasets, per function of
# lerobot_train that concertina-train does not replace. `image_transforms` and `meta.depth_keys` are only read
# when `cfg.dataset.image_transforms` enables the "gpu" backend.
EXPECTED_DATASET_READS = {
    "_make_image_augmentation": {
        "dataset.image_transforms",
        "dataset.meta",
        "dataset.meta.camera_keys",
        "dataset.meta.depth_keys",
    },
    "train": {
        "dataset.meta",
        "dataset.meta.camera_keys",
        "dataset.meta.stats",
        "dataset.num_episodes",
        "dataset.num_frames",
    },
}
EXPECTED_DATASET_HANDOFFS = {
    ("train", "_make_image_augmentation", "dataset"),
    ("train", "make_dataloaders", "dataset"),
    ("train", "make_dataloaders", "eval_dataset"),
    ("train", "make_policy", "dataset.meta"),
    ("train", "make_reward_model", "dataset.meta"),
    ("train", "publish_trained_model", "dataset.meta"),
    ("train", "unwrapped.push_model_to_hub", "dataset.meta"),
}
REPLACED_FUNCTIONS = {"make_dataloaders"}
DATASET_NAMES = {"dataset", "eval_dataset"}


def _attribute_chain(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _dataset_surface() -> tuple[dict[str, set[str]], set[tuple[str, str, str]]]:
    tree = ast.parse(inspect.getsource(lerobot_train))
    reads: dict[str, set[str]] = {}
    handoffs: set[tuple[str, str, str]] = set()
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name in REPLACED_FUNCTIONS:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute):
                chain = _attribute_chain(node)
                if chain and chain.split(".")[0] in DATASET_NAMES:
                    reads.setdefault(fn.name, set()).add(chain)
            if isinstance(node, ast.Call):
                for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                    chain = _attribute_chain(arg)
                    if chain in {*DATASET_NAMES, *(f"{n}.meta" for n in DATASET_NAMES)}:
                        handoffs.add((fn.name, ast.unparse(node.func), chain))
    return reads, handoffs


def _global_loads(fn) -> set[str]:
    return {ins.argval for ins in dis.get_instructions(fn) if ins.opname == "LOAD_GLOBAL"}


def _nominal_train_config(config_cls: type[TrainPipelineConfig] = TrainPipelineConfig, **kwargs):
    return config_cls(
        dataset=DatasetConfig(repo_id="concertina/nominal"),
        policy=ConcertinaConfig(device="cpu", push_to_hub=False),
        **kwargs,
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_SIGNATURES))
def test_train_entrypoint_signatures(name, assert_signature):
    assert_signature(getattr(lerobot_train, name), EXPECTED_SIGNATURES[name], OVERRIDES_CONSEQUENCE)


def test_train_resolves_replaced_functions_as_module_globals():
    loads = _global_loads(inspect.unwrap(lerobot_train.train))
    missing = {"make_train_eval_datasets", "make_dataloaders", "make_policy", "update_policy"} - loads
    assert not missing, (
        f"lerobot_train.train no longer looks up {sorted(missing)} as module globals, so assigning "
        "lerobot.scripts.lerobot_train.<name> in concertina/train/overrides.py would have no effect."
    )


def test_train_is_wrapped_by_parser_wrap():
    train = lerobot_train.train
    assert train.__code__.co_filename == inspect.getsourcefile(parser), (
        "lerobot_train.train is no longer the parser.wrap wrapper; concertina-train calls it with a config "
        "instance and relies on parser.wrap passing that instance through."
    )
    spec = inspect.getfullargspec(inspect.unwrap(train))
    assert spec.annotations[spec.args[0]] is TrainPipelineConfig, (
        "parser.wrap compares type(args[0]) against the evaluated annotation of train's first argument; "
        f"it is {spec.annotations[spec.args[0]]!r}, not the TrainPipelineConfig class."
    )


def test_parser_wrap_passes_through_exact_train_pipeline_config_only(monkeypatch):
    received = []

    @parser.wrap()
    def probe(cfg: TrainPipelineConfig) -> None:
        received.append(cfg)

    cfg = _nominal_train_config()
    probe(cfg)
    assert received == [cfg] and received[0] is cfg, "parser.wrap did not pass the config instance through"

    @dataclass
    class TrainPipelineConfigSubclass(TrainPipelineConfig):
        pass

    class CliParseReachedError(Exception):
        pass

    def fail_parse(*args, **kwargs):
        raise CliParseReachedError

    monkeypatch.setattr(sys, "argv", ["concertina-train"])
    monkeypatch.setattr(draccus, "parse", fail_parse)
    with pytest.raises(CliParseReachedError):
        probe(_nominal_train_config(TrainPipelineConfigSubclass))
    assert len(received) == 1, (
        "parser.wrap passed a TrainPipelineConfig subclass instance through; the strict "
        "`type(args[0]) is argtype` check this repository works around has changed."
    )


def test_nominal_train_config_with_concertina_policy_validates(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["concertina-train"])
    cfg = _nominal_train_config(output_dir=tmp_path / "run")
    cfg.validate()
    assert cfg.trainable_config is cfg.policy
    assert cfg.job_name == "concertina"


def test_train_dataset_surface_is_unchanged():
    reads, handoffs = _dataset_surface()
    assert reads == EXPECTED_DATASET_READS and handoffs == EXPECTED_DATASET_HANDOFFS, (
        "The attributes lerobot_train reads from (or hands off on) the dataset returned by "
        "make_train_eval_datasets changed. The Concertina mixture dataset must provide every one of them.\n"
        f"  reads expected: {EXPECTED_DATASET_READS}\n  reads actual:   {reads}\n"
        f"  handoffs added:   {sorted(handoffs - EXPECTED_DATASET_HANDOFFS)}\n"
        f"  handoffs removed: {sorted(EXPECTED_DATASET_HANDOFFS - handoffs)}"
    )


def test_train_builds_the_dataset_before_the_policy():
    tree = ast.parse(inspect.getsource(inspect.unwrap(lerobot_train.train)))
    first_call = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            first_call.setdefault(node.func.id, node.lineno)
            first_call[node.func.id] = min(first_call[node.func.id], node.lineno)
    assert first_call["make_train_eval_datasets"] < first_call["make_policy"], (
        "lerobot_train.train now creates the policy before the dataset; concertina/train/factories.py "
        "TrainingRunHooks records the resume flag in make_train_eval_datasets for make_policy."
    )
