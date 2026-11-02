"""Install Concertina's dataset, dataloader and policy factories into ``lerobot.scripts.lerobot_train``.

``concertina-train`` builds a ``TrainPipelineConfig``, replaces exactly three module globals that
``train()`` resolves at call time, then calls the official ``train(cfg)``. Nothing here runs at
import time; the replacement is explicit, scoped to the entry point, and verified against the
pinned upstream signatures before anything is touched.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType

OVERRIDE_TARGETS: tuple[str, ...] = ("make_train_eval_datasets", "make_dataloaders", "make_policy")

# str(inspect.signature(fn)) of the pinned upstream functions. Any drift means the override
# contract must be re-audited against the new upstream source.
EXPECTED_SIGNATURES: dict[str, str] = {
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
}


class OverrideContractError(RuntimeError):
    """The pinned lerobot training module no longer matches the contract Concertina relies on."""


def assert_signature(fn: Callable, expected: str, *, name: str) -> None:
    actual = str(inspect.signature(fn))
    if actual != expected:
        raise OverrideContractError(
            f"Signature of {name} does not match the pinned lerobot contract.\n"
            f"  expected: {expected}\n  actual:   {actual}"
        )


def check_module_contract(module: ModuleType, expected: dict[str, str] | None = None) -> None:
    """Verify every function Concertina replaces or calls has the pinned signature."""
    expected = EXPECTED_SIGNATURES if expected is None else expected
    for name, signature in expected.items():
        fn = getattr(module, name, None)
        if fn is None or not callable(fn):
            raise OverrideContractError(f"{module.__name__} has no callable {name!r}")
        assert_signature(fn, signature, name=f"{module.__name__}.{name}")


def _parameter_layout(fn: Callable) -> list[tuple[str, inspect._ParameterKind]]:
    return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]


@dataclass(frozen=True)
class InstalledOverrides:
    """Handle to the replaced globals; ``restore()`` puts the upstream functions back."""

    module: ModuleType
    originals: dict[str, Callable]

    def restore(self) -> None:
        for name, fn in self.originals.items():
            setattr(self.module, name, fn)

    def __enter__(self) -> InstalledOverrides:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.restore()


def install_overrides(
    module: ModuleType,
    *,
    make_train_eval_datasets: Callable,
    make_dataloaders: Callable,
    make_policy: Callable,
    expected: dict[str, str] | None = None,
) -> InstalledOverrides:
    """Replace ``module.make_train_eval_datasets``, ``module.make_dataloaders`` and ``module.make_policy``.

    The module contract is checked first (nothing is modified on failure), and each replacement
    must take the upstream parameters by name and kind so ``train()`` can call it unchanged.
    """
    check_module_contract(module, expected)
    replacements = {
        "make_train_eval_datasets": make_train_eval_datasets,
        "make_dataloaders": make_dataloaders,
        "make_policy": make_policy,
    }
    originals: dict[str, Callable] = {}
    for name, replacement in replacements.items():
        original = getattr(module, name)
        if _parameter_layout(replacement) != _parameter_layout(original):
            raise OverrideContractError(
                f"Replacement for {name} must accept the upstream parameters "
                f"{[p for p, _ in _parameter_layout(original)]}, got "
                f"{[p for p, _ in _parameter_layout(replacement)]}"
            )
        originals[name] = original
    for name, replacement in replacements.items():
        setattr(module, name, replacement)
    return InstalledOverrides(module=module, originals=originals)
