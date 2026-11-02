import inspect
import types

import pytest

from concertina.train import overrides
from concertina.train.overrides import (
    EXPECTED_SIGNATURES,
    OVERRIDE_TARGETS,
    OverrideContractError,
    check_module_contract,
    install_overrides,
)


def _fake_train_module() -> types.ModuleType:
    """A stand-in for lerobot.scripts.lerobot_train with the pinned parameter layout."""
    module = types.ModuleType("fake_lerobot_train")

    def make_train_eval_datasets(cfg):
        return ("upstream_dataset", None)

    def make_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
        return ("upstream_loader", None)

    def make_policy(
        cfg, ds_meta=None, env_cfg=None, rename_map=None, defer_weight_load=False, pretrained_path=None
    ):
        return "upstream_policy"

    def train(cfg):
        dataset, eval_dataset = module.make_train_eval_datasets(cfg)
        policy = module.make_policy(cfg=cfg, ds_meta=dataset)
        loader, _ = module.make_dataloaders(cfg, dataset, eval_dataset, 0, None)
        return dataset, loader, policy

    for fn in (make_train_eval_datasets, make_dataloaders, make_policy, train):
        setattr(module, fn.__name__, fn)
    return module


def _expected_for(module: types.ModuleType) -> dict[str, str]:
    return {name: str(inspect.signature(getattr(module, name))) for name in EXPECTED_SIGNATURES}


def _concertina_datasets(cfg):
    return ("concertina_dataset", None)


def _concertina_dataloaders(cfg, dataset, eval_dataset, step, parallel_dims):
    return (f"concertina_loader[{dataset}]", None)


def _concertina_policy(
    cfg, ds_meta=None, env_cfg=None, rename_map=None, defer_weight_load=False, pretrained_path=None
):
    return f"concertina_policy[{ds_meta}]"


REPLACEMENTS = {
    "make_train_eval_datasets": _concertina_datasets,
    "make_dataloaders": _concertina_dataloaders,
    "make_policy": _concertina_policy,
}


def test_override_targets_are_the_three_sanctioned_functions():
    assert OVERRIDE_TARGETS == ("make_train_eval_datasets", "make_dataloaders", "make_policy")
    assert set(OVERRIDE_TARGETS) <= set(EXPECTED_SIGNATURES)


def test_check_module_contract_detects_missing_and_drifted_functions():
    module = _fake_train_module()
    check_module_contract(module, _expected_for(module))

    drifted = _expected_for(module)
    drifted["make_dataloaders"] = drifted["make_dataloaders"].replace("step", "micro_step")
    with pytest.raises(OverrideContractError, match="make_dataloaders"):
        check_module_contract(module, drifted)

    del module.make_dataloaders
    with pytest.raises(OverrideContractError, match="no callable 'make_dataloaders'"):
        check_module_contract(module, _expected_for(_fake_train_module()))


def test_install_replaces_globals_seen_by_train_and_restores_them():
    module = _fake_train_module()
    originals = {name: getattr(module, name) for name in OVERRIDE_TARGETS}

    handle = install_overrides(
        module,
        **REPLACEMENTS,
        expected=_expected_for(module),
    )
    assert all(getattr(module, name) is REPLACEMENTS[name] for name in OVERRIDE_TARGETS)
    assert module.train(cfg=None) == (
        "concertina_dataset",
        "concertina_loader[concertina_dataset]",
        "concertina_policy[concertina_dataset]",
    )

    handle.restore()
    assert {name: getattr(module, name) for name in OVERRIDE_TARGETS} == originals
    assert module.train(cfg=None) == ("upstream_dataset", "upstream_loader", "upstream_policy")


def test_install_is_a_context_manager():
    module = _fake_train_module()
    with install_overrides(
        module,
        **REPLACEMENTS,
        expected=_expected_for(module),
    ):
        assert module.train(cfg=None)[0] == "concertina_dataset"
    assert module.train(cfg=None)[0] == "upstream_dataset"


def test_install_rejects_replacements_with_a_different_parameter_layout():
    module = _fake_train_module()

    def wrong_dataloaders(cfg, dataset, step, parallel_dims):
        return None

    with pytest.raises(OverrideContractError, match="Replacement for make_dataloaders"):
        install_overrides(
            module,
            **{**REPLACEMENTS, "make_dataloaders": wrong_dataloaders},
            expected=_expected_for(module),
        )
    # Nothing was modified.
    assert module.make_train_eval_datasets is not _concertina_datasets


def test_install_does_not_modify_the_module_when_the_contract_is_broken():
    module = _fake_train_module()
    drifted = _expected_for(module)
    drifted["train"] = "(cfg, extra)"
    with pytest.raises(OverrideContractError):
        install_overrides(
            module,
            **REPLACEMENTS,
            expected=drifted,
        )
    assert module.make_train_eval_datasets is not _concertina_datasets
    assert module.make_dataloaders is not _concertina_dataloaders
    assert module.make_policy is not _concertina_policy


def test_pinned_lerobot_matches_the_recorded_signatures():
    lerobot_train = pytest.importorskip("lerobot.scripts.lerobot_train")
    if not all(hasattr(lerobot_train, name) for name in EXPECTED_SIGNATURES):
        pytest.skip("installed lerobot predates the pinned training entry point")
    check_module_contract(lerobot_train)
    with install_overrides(lerobot_train, **REPLACEMENTS):
        assert lerobot_train.make_dataloaders is _concertina_dataloaders
        assert lerobot_train.make_policy is _concertina_policy
    assert lerobot_train.make_dataloaders is not _concertina_dataloaders
    assert lerobot_train.make_policy is not _concertina_policy


def test_module_has_no_import_time_side_effects():
    source = inspect.getsource(overrides)
    assert "import lerobot" not in source and "from lerobot" not in source
