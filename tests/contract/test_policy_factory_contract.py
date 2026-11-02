import importlib
import sys
import textwrap
from types import SimpleNamespace

import pytest
from lerobot.configs import PreTrainedConfig
from lerobot.distributed import set_fsdp_wrap_modules
from lerobot.policies import PreTrainedPolicy, factory as policy_factory

FACTORY_CONSEQUENCE = (
    "concertina/policy/ is resolved through this lerobot.policies.factory convention; re-audit "
    "concertina/policy/{modeling,processor}_concertina.py before updating the expected value here."
)

EXPECTED_SIBLING_IMPORT = "(config_cls: 'type[PreTrainedConfig]', prefix: 'str') -> 'ModuleType | None'"
EXPECTED_FRESH_FACTORY = (
    "(config: 'PreTrainedConfig', dataset_stats: 'dict[str, dict[str, torch.Tensor]] | None' = None, "
    "dataset_meta: 'Any | None' = None) -> 'tuple[Any, Any]'"
)
EXPECTED_PRETRAINED_FACTORY = (
    "(config: 'PreTrainedConfig', pretrained_path: 'str | Path', *, revision: 'str | None', "
    "dataset_stats: 'dict[str, dict[str, torch.Tensor]] | None', dataset_meta: 'Any | None', "
    "preprocessor_overrides: 'dict[str, Any] | None', postprocessor_overrides: 'dict[str, Any] | None', "
    "preprocessor_config_filename: 'str', postprocessor_config_filename: 'str') -> 'tuple[Any, Any] | None'"
)
EXPECTED_SET_FSDP_WRAP_MODULES = "(accelerator: 'Accelerator', policy: 'PreTrainedPolicy') -> None"

PROBE_PACKAGE = "concertina_contract_probe_policy"
PROBE_TYPE = "concertina_contract_probe"
PROBE_MODULES = {
    "__init__.py": "",
    "configuration_probe.py": f"""
        from dataclasses import dataclass

        from lerobot.configs import PreTrainedConfig
        from lerobot.optim import AdamWConfig


        @PreTrainedConfig.register_subclass("{PROBE_TYPE}")
        @dataclass
        class ProbeConfig(PreTrainedConfig):
            @property
            def observation_delta_indices(self):
                return None

            @property
            def action_delta_indices(self):
                return None

            @property
            def reward_delta_indices(self):
                return None

            def get_optimizer_preset(self):
                return AdamWConfig()

            def get_scheduler_preset(self):
                return None

            def validate_features(self):
                pass
        """,
    "modeling_probe.py": f"""
        from lerobot.policies import PreTrainedPolicy

        from .configuration_probe import ProbeConfig


        class ProbePolicy(PreTrainedPolicy):
            config_class = ProbeConfig
            name = "{PROBE_TYPE}"
        """,
    "processor_probe.py": f"""
        CALLS = []


        def make_{PROBE_TYPE}_pre_post_processors(config, dataset_stats=None, dataset_meta=None):
            CALLS.append(("fresh", dict(config=config, dataset_stats=dataset_stats, dataset_meta=dataset_meta)))
            return "fresh_pre", "fresh_post"


        def make_{PROBE_TYPE}_pre_post_processors_from_pretrained(**kwargs):
            CALLS.append(("pretrained", kwargs))
            return "pretrained_pre", "pretrained_post"
        """,
}


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    root = tmp_path_factory.mktemp("probe_policy")
    package_dir = root / PROBE_PACKAGE
    package_dir.mkdir()
    for filename, source in PROBE_MODULES.items():
        (package_dir / filename).write_text(textwrap.dedent(source))
    sys.path.insert(0, str(root))
    try:
        configuration = importlib.import_module(f"{PROBE_PACKAGE}.configuration_probe")
        processor = importlib.import_module(f"{PROBE_PACKAGE}.processor_probe")
        yield SimpleNamespace(config=configuration.ProbeConfig(device="cpu"), calls=processor.CALLS)
    finally:
        sys.path.remove(str(root))


def test_factory_hook_signatures(assert_signature):
    assert_signature(
        policy_factory._import_sibling_policy_module, EXPECTED_SIBLING_IMPORT, FACTORY_CONSEQUENCE
    )
    assert_signature(
        policy_factory._make_processors_from_policy_config, EXPECTED_FRESH_FACTORY, FACTORY_CONSEQUENCE
    )
    assert_signature(
        policy_factory._make_pretrained_processors_from_policy_config,
        EXPECTED_PRETRAINED_FACTORY,
        FACTORY_CONSEQUENCE,
    )


def test_policy_class_resolves_from_sibling_modeling_module(probe):
    policy_cls = policy_factory.get_policy_class(PROBE_TYPE)
    assert policy_cls.__module__ == f"{PROBE_PACKAGE}.modeling_probe", FACTORY_CONSEQUENCE
    assert policy_cls.__name__ == "ProbePolicy", FACTORY_CONSEQUENCE


def test_fresh_processors_resolve_by_naming_convention(probe):
    probe.calls.clear()
    stats, meta = {"action": {}}, object()
    processors = policy_factory.make_pre_post_processors(probe.config, dataset_stats=stats, dataset_meta=meta)
    assert processors == ("fresh_pre", "fresh_post"), (
        f"make_pre_post_processors did not dispatch to make_{PROBE_TYPE}_pre_post_processors. "
        f"{FACTORY_CONSEQUENCE}"
    )
    assert probe.calls == [("fresh", {"config": probe.config, "dataset_stats": stats, "dataset_meta": meta})]


def test_pretrained_processors_resolve_through_from_pretrained_hook(probe):
    probe.calls.clear()
    processors = policy_factory.make_pre_post_processors(
        probe.config, pretrained_path="/nonexistent/pretrained_model", pretrained_revision="rev"
    )
    assert processors == ("pretrained_pre", "pretrained_post"), (
        f"make_pre_post_processors(pretrained_path=...) did not dispatch to "
        f"make_{PROBE_TYPE}_pre_post_processors_from_pretrained. {FACTORY_CONSEQUENCE}"
    )
    assert probe.calls == [
        (
            "pretrained",
            {
                "config": probe.config,
                "pretrained_path": "/nonexistent/pretrained_model",
                "revision": "rev",
                "dataset_stats": None,
                "dataset_meta": None,
                "preprocessor_overrides": None,
                "postprocessor_overrides": None,
                "preprocessor_config_filename": "policy_preprocessor.json",
                "postprocessor_config_filename": "policy_postprocessor.json",
            },
        )
    ], f"The keyword arguments passed to the _from_pretrained hook changed. {FACTORY_CONSEQUENCE}"


def test_concertina_resolves_to_its_own_sibling_modules():
    config_cls = PreTrainedConfig.get_choice_class("concertina")
    module_path = config_cls.__module__
    assert module_path.replace("configuration_", "modeling_") == "concertina.policy.modeling_concertina"
    assert module_path.replace("configuration_", "processor_") == "concertina.policy.processor_concertina"


def test_fsdp_wrap_units_come_from_policy_class_attribute(assert_signature):
    assert_signature(set_fsdp_wrap_modules, EXPECTED_SET_FSDP_WRAP_MODULES, FACTORY_CONSEQUENCE)
    assert PreTrainedPolicy._fsdp_wrap_modules is None

    class DeclaresWrapUnits:
        _fsdp_wrap_modules = ["DecoderLayer", "VisionBlock"]

    plugin = SimpleNamespace(min_num_params=0, transformer_cls_names_to_wrap=None)
    accelerator = SimpleNamespace(state=SimpleNamespace(fsdp_plugin=plugin))
    set_fsdp_wrap_modules(accelerator, DeclaresWrapUnits())
    assert plugin.transformer_cls_names_to_wrap == ["DecoderLayer", "VisionBlock"], (
        "set_fsdp_wrap_modules no longer reads the policy class attribute `_fsdp_wrap_modules`. "
        f"{FACTORY_CONSEQUENCE}"
    )
