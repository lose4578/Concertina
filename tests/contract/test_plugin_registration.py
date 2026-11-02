import subprocess
import sys
import textwrap
from importlib import metadata

from lerobot.configs import PreTrainedConfig

DIST_NAME = "lerobot_policy_concertina"

FRESH_INTERPRETER_PROBE = textwrap.dedent(
    """
    import sys

    import lerobot.datasets.factory as dataset_factory
    import lerobot.scripts.lerobot_train as lerobot_train
    from lerobot.configs import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    upstream = {
        name: getattr(lerobot_train, name)
        for name in ("make_train_eval_datasets", "make_dataloaders", "train", "update_policy")
    }
    assert "concertina" not in PreTrainedConfig.get_known_choices(), "'concertina' registered before plugin discovery"

    register_third_party_plugins()

    assert "lerobot_policy_concertina" in sys.modules, (
        "register_third_party_plugins() did not import lerobot_policy_concertina (see the logged traceback)"
    )
    for name, fn in upstream.items():
        assert getattr(lerobot_train, name) is fn, f"plugin import replaced lerobot_train.{name}"
    assert lerobot_train.make_train_eval_datasets is dataset_factory.make_train_eval_datasets
    assert "concertina.train" not in sys.modules, "plugin import pulled in concertina.train (training overrides)"
    config_cls = PreTrainedConfig.get_choice_class("concertina")
    print(f"{config_cls.__module__}.{config_cls.__qualname__}")
    """
)


def test_distribution_name_matches_plugin_discovery_prefix():
    name = metadata.distribution(DIST_NAME).metadata["Name"]
    assert name == DIST_NAME, (
        f"Installed distribution metadata Name is {name!r}, expected {DIST_NAME!r}. "
        "lerobot.utils.import_utils.register_third_party_plugins matches the raw metadata Name against "
        "the 'lerobot_policy_' prefix and imports it as a module, so a build backend that normalizes "
        "underscores to dashes silently disables the plugin."
    )


def test_register_third_party_plugins_registers_concertina_without_side_effects():
    result = subprocess.run(
        [sys.executable, "-I", "-c", FRESH_INTERPRETER_PROBE],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, (
        "Plugin discovery contract failed in a fresh interpreter.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    resolved = result.stdout.strip().splitlines()[-1]
    assert resolved == "concertina.policy.configuration_concertina.ConcertinaConfig", (
        f"PreTrainedConfig.get_choice_class('concertina') resolved to {resolved!r}"
    )


def test_concertina_config_follows_factory_naming_convention():
    config_cls = PreTrainedConfig.get_choice_class("concertina")
    assert config_cls.__name__ == "ConcertinaConfig", (
        "lerobot.policies.factory.get_policy_class derives the policy class name by stripping 'Config' "
        f"from the config class name; got {config_cls.__name__!r}"
    )
    assert config_cls.__module__ == "concertina.policy.configuration_concertina", (
        "lerobot.policies.factory resolves sibling modules by replacing 'configuration_' in the config "
        f"module path with 'modeling_' / 'processor_'; got {config_cls.__module__!r}"
    )
    config = config_cls(device="cpu")
    assert config.type == "concertina"
