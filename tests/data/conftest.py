import importlib.util
import sys
from pathlib import Path

import pytest

TESTKIT = "concertina_testkit"


def load_testkit():
    if TESTKIT not in sys.modules:
        spec = importlib.util.spec_from_file_location(TESTKIT, Path(__file__).with_name("testkit.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[TESTKIT] = module
        spec.loader.exec_module(module)
    return sys.modules[TESTKIT]


@pytest.fixture(scope="session")
def testkit():
    return load_testkit()


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory, testkit) -> Path:
    """Synthetic datasets of every source under one root (read-only for tests)."""
    root = tmp_path_factory.mktemp("lerobot_datasets")
    for name in ("libero", "calvin", "droid", "agibot"):
        getattr(testkit, f"make_{name}")(root / name)
    return root


@pytest.fixture(scope="session")
def shim_config_cls(tmp_path_factory, testkit):
    configuration, _ = testkit.import_shim(tmp_path_factory.mktemp("shim"))
    return configuration.ConcertinaShimConfig


@pytest.fixture(scope="session", autouse=True)
def dataset_view_cache(tmp_path_factory):
    """Dataset views (`concertina.data.mixture.dataset_view`) go to a temporary cache instead of the user's."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CONCERTINA_CACHE_DIR", str(tmp_path_factory.mktemp("concertina_cache")))
        yield
