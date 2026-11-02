import importlib.util
import sys
from pathlib import Path

import pytest

TESTKIT = "concertina_testkit"


def load_testkit():
    if TESTKIT not in sys.modules:
        path = Path(__file__).parents[1] / "data" / "testkit.py"
        spec = importlib.util.spec_from_file_location(TESTKIT, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[TESTKIT] = module
        spec.loader.exec_module(module)
    return sys.modules[TESTKIT]


@pytest.fixture(scope="session")
def testkit():
    return load_testkit()


@pytest.fixture(scope="session")
def shim(tmp_path_factory, testkit):
    """(configuration module, modeling module) of the "concertina_shim" policy type."""
    return testkit.import_shim(tmp_path_factory.mktemp("shim"))


@pytest.fixture(autouse=True)
def reset_accelerate_state():
    """The training loop leaves accelerate's process-wide state behind; do not leak it into other tests."""
    yield
    from accelerate.state import AcceleratorState, GradientState

    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._reset_state()


@pytest.fixture(scope="session", autouse=True)
def dataset_view_cache(tmp_path_factory):
    """Dataset views (`concertina.data.mixture.dataset_view`) go to a temporary cache instead of the user's."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("CONCERTINA_CACHE_DIR", str(tmp_path_factory.mktemp("concertina_cache")))
        yield
