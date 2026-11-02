import importlib.util
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).parents[1]
TESTKIT = "concertina_testkit"


def _load(name: str, path: Path):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


# The tiny-backbone fixture lives with the model tests; load that conftest as a plain module and re-export it.
tiny_backbone = _load("concertina_model_conftest", TESTS / "model" / "conftest.py").tiny_backbone


@pytest.fixture(scope="session")
def testkit():
    return _load(TESTKIT, TESTS / "data" / "testkit.py")


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
