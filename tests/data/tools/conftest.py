import importlib.util
import sys
from pathlib import Path

import pytest

SOURCES = "concertina_tools_sources"


def load_sources():
    if SOURCES not in sys.modules:
        spec = importlib.util.spec_from_file_location(SOURCES, Path(__file__).with_name("sources.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[SOURCES] = module
        spec.loader.exec_module(module)
    return sys.modules[SOURCES]


@pytest.fixture
def sources():
    return load_sources()


@pytest.fixture
def calvin_root(tmp_path, sources) -> Path:
    return sources.make_calvin_root(tmp_path / "raw")
