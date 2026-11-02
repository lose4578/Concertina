"""Synthetic dataset fixtures; `synthetic.py` is loaded by path since tests use ``--import-mode=importlib``."""

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE = "concertina_quality_synthetic"


def load_synthetic():
    if MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE, Path(__file__).with_name("synthetic.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        spec.loader.exec_module(module)
    return sys.modules[MODULE]


@pytest.fixture(scope="session")
def synthetic():
    return load_synthetic()


@pytest.fixture(scope="session")
def droid_like(tmp_path_factory, synthetic) -> Path:
    return synthetic.make_droid_like(tmp_path_factory.mktemp("datasets") / "droid_like")


@pytest.fixture(scope="session")
def agibot_like(tmp_path_factory, synthetic) -> Path:
    return synthetic.make_agibot_like(tmp_path_factory.mktemp("datasets") / "agibot_like")
