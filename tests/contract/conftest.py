import inspect
from collections.abc import Callable

import pytest


@pytest.fixture
def assert_signature() -> Callable[[Callable, str, str], None]:
    def check(fn: Callable, expected: str, consequence: str) -> None:
        actual = str(inspect.signature(fn))
        assert actual == expected, (
            f"Signature of {fn.__module__}.{fn.__qualname__} no longer matches the pinned lerobot contract.\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"{consequence}"
        )

    return check
