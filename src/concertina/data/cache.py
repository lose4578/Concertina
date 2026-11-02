"""Per-user cache of derived dataset files (``$CONCERTINA_CACHE_DIR``, default ``$XDG_CACHE_HOME/concertina`` or
``~/.cache/concertina``).

Datasets are never written to: views of their metadata and statistics computed from them live here.
"""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def cache_dir(*parts: str) -> Path:
    base = (
        os.environ.get("CONCERTINA_CACHE_DIR")
        or Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "concertina"
    )
    return Path(base).joinpath(*parts)


@contextmanager
def file_lock(path: Path) -> Iterator[None]:
    """Exclusive lock between processes on one machine (e.g. the ranks of a training job)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
