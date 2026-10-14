"""Atomic report publication without opening an existing destination entry.

The caller must trust the output directory and its ancestors against concurrent
replacement. This protects preplanted destination symlinks/hardlinks; it does not
isolate writers sharing an attacker-controlled directory.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO


@contextmanager
def atomic_output(destination: Path) -> Iterator[BinaryIO]:
    """Write through an exclusively created sibling, then replace the leaf entry."""
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomic(destination: Path, text: str) -> None:
    with atomic_output(destination) as handle:
        handle.write(text.encode("utf-8"))


def figure_filename(query_type: str) -> str:
    """Keep legitimate labels unchanged while requiring one bounded filename part."""
    if (
        not query_type
        or query_type in {".", ".."}
        or any(
            character in "/\\" or ord(character) < 32 or ord(character) == 127
            for character in query_type
        )
        or len(query_type.encode("utf-8")) > 180
    ):
        raise ValueError(
            "query_type must be a nonempty filename label without separators or controls"
        )
    return f"figure_{query_type}.png"
