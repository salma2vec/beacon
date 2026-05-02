from __future__ import annotations

from cegraph.native import bfs_neighbors, has_native_backend


def test_native_fallback_bfs() -> None:
    adjacency = {"a": ["b", "c"], "b": ["d"], "c": [], "d": []}
    assert bfs_neighbors(adjacency, "a", depth=2) == [("b", 1), ("c", 1), ("d", 2)]
    assert has_native_backend() is False

