from __future__ import annotations

from pathlib import Path

from cegraph.graph import CodebaseGraph


def write_repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "helpers.py").write_text(
        """
class Base:
    pass

def traced(fn):
    return fn

def helper():
    return 1
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "pkg" / "main.py").write_text(
        """
from pkg.helpers import Base, helper, traced

@traced
class Child(Base):
    @traced
    def run(self) -> int:
        value = helper()
        return value
""".strip(),
        encoding="utf-8",
    )
    return tmp_path


def test_graph_builds_nodes_and_edges(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(write_repo(tmp_path))
    summary = graph.summary()

    assert summary.node_count >= 7
    assert summary.node_kind_counts["class"] >= 2
    assert summary.edge_kind_counts["imports"] >= 1
    assert summary.edge_kind_counts["calls"] >= 1
    assert summary.edge_kind_counts["inherits_from"] >= 1


def test_graph_neighbors_and_nearest_node(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(write_repo(tmp_path))
    node_id = graph.nearest_node("pkg/main.py:6")
    assert node_id is not None
    neighbors = graph.graph_neighbors(node_id, depth=2)
    assert neighbors
    assert all(hop_count >= 1 for _, hop_count in neighbors)
