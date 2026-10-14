from __future__ import annotations

from pathlib import Path

from cegraph import CodebaseGraph, assemble_context, available_methods


def build_repo(tmp_path: Path) -> Path:
    (tmp_path / "app.py").write_text(
        """
def helper():
    return 1

def broken():
    return helper()
""".strip(),
        encoding="utf-8",
    )
    return tmp_path


def test_method_registry_is_complete() -> None:
    assert available_methods() == [
        "no_retrieval",
        "bm25",
        "vector",
        "embedding",
        "keyword_map",
        "bca_d1",
        "bca",
        "bca_d5",
        "bca_no_closure",
        "bca_no_scoring",
        "target_file",
        "compact_bm25",
        "grounded_bm25",
    ]


def test_budget_enforcement_and_empty_method(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(build_repo(tmp_path))
    empty = assemble_context(
        graph, "fix broken", "no_retrieval", budget=50, bug_location="app.py:4"
    )
    assert empty.assembled_context.total_tokens == 0
    assert empty.selected_node_ids == []

    result = assemble_context(graph, "fix broken", "bca", budget=20, bug_location="app.py:4")
    assert result.assembled_context.total_tokens <= 20


def test_target_file_and_keyword_map_return_context(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(build_repo(tmp_path))
    target = assemble_context(graph, "helper", "target_file", budget=500, bug_location="app.py:1")
    keyword = assemble_context(graph, "helper", "keyword_map", budget=500, bug_location="app.py:1")
    assert target.selected_node_ids
    assert keyword.selected_node_ids


def test_grounded_bm25_reserves_named_symbol_under_tight_budget(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(build_repo(tmp_path))
    baseline = assemble_context(graph, "broken regression", "compact_bm25", budget=15)
    grounded = assemble_context(graph, "broken regression", "grounded_bm25", budget=15)
    assert grounded.selected_node_ids == ["app.py::broken"]
    assert grounded.context.render().startswith("# app.py::broken\n")
    assert graph.token_counter.count(grounded.context.render()) == grounded.context.total_tokens
    assert grounded.context.total_tokens <= 15
    assert grounded.metadata["query_anchor_node_ids"] == ["app.py::broken"]
    assert baseline.context.total_tokens <= 15


def test_grounded_bm25_preserves_query_order_for_multiple_symbol_names(tmp_path: Path) -> None:
    graph = CodebaseGraph.from_repo(build_repo(tmp_path))
    grounded = assemble_context(graph, "broken then helper", "grounded_bm25", budget=15)
    assert grounded.selected_node_ids == ["app.py::broken"]
