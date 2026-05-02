from __future__ import annotations

from pathlib import Path

from cegraph.graph import CodebaseGraph


def test_lexical_and_semantic_search(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        """
def alpha_processor():
    return "alpha"

def beta_worker():
    return "beta"
""".strip(),
        encoding="utf-8",
    )
    graph = CodebaseGraph.from_repo(tmp_path)
    lexical = graph.lexical_search("alpha processor", k=2)
    assert lexical
    semantic = graph.semantic_neighbors(graph.embed_query("beta worker"), k=2)
    assert semantic
    assert semantic[0][1] >= semantic[-1][1]
