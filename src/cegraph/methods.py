"""Context assembly methods for beacon."""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Iterable

from .graph import CodebaseGraph
from .models import AssembledContext, CandidateScore, ContextChunk, MethodResult
from .optimize import greedy_budgeted_selection, relevance_score

METHODS = [
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


def available_methods() -> list[str]:
    return list(METHODS)


def _seed_node_ids(graph: CodebaseGraph, bug_location: str | None, query: str) -> list[str]:
    seeds: list[str] = []
    if bug_location:
        nearest = graph.nearest_node(bug_location)
        if nearest:
            seeds.append(nearest)
    seeds.extend(node_id for node_id, _ in graph.lexical_search(query, k=3))
    return list(dict.fromkeys(seeds))


def _candidate_pool(
    graph: CodebaseGraph,
    query: str,
    bug_location: str | None,
    depth: int,
    include_closure: bool,
) -> list[tuple[str, int]]:
    seeds = _seed_node_ids(graph, bug_location, query)
    if not include_closure:
        return [(node_id, 0) for node_id in seeds]
    return graph.candidate_closure(seeds, depth=depth)


def _build_candidates(
    graph: CodebaseGraph,
    node_hops: Iterable[tuple[str, int]],
    query: str,
) -> list[CandidateScore]:
    query_embedding = graph.embed_query(query)
    hop_lookup = {node_id: hop for node_id, hop in node_hops}
    semantic_lookup = defaultdict(
        float, graph.semantic_neighbors(query_embedding, k=len(graph.nodes))
    )
    candidates: list[CandidateScore] = []
    for node_id, hop_count in hop_lookup.items():
        coverage = {node_id, *(neighbor for neighbor, _ in graph.graph_neighbors(node_id, depth=1))}
        candidates.append(
            CandidateScore(
                node_id=node_id,
                token_cost=graph.token_count(node_id),
                semantic_similarity=semantic_lookup.get(node_id, 0.0),
                hop_count=hop_count,
                coverage=frozenset(coverage),
            )
        )
    return candidates


def _chunks_from_selection(
    graph: CodebaseGraph,
    selected: list[CandidateScore],
    budget: int,
    method: str,
    alpha: float,
    render_format: str = "verbose_v1",
    result_metadata: dict[str, object] | None = None,
) -> MethodResult:
    chunks: list[ContextChunk] = []
    total_tokens = 0
    for candidate in selected:
        score = relevance_score(candidate.semantic_similarity, candidate.hop_count, alpha=alpha)
        chunk = ContextChunk(
            node_id=candidate.node_id,
            text=graph.node_text(candidate.node_id),
            token_cost=candidate.token_cost,
            score=score,
            metadata={
                "hop_count": candidate.hop_count,
                "semantic_similarity": candidate.semantic_similarity,
            },
        )
        trial = AssembledContext(
            method=method,
            budget=budget,
            chunks=[*chunks, chunk],
            total_tokens=0,
            metadata={"render_format": render_format},
        )
        rendered_cost = graph.token_counter.count(trial.render())
        if rendered_cost > budget:
            continue
        chunks.append(chunk)
        total_tokens = rendered_cost
    context = AssembledContext(
        method=method,
        budget=budget,
        chunks=chunks,
        total_tokens=total_tokens,
        metadata={"render_format": render_format},
    )
    return MethodResult(
        method=method,
        assembled_context=context,
        selected_node_ids=[chunk.node_id for chunk in chunks],
        metadata=dict(result_metadata or {}),
    )


def _exact_query_anchors(graph: CodebaseGraph, query: str) -> list[str]:
    """Find symbols named verbatim in the report, without task-label access."""
    identifiers = list(dict.fromkeys(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)))
    by_name: dict[str, list[str]] = defaultdict(list)
    for node_id, node in graph.nodes.items():
        if node.kind.value != "module":
            by_name[node.basename].append(node_id)
    return [node_id for name in identifiers for node_id in by_name.get(name, [])]


def assemble_context(
    graph: CodebaseGraph,
    query: str,
    method: str,
    budget: int,
    bug_location: str | None = None,
    alpha: float = 0.5,
) -> MethodResult:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if method not in METHODS:
        available = ", ".join(METHODS)
        raise ValueError(f"Unknown method '{method}'. Available methods: {available}")

    if method == "no_retrieval":
        context = AssembledContext(method=method, budget=budget, chunks=[], total_tokens=0)
        return MethodResult(method=method, assembled_context=context, selected_node_ids=[])

    selected: list[CandidateScore]
    if method == "target_file":
        node_ids = graph.node_ids_for_file((bug_location or "").partition(":")[0])
        selected = []
        spent = 0
        for node_id in node_ids:
            token_cost = graph.token_count(node_id)
            if spent + token_cost > budget:
                break
            selected.append(
                CandidateScore(
                    node_id=node_id,
                    token_cost=token_cost,
                    semantic_similarity=1.0,
                    hop_count=0,
                    coverage=frozenset({node_id}),
                )
            )
            spent += token_cost
        return _chunks_from_selection(graph, selected, budget, method, alpha)

    if method in {
        "bm25",
        "compact_bm25",
        "grounded_bm25",
        "keyword_map",
        "vector",
        "embedding",
    }:
        if method in {"bm25", "compact_bm25", "grounded_bm25"}:
            ranked = graph.lexical_search(query, k=len(graph.nodes))
        elif method == "keyword_map":
            ranked = graph.keyword_search(query, k=len(graph.nodes))
        else:
            ranked = graph.semantic_neighbors(graph.embed_query(query), k=len(graph.nodes))
        # Baselines preserve their own ranking; graph-aware optimization is BCA-only.
        anchors: list[str] = []
        if method == "grounded_bm25":
            anchors = _exact_query_anchors(graph, query)
            anchor_ids = set(anchors)
            score_lookup = dict(ranked)
            ranked = [
                *((node_id, score_lookup.get(node_id, 0.0)) for node_id in anchors),
                *((node_id, score) for node_id, score in ranked if node_id not in anchor_ids),
            ]
        candidates = [
            CandidateScore(node_id, graph.token_count(node_id), score, 0)
            for node_id, score in ranked
        ]
        compact = method in {"compact_bm25", "grounded_bm25"}
        if method == "grounded_bm25":
            selection_policy = "exact_query_symbol_then_bm25"
        elif method == "compact_bm25":
            selection_policy = "bm25"
        else:
            selection_policy = method
        return _chunks_from_selection(
            graph,
            candidates,
            budget,
            method,
            alpha,
            render_format="recoverable_v1" if compact else "verbose_v1",
            result_metadata={
                "selection_policy": selection_policy,
                "query_anchor_node_ids": anchors,
            },
        )

    if method == "bca_no_scoring":
        pool = _candidate_pool(graph, query, bug_location, depth=2, include_closure=True)
        rng = random.Random(7)
        shuffled = list(pool)
        rng.shuffle(shuffled)
        selected = []
        spent = 0
        for node_id, hop_count in shuffled:
            token_cost = graph.token_count(node_id)
            if spent + token_cost > budget:
                continue
            selected.append(
                CandidateScore(
                    node_id=node_id,
                    token_cost=token_cost,
                    semantic_similarity=0.0,
                    hop_count=hop_count,
                    coverage=frozenset({node_id}),
                )
            )
            spent += token_cost
        return _chunks_from_selection(graph, selected, budget, method, alpha)

    depth = 1 if method == "bca_d1" else 5 if method == "bca_d5" else 2
    include_closure = method != "bca_no_closure"
    pool = _candidate_pool(graph, query, bug_location, depth=depth, include_closure=include_closure)
    candidates = _build_candidates(graph, pool, query)
    selected = greedy_budgeted_selection(candidates, budget, alpha=alpha)
    return _chunks_from_selection(graph, selected, budget, method, alpha)
