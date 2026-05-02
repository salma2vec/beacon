"""Budget-aware candidate selection."""

from __future__ import annotations

from .models import CandidateScore


def relevance_score(semantic_similarity: float, hop_count: int, alpha: float = 0.5) -> float:
    graph_score = 1.0 / (hop_count + 1)
    return alpha * semantic_similarity + (1 - alpha) * graph_score


def marginal_gain(candidate: CandidateScore, covered: set[str], alpha: float = 0.5) -> float:
    base = relevance_score(candidate.semantic_similarity, candidate.hop_count, alpha=alpha)
    if not candidate.coverage:
        return base
    overlap = len(candidate.coverage & covered)
    novelty = 1.0 / (1 + overlap)
    return base * novelty


def greedy_budgeted_selection(
    candidates: list[CandidateScore],
    budget: int,
    alpha: float = 0.5,
) -> list[CandidateScore]:
    selected: list[CandidateScore] = []
    remaining = {candidate.node_id: candidate for candidate in candidates}
    covered: set[str] = set()
    spent = 0
    while remaining:
        best_id: str | None = None
        best_ratio = -1.0
        for candidate in remaining.values():
            if spent + candidate.token_cost > budget:
                continue
            gain = marginal_gain(candidate, covered, alpha=alpha)
            ratio = gain / max(candidate.token_cost, 1)
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = candidate.node_id
        if best_id is None:
            break
        choice = remaining.pop(best_id)
        selected.append(choice)
        spent += choice.token_cost
        covered.update(choice.coverage or {choice.node_id})
    return selected
