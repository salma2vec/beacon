"""Lexical and semantic indexing for code nodes."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from hashlib import sha256

from .models import Node, RankedNode

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


class LexicalIndex:
    def __init__(self, documents: dict[str, str]) -> None:
        self.documents = documents
        self.tokens = {node_id: tokenize(text) for node_id, text in documents.items()}
        self.doc_freqs: dict[str, int] = defaultdict(int)
        self.doc_lengths = {node_id: len(tokens) for node_id, tokens in self.tokens.items()}
        self.avg_doc_length = (
            sum(self.doc_lengths.values()) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )
        self.term_freqs = {node_id: Counter(tokens) for node_id, tokens in self.tokens.items()}
        for terms in self.tokens.values():
            for term in set(terms):
                self.doc_freqs[term] += 1

    def bm25(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms:
            return []
        scores: dict[str, float] = defaultdict(float)
        doc_count = max(len(self.documents), 1)
        for term in terms:
            doc_freq = self.doc_freqs.get(term, 0)
            inverse_doc = math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
            for node_id, frequencies in self.term_freqs.items():
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                length = self.doc_lengths.get(node_id, 0)
                denom = frequency + 1.5 * (1 - 0.75 + 0.75 * length / (self.avg_doc_length or 1))
                scores[node_id] += inverse_doc * (frequency * 2.5) / denom
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

    def keyword_overlap(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = Counter(tokenize(query))
        if not terms:
            return []
        scores: dict[str, float] = {}
        for node_id, frequencies in self.term_freqs.items():
            overlap = sum(min(freq, terms[term]) for term, freq in frequencies.items())
            if overlap:
                scores[node_id] = overlap / max(len(self.tokens[node_id]), 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]


class EmbeddingIndex:
    def __init__(self, dimensions: int = 96) -> None:
        self.dimensions = dimensions
        self._embeddings: dict[str, list[float]] = {}

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = sha256(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:12], 16) % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def fit(self, documents: dict[str, str]) -> None:
        self._embeddings = {node_id: self.embed_text(text) for node_id, text in documents.items()}

    def get(self, node_id: str) -> list[float]:
        return self._embeddings[node_id]

    def semantic_neighbors(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        scores = [
            (node_id, cosine_similarity(query_embedding, embedding))
            for node_id, embedding in self._embeddings.items()
        ]
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores[:k]


def build_node_document(node: Node) -> str:
    doc_parts = [
        node.qualified_name,
        node.kind.value,
        node.docstring or "",
        node.source,
    ]
    return "\n".join(part for part in doc_parts if part)


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


class CodebaseIndex:
    def __init__(self, graph: object) -> None:
        self.graph = graph

    @classmethod
    def from_graph(cls, graph: object) -> CodebaseIndex:
        return cls(graph)

    def search(self, query: str, k: int = 10) -> list[RankedNode]:
        if hasattr(self.graph, "lexical_search"):
            results = self.graph.lexical_search(query, k)
        else:
            results = []
        ranked: list[RankedNode] = []
        query_lower = query.lower()
        for node_id, score in results:
            node = self.graph.nodes[node_id]  # type: ignore[attr-defined]
            node_name = getattr(node, "basename", getattr(node, "qualified_name", node_id)).lower()
            bonus = 0.0
            if query_lower == node_name:
                bonus = 10.0
            elif query_lower in node_name:
                bonus = 5.0
            ranked.append(
                RankedNode(
                    node_id=node_id,
                    score=float(score) + bonus,
                    text=self.graph.node_text(node_id),  # type: ignore[attr-defined]
                    token_count=self.graph.token_count(node_id),  # type: ignore[attr-defined]
                    kind=node.kind.value if hasattr(node.kind, "value") else str(node.kind),
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.token_count, item.node_id))
        return ranked

    def semantic_neighbors(self, query_embedding: list[float], k: int) -> list[RankedNode]:
        results = self.graph.semantic_neighbors(query_embedding, k)  # type: ignore[attr-defined]
        ranked: list[RankedNode] = []
        for node_id, score in results:
            node = self.graph.nodes[node_id]  # type: ignore[attr-defined]
            ranked.append(
                RankedNode(
                    node_id=node_id,
                    score=score,
                    text=self.graph.node_text(node_id),  # type: ignore[attr-defined]
                    token_count=self.graph.token_count(node_id),  # type: ignore[attr-defined]
                    kind=node.kind.value if hasattr(node.kind, "value") else str(node.kind),
                )
            )
        return ranked
