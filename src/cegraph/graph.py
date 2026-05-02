"""Repository graph construction and traversal."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from .index import EmbeddingIndex, LexicalIndex, build_node_document, unique_preserving_order
from .models import Edge, EdgeKind, GraphSummary, Node
from .parser import ParsedModule, PendingEdge, iter_python_files, parse_python_file
from .tokens import TokenCounter


def _resolve_relative_module(source_module: str, reference: str) -> str:
    if not reference.startswith("."):
        return reference
    dot_count = len(reference) - len(reference.lstrip("."))
    suffix = reference[dot_count:]
    source_parts = source_module.split(".") if source_module else []
    base_parts = source_parts[:-dot_count] if dot_count <= len(source_parts) else []
    if suffix:
        base_parts.extend(suffix.split("."))
    return ".".join(part for part in base_parts if part)


@dataclass(slots=True)
class CodebaseGraph:
    repo_root: Path
    nodes: dict[str, Node]
    edges: list[Edge]
    lexical_index: LexicalIndex
    embedding_index: EmbeddingIndex
    token_counter: TokenCounter
    _outgoing: dict[str, list[Edge]] = field(init=False, repr=False, default_factory=dict)
    _incoming: dict[str, list[Edge]] = field(init=False, repr=False, default_factory=dict)
    _file_index: dict[str, list[str]] = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        outgoing: dict[str, list[Edge]] = defaultdict(list)
        incoming: dict[str, list[Edge]] = defaultdict(list)
        for edge in self.edges:
            outgoing[edge.source].append(edge)
            incoming[edge.target].append(edge)
        file_index: dict[str, list[str]] = defaultdict(list)
        for node_id, node in self.nodes.items():
            file_index[node.file_path].append(node_id)
            if node.parent_id and node.parent_id in self.nodes and node.parent_id != node_id:
                outgoing[node_id].append(Edge(source=node_id, target=node.parent_id, kind=EdgeKind.ASSIGNS_TO))
                incoming[node.parent_id].append(Edge(source=node_id, target=node.parent_id, kind=EdgeKind.ASSIGNS_TO))
                outgoing[node.parent_id].append(Edge(source=node.parent_id, target=node_id, kind=EdgeKind.ASSIGNS_TO))
                incoming[node_id].append(Edge(source=node.parent_id, target=node_id, kind=EdgeKind.ASSIGNS_TO))
        for node_ids in file_index.values():
            node_ids.sort(key=lambda identifier: self.nodes[identifier].lineno)
        object.__setattr__(self, '_outgoing', outgoing)
        object.__setattr__(self, '_incoming', incoming)
        object.__setattr__(self, '_file_index', file_index)

    @classmethod
    def from_repo(cls, repo_root: Path) -> "CodebaseGraph":
        parsed_modules = [parse_python_file(repo_root, path) for path in iter_python_files(repo_root)]
        nodes = {node.id: node for parsed in parsed_modules for node in parsed.nodes}
        edges = cls._resolve_edges(parsed_modules, nodes)
        documents = {node_id: build_node_document(node) for node_id, node in nodes.items()}
        lexical_index = LexicalIndex(documents)
        embedding_index = EmbeddingIndex()
        embedding_index.fit(documents)
        return cls(
            repo_root=repo_root,
            nodes=nodes,
            edges=edges,
            lexical_index=lexical_index,
            embedding_index=embedding_index,
            token_counter=TokenCounter(),
        )

    @staticmethod
    def _resolve_edges(parsed_modules: list[ParsedModule], nodes: dict[str, Node]) -> list[Edge]:
        module_nodes = {
            parsed.module_name: next(node.id for node in parsed.nodes if node.kind.value == "module")
            for parsed in parsed_modules
        }
        scoped_symbols: dict[tuple[str, str], str] = {}
        global_symbols: dict[str, list[str]] = defaultdict(list)
        for parsed in parsed_modules:
            for symbol_name, symbol_id in parsed.local_symbols.items():
                scoped_symbols[(parsed.module_name, symbol_name)] = symbol_id
                global_symbols[symbol_name].append(symbol_id)

        def resolve_target(pending: PendingEdge) -> str | None:
            target_ref = pending.target_ref
            if pending.kind == EdgeKind.IMPORTS:
                module_name, _, imported_name = target_ref.partition(":")
                resolved_module = _resolve_relative_module(pending.source_module, module_name)
                if imported_name:
                    return scoped_symbols.get((resolved_module, imported_name))
                return module_nodes.get(resolved_module)

            if "." in target_ref and (pending.source_module, target_ref.split(".")[0]) in scoped_symbols:
                first, _, rest = target_ref.partition(".")
                prefix = scoped_symbols[(pending.source_module, first)]
                candidate = f"{nodes[prefix].qualified_name}.{rest}"
                candidate_id = f"{nodes[prefix].file_path}::{candidate}"
                if candidate_id in nodes:
                    return candidate_id

            local = scoped_symbols.get((pending.source_module, target_ref))
            if local is not None:
                return local
            global_candidates = global_symbols.get(target_ref, [])
            if len(global_candidates) == 1:
                return global_candidates[0]
            return module_nodes.get(target_ref) or target_ref

        resolved_edges: list[Edge] = []
        seen: set[tuple[str, str, str]] = set()
        for parsed in parsed_modules:
            for pending in parsed.pending_edges:
                target_id = resolve_target(pending)
                if target_id is None or target_id == pending.source_id:
                    continue
                signature = (pending.source_id, target_id, pending.kind.value)
                if signature in seen:
                    continue
                seen.add(signature)
                resolved_edges.append(
                    Edge(
                        source=pending.source_id,
                        target=target_id,
                        kind=pending.kind,
                        metadata=pending.metadata,
                    )
                )
        return resolved_edges

    def get_node(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def summary(self) -> GraphSummary:
        node_counts = Counter(node.kind.value for node in self.nodes.values())
        edge_counts = Counter(edge.kind.value for edge in self.edges)
        return GraphSummary(
            node_count=len(self.nodes),
            edge_count=len(self.edges),
            file_count=len(self._file_index),
            node_kind_counts=dict(node_counts),
            edge_kind_counts=dict(edge_counts),
        )

    def graph_neighbors(self, node_id: str, depth: int) -> list[tuple[str, int]]:
        if node_id not in self.nodes:
            return []
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        visited = {node_id}
        ordered: list[tuple[str, int]] = []
        while queue:
            current, hop_count = queue.popleft()
            if hop_count >= depth:
                continue
            adjacent_edges = [*self._outgoing.get(current, []), *self._incoming.get(current, [])]
            for edge in adjacent_edges:
                neighbor = edge.target if edge.source == current else edge.source
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                next_depth = hop_count + 1
                ordered.append((neighbor, next_depth))
                queue.append((neighbor, next_depth))
        return ordered

    def semantic_neighbors(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        return self.embedding_index.semantic_neighbors(query_embedding, k)

    def embed_query(self, query: str) -> list[float]:
        return self.embedding_index.embed_text(query)

    def token_count(self, node_id: str) -> int:
        return self.token_counter.count(self.nodes[node_id].source)

    def node_text(self, node_id: str) -> str:
        return self.nodes[node_id].source

    def node_ids_for_file(self, file_path: str) -> list[str]:
        normalized = file_path.replace("\\", "/")
        return list(self._file_index.get(normalized, []))

    def nearest_node(self, location: str) -> str | None:
        file_path, _, maybe_line = location.partition(":")
        node_ids = self.node_ids_for_file(file_path)
        if not node_ids:
            return None
        if not maybe_line.isdigit():
            return node_ids[0]
        line_number = int(maybe_line)
        ranked = sorted(
            node_ids,
            key=lambda identifier: abs(self.nodes[identifier].lineno - line_number),
        )
        return ranked[0]

    def lexical_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.lexical_index.bm25(query, k)

    def keyword_search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        return self.lexical_index.keyword_overlap(query, k)

    def candidate_closure(self, seeds: list[str], depth: int) -> list[tuple[str, int]]:
        results: list[tuple[str, int]] = []
        for seed in unique_preserving_order(seeds):
            results.append((seed, 0))
            results.extend(self.graph_neighbors(seed, depth))
        ordered = unique_preserving_order(node_id for node_id, _ in results)
        hop_lookup = {node_id: hop for node_id, hop in reversed(results)}
        return [(node_id, hop_lookup[node_id]) for node_id in ordered]

