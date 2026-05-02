"""Typed data models for beacon."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class NodeKind(str, Enum):
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


class EdgeKind(str, Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    ASSIGNS_TO = "assigns_to"
    INHERITS_FROM = "inherits_from"
    DECORATES = "decorates"


@dataclass(slots=True, frozen=True)
class Node:
    id: str
    file_path: str
    qualified_name: str
    kind: NodeKind
    source: str
    lineno: int
    end_lineno: int
    parent_id: str | None = None
    docstring: str | None = None

    @property
    def location(self) -> str:
        return f"{self.file_path}:{self.lineno}"

    @property
    def basename(self) -> str:
        return self.qualified_name.rsplit(".", maxsplit=1)[-1]


@dataclass(slots=True, frozen=True)
class Edge:
    source: str
    target: str
    kind: EdgeKind
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class GraphSummary:
    node_count: int
    edge_count: int
    file_count: int
    node_kind_counts: dict[str, int]
    edge_kind_counts: dict[str, int]


@dataclass(slots=True, frozen=True)
class TaskRecord:
    repo: str
    commit: str
    bug_location: str
    mutation_type: str
    original_line: str
    mutated_line: str
    token_budget: int
    query_type: str = "exact"
    prompt: str | None = None
    test_command: str | None = None
    prompt_metadata: dict[str, Any] = field(default_factory=dict)
    test_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ContextChunk:
    node_id: str
    text: str
    token_cost: int
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.token_cost


@dataclass(slots=True, frozen=True)
class AssembledContext:
    method: str
    budget: int
    chunks: list[ContextChunk]
    total_tokens: int
    task: TaskRecord | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return self.total_tokens

    def render(self) -> str:
        parts: list[str] = []
        for chunk in self.chunks:
            header = f"# {chunk.node_id} ({chunk.token_cost} tokens, score={chunk.score:.3f})"
            parts.append(f"{header}\n{chunk.text}".strip())
        return "\n\n".join(parts)


@dataclass(slots=True, frozen=True)
class CandidateScore:
    node_id: str
    token_cost: int
    semantic_similarity: float
    hop_count: int
    coverage: frozenset[str] = frozenset()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class MethodResult:
    method: str
    assembled_context: AssembledContext
    selected_node_ids: list[str]
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task(self) -> TaskRecord | None:
        return self.assembled_context.task

    @property
    def context(self) -> AssembledContext:
        return self.assembled_context


@dataclass(slots=True, frozen=True)
class RepairPrompt:
    system: str
    user: str


@dataclass(slots=True, frozen=True)
class RepairResponse:
    patch_text: str
    raw_response: str
    model: str


@dataclass(slots=True, frozen=True)
class EvaluationResult:
    compiled: bool
    tests_passed: bool
    pass_at_1: bool
    exit_code: int
    stdout: str
    stderr: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BenchmarkResult:
    repo: str
    commit: str
    task_id: str
    method: str
    budget: int
    query_type: str
    pass_at_1: bool
    compiled: bool
    tests_passed: bool
    context_tokens: int
    selected_node_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "repo": self.repo,
            "commit": self.commit,
            "task_id": self.task_id,
            "method": self.method,
            "budget": self.budget,
            "query_type": self.query_type,
            "pass_at_1": self.pass_at_1,
            "compiled": self.compiled,
            "tests_passed": self.tests_passed,
            "context_tokens": self.context_tokens,
            "selected_node_ids": self.selected_node_ids,
            "metadata": self.metadata,
        }


def task_identifier(task: TaskRecord) -> str:
    location = task.bug_location.replace("/", "_").replace(":", "_")
    return f"{task.repo}-{task.commit[:8]}-{location}-{task.mutation_type}"


def repo_path(base_dir: Path, repo_name: str) -> Path:
    return base_dir / repo_name



@dataclass(slots=True, frozen=True)
class RankedNode:
    node_id: str
    score: float
    text: str
    token_count: int
    kind: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
