"""AST parsing and edge extraction for Python codebases."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .models import EdgeKind, Node, NodeKind


@dataclass(slots=True, frozen=True)
class PendingEdge:
    source_id: str
    kind: EdgeKind
    target_ref: str
    source_module: str
    metadata: dict[str, str]


@dataclass(slots=True, frozen=True)
class ParsedModule:
    relative_path: str
    module_name: str
    nodes: list[Node]
    pending_edges: list[PendingEdge]
    local_symbols: dict[str, str]


def module_name_for_path(file_path: Path, repo_root: Path) -> str:
    relative = file_path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def node_id(relative_path: str, qualified_name: str) -> str:
    return f"{relative_path}::{qualified_name}"


def expression_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = expression_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Call):
        return expression_name(node.func)
    if isinstance(node, ast.Subscript):
        return expression_name(node.value)
    return None


class ModuleParser(ast.NodeVisitor):
    def __init__(self, repo_root: Path, file_path: Path, source: str) -> None:
        self.repo_root = repo_root
        self.file_path = file_path
        self.source = source
        self.relative_path = file_path.relative_to(repo_root).as_posix()
        self.module_name = module_name_for_path(file_path, repo_root)
        self.tree = ast.parse(source)
        self.lines = source.splitlines()
        self.nodes: list[Node] = []
        self.pending_edges: list[PendingEdge] = []
        self.local_symbols: dict[str, str] = {}
        self.scope_stack: list[str] = []
        self.node_kinds: dict[str, NodeKind] = {}
        self._register_module()

    def _register_module(self) -> None:
        module_qualified_name = self.module_name or self.file_path.stem
        module_node = Node(
            id=node_id(self.relative_path, module_qualified_name),
            file_path=self.relative_path,
            qualified_name=module_qualified_name,
            kind=NodeKind.MODULE,
            source=self.source,
            lineno=1,
            end_lineno=max(len(self.lines), 1),
            docstring=ast.get_docstring(self.tree),
        )
        self.nodes.append(module_node)
        self.local_symbols[module_qualified_name] = module_node.id
        self.node_kinds[module_node.id] = module_node.kind

    @property
    def current_scope(self) -> str:
        return self.scope_stack[-1] if self.scope_stack else self.nodes[0].id

    def parse(self) -> ParsedModule:
        self.visit(self.tree)
        return ParsedModule(
            relative_path=self.relative_path,
            module_name=self.module_name,
            nodes=self.nodes,
            pending_edges=self.pending_edges,
            local_symbols=self.local_symbols,
        )

    def _source_segment(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        if segment is not None:
            return segment
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(self.lines[start:end])

    def _add_node(self, node_name: str, kind: NodeKind, node: ast.AST, parent_id: str) -> str:
        parent_name = self.nodes_by_id[parent_id].qualified_name
        qualified_name = (
            f"{parent_name}.{node_name}" if self.node_kinds[parent_id] != NodeKind.MODULE else node_name
        )
        identifier = node_id(self.relative_path, qualified_name)
        created = Node(
            id=identifier,
            file_path=self.relative_path,
            qualified_name=qualified_name,
            kind=kind,
            source=self._source_segment(node),
            lineno=getattr(node, "lineno", 1),
            end_lineno=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            parent_id=parent_id,
            docstring=ast.get_docstring(node) if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) else None,
        )
        self.nodes.append(created)
        self.local_symbols[node_name] = identifier
        self.node_kinds[identifier] = kind
        return identifier

    @property
    def nodes_by_id(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def _add_pending_edge(
        self,
        source_id: str,
        kind: EdgeKind,
        target_ref: str | None,
        **metadata: str,
    ) -> None:
        if not target_ref:
            return
        self.pending_edges.append(
            PendingEdge(
                source_id=source_id,
                kind=kind,
                target_ref=target_ref,
                source_module=self.module_name,
                metadata=metadata,
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        created_id = self._add_node(node.name, NodeKind.CLASS, node, self.current_scope)
        for base in node.bases:
            self._add_pending_edge(created_id, EdgeKind.INHERITS_FROM, expression_name(base))
        for decorator in node.decorator_list:
            self._add_pending_edge(created_id, EdgeKind.DECORATES, expression_name(decorator))
        self.scope_stack.append(created_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        kind = NodeKind.METHOD if self.node_kinds[self.current_scope] == NodeKind.CLASS else NodeKind.FUNCTION
        created_id = self._add_node(node.name, kind, node, self.current_scope)
        for decorator in node.decorator_list:
            self._add_pending_edge(created_id, EdgeKind.DECORATES, expression_name(decorator))
        self.scope_stack.append(created_id)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        sync_version = ast.FunctionDef(
            name=node.name,
            args=node.args,
            body=node.body,
            decorator_list=node.decorator_list,
            returns=node.returns,
            type_comment=node.type_comment,
        )
        sync_version.lineno = node.lineno
        sync_version.end_lineno = node.end_lineno
        self.visit_FunctionDef(sync_version)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for imported in node.names:
            self._add_pending_edge(
                self.current_scope,
                EdgeKind.IMPORTS,
                imported.name,
                alias=imported.asname or "",
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if node.level:
            module = "." * node.level + module
        for imported in node.names:
            self._add_pending_edge(
                self.current_scope,
                EdgeKind.IMPORTS,
                f"{module}:{imported.name}",
                alias=imported.asname or "",
            )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        target_name = expression_name(node.value)
        self._add_pending_edge(self.current_scope, EdgeKind.ASSIGNS_TO, target_name)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        target_name = expression_name(node.annotation) or expression_name(node.value)
        self._add_pending_edge(self.current_scope, EdgeKind.ASSIGNS_TO, target_name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._add_pending_edge(self.current_scope, EdgeKind.CALLS, expression_name(node.func))
        self.generic_visit(node)


def parse_python_file(repo_root: Path, file_path: Path) -> ParsedModule:
    source = file_path.read_text(encoding="utf-8")
    return ModuleParser(repo_root, file_path, source).parse()


def iter_python_files(repo_root: Path) -> Iterable[Path]:
    for file_path in sorted(repo_root.rglob("*.py")):
        if any(part.startswith(".") for part in file_path.relative_to(repo_root).parts):
            continue
        yield file_path

