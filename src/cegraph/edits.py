"""Compile one validated exact edit into a diff, without filesystem side effects."""

from __future__ import annotations

import difflib
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from .llm import RepairClient
from .models import RepairPrompt, RepairResponse

EDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "maxLength": 512},
        "search": {"type": "string", "minLength": 1, "maxLength": 8192},
        "replace": {"type": "string", "maxLength": 8192},
    },
    "required": ["path", "search", "replace"],
    "additionalProperties": False,
}

LINE_SPAN_EDIT_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "maxLength": 512},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "replacement": {"type": "string", "maxLength": 8192},
            },
            "required": ["path", "start_line", "end_line", "replacement"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"abstain": {"const": True}},
            "required": ["abstain"],
            "additionalProperties": False,
        },
    ]
}

EDIT_SYSTEM = (
    "Repair the reported Python bug with one exact source edit. Return only a JSON object "
    'with string fields "path", "search", "replace". Use the repository-relative file path. '
    "Copy search text exactly from the current source, including indentation; it must occur "
    "exactly once in that file. Replace it with corrected code. Do not edit tests or test "
    "support. Do not include explanations or a diff."
)

LINE_SPAN_EDIT_SYSTEM = (
    "Repair the reported Python bug with one line-span source edit. The context shows "
    "authorized repository-relative paths, inclusive source line numbers, and the indexed "
    "source SHA-256. Return only JSON with path, start_line, end_line and replacement, or "
    '{"abstain":true} when no justified edit is possible. The range must lie wholly inside '
    "one visible span. Replacement is literal source text for the complete range; include a "
    "trailing newline when needed. Do not edit tests or test support."
)


@dataclass(slots=True, frozen=True)
class ExactEditContract:
    """A schema and compiler sources derived only from visible context."""

    schema: dict[str, Any]
    system: str
    sources: dict[str, str]
    source_sha256: dict[str, str]


@dataclass(slots=True, frozen=True)
class LineSpanEditContract:
    """A line-numbered context and compiler authority from the same indexed spans."""

    schema: dict[str, Any]
    system: str
    sources: dict[str, str]
    source_sha256: dict[str, str]
    visible_spans: dict[str, tuple[tuple[int, int], ...]]
    rendered_context: str


def _authorized_source_path(path: str) -> bool:
    relative = PurePosixPath(path)
    return bool(
        path
        and len(path) <= 512
        and "\\" not in path
        and all(ord(char) >= 32 and ord(char) != 127 for char in path)
        and not relative.is_absolute()
        and relative.as_posix() == path
        and relative.suffix == ".py"
        and all(not part.startswith(".") for part in relative.parts)
        and relative.name not in {"conftest.py", "test.py", "tests.py"}
        and not relative.name.endswith("_test.py")
        and all(
            part not in {"tests", "test"} and not part.startswith("test_")
            for part in relative.parts
        )
    )


def exact_edit_schema(visible_paths: list[str]) -> dict[str, Any]:
    """Constrain decoding to source paths already present in model context."""
    paths = list(dict.fromkeys(visible_paths))
    if not paths or any(not _authorized_source_path(path) for path in paths):
        raise ValueError("Visible edit paths must be authorized Python source files")
    return {
        **EDIT_SCHEMA,
        "properties": {
            **EDIT_SCHEMA["properties"],
            "path": {"type": "string", "enum": paths},
        },
    }


def exact_edit_system(visible_paths: list[str]) -> str:
    """Tell the model the same visible-path constraint enforced by the schema."""
    paths = list(dict.fromkeys(visible_paths))
    exact_edit_schema(paths)
    rendered = ", ".join(json.dumps(path) for path in paths)
    return f"{EDIT_SYSTEM} The path must be one of these visible source files: {rendered}."


def line_span_edit_schema(visible_paths: list[str]) -> dict[str, Any]:
    """Constrain line-span decoding to source paths already rendered to the model."""
    paths = list(dict.fromkeys(visible_paths))
    if not paths or any(not _authorized_source_path(path) for path in paths):
        raise ValueError("Visible edit paths must be authorized Python source files")
    edit, abstain = LINE_SPAN_EDIT_SCHEMA["oneOf"]
    return {
        "oneOf": [
            {
                **edit,
                "properties": {
                    **edit["properties"],
                    "path": {"type": "string", "enum": paths},
                },
            },
            abstain,
        ]
    }


def _physical_lines(source: str) -> list[str]:
    """Split normalized indexed text only at Python physical newline characters."""
    parts = source.split("\n")
    if len(parts) == 1:
        return parts
    return [part + "\n" for part in parts[:-1]] + ([] if parts[-1] == "" else [parts[-1]])


def _merge_spans(spans: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(spans)):
        if start < 1 or end < start:
            raise ValueError("Selected context contains an invalid source span")
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def grounded_line_span_contract(graph: Any, selected_node_ids: list[str]) -> LineSpanEditContract:
    """Render numbered lines and bind edit authority to exactly those indexed spans."""
    sources: dict[str, str] = {}
    raw_spans: dict[str, list[tuple[int, int]]] = {}
    for node_id in selected_node_ids:
        try:
            node = graph.get_node(node_id)
        except KeyError as exc:
            raise ValueError("Selected context node is absent from the indexed graph") from exc
        if not _authorized_source_path(node.file_path):
            continue
        source = sources.setdefault(node.file_path, graph.source_for_file(node.file_path))
        end_line = (
            len(_physical_lines(source)) if node.kind.value == "module" else node.end_lineno
        )
        raw_spans.setdefault(node.file_path, []).append((node.lineno, end_line))
    if not sources:
        raise ValueError("Selected context has no authorized Python source spans")

    hashes = {
        path: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for path, source in sources.items()
    }
    visible_spans: dict[str, tuple[tuple[int, int], ...]] = {}
    rendered: list[str] = []
    for path, source in sources.items():
        lines = _physical_lines(source)
        spans = _merge_spans(raw_spans[path])
        if any(end > len(lines) for _, end in spans):
            raise ValueError("Selected context span is outside indexed source")
        visible_spans[path] = spans
        for start, end in spans:
            width = len(str(end))
            numbered = "".join(
                f"{number:>{width}}|{lines[number - 1]}" for number in range(start, end + 1)
            ).rstrip("\n")
            rendered.append(
                f"# {path}:{start}-{end} source_sha256={hashes[path]}\n{numbered}"
            )
    paths = list(sources)
    return LineSpanEditContract(
        schema=line_span_edit_schema(paths),
        system=(
            f"{LINE_SPAN_EDIT_SYSTEM} Authorized visible paths: "
            + ", ".join(json.dumps(path) for path in paths)
            + "."
        ),
        sources=sources,
        source_sha256=hashes,
        visible_spans=visible_spans,
        rendered_context="\n\n".join(rendered),
    )


def grounded_edit_contract(graph: Any, selected_node_ids: list[str]) -> ExactEditContract:
    """Bind exact edits to non-test source files represented in rendered context.

    This reads no task manifest or bug location. Full indexed file text is trusted
    compiler state; only selected exact indexed-source spans are model context.
    """
    paths: list[str] = []
    for node_id in selected_node_ids:
        try:
            node = graph.get_node(node_id)
        except KeyError as exc:
            raise ValueError("Selected context node is absent from the indexed graph") from exc
        if _authorized_source_path(node.file_path):
            paths.append(node.file_path)
    paths = list(dict.fromkeys(paths))
    schema = exact_edit_schema(paths)
    sources = {path: graph.source_for_file(path) for path in paths}
    for node_id in selected_node_ids:
        node = graph.get_node(node_id)
        if node.file_path in sources and node.source not in sources[node.file_path]:
            raise ValueError("Selected context span is not recoverable from indexed source")
    hashes = {
        path: hashlib.sha256(source.encode("utf-8")).hexdigest()
        for path, source in sources.items()
    }
    return ExactEditContract(schema, exact_edit_system(paths), sources, hashes)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def _unified_diff(path: str, source: str, changed: str) -> str:
    lines = difflib.unified_diff(
        source.splitlines(keepends=True),
        changed.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
        for line in lines
    )


def compile_exact_edit(raw: str, sources: Mapping[str, str]) -> str:
    """Reject ambiguity and invalid edits; never infer a replacement or read a file.

    ``sources`` is trusted evaluator state, not additional model context. It may
    contain only authorized source files. Patch evaluation still enforces its
    independent target and verifier exclusions before any code is executed.
    """
    if len(raw.encode("utf-8")) > 32768:
        raise ValueError("Edit response exceeds 32 KiB")
    payload = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(payload, dict) or set(payload) != {"path", "search", "replace"}:
        raise ValueError("Expected exactly path, search and replace")
    if not all(isinstance(value, str) for value in payload.values()):
        raise ValueError("Edit fields must be strings")
    path, search, replacement = payload["path"], payload["search"], payload["replace"]
    if (
        not _authorized_source_path(path)
        or path not in sources
    ):
        raise ValueError("Edit path must name an authorized Python source file")
    if not search or len(search) > 8192 or len(replacement) > 8192:
        raise ValueError("Edit text is empty or exceeds its limit")
    source = sources[path]
    start = source.find(search)
    if start < 0 or source.find(search, start + 1) >= 0:
        raise ValueError("Search text must match exactly once")
    if search == replacement:
        raise ValueError("Edit makes no change")
    changed = source[:start] + replacement + source[start + len(search) :]
    try:
        compile(changed, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Edit produces invalid Python") from exc
    return _unified_diff(path, source, changed)


def compile_line_span_edit(raw: str, contract: LineSpanEditContract) -> str | None:
    """Compile one visible line-range edit; return None for explicit abstention."""
    if len(raw.encode("utf-8")) > 32768:
        raise ValueError("Edit response exceeds 32 KiB")
    payload = json.loads(raw, object_pairs_hook=_unique_object)
    if payload == {"abstain": True}:
        return None
    required = {"path", "start_line", "end_line", "replacement"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Expected a line-span edit or explicit abstention")
    path = payload["path"]
    start = payload["start_line"]
    end = payload["end_line"]
    replacement = payload["replacement"]
    if not isinstance(path, str) or not isinstance(replacement, str):
        raise ValueError("Edit path and replacement must be strings")
    if type(start) is not int or type(end) is not int:
        raise ValueError("Edit line bounds must be integers")
    if not _authorized_source_path(path) or path not in contract.sources:
        raise ValueError("Edit path must name an authorized visible Python source file")
    if len(replacement) > 8192:
        raise ValueError("Replacement exceeds its limit")
    source = contract.sources[path]
    actual_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if actual_hash != contract.source_sha256.get(path):
        raise ValueError("Indexed source does not match the bound source hash")
    containing = [
        span for span in contract.visible_spans.get(path, ()) if span[0] <= start <= end <= span[1]
    ]
    if len(containing) != 1:
        raise ValueError("Edit range must bind to exactly one authorized visible span")
    lines = _physical_lines(source)
    if end > len(lines):
        raise ValueError("Edit range is outside indexed source")
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    first, last = offsets[start - 1], offsets[end]
    if source[first:last] == replacement:
        raise ValueError("Edit makes no change")
    changed = source[:first] + replacement + source[last:]
    try:
        compile(changed, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("Edit produces invalid Python") from exc
    return _unified_diff(path, source, changed)


class ExactEditRepairClient:
    """Compose a schema-configured transport with a deterministic edit compiler."""

    def __init__(self, transport: RepairClient, sources: Mapping[str, str]) -> None:
        self.transport = transport
        self.sources = dict(sources)

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        response = self.transport.generate_patch(prompt)
        metadata = {**response.metadata, "edit_format": "exact_search_replace_v1"}
        patch = ""
        try:
            if not response.patch_text:
                raise ValueError("Transport returned no complete edit")
            patch = compile_exact_edit(response.patch_text, self.sources)
            metadata["edit_valid"] = True
        except (ValueError, UnicodeError, RecursionError) as exc:
            metadata.update(edit_valid=False, edit_error=str(exc))
        return RepairResponse(patch, response.raw_response, response.model, metadata)


class LineSpanEditRepairClient:
    """Compose a schema transport with the source-hash-bound line compiler."""

    def __init__(self, transport: RepairClient, contract: LineSpanEditContract) -> None:
        self.transport = transport
        self.contract = contract

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        if (
            prompt.system != self.contract.system
            or self.contract.rendered_context not in prompt.user
        ):
            raise ValueError("Line-span prompt must contain its bound system and rendered context")
        response = self.transport.generate_patch(prompt)
        metadata = {
            **response.metadata,
            "edit_format": "visible_line_span_v1",
            "visible_source_sha256": dict(self.contract.source_sha256),
        }
        patch = ""
        try:
            if not response.patch_text:
                raise ValueError("Transport returned no complete edit")
            compiled = compile_line_span_edit(response.patch_text, self.contract)
            metadata["edit_valid"] = True
            metadata["edit_abstained"] = compiled is None
            patch = compiled or ""
        except (ValueError, UnicodeError, RecursionError) as exc:
            metadata.update(edit_valid=False, edit_abstained=False, edit_error=str(exc))
        return RepairResponse(patch, response.raw_response, response.model, metadata)
