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

EDIT_SYSTEM = (
    "Repair the reported Python bug with one exact source edit. Return only a JSON object "
    'with string fields "path", "search", "replace". Use the repository-relative file path. '
    "Copy search text exactly from the current source, including indentation; it must occur "
    "exactly once in that file. Replace it with corrected code. Do not edit tests or test "
    "support. Do not include explanations or a diff."
)


@dataclass(slots=True, frozen=True)
class ExactEditContract:
    """A schema and compiler sources derived only from visible context."""

    schema: dict[str, Any]
    system: str
    sources: dict[str, str]
    source_sha256: dict[str, str]


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
    lines = difflib.unified_diff(
        source.splitlines(keepends=True),
        changed.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(
        line if line.endswith("\n") else line + "\n\\ No newline at end of file\n" for line in lines
    )


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
