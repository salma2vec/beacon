"""Compile one validated exact edit into a diff, without filesystem side effects."""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping
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
    relative = PurePosixPath(path)
    if (
        not path
        or len(path) > 512
        or "\\" in path
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or relative.is_absolute()
        or relative.as_posix() != path
        or relative.suffix != ".py"
        or any(part.startswith(".") for part in relative.parts)
        or relative.name in {"conftest.py", "test.py", "tests.py"}
        or relative.name.endswith("_test.py")
        or any(part in {"tests", "test"} or part.startswith("test_") for part in relative.parts)
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
