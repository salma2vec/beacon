from __future__ import annotations

import json
import subprocess

import pytest

from cegraph.edits import EDIT_SCHEMA, ExactEditRepairClient, compile_exact_edit
from cegraph.llm import OllamaRepairClient
from cegraph.models import RepairPrompt, RepairResponse

SOURCE = "def clip(x):\n    return max(x, 5)\n"


def edit(**overrides):
    return json.dumps(
        {"path": "maths.py", "search": "max(x, 5)", "replace": "min(x, 5)", **overrides}
    )


@pytest.mark.parametrize("newline", [True, False])
def test_edit_applies_without_changing_unrelated_source(tmp_path, newline):
    original = SOURCE if newline else SOURCE.rstrip("\n")
    (tmp_path / "maths.py").write_text(original)
    patch = compile_exact_edit(edit(), {"maths.py": original})
    result = subprocess.run(
        ["git", "apply", "-"], input=patch, text=True, cwd=tmp_path, capture_output=True
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "maths.py").read_text() == original.replace("max", "min")
    assert original == (SOURCE if newline else SOURCE.rstrip("\n"))


@pytest.mark.parametrize(
    "path",
    [
        "../maths.py",
        "/maths.py",
        "a/../maths.py",
        "./maths.py",
        "a//b.py",
        "a\\b.py",
        "test.py",
        "tests.py",
        "conftest.py",
        "test_helpers.py",
        "x_test.py",
        "tests/a.py",
        ".hidden/a.py",
        "a\nb.py",
        "maths.txt",
    ],
)
def test_paths_fail_even_if_present_in_mapping(path):
    with pytest.raises(ValueError, match="path"):
        compile_exact_edit(edit(path=path), {path: SOURCE})


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        '{"path":"maths.py","search":"max","replace":"min","path":"x.py"}',
        edit(extra="ignored?"),
        edit(search=3),
        edit(search=""),
        edit(search="missing"),
        edit(replace="max(x, 5)"),
        edit(replace="return return"),
        edit(replace="x" * 8193),
        edit(path="other.py"),
    ],
)
def test_bad_edits_fail_without_mutating_sources(raw):
    sources = {"maths.py": SOURCE}
    with pytest.raises(ValueError):
        compile_exact_edit(raw, sources)
    assert sources == {"maths.py": SOURCE}


def test_overlapping_matches_are_ambiguous():
    with pytest.raises(ValueError, match="exactly once"):
        compile_exact_edit(edit(search="aa", replace="bb"), {"maths.py": "aaa = 1\n"})


def test_edit_is_not_executed_by_compiler(tmp_path):
    marker = tmp_path / "never-created"
    payload = edit(replace=f"__import__('pathlib').Path({str(marker)!r}).touch()")
    assert compile_exact_edit(payload, {"maths.py": SOURCE})
    assert not marker.exists()


def test_compiler_keeps_raw_failures():
    class Fake:
        def generate_patch(self, prompt):
            return RepairResponse(edit(search="unknown"), "original completion", "stub")

    result = ExactEditRepairClient(Fake(), {"maths.py": SOURCE}).generate_patch(
        RepairPrompt("s", "u")
    )
    assert result.patch_text == ""
    assert result.raw_response == "original completion"
    assert result.metadata["edit_valid"] is False


def test_schema_reaches_transport_without_fence_repair(monkeypatch):
    requests = []

    def request(self, path, payload=None):
        if path == "/api/tags":
            return {"models": [{"name": "local", "digest": "a" * 64}]}
        if path != "/api/chat":
            return {}
        requests.append(payload)
        return {
            "done": True,
            "done_reason": "stop",
            "message": {"content": "```json\n{}\n```"},
            "prompt_eval_count": 10,
        }

    monkeypatch.setattr(OllamaRepairClient, "_request", request)
    client = OllamaRepairClient("local", response_schema=EDIT_SCHEMA)
    result = client.generate_patch(RepairPrompt("s", "u"))
    assert requests[0]["format"] == EDIT_SCHEMA
    assert result.patch_text.startswith("```json")
    assert result.metadata["output_format"] == "json_schema"
