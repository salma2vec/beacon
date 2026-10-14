"""Deterministic controls for visible, source-bound line-span edits."""

from __future__ import annotations

import json
import subprocess

import pytest

from cegraph import CodebaseGraph, grounded_line_span_contract
from cegraph.edits import LineSpanEditRepairClient, compile_line_span_edit
from cegraph.models import RepairPrompt, RepairResponse

SOURCE = (
    "def first(x):\n"
    "    return max(x, 5)\n"
    "\n"
    "def second(x):\n"
    "    return max(x, 5)\n"
)


def line_edit(**overrides: object) -> str:
    payload: dict[str, object] = {
        "path": "maths.py",
        "start_line": 5,
        "end_line": 5,
        "replacement": "    return min(x, 5)\n",
    }
    payload.update(overrides)
    return json.dumps(payload)


def second_contract(tmp_path):
    (tmp_path / "maths.py").write_text(SOURCE)
    graph = CodebaseGraph.from_repo(tmp_path)
    return grounded_line_span_contract(graph, ["maths.py::second"])


def test_numbered_range_uniquely_binds_identical_source_text(tmp_path) -> None:
    contract = second_contract(tmp_path)
    digest = contract.source_sha256["maths.py"]
    assert contract.visible_spans == {"maths.py": ((4, 5),)}
    assert f"# maths.py:4-5 source_sha256={digest}" in contract.rendered_context
    assert "5|    return max(x, 5)" in contract.rendered_context
    assert contract.schema["oneOf"][0]["properties"]["path"]["enum"] == ["maths.py"]

    patch = compile_line_span_edit(line_edit(), contract)
    assert patch is not None
    result = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        text=True,
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    expected = SOURCE.splitlines(keepends=True)
    expected[4] = "    return min(x, 5)\n"
    assert (tmp_path / "maths.py").read_text() == "".join(expected)
    # The identical first occurrence was outside the selected visible span.
    assert (tmp_path / "maths.py").read_text().splitlines()[1] == "    return max(x, 5)"


def test_range_outside_visible_span_and_stale_source_hash_fail(tmp_path) -> None:
    contract = second_contract(tmp_path)
    with pytest.raises(ValueError, match="exactly one authorized visible span"):
        compile_line_span_edit(line_edit(start_line=2, end_line=2), contract)

    contract.sources["maths.py"] += "# changed after indexing\n"
    with pytest.raises(ValueError, match="bound source hash"):
        compile_line_span_edit(line_edit(), contract)


def test_line_span_compiler_rejects_invalid_python(tmp_path) -> None:
    contract = second_contract(tmp_path)
    with pytest.raises(ValueError, match="invalid Python"):
        compile_line_span_edit(line_edit(replacement="    return (\n"), contract)


def test_verifier_paths_never_enter_line_span_authority(tmp_path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_maths.py").write_text("def test_value():\n    assert True\n")
    graph = CodebaseGraph.from_repo(tmp_path)
    with pytest.raises(ValueError, match="no authorized Python source spans"):
        grounded_line_span_contract(graph, ["tests/test_maths.py::test_value"])


def test_explicit_abstention_is_valid_and_preserves_raw_response(tmp_path) -> None:
    contract = second_contract(tmp_path)

    class Fake:
        def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
            return RepairResponse('{"abstain":true}', "raw abstention", "stub")

    client = LineSpanEditRepairClient(Fake(), contract)
    with pytest.raises(ValueError, match="bound system and rendered context"):
        client.generate_patch(RepairPrompt("s", "u"))
    response = client.generate_patch(
        RepairPrompt(contract.system, f"Repair this issue:\n{contract.rendered_context}")
    )
    assert response.patch_text == ""
    assert response.raw_response == "raw abstention"
    assert response.metadata["edit_valid"] is True
    assert response.metadata["edit_abstained"] is True
