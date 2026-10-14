from __future__ import annotations

import difflib
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from cegraph import CodebaseGraph, assemble_context, available_methods
from cegraph.evaluation import (
    apply_mutation,
    apply_patch_text,
    compile_python_repo,
    evaluate_patch,
    repo_snapshot,
    run_pytest_tests,
    run_test_command,
)
from cegraph.models import RepairPrompt, RepairResponse, TaskRecord
from paper.experiments.benchmark import build_parser, run_benchmark
from paper.experiments.merge_and_report import aggregate_pass_rates


def test_diff_preserves_trailing_blank_context_line(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    old = "def add(a, b):\n    return a - b\n\n"
    fixed = "def add(a, b):\n    return a + b\n\n"
    (root / "calc.py").write_text(old)
    patch = "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            "a/calc.py",
            "b/calc.py",
        )
    )
    assert patch.endswith(" \n")
    passed, reason = apply_patch_text(root, task, RepairResponse(patch, patch, "control"))
    assert passed, reason
    assert (root / "calc.py").read_text() == fixed


def fixture_repo(tmp_path: Path) -> tuple[Path, TaskRecord]:
    root = tmp_path / "fixture"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n")
    (root / "test_calc.py").write_text(
        "from calc import add\n\ndef test_positive():\n    assert add(2, 3) == 5\n"
        "\ndef test_negative():\n    assert add(-2, 3) == 1\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=root,
        check=True,
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return root, TaskRecord(
        "fixture",
        commit,
        "calc.py:2",
        "operator",
        "    return a + b",
        "    return a - b",
        256,
        test_command=f"{shlex.quote(sys.executable)} -m pytest -q test_calc.py",
    )


def response(path: str, content: str) -> RepairResponse:
    return RepairResponse(json.dumps({"file_path": path, "content": content}), "", "fixture-oracle")


@pytest.mark.parametrize(
    "path", ["../outside.py", "/tmp/outside.py", ".git/config", "test_calc.py"]
)
def test_patch_rejects_other_paths(tmp_path: Path, path: str) -> None:
    root, task = fixture_repo(tmp_path)
    before = (root / "calc.py").read_text()
    assert not apply_patch_text(root, task, response(path, "bad"))[0]
    assert (root / "calc.py").read_text() == before


def test_symlink_rejected(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("original")
    (root / "calc.py").unlink()
    (root / "calc.py").symlink_to(outside)
    assert not apply_patch_text(root, task, response("calc.py", "bad"))[0]
    assert outside.read_text() == "original"


def test_pinned_mutation_and_evaluation(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    (root / "calc.py").write_text("dirty checkout")
    with repo_snapshot(root, task.commit) as snapshot:
        baseline = run_pytest_tests(snapshot, task.test_command)
        assert baseline.passed
        apply_mutation(snapshot, task)
        assert run_test_command(snapshot, task.test_command)[0] == 1
        result = evaluate_patch(
            snapshot,
            task,
            response("calc.py", "def add(a, b):\n    return a + b\n"),
            required_test_ids=baseline.test_ids,
        )
        assert result.pass_at_1
    assert not snapshot.exists()
    assert (root / "calc.py").read_text() == "dirty checkout"


def test_missing_test_and_timeout_are_failures(tmp_path: Path) -> None:
    assert run_test_command(tmp_path, None)[0] != 0
    command = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(10)"'
    assert run_test_command(tmp_path, command, timeout=0.1)[0] == 124


def test_mutation_requires_matching_source(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    with pytest.raises(ValueError, match="match"):
        apply_mutation(root, replace(task, original_line="wrong"))


def test_rendered_context_budget_includes_headers(tmp_path: Path) -> None:
    root, _ = fixture_repo(tmp_path)
    graph = CodebaseGraph.from_repo(root)
    for method in available_methods():
        for budget in [0, 1, 20, 40, 100, 256]:
            context = assemble_context(graph, "add", method, budget, "calc.py:2").context
            assert graph.token_counter.count(context.render()) == context.total_tokens
            assert context.total_tokens <= budget


def test_end_to_end_oracle_control(tmp_path: Path) -> None:
    from dataclasses import asdict

    root, task = fixture_repo(tmp_path)
    manifest = tmp_path / "tasks.jsonl"
    manifest.write_text(json.dumps(asdict(task)) + "\n")

    class Oracle:
        def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
            return response("calc.py", "def add(a, b):\n    return a + b\n")

    args = build_parser().parse_args(
        [
            "--tasks-file",
            str(manifest),
            "--repos-dir",
            str(tmp_path),
            "--budgets",
            "256",
            "--methods",
            "bm25,bca",
            "--query-types",
            "vague",
            "--output-dir",
            str(tmp_path / "output"),
            "--allow-local-execution",
        ]
    )
    result = run_benchmark(args, Oracle())
    rows = [json.loads(line) for line in result.read_text().splitlines()]
    assert len(rows) == 2 and all(row["pass_at_1"] for row in rows)
    assert all(
        row["metadata"]["baseline_exit_code"] == 0 and row["metadata"]["mutant_exit_code"] == 1
        for row in rows
    )
    assert all(not row["metadata"]["oracle_location"] for row in rows)
    assert (root / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_unscored_rows_excluded() -> None:
    assert aggregate_pass_rates([{"pass_at_1": None}]) == {}


def test_subprocess_credentials_not_inherited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "private-test-value")
    code = (
        "import os; assert not os.getenv('OPENAI_API_KEY'); "
        "assert not os.getenv('AWS_SECRET_ACCESS_KEY')"
    )
    command = shlex.join([sys.executable, "-c", code])
    assert run_test_command(tmp_path, command)[0] == 0


def test_output_is_bounded(tmp_path: Path) -> None:
    command = shlex.join([sys.executable, "-c", "print('x' * 100000)"])
    code, stdout, stderr = run_test_command(tmp_path, command, output_limit=1024)
    assert code == 124 and len(stdout) <= 1024
    assert "output limit" in stderr


def test_index_does_not_read_external_symlink(tmp_path: Path) -> None:
    root, _ = fixture_repo(tmp_path)
    outside = tmp_path / "private.py"
    outside.write_text("secret = 'do not index'\n")
    (root / "linked.py").symlink_to(outside)
    graph = CodebaseGraph.from_repo(root)
    assert not any(node.file_path == "linked.py" for node in graph.nodes.values())


def test_no_gold_source_from_working_tree(tmp_path: Path) -> None:
    from paper.experiments.discover_mutations import main

    root, _ = fixture_repo(tmp_path)
    (root / "calc.py").write_text("not valid python")
    output = main(
        [
            "--repo-name",
            "fixture",
            "--repo-root",
            str(root),
            "--budgets",
            "100",
            "--output-file",
            str(tmp_path / "mutations.jsonl"),
        ]
    )
    assert output.read_text()


@pytest.mark.parametrize("patch", ["import sys\nsys.exit(0)\n", "import os\nos._exit(0)\n"])
def test_successful_early_exit_cannot_score(tmp_path: Path, patch: str) -> None:
    root, task = fixture_repo(tmp_path)
    baseline = run_pytest_tests(root, task.test_command)
    assert baseline.passed and len(baseline.test_ids) == 2
    apply_mutation(root, task)
    result = evaluate_patch(
        root, task, response("calc.py", patch), required_test_ids=baseline.test_ids
    )
    assert not result.pass_at_1 and not result.tests_passed
    assert result.metadata["test_run_completed"] is False


def test_custom_test_command_cannot_score(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    baseline = run_pytest_tests(root, task.test_command)
    custom = replace(task, test_command=shlex.join([sys.executable, "test_calc.py"]))
    result = evaluate_patch(
        root,
        custom,
        response("calc.py", "import sys\nsys.exit(0)\n"),
        required_test_ids=baseline.test_ids,
    )
    assert not result.pass_at_1 and "requires pytest" in result.stderr


def test_empty_or_skipped_tests_cannot_complete(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    (root / "test_calc.py").write_text("pass\n")
    result = run_pytest_tests(root, task.test_command)
    assert not result.completed and not result.passed
    (root / "test_calc.py").write_text(
        "import pytest\n@pytest.mark.skip(reason='bypass')\ndef test_skip():\n    assert False\n"
    )
    result = run_pytest_tests(root, task.test_command)
    assert result.exit_code == 0
    assert not result.completed and not result.passed


def test_required_original_test_ids_must_all_complete(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    baseline = run_pytest_tests(root, task.test_command)
    (root / "test_calc.py").write_text("def test_positive():\n    assert True\n")
    result = run_pytest_tests(root, task.test_command, required_test_ids=baseline.test_ids)
    assert result.exit_code == 0
    assert not result.completed and "test IDs differ" in result.reason


def test_missing_report_even_after_exit_zero_cannot_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import cegraph.evaluation as evaluation

    root, task = fixture_repo(tmp_path)
    monkeypatch.setattr(evaluation, "run_test_command", lambda *args: (0, "all tests passed", ""))
    result = run_pytest_tests(root, task.test_command)
    assert result.exit_code == 0 and not result.passed
    assert "report is missing" in result.reason


def test_evaluation_requires_trusted_original_test_ids(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    result = evaluate_patch(root, task, response("calc.py", "import sys\nsys.exit(0)\n"))
    assert not result.pass_at_1 and "baseline test IDs" in result.stderr
    assert (root / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_runner_is_not_shadowed_by_repository_pytest_module(tmp_path: Path) -> None:
    root, task = fixture_repo(tmp_path)
    (root / "pytest.py").write_text("raise RuntimeError('untrusted pytest module')\n")
    result = run_pytest_tests(root, task.test_command)
    assert result.passed and len(result.test_ids) == 2


def test_syntax_check_never_writes_through_bytecode_directory_symlink(tmp_path: Path) -> None:
    root, _ = fixture_repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_text("unchanged")
    (root / "__pycache__").symlink_to(outside, target_is_directory=True)
    assert compile_python_repo(root) == (True, "")
    assert list(outside.iterdir()) == [marker]
    assert marker.read_text() == "unchanged"
    (root / "calc.py").write_text("def invalid(:\n")
    passed, reason = compile_python_repo(root)
    assert not passed and "calc.py" in reason
    assert list(outside.iterdir()) == [marker]


@pytest.mark.parametrize("filename", ["conftest.py", "calc_test.py", "pkg/conftest.py"])
def test_task_cannot_designate_pytest_verifier_as_repair_source(
    tmp_path: Path, filename: str
) -> None:
    root, task = fixture_repo(tmp_path)
    verifier = root / filename
    verifier.parent.mkdir(exist_ok=True)
    verifier.write_text("x = 1\n")
    target_task = replace(
        task, bug_location=f"{filename}:1", original_line="x = 1", mutated_line="x = 2"
    )
    with pytest.raises(ValueError, match="evaluation tests"):
        apply_mutation(root, target_task)
    assert not apply_patch_text(root, target_task, response(filename, "x = 2\n"))[0]
    assert verifier.read_text() == "x = 1\n"


@pytest.mark.parametrize("selection", ["checks.py", "./checks.py::test_required"])
def test_task_cannot_edit_explicitly_selected_test_with_custom_filename(
    tmp_path: Path, selection: str
) -> None:
    root, task = fixture_repo(tmp_path)
    content = "def test_required():\n    assert 1 == 1\n"
    (root / "checks.py").write_text(content)
    target_task = replace(
        task,
        bug_location="checks.py:2",
        original_line="    assert 1 == 1",
        mutated_line="    assert 1 == 2",
        test_command=f"python -m pytest -q {selection}",
    )
    with pytest.raises(ValueError, match="explicitly selected evaluation tests"):
        apply_mutation(root, target_task)
    assert not apply_patch_text(root, target_task, response("checks.py", "assert True\n"))[0]
    assert (root / "checks.py").read_text() == content


@pytest.mark.parametrize(
    ("filename", "content", "config"),
    [
        ("checks_plugin.py", "def pytest_pyfunc_call(item):\n    return None\n", ""),
        (
            "checks_plugin.py",
            "def custom_hook(item):\n    return None\n",
            "pytest_plugins = ['checks_plugin']\n",
        ),
        (
            "src/validation/plugin.py",
            "x = 1\n",
            "pytest_plugins: list[str] = ['validation.plugin']\n",
        ),
        ("validation/__init__.py", "x = 1\n", "pytest_plugins = 'validation'\n"),
    ],
)
def test_task_cannot_edit_named_hooks_or_registered_local_pytest_plugins(
    tmp_path: Path, filename: str, content: str, config: str
) -> None:
    root, task = fixture_repo(tmp_path)
    path = root / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    (root / "conftest.py").write_text(config)
    target_task = replace(
        task,
        bug_location=f"{filename}:1",
        original_line=content.splitlines()[0],
        mutated_line="raise AssertionError",
    )
    with pytest.raises(ValueError, match="pytest verifier hooks or plugins"):
        apply_mutation(root, target_task)
    assert not apply_patch_text(root, target_task, response(filename, "x = 0\n"))[0]
    assert path.read_text() == content


def test_source_with_directory_test_selection_ignores_hidden_environment_hooks(
    tmp_path: Path,
) -> None:
    root, task = fixture_repo(tmp_path)
    hidden = root / ".venv" / "conftest.py"
    hidden.parent.mkdir()
    hidden.write_text("not valid Python !\n")
    apply_mutation(root, replace(task, test_command="python -m pytest -q ."))
    assert "return a - b" in (root / "calc.py").read_text()


def test_exited_process_group_cleanup_denial_cannot_score_success(tmp_path: Path) -> None:
    command = shlex.join([sys.executable, "-c", "print('completed')"])
    with patch("cegraph.evaluation.os.killpg", side_effect=PermissionError):
        code, stdout, stderr = run_test_command(tmp_path, command)
    assert code == 124 and "completed" in stdout
    assert "process-group cleanup denied" in stderr
