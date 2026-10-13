"""Trusted-local evaluation with path-constrained edits and bounded subprocesses.

This module is not an OS sandbox. Run untrusted code in an external container.
"""

from __future__ import annotations

import json
import math
import os
import py_compile
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .models import EvaluationResult, RepairResponse, TaskRecord


def task_target(repo_root: Path, task: TaskRecord) -> Path:
    """Resolve the single allowed source edit without following symlinks."""
    relative = Path(task.bug_location.rsplit(":", 1)[0])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Task source path must be relative and contained in the repository")
    if relative.suffix != ".py" or any(p.startswith(".") for p in relative.parts):
        raise ValueError("Task target must be a visible Python source file")
    if any(p in {"tests", "test"} or p.startswith("test_") for p in relative.parts):
        raise ValueError("Task target cannot modify the evaluation tests")
    target = repo_root / relative
    if any(
        (repo_root / Path(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("Task target cannot traverse a symlink")
    if not target.resolve().is_relative_to(repo_root.resolve()) or not target.is_file():
        raise ValueError("Task target must be an existing file inside the repository")
    return target


def apply_mutation(repo_root: Path, task: TaskRecord) -> None:
    """Apply exactly the declared mutation to the pinned original source."""
    target = task_target(repo_root, task)
    line_number = int(task.bug_location.rsplit(":", 1)[1])
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    if not 1 <= line_number <= len(lines):
        raise ValueError("Mutation line is outside the source file")
    original = lines[line_number - 1].rstrip("\r\n")
    if original != task.original_line or task.original_line == task.mutated_line:
        raise ValueError("Mutation does not match the pinned original line")
    if "\n" in task.mutated_line or "\r" in task.mutated_line:
        raise ValueError("Only single-line mutations are supported")
    newline = "\n" if lines[line_number - 1].endswith("\n") else ""
    lines[line_number - 1] = task.mutated_line + newline
    target.write_text("".join(lines), encoding="utf-8")


def apply_patch_text(
    repo_root: Path, task: TaskRecord, response: RepairResponse
) -> tuple[bool, str]:
    patch_text = response.patch_text.strip()
    if not patch_text:
        return False, "empty patch"
    try:
        target = task_target(repo_root, task)
        allowed = target.relative_to(repo_root).as_posix()
        if patch_text.startswith("{"):
            payload = json.loads(patch_text)
            if not isinstance(payload, dict) or set(payload) != {"file_path", "content"}:
                return False, "replacement must contain only file_path and content"
            if payload["file_path"] != allowed or not isinstance(payload["content"], str):
                return False, "replacement must edit only the declared task source"
            target.write_text(payload["content"], encoding="utf-8")
            return True, "json file replacement"
        patch = patch_text + "\n"
        summary = subprocess.run(
            ["git", "apply", "--numstat", "-"],
            input=patch,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        paths = [line.split("\t")[-1] for line in summary.stdout.splitlines()]
        if summary.returncode or paths != [allowed]:
            return False, "patch must edit only the declared task source"
        if any(
            word in patch
            for word in (
                "new file mode",
                "deleted file mode",
                "old mode",
                "new mode",
                "rename from",
                "copy from",
            )
        ):
            return False, "patch cannot change file modes or paths"
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            input=patch,
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return result.returncode == 0, result.stderr.strip() or "git apply"
    except (ValueError, TypeError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)


def compile_python_repo(repo_root: Path) -> tuple[bool, str]:
    failures: list[str] = []
    for path in repo_root.rglob("*.py"):
        if path.is_symlink() or any(
            part.startswith(".") for part in path.relative_to(repo_root).parts
        ):
            continue
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(str(error))
    return not failures, "\n".join(failures)


def run_test_command(
    repo_root: Path,
    test_command: str | None,
    timeout: float = 60,
    output_limit: int = 1_048_576,
) -> tuple[int, str, str]:
    """Run trusted local code with bounded output/time and no inherited credentials.

    This is resource hygiene, not a sandbox: code retains the current user's file
    and network permissions. The benchmark CLI requires explicit local execution.
    """
    if not test_command or not shlex.split(test_command):
        return 2, "", "test_command is required for scored evaluation"
    if not math.isfinite(timeout) or timeout <= 0 or output_limit <= 0:
        raise ValueError("timeout and output_limit must be positive and finite")
    for directory in repo_root.rglob("__pycache__"):
        if directory.is_dir() and not directory.is_symlink():
            shutil.rmtree(directory)
    with tempfile.TemporaryDirectory(prefix="beacon-process-") as process_home:
        environment = {
            "PATH": os.environ.get("PATH", os.defpath),
            "LANG": "C.UTF-8",
            "HOME": process_home,
            "TMPDIR": process_home,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_ADDOPTS": "",
        }
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    shlex.split(test_command),
                    cwd=repo_root,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    env=environment,
                    start_new_session=True,
                )
            except OSError as error:
                return 127, "", str(error)
            deadline = time.monotonic() + timeout
            reason = ""
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    reason = "test command timed out"
                if (
                    os.fstat(stdout_file.fileno()).st_size + os.fstat(stderr_file.fileno()).st_size
                    > output_limit
                ):
                    reason = "test output limit exceeded"
                if reason:
                    break
                time.sleep(min(0.02, timeout))
            # Terminate any children still holding output handles after the parent exits.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(output_limit).decode("utf-8", errors="replace")
            stderr = stderr_file.read(output_limit).decode("utf-8", errors="replace")
            if (
                os.fstat(stdout_file.fileno()).st_size + os.fstat(stderr_file.fileno()).st_size
                > output_limit
            ):
                reason = "test output limit exceeded"
            if reason:
                return 124, stdout, stderr + "\n" + reason
            return process.returncode, stdout, stderr


@dataclass(frozen=True)
class PytestRun:
    """Validated pytest completion, separate from process exit success."""

    exit_code: int
    stdout: str
    stderr: str
    completed: bool = False
    test_ids: frozenset[str] = frozenset()
    failed_ids: frozenset[str] = frozenset()
    reason: str = ""

    @property
    def passed(self) -> bool:
        return self.completed and self.exit_code == 0 and not self.failed_ids


def _pytest_arguments(test_command: str | None) -> list[str]:
    """Accept a narrow pytest command; the harness selects its own interpreter."""
    command = shlex.split(test_command or "")
    if command and command[0] == "pytest":
        arguments = command[1:]
    elif len(command) >= 3 and command[1:3] == ["-m", "pytest"]:
        executable = command[0]
        if "/" in executable:
            if Path(executable).resolve() != Path(sys.executable).resolve():
                raise ValueError("Scored pytest must use the harness Python interpreter")
        elif not re.fullmatch(r"python(?:3(?:\.\d+)?)?", executable):
            raise ValueError("Scored evaluation supports only python -m pytest")
        arguments = command[3:]
    else:
        raise ValueError(
            "Scored evaluation requires pytest; custom commands need a structured adapter"
        )
    flags = {
        "-q",
        "-v",
        "--disable-warnings",
        "--strict-markers",
        "--strict-config",
        "--tb=short",
        "--tb=no",
        "--tb=long",
    }
    for argument in arguments:
        if argument.startswith("-") and argument not in flags:
            raise ValueError(f"Unsupported scored pytest option: {argument}")
        if not argument or "\x00" in argument:
            raise ValueError("Invalid pytest argument")
    return arguments


def run_pytest_tests(
    repo_root: Path,
    test_command: str | None,
    timeout: float = 60,
    required_test_ids: frozenset[str] | None = None,
) -> PytestRun:
    """Require a complete, nonempty JUnit run with unchanged original test IDs.

    Skips, collection/setup errors, missing reports, and early successful exits are
    failures. The report lives outside the checkout. Same-process adversarial code
    can still forge runner output; this is not tamper-proof sandboxing.
    """
    try:
        arguments = _pytest_arguments(test_command)
    except ValueError as error:
        return PytestRun(2, "", str(error), reason=str(error))
    with tempfile.TemporaryDirectory(prefix="beacon-pytest-") as directory:
        report = Path(directory) / "results.xml"
        command = shlex.join(
            [
                sys.executable,
                "-I",
                "-m",
                "pytest",
                *arguments,
                "-o",
                "addopts=",
                "-o",
                "junit_family=xunit2",
                "-o",
                "junit_suite_name=beacon",
                "--junitxml",
                str(report),
            ]
        )
        code, stdout, stderr = run_test_command(repo_root, command, timeout)
        try:
            if code not in {0, 1}:
                raise ValueError(f"pytest did not finish normally (exit {code})")
            if not report.is_file() or report.is_symlink():
                raise ValueError("pytest completion report is missing")
            if report.stat().st_size > 4_194_304:
                raise ValueError("pytest completion report exceeds size limit")
            root = ET.parse(report).getroot()
            suites = root.findall("testsuite")
            if root.tag != "testsuites" or len(suites) != 1:
                raise ValueError("Invalid pytest completion report structure")
            suite = suites[0]
            cases = suite.findall("testcase")
            count = int(suite.attrib["tests"])
            failures = int(suite.attrib["failures"])
            if count <= 0 or count != len(cases):
                raise ValueError("pytest must complete a nonempty set of tests")
            if int(suite.attrib["errors"]) != 0 or int(suite.attrib["skipped"]) != 0:
                raise ValueError(
                    "pytest errors or skipped tests cannot count as behavioral completion"
                )
            ids: set[str] = set()
            failed: set[str] = set()
            for case in cases:
                identifier = json.dumps([case.attrib["classname"], case.attrib["name"]])
                if identifier in ids or not case.attrib["name"]:
                    raise ValueError("pytest completion has duplicate or missing test IDs")
                if case.find("error") is not None or case.find("skipped") is not None:
                    raise ValueError("pytest did not execute every required test")
                ids.add(identifier)
                if case.find("failure") is not None:
                    failed.add(identifier)
            if failures != len(failed) or (code == 0) != (not failed):
                raise ValueError("pytest process status and completion report disagree")
            if required_test_ids is not None and ids != required_test_ids:
                raise ValueError("pytest completed test IDs differ from the original baseline")
            return PytestRun(code, stdout, stderr, True, frozenset(ids), frozenset(failed))
        except (OSError, ValueError, KeyError, ET.ParseError) as error:
            reason = str(error)
            return PytestRun(code, stdout, stderr + "\n" + reason, reason=reason)


def prepare_repo_snapshot(repo_root: Path, commit: str) -> Path:
    """Check out a verified commit, never a mutable working-tree fallback."""
    temp_dir = Path(tempfile.mkdtemp(prefix="beacon-eval-"))
    snapshot = temp_dir / "repo"
    try:
        resolved = subprocess.run(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{commit}^{{commit}}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        subprocess.run(
            [
                "git",
                "clone",
                "--no-hardlinks",
                "--no-checkout",
                str(repo_root.resolve()),
                str(snapshot),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "checkout", "--detach", resolved],
            cwd=snapshot,
            capture_output=True,
            text=True,
            check=True,
        )
        return snapshot
    except BaseException:
        shutil.rmtree(temp_dir)
        raise


@contextmanager
def repo_snapshot(repo_root: Path, commit: str) -> Iterator[Path]:
    """Clean up an isolated pinned checkout on success or failure."""
    snapshot = prepare_repo_snapshot(repo_root, commit)
    try:
        yield snapshot
    finally:
        shutil.rmtree(snapshot.parent)


def evaluate_patch(
    repo_root: Path,
    task: TaskRecord,
    response: RepairResponse,
    timeout: float = 60,
    required_test_ids: frozenset[str] | None = None,
) -> EvaluationResult:
    if not required_test_ids:
        return EvaluationResult(
            False, False, False, 2, "", "Original baseline test IDs are required"
        )
    applied, reason = apply_patch_text(repo_root, task, response)
    if not applied:
        return EvaluationResult(False, False, False, 1, "", reason)
    compiled, compile_output = compile_python_repo(repo_root)
    if not compiled:
        return EvaluationResult(False, False, False, 1, "", compile_output)
    result = run_pytest_tests(repo_root, task.test_command, timeout, required_test_ids)
    return EvaluationResult(
        True,
        result.passed,
        result.passed,
        result.exit_code,
        result.stdout,
        result.stderr,
        {
            "apply_reason": reason,
            "evaluated": True,
            "test_runner": "pytest-junit",
            "test_run_completed": result.completed,
            "test_ids": sorted(result.test_ids),
            "failed_test_ids": sorted(result.failed_ids),
            "completion_error": result.reason,
        },
    )
