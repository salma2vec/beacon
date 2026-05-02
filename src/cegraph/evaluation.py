"""Patch application and benchmark evaluation."""

from __future__ import annotations

import json
import os
import py_compile
import shutil
import subprocess
import tempfile
from pathlib import Path

from .models import EvaluationResult, RepairResponse, TaskRecord


def apply_patch_text(repo_root: Path, task: TaskRecord, response: RepairResponse) -> tuple[bool, str]:
    patch_text = response.patch_text.strip()
    if not patch_text:
        return False, "empty patch"

    patch_file = repo_root / ".beacon.patch"
    patch_file.write_text(patch_text, encoding="utf-8")
    try:
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", patch_file.name],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return True, "git apply"
        if patch_text.startswith("{"):
            payload = json.loads(patch_text)
            target = repo_root / payload["file_path"]
            target.write_text(payload["content"], encoding="utf-8")
            return True, "json file replacement"
        return False, result.stderr.strip() or result.stdout.strip()
    finally:
        patch_file.unlink(missing_ok=True)


def compile_python_repo(repo_root: Path) -> tuple[bool, str]:
    failures: list[str] = []
    for path in repo_root.rglob("*.py"):
        if any(part.startswith(".") for part in path.relative_to(repo_root).parts):
            continue
        try:
            py_compile.compile(path, doraise=True)
        except py_compile.PyCompileError as error:
            failures.append(str(error))
    return (not failures, "\n".join(failures))


def run_test_command(repo_root: Path, test_command: str | None) -> tuple[int, str, str]:
    if not test_command:
        return 0, "", ""
    result = subprocess.run(
        test_command,
        cwd=repo_root,
        shell=True,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def prepare_repo_snapshot(repo_root: Path, commit: str) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="beacon-eval-"))
    snapshot = temp_dir / repo_root.name
    git_dir = repo_root / ".git"
    if git_dir.exists():
        subprocess.run(
            ["git", "clone", "--shared", str(repo_root), str(snapshot)],
            capture_output=True,
            text=True,
            check=True,
        )
        subprocess.run(
            ["git", "checkout", commit],
            cwd=snapshot,
            capture_output=True,
            text=True,
            check=True,
        )
    else:
        shutil.copytree(repo_root, snapshot)
    return snapshot


def evaluate_patch(
    repo_root: Path,
    task: TaskRecord,
    response: RepairResponse,
) -> EvaluationResult:
    applied, reason = apply_patch_text(repo_root, task, response)
    if not applied:
        return EvaluationResult(
            compiled=False,
            tests_passed=False,
            pass_at_1=False,
            exit_code=1,
            stdout="",
            stderr=reason,
        )

    compiled, compile_output = compile_python_repo(repo_root)
    if not compiled:
        return EvaluationResult(
            compiled=False,
            tests_passed=False,
            pass_at_1=False,
            exit_code=1,
            stdout="",
            stderr=compile_output,
        )

    exit_code, stdout, stderr = run_test_command(repo_root, task.test_command)
    tests_passed = exit_code == 0
    return EvaluationResult(
        compiled=True,
        tests_passed=tests_passed,
        pass_at_1=compiled and tests_passed,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        metadata={"apply_reason": reason, "cwd": os.fspath(repo_root)},
    )
