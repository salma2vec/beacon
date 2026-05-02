"""Mutation generation for benchmark task discovery."""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass
from pathlib import Path

from .models import TaskRecord


@dataclass(slots=True, frozen=True)
class MutationCandidate:
    file_path: str
    line_number: int
    mutation_type: str
    original_line: str
    mutated_line: str


WRONG_LITERALS = {
    "True": "False",
    "False": "True",
    "None": "0",
    "0": "1",
    "1": "0",
    "[]": "{}",
    "{}": "[]",
    '""': "0",
}

WRONG_TYPES = {
    "int": "str",
    "str": "int",
    "list": "dict",
    "dict": "list",
    "bool": "str",
}


def _mutate_line(line: str) -> list[tuple[str, str]]:
    stripped = line.strip()
    mutations: list[tuple[str, str]] = []
    if stripped:
        mutations.append(("deletion", ""))
    for original, replacement in WRONG_LITERALS.items():
        if original in line:
            mutations.append(("substitution", line.replace(original, replacement, 1)))
    for original, replacement in WRONG_TYPES.items():
        annotation = f": {original}"
        if annotation in line:
            mutations.append(("type_change", line.replace(annotation, f": {replacement}", 1)))
    return mutations


def syntactically_valid_after_change(source_lines: list[str], line_number: int, replacement: str) -> bool:
    mutated = source_lines.copy()
    mutated[line_number - 1] = replacement
    try:
        ast.parse("\n".join(mutated))
    except SyntaxError:
        return False
    return True


def discover_mutations_in_file(file_path: Path, repo_root: Path) -> list[MutationCandidate]:
    source = file_path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    candidates: list[MutationCandidate] = []
    for line_number, line in enumerate(source_lines, start=1):
        for mutation_type, replacement in _mutate_line(line):
            if syntactically_valid_after_change(source_lines, line_number, replacement):
                candidates.append(
                    MutationCandidate(
                        file_path=file_path.relative_to(repo_root).as_posix(),
                        line_number=line_number,
                        mutation_type=mutation_type,
                        original_line=line,
                        mutated_line=replacement,
                    )
                )
    return candidates


def build_task_records(
    repo: str,
    commit: str,
    repo_root: Path,
    budgets: list[int],
    query_type: str = "exact",
) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    for file_path in repo_root.rglob("*.py"):
        if any(part.startswith(".") for part in file_path.relative_to(repo_root).parts):
            continue
        for candidate, budget in itertools.product(
            discover_mutations_in_file(file_path, repo_root),
            budgets,
        ):
            tasks.append(
                TaskRecord(
                    repo=repo,
                    commit=commit,
                    bug_location=f"{candidate.file_path}:{candidate.line_number}",
                    mutation_type=candidate.mutation_type,
                    original_line=candidate.original_line,
                    mutated_line=candidate.mutated_line,
                    token_budget=budget,
                    query_type=query_type,
                    metadata={"line_number": candidate.line_number},
                )
            )
    return tasks

