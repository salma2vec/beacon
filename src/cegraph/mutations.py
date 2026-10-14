"""Mutation generation for benchmark task discovery."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .models import TaskRecord


class MutationDiscoveryLimitError(ValueError):
    """The requested exhaustive discovery cannot complete within its budget."""


@dataclass(slots=True, frozen=True)
class MutationDiscoveryLimits:
    """Explicit resource bounds, not sampling/truncation thresholds.

    Syntax validation still parses a complete candidate file. The byte-work
    limit bounds the sum of those inputs, rather than merely the file count.
    Increase limits deliberately for larger corpora and retain them with results.
    """

    max_entries: int = 10_000
    max_file_bytes: int = 256 * 1024
    max_total_source_bytes: int = 8 * 1024 * 1024
    max_syntax_checks: int = 10_000
    max_validation_bytes: int = 64 * 1024 * 1024
    max_task_records: int = 10_000
    max_output_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(slots=True)
class _DiscoveryBudget:
    limits: MutationDiscoveryLimits
    used: dict[str, int] = field(default_factory=dict)

    def reserve(self, name: str, amount: int = 1) -> None:
        total = self.used.get(name, 0) + amount
        maximum = getattr(self.limits, name)
        if total > maximum:
            raise MutationDiscoveryLimitError(
                f"Mutation discovery exceeded {name}={maximum}; "
                "no complete enumeration was produced. Increase the explicit limit "
                "or select a smaller corpus; do not treat a prefix as a complete dataset."
            )
        self.used[name] = total


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


def syntactically_valid_after_change(
    source_lines: list[str], line_number: int, replacement: str
) -> bool:
    mutated = source_lines.copy()
    mutated[line_number - 1] = replacement
    try:
        ast.parse("\n".join(mutated))
    except SyntaxError:
        return False
    return True


def _python_files(repo_root: Path, budget: _DiscoveryBudget) -> list[Path]:
    # rglob can eagerly materialize wide directories before the caller can count
    # their entries. Count every encountered entry through scandir instead.
    pending = [repo_root]
    files: list[Path] = []
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                budget.reserve("max_entries")
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                path = Path(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False) and path.suffix == ".py":
                    files.append(path)
    return sorted(files)


def _discover_mutations_in_file(
    file_path: Path, repo_root: Path, budget: _DiscoveryBudget
) -> list[MutationCandidate]:
    if file_path.is_symlink() or not file_path.resolve().is_relative_to(repo_root.resolve()):
        raise ValueError("Source file must be contained in the repository without symlinks")
    with file_path.open("rb") as handle:
        raw = handle.read(budget.limits.max_file_bytes + 1)
    if len(raw) > budget.limits.max_file_bytes:
        # File size is a per-file bound, unlike the cumulative counters.
        raise MutationDiscoveryLimitError(
            f"Mutation discovery exceeded max_file_bytes={budget.limits.max_file_bytes}; "
            "no complete enumeration was produced."
        )
    budget.reserve("max_total_source_bytes", len(raw))
    source = raw.decode("utf-8")
    source_lines = source.splitlines()
    normalized = "\n".join(source_lines)
    normalized_bytes = len(normalized.encode("utf-8"))
    offset = 0
    candidates: list[MutationCandidate] = []
    for line_number, line in enumerate(source_lines, start=1):
        line_bytes = len(line.encode("utf-8"))
        for mutation_type, replacement in _mutate_line(line):
            budget.reserve("max_syntax_checks")
            budget.reserve(
                "max_validation_bytes",
                normalized_bytes - line_bytes + len(replacement.encode("utf-8")),
            )
            # Offsets here index Python strings, unlike AST's byte columns.
            mutated = normalized[:offset] + replacement + normalized[offset + len(line) :]
            try:
                ast.parse(mutated)
            except SyntaxError:
                continue
            candidates.append(
                MutationCandidate(
                    file_path=file_path.relative_to(repo_root).as_posix(),
                    line_number=line_number,
                    mutation_type=mutation_type,
                    original_line=line,
                    mutated_line=replacement,
                )
            )
        offset += len(line) + 1
    return candidates


def discover_mutations_in_file(
    file_path: Path, repo_root: Path, *, limits: MutationDiscoveryLimits | None = None
) -> list[MutationCandidate]:
    """Return all valid candidates, or raise rather than return a truncated list."""
    return _discover_mutations_in_file(
        file_path, repo_root, _DiscoveryBudget(limits or MutationDiscoveryLimits())
    )


def task_record_dict(task: TaskRecord) -> dict[str, Any]:
    """The discovery JSONL schema, shared with serialized-output accounting."""
    return {
        "repo": task.repo,
        "commit": task.commit,
        "bug_location": task.bug_location,
        "mutation_type": task.mutation_type,
        "original_line": task.original_line,
        "mutated_line": task.mutated_line,
        "token_budget": task.token_budget,
        "query_type": task.query_type,
        "metadata": task.metadata,
    }


def build_task_records(
    repo: str,
    commit: str,
    repo_root: Path,
    budgets: list[int],
    query_type: str = "exact",
    *,
    limits: MutationDiscoveryLimits | None = None,
) -> list[TaskRecord]:
    """Exhaustively enumerate within explicit limits; never return partial tasks.

    Limits are saved in every record's metadata. No output is returned until all
    files have completed, so a budget failure cannot masquerade as a full corpus.
    Task identifiers do not depend on this additional provenance metadata.
    """
    if not budgets or any(type(value) is not int or value <= 0 for value in budgets):
        raise ValueError("Token budgets must be a nonempty list of positive integers")
    work = _DiscoveryBudget(limits or MutationDiscoveryLimits())
    provenance = {"complete": True, "limits": asdict(work.limits)}
    tasks: list[TaskRecord] = []
    for file_path in _python_files(repo_root, work):
        for candidate in _discover_mutations_in_file(file_path, repo_root, work):
            for budget in budgets:
                work.reserve("max_task_records")
                task = TaskRecord(
                    repo=repo,
                    commit=commit,
                    bug_location=f"{candidate.file_path}:{candidate.line_number}",
                    mutation_type=candidate.mutation_type,
                    original_line=candidate.original_line,
                    mutated_line=candidate.mutated_line,
                    token_budget=budget,
                    query_type=query_type,
                    metadata={"line_number": candidate.line_number, "discovery": provenance},
                )
                work.reserve(
                    "max_output_bytes",
                    len((json.dumps(task_record_dict(task)) + "\n").encode("utf-8")),
                )
                tasks.append(task)
    return tasks
