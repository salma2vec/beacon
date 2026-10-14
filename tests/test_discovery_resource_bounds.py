"""Compatibility and fail-closed work bounds for source/task discovery."""

from __future__ import annotations

import ast
import itertools
import json
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import pytest

import cegraph.mutations as mutations
from cegraph.models import Node, TaskRecord, task_identifier
from cegraph.mutations import (
    MutationCandidate,
    MutationDiscoveryLimitError,
    MutationDiscoveryLimits,
    build_task_records,
    discover_mutations_in_file,
    task_record_dict,
)
from cegraph.parser import ModuleParser
from paper.experiments import discover_mutations as discovery_cli


class LegacyModuleParser(ModuleParser):
    """The pre-fix lookup/slicing behavior, for exact compatibility checks."""

    @property
    def nodes_by_id(self) -> dict[str, Node]:
        return {node.id: node for node in self.nodes}

    def _source_segment(self, node: ast.AST) -> str:
        segment = ast.get_source_segment(self.source, node)
        if segment is not None:
            return segment
        start = getattr(node, "lineno", 1) - 1
        end = getattr(node, "end_lineno", start + 1)
        return "\n".join(self.lines[start:end])


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_parser_preserves_ids_edges_and_exact_source(newline: str, tmp_path: Path) -> None:
    source = """import package as p
é = "雪"; résult = é.upper()
@décorateur(
    "é\u2028x\f雪",
)
class Café(Base):
    @wrapper(
        mode="雪"
    )
    def méthode(self, entrée: str = "é"):
        def nested(): return entrée
        return nested()

    async def awaitable(self):
        return await p.call()

def duplicate(): return 1
def duplicate(): return 2
\fdef after_form_feed(): return "雪"
""".replace("\n", newline)
    path = tmp_path / "sample.py"
    actual = ModuleParser(tmp_path, path, source)
    expected = LegacyModuleParser(tmp_path, path, source)
    assert actual.parse() == expected.parse()
    # Check nonzero byte columns too: Unicode precedes expression nodes on a line.
    for node in ast.walk(actual.tree):
        segment = ast.get_source_segment(source, node)
        if segment is not None:
            assert actual._source_segment(node) == segment


def test_symbol_registration_does_not_revisit_prior_nodes(tmp_path: Path) -> None:
    class CountingNodes(list[Node]):
        visited = 0

        def __iter__(self):
            for node in super().__iter__():
                self.visited += 1
                yield node

    count = 500
    source = "".join(f"def f{i}(): return {i}\n" for i in range(count))
    parser = ModuleParser(tmp_path, tmp_path / "sample.py", source)
    legacy = LegacyModuleParser(tmp_path, tmp_path / "sample.py", source)
    parser.nodes = CountingNodes(parser.nodes)
    legacy.nodes = CountingNodes(legacy.nodes)
    parsed = parser.parse()
    expected = legacy.parse()
    assert len(parsed.nodes) == count + 1
    assert parser.nodes.visited == 0
    assert legacy.nodes.visited == count * (count + 1) // 2
    assert parsed == expected


def legacy_candidates(path: Path, root: Path) -> list[MutationCandidate]:
    lines = path.read_text(encoding="utf-8").splitlines()
    candidates = []
    for number, line in enumerate(lines, 1):
        for kind, replacement in mutations._mutate_line(line):
            changed = lines.copy()
            changed[number - 1] = replacement
            try:
                ast.parse("\n".join(changed))
            except SyntaxError:
                continue
            candidates.append(
                MutationCandidate(
                    path.relative_to(root).as_posix(), number, kind, line, replacement
                )
            )
    return candidates


def test_discovery_preserves_exhaustive_order_and_task_ids(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    sources = {
        "z.py": "é: int = 1\r\ntext = '雪 True'\r\n",
        "pkg/a.py": "@deco(\n    True\n)\ndef f(x: int = 0):\n    return [x]\n",
        # Preserve the old behavior where a deletion can repair invalid source.
        "bad.py": "invalid syntax\nvalue = 1\n",
    }
    expected = []
    for relative, source in sources.items():
        (tmp_path / relative).write_bytes(source.encode("utf-8"))
    for relative in sorted(sources):
        path = tmp_path / relative
        candidates = legacy_candidates(path, tmp_path)
        assert discover_mutations_in_file(path, tmp_path) == candidates
        for candidate, token_budget in itertools.product(candidates, [256, 1024]):
            expected.append(
                TaskRecord(
                    repo="example",
                    commit="a" * 40,
                    bug_location=f"{relative}:{candidate.line_number}",
                    mutation_type=candidate.mutation_type,
                    original_line=candidate.original_line,
                    mutated_line=candidate.mutated_line,
                    token_budget=token_budget,
                    query_type="exact",
                    metadata={"line_number": candidate.line_number},
                )
            )
    actual = build_task_records("example", "a" * 40, tmp_path, [256, 1024])
    assert [task_identifier(task) for task in actual] == [task_identifier(t) for t in expected]
    for task, prior in zip(actual, expected, strict=True):
        record = task_record_dict(task)
        provenance = record["metadata"].pop("discovery")
        assert provenance == {"complete": True, "limits": asdict(MutationDiscoveryLimits())}
        assert record == task_record_dict(prior)


@pytest.mark.parametrize(
    ("field", "maximum", "expected_parses"),
    [
        ("max_file_bytes", 1, 0),
        ("max_total_source_bytes", 1, 0),
        ("max_syntax_checks", 1, 1),
        ("max_validation_bytes", 1, 0),
    ],
)
def test_file_work_limits_precede_additional_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, maximum: int, expected_parses: int
) -> None:
    path = tmp_path / "sample.py"
    path.write_text("x = 1\ny = 0\n")
    parse = ast.parse
    parsed = []

    def counted(source, *args, **kwargs):
        parsed.append(source)
        return parse(source, *args, **kwargs)

    monkeypatch.setattr(mutations.ast, "parse", counted)
    limits = replace(MutationDiscoveryLimits(), **{field: maximum})
    with pytest.raises(MutationDiscoveryLimitError, match=field):
        discover_mutations_in_file(path, tmp_path, limits=limits)
    assert len(parsed) == expected_parses


@pytest.mark.parametrize(
    ("field", "maximum"),
    [("max_total_source_bytes", 7), ("max_syntax_checks", 1), ("max_validation_bytes", 9)],
)
def test_repository_work_limits_are_shared_between_files(
    tmp_path: Path, field: str, maximum: int
) -> None:
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("x = 2\n")
    limits = replace(MutationDiscoveryLimits(), **{field: maximum})
    # A deletion's source is empty; for the byte-work case use a second line.
    if field == "max_validation_bytes":
        for name in ("a.py", "b.py"):
            (tmp_path / name).write_text("#abc\nx = 2\n")
    with pytest.raises(MutationDiscoveryLimitError, match=field):
        build_task_records("repo", "a" * 40, tmp_path, [256], limits=limits)


def test_entry_budget_is_checked_while_enumerating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(20):
        (tmp_path / f"unrelated{index}.txt").touch()
    scandir = mutations.os.scandir
    yielded = []

    @contextmanager
    def counted(path):
        with scandir(path) as entries:

            def iterator():
                for entry in entries:
                    yielded.append(entry.name)
                    yield entry

            yield iterator()

    monkeypatch.setattr(mutations.os, "scandir", counted)
    with pytest.raises(MutationDiscoveryLimitError, match="max_entries"):
        build_task_records(
            "repo", "a" * 40, tmp_path, [256], limits=MutationDiscoveryLimits(max_entries=3)
        )
    assert len(yielded) == 4


def test_discovery_skips_hidden_and_symlink_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("pass\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "bad.py").write_text("x = 1\n" * 100)
    (tmp_path / "link.py").symlink_to(tmp_path / "a.py")
    (tmp_path / "dirlink").symlink_to(tmp_path / ".hidden", target_is_directory=True)
    tasks = build_task_records("repo", "a" * 40, tmp_path, [256])
    assert [task.bug_location for task in tasks] == ["a.py:1"]


@pytest.mark.parametrize("field", ["max_task_records", "max_output_bytes"])
def test_output_limit_never_returns_a_task_prefix(tmp_path: Path, field: str) -> None:
    (tmp_path / "a.py").write_text("pass\n")
    limits = replace(MutationDiscoveryLimits(), **{field: 1})
    with pytest.raises(MutationDiscoveryLimitError, match=field):
        build_task_records("repo", "a" * 40, tmp_path, [256, 1024], limits=limits)


def test_output_size_accounting_matches_serialized_records(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text('é = "雪"\n')
    tasks = build_task_records("repo", "a" * 40, tmp_path, [256])
    # The limit itself is recorded, so use an equal-width value to keep the
    # serialized length stable while checking the exact accounting boundary.
    size = sum(len((json.dumps(task_record_dict(task)) + "\n").encode()) for task in tasks)
    limits = MutationDiscoveryLimits(max_output_bytes=size)
    tasks = build_task_records("repo", "a" * 40, tmp_path, [256], limits=limits)
    size = sum(len((json.dumps(task_record_dict(task)) + "\n").encode()) for task in tasks)
    assert build_task_records(
        "repo", "a" * 40, tmp_path, [256], limits=replace(limits, max_output_bytes=size)
    )
    with pytest.raises(MutationDiscoveryLimitError, match="max_output_bytes"):
        build_task_records(
            "repo", "a" * 40, tmp_path, [256], limits=replace(limits, max_output_bytes=size - 1)
        )


def test_cli_limit_failure_preserves_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "a.py").write_text("pass\n")
    output = tmp_path / "tasks.jsonl"
    output.write_text("existing completed dataset\n")
    monkeypatch.setattr(discovery_cli, "current_commit", lambda root: "a" * 40)

    @contextmanager
    def snapshot(root, commit):
        yield root

    monkeypatch.setattr(discovery_cli, "repo_snapshot", snapshot)
    with pytest.raises(SystemExit) as error:
        discovery_cli.main(
            [
                "--repo-name",
                "repo",
                "--repo-root",
                str(tmp_path),
                "--budgets",
                "256,1024",
                "--output-file",
                str(output),
                "--max-task-records",
                "1",
            ]
        )
    assert error.value.code == 2
    assert output.read_text() == "existing completed dataset\n"


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_work_limits_are_rejected(value) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        MutationDiscoveryLimits(max_syntax_checks=value)


@pytest.mark.parametrize("budgets", [[], [0], [-1], [True], [1.5]])
def test_invalid_token_budgets_are_rejected(tmp_path: Path, budgets) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        build_task_records("repo", "a" * 40, tmp_path, budgets)
