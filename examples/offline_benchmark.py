"""Exercise the entire evaluator with a known repair and a fresh local Git fixture.

This is a harness control, not an LLM result or a retrieval quality benchmark.
Run from the checkout: python -m examples.offline_benchmark --output /tmp/beacon-control
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from cegraph.models import RepairPrompt, RepairResponse, TaskRecord
from paper.experiments.benchmark import build_parser, run_benchmark


class FixtureOracle:
    """Known correct answer for checking mutation, patching, and tests."""

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        patch = json.dumps(
            {"file_path": "calc.py", "content": "def add(a, b):\n    return a + b\n"}
        )
        return RepairResponse(patch, patch, "fixture-oracle-not-an-llm")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="beacon-control-") as directory:
        base = Path(directory)
        repo = base / "fixture"
        repo.mkdir()
        (repo / "calc.py").write_text("def add(a, b):\n    return a + b\n")
        (repo / "test_calc.py").write_text(
            "from calc import add\nassert add(2, 3) == 5\nassert add(-2, 3) == 1\n"
        )
        for command in (
            ["git", "init", "-q"],
            ["git", "add", "."],
            [
                "git",
                "-c",
                "user.name=Fixture",
                "-c",
                "user.email=fixture@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
        ):
            subprocess.run(command, cwd=repo, check=True)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        task = TaskRecord(
            "fixture",
            commit,
            "calc.py:2",
            "operator",
            "    return a + b",
            "    return a - b",
            256,
            test_command=f"{shlex.quote(sys.executable)} test_calc.py",
        )
        manifest = base / "tasks.jsonl"
        manifest.write_text(json.dumps(asdict(task)) + "\n")
        config = build_parser().parse_args(
            [
                "--tasks-file",
                str(manifest),
                "--repos-dir",
                str(base),
                "--budgets",
                "64,256",
                "--methods",
                "no_retrieval,bm25,bca",
                "--query-types",
                "vague",
                "--output-dir",
                str(args.output),
                "--allow-local-execution",
            ]
        )
        output = run_benchmark(config, FixtureOracle())
        rows = [json.loads(line) for line in output.read_text().splitlines()]
        assert all(row["pass_at_1"] for row in rows), "Harness control failed"
        print(f"{len(rows)} fixture-oracle controls passed; results: {output}")


if __name__ == "__main__":
    main()
