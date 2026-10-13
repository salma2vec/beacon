from __future__ import annotations

import json
import subprocess
from pathlib import Path

from paper.experiments.benchmark import main as benchmark_main
from paper.experiments.merge_and_report import main as report_main
from paper.experiments.posthoc_analysis import main as posthoc_main
from paper.experiments.router_ab import main as router_main


def build_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path / "demo_repo"
    repo_root.mkdir()
    (repo_root / "demo.py").write_text(
        """
def helper():
    return 1

def broken():
    return helper()
""".strip(),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "init",
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tasks_file = tmp_path / "tasks.jsonl"
    tasks_file.write_text(
        json.dumps(
            {
                "repo": "demo_repo",
                "commit": commit,
                "bug_location": "demo.py:5",
                "mutation_type": "substitution",
                "original_line": "    return helper()",
                "mutated_line": "    return unknown()",
                "token_budget": 128,
                "query_type": "exact",
                "test_command": "python -m py_compile demo.py",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return tasks_file


def test_benchmark_cli_and_reports(tmp_path: Path) -> None:
    tasks_file = build_repo(tmp_path)
    output_dir = tmp_path / "results"
    results_path = benchmark_main(
        [
            "--tasks-file",
            str(tasks_file),
            "--repos-dir",
            str(tmp_path),
            "--budgets",
            "64",
            "--methods",
            "no_retrieval,bca",
            "--query-types",
            "exact,vague",
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--skip-eval",
        ]
    )
    assert results_path.exists()
    assert all(
        row["pass_at_1"] is None for row in map(json.loads, results_path.read_text().splitlines())
    )
    assert results_path.read_text(encoding="utf-8").count("\n") == 4

    report_paths = report_main(["--run-dir", str(output_dir)])
    posthoc_path = posthoc_main(["--run-dir", str(output_dir)])
    router_path = router_main(["--run-dir", str(output_dir)])
    assert report_paths["markdown"].exists()
    assert report_paths["latex"].exists()
    assert posthoc_path.exists()
    assert router_path.exists()
