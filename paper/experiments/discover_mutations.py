"""Generate JSONL task files from pinned repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from cegraph.evaluation import repo_snapshot
from cegraph.mutations import build_task_records


def current_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-name", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--query-type", default="exact")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    budgets = [int(part) for part in args.budgets.split(",") if part]
    commit = current_commit(repo_root)
    with repo_snapshot(repo_root, commit) as snapshot:
        tasks = build_task_records(
            repo=args.repo_name,
            commit=commit,
            repo_root=snapshot,
            budgets=budgets,
            query_type=args.query_type,
        )
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(
                json.dumps(
                    {
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
                )
                + "\n"
            )
    return output_path


if __name__ == "__main__":
    main()
