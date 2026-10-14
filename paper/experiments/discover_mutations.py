"""Generate JSONL task files from pinned repositories."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from cegraph.evaluation import repo_snapshot
from cegraph.mutations import (
    MutationDiscoveryLimitError,
    MutationDiscoveryLimits,
    build_task_records,
    task_record_dict,
)


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
    for name, default in asdict(MutationDiscoveryLimits()).items():
        parser.add_argument(
            "--" + name.replace("_", "-"),
            type=int,
            default=default,
            help=f"Exhaustive discovery budget (default: {default}); exceeding it fails the run",
        )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root)
    try:
        budgets = [int(part) for part in args.budgets.split(",") if part]
        limits = MutationDiscoveryLimits(
            **{name: getattr(args, name) for name in asdict(MutationDiscoveryLimits())}
        )
    except ValueError as error:
        parser.error(str(error))
    commit = current_commit(repo_root)
    with repo_snapshot(repo_root, commit) as snapshot:
        try:
            tasks = build_task_records(
                repo=args.repo_name,
                commit=commit,
                repo_root=snapshot,
                budgets=budgets,
                query_type=args.query_type,
                limits=limits,
            )
        except (MutationDiscoveryLimitError, ValueError) as error:
            parser.error(str(error))
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task_record_dict(task)) + "\n")
    return output_path


if __name__ == "__main__":
    main()
