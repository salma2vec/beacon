"""Pairwise win/loss/tie comparison for routing methods."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_results(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def adaptive_choices(results: list[dict[str, object]]) -> dict[tuple[int, str], str]:
    grouped: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for row in results:
        grouped[(int(row["budget"]), str(row["query_type"]), str(row["method"]))].append(
            1 if row["pass_at_1"] else 0
        )
    winners: dict[tuple[int, str], str] = {}
    for budget, query_type, method in grouped:
        key = (budget, query_type)
        current = sum(grouped[(budget, query_type, method)]) / len(grouped[(budget, query_type, method)])
        previous_method = winners.get(key)
        if previous_method is None:
            winners[key] = method
            continue
        previous = sum(grouped[(budget, query_type, previous_method)]) / len(
            grouped[(budget, query_type, previous_method)]
        )
        if current > previous:
            winners[key] = method
    return winners


def compare_methods(
    results: list[dict[str, object]],
    left_method: str,
    right_method: str,
) -> dict[str, int]:
    left_wins = right_wins = ties = 0
    adaptive = adaptive_choices(results)
    index = {
        (str(row["task_id"]), int(row["budget"]), str(row["query_type"]), str(row["method"])): 1
        if row["pass_at_1"]
        else 0
        for row in results
    }
    tasks = {(task_id, budget, query_type) for task_id, budget, query_type, _method in index}
    for task_id, budget, query_type in tasks:
        left = adaptive[(budget, query_type)] if left_method == "adaptive" else left_method
        right = adaptive[(budget, query_type)] if right_method == "adaptive" else right_method
        left_score = index.get((task_id, budget, query_type, left), 0)
        right_score = index.get((task_id, budget, query_type, right), 0)
        if left_score > right_score:
            left_wins += 1
        elif right_score > left_score:
            right_wins += 1
        else:
            ties += 1
    return {"left_wins": left_wins, "right_wins": right_wins, "ties": ties}


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--left-method", default="adaptive")
    parser.add_argument("--right-method", default="bca")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = compare_methods(load_results(run_dir), args.left_method, args.right_method)
    output_path = run_dir / "router_ab.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()

