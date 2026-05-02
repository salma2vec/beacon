"""Posthoc adaptive-vs-fixed router analysis."""

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


def summarize_router_study(results: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    overall: dict[str, list[int]] = defaultdict(list)
    by_task: dict[tuple[str, int, str, str], int] = {}
    for row in results:
        method = str(row["method"])
        budget = int(row["budget"])
        query_type = str(row["query_type"])
        score = 1 if row["pass_at_1"] else 0
        grouped[(budget, query_type, method)].append(score)
        overall[method].append(score)
        by_task[(str(row["task_id"]), budget, query_type, method)] = score

    best_fixed = max(overall.items(), key=lambda item: sum(item[1]) / len(item[1]))[0]
    adaptive_by_cell: dict[tuple[int, str], str] = {}
    for budget, query_type, method in grouped:
        current_score = sum(grouped[(budget, query_type, method)]) / len(grouped[(budget, query_type, method)])
        key = (budget, query_type)
        previous = adaptive_by_cell.get(key)
        if previous is None:
            adaptive_by_cell[key] = method
            continue
        previous_score = sum(grouped[(budget, query_type, previous)]) / len(grouped[(budget, query_type, previous)])
        if current_score > previous_score:
            adaptive_by_cell[key] = method

    adaptive_scores: list[int] = []
    fixed_scores: list[int] = []
    for task_id, budget, query_type, _method in sorted(by_task):
        adaptive_method = adaptive_by_cell[(budget, query_type)]
        adaptive_scores.append(by_task[(task_id, budget, query_type, adaptive_method)])
        fixed_scores.append(by_task[(task_id, budget, query_type, best_fixed)])

    adaptive_rate = sum(adaptive_scores) / len(adaptive_scores) if adaptive_scores else 0.0
    fixed_rate = sum(fixed_scores) / len(fixed_scores) if fixed_scores else 0.0
    return {
        "best_fixed_method": best_fixed,
        "adaptive_pass_rate": adaptive_rate,
        "fixed_pass_rate": fixed_rate,
        "relative_improvement": adaptive_rate - fixed_rate,
        "adaptive_by_cell": {f"{budget}:{query_type}": method for (budget, query_type), method in adaptive_by_cell.items()},
    }


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    summary = summarize_router_study(load_results(run_dir))
    output_path = run_dir / "posthoc_analysis.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_path


if __name__ == "__main__":
    main()

