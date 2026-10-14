"""Aggregate benchmark results into markdown and LaTeX tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

if __package__:
    from .safe_output import write_text_atomic
else:  # Retain direct script usage as well as `python -m`.
    from safe_output import write_text_atomic


def load_results(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate_pass_rates(results: list[dict[str, object]]) -> dict[tuple[str, int, str], float]:
    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row in results:
        if row["pass_at_1"] is None:
            continue
        key = (str(row["method"]), int(row["budget"]), str(row["query_type"]))
        grouped[key].append(1 if row["pass_at_1"] else 0)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def build_markdown_table(aggregates: dict[tuple[str, int, str], float]) -> str:
    rows = ["| method | budget | query_type | pass@1 |", "|---|---:|---|---:|"]
    for key in sorted(aggregates):
        method, budget, query_type = key
        rows.append(f"| {method} | {budget} | {query_type} | {aggregates[key]:.3f} |")
    return "\n".join(rows)


def build_latex_table(aggregates: dict[tuple[str, int, str], float]) -> str:
    rows = ["\\begin{tabular}{lllr}", "Method & Budget & Query & Pass@1\\\\", "\\hline"]
    for key in sorted(aggregates):
        method, budget, query_type = key
        rows.append(f"{method} & {budget} & {query_type} & {aggregates[key]:.3f}\\\\")
    rows.append("\\end{tabular}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> dict[str, Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    aggregates = aggregate_pass_rates(load_results(run_dir))
    markdown = build_markdown_table(aggregates)
    latex = build_latex_table(aggregates)
    markdown_path = run_dir / "report.md"
    latex_path = run_dir / "report.tex"
    write_text_atomic(markdown_path, markdown + "\n")
    write_text_atomic(latex_path, latex + "\n")
    return {"markdown": markdown_path, "latex": latex_path}


if __name__ == "__main__":
    main()
