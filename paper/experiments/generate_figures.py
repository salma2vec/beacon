"""Generate benchmark figures when matplotlib is available."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional
    plt = None


def load_results(run_dir: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def aggregate(results: list[dict[str, object]]) -> dict[tuple[str, str], list[tuple[int, float]]]:
    grouped: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for row in results:
        grouped[(str(row["method"]), str(row["query_type"]), int(row["budget"]))].append(
            1 if row["pass_at_1"] else 0
        )
    output: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (method, query_type, budget), values in grouped.items():
        output[(method, query_type)].append((budget, sum(values) / len(values)))
    for key in output:
        output[key].sort()
    return output


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    aggregated = aggregate(load_results(run_dir))
    outputs: list[Path] = []
    if plt is None:
        fallback = run_dir / "figures.tsv"
        lines = ["method\tquery_type\tbudget\tpass_at_1"]
        for (method, query_type), values in aggregated.items():
            for budget, score in values:
                lines.append(f"{method}\t{query_type}\t{budget}\t{score:.3f}")
        fallback.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return [fallback]

    by_query: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(dict)
    for (method, query_type), values in aggregated.items():
        by_query[query_type][method] = values

    for query_type, series in by_query.items():
        figure, axis = plt.subplots(figsize=(8, 5))
        for method, values in sorted(series.items()):
            axis.plot([budget for budget, _ in values], [score for _, score in values], label=method)
        axis.set_title(f"Pass@1 vs budget ({query_type})")
        axis.set_xlabel("Token budget")
        axis.set_ylabel("Pass@1")
        axis.legend(fontsize="small", ncol=2)
        output_path = run_dir / f"figure_{query_type}.png"
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        outputs.append(output_path)
    return outputs


if __name__ == "__main__":
    main()

