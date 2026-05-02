"""Benchmark runner for beacon."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cegraph import CodebaseGraph, assemble_context, available_methods
from cegraph.evaluation import evaluate_patch, prepare_repo_snapshot
from cegraph.llm import EchoRepairClient, OpenAIRepairClient, RepairClient
from cegraph.models import BenchmarkResult, RepairPrompt, TaskRecord, task_identifier


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_tasks(path: Path) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        tasks.append(TaskRecord(**payload))
    return tasks


def build_query(task: TaskRecord, query_type: str) -> str:
    if query_type == "exact":
        return (
            f"Fix the bug at {task.bug_location}. "
            f"Original line: {task.original_line}. Mutated line: {task.mutated_line}."
        )
    if query_type == "vague":
        return (
            f"There is a {task.mutation_type} bug near {task.bug_location.split(':', maxsplit=1)[0]}. "
            "Find the relevant context and repair it."
        )
    if query_type == "dev_report":
        return task.prompt or (
            f"Regression report: behavior changed around {task.bug_location}. "
            f"Expected `{task.original_line}` but saw `{task.mutated_line}`."
        )
    raise ValueError(f"Unknown query type: {query_type}")


def build_prompt(task: TaskRecord, query: str, context_text: str) -> RepairPrompt:
    system = (
        "You repair Python code. Return a unified diff patch that fixes the bug while changing as "
        "little as possible."
    )
    user = f"{query}\n\nContext:\n{context_text}\n"
    return RepairPrompt(system=system, user=user)


def get_repair_client(dry_run: bool, model: str) -> RepairClient:
    if dry_run:
        return EchoRepairClient()
    return OpenAIRepairClient(model=model)


def run_benchmark(args: argparse.Namespace) -> Path:
    tasks = load_tasks(Path(args.tasks_file))
    methods = parse_csv_strings(args.methods)
    invalid = sorted(set(methods) - set(available_methods()))
    if invalid:
        raise ValueError(f"Unsupported methods requested: {', '.join(invalid)}")
    budgets = parse_csv_ints(args.budgets)
    query_types = parse_csv_strings(args.query_types)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    repair_client = get_repair_client(args.dry_run, args.model)

    graph_cache: dict[tuple[str, str], CodebaseGraph] = {}
    with results_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            repo_root = Path(args.repos_dir) / task.repo
            graph_key = (task.repo, task.commit)
            if graph_key not in graph_cache:
                graph_cache[graph_key] = CodebaseGraph.from_repo(repo_root)
            graph = graph_cache[graph_key]
            task_id = task_identifier(task)
            for query_type in query_types:
                query = build_query(task, query_type)
                for budget in budgets:
                    for method in methods:
                        assembled = assemble_context(
                            graph=graph,
                            query=query,
                            method=method,
                            budget=budget,
                            bug_location=task.bug_location,
                            alpha=args.alpha,
                        )
                        prompt = build_prompt(task, query, assembled.assembled_context.render())
                        response = repair_client.generate_patch(prompt)
                        if args.skip_eval:
                            compiled = args.dry_run
                            tests_passed = args.dry_run
                            pass_at_1 = args.dry_run
                            metadata = {"mode": "skip_eval"}
                        else:
                            snapshot = prepare_repo_snapshot(repo_root, task.commit)
                            evaluation = evaluate_patch(snapshot, task, response)
                            compiled = evaluation.compiled
                            tests_passed = evaluation.tests_passed
                            pass_at_1 = evaluation.pass_at_1
                            metadata = evaluation.metadata
                        result = BenchmarkResult(
                            repo=task.repo,
                            commit=task.commit,
                            task_id=task_id,
                            method=method,
                            budget=budget,
                            query_type=query_type,
                            pass_at_1=pass_at_1,
                            compiled=compiled,
                            tests_passed=tests_passed,
                            context_tokens=assembled.assembled_context.total_tokens,
                            selected_node_ids=assembled.selected_node_ids,
                            metadata={
                                **metadata,
                                "bug_location": task.bug_location,
                                "mutation_type": task.mutation_type,
                            },
                        )
                        handle.write(json.dumps(result.to_json()) + "\n")
    return results_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-file", required=True)
    parser.add_argument("--repos-dir", required=True)
    parser.add_argument("--budgets", required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--query-types", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    main()

