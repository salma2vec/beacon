"""Benchmark runner for beacon."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import TextIO

from cegraph import CodebaseGraph, assemble_context, available_methods
from cegraph.evaluation import apply_mutation, evaluate_patch, repo_snapshot, run_pytest_tests
from cegraph.llm import EchoRepairClient, OllamaRepairClient, OpenAIRepairClient, RepairClient
from cegraph.models import BenchmarkResult, RepairPrompt, TaskRecord, task_identifier


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def load_tasks(path: Path) -> list[TaskRecord]:
    tasks: list[TaskRecord] = []
    seen: dict[str, TaskRecord] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task = TaskRecord(**payload)
        identifier = task_identifier(task)
        normalized = replace(task, token_budget=0)
        if identifier in seen:
            if seen[identifier] != normalized:
                raise ValueError("Conflicting task metadata for the same mutation")
            continue
        seen[identifier] = normalized
        tasks.append(task)
    return tasks


def build_query(task: TaskRecord, query_type: str) -> str:
    if query_type == "exact":
        return (
            f"Fix the bug at {task.bug_location}. "
            f"Original line: {task.original_line}. Mutated line: {task.mutated_line}."
        )
    if query_type == "vague":
        return (
            f"There is a {task.mutation_type} bug near "
            f"{task.bug_location.split(':', maxsplit=1)[0]}. "
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


def get_repair_client(dry_run: bool, model: str, provider: str = "openai") -> RepairClient:
    if dry_run:
        return EchoRepairClient()
    if provider == "ollama":
        return OllamaRepairClient(model=model)
    return OpenAIRepairClient(model=model)


def create_results_file(path: Path) -> TextIO:
    """Exclusively create a single-link regular result file without following links."""
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            raise ValueError("results.jsonl must be a newly created single-link regular file")
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        return handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)
            path.unlink(missing_ok=True)


def run_benchmark(args: argparse.Namespace, repair_client: RepairClient | None = None) -> Path:
    tasks = load_tasks(Path(args.tasks_file))
    if not tasks:
        raise ValueError(
            "Task manifest is empty; generate the offline fixture or supply validated tasks"
        )
    methods = parse_csv_strings(args.methods)
    if not methods or set(methods) - set(available_methods()):
        raise ValueError("Specify at least one supported retrieval method")
    budgets = parse_csv_ints(args.budgets)
    if not budgets or any(b < 0 for b in budgets):
        raise ValueError("budgets must be non-negative integers")
    query_types = parse_csv_strings(args.query_types)
    if not query_types or set(query_types) - {"exact", "vague", "dev_report"}:
        raise ValueError("Unsupported query type")
    scored = not (args.skip_eval or args.dry_run)
    if scored and not args.allow_local_execution:
        raise ValueError(
            "Scoring executes source code locally; use --allow-local-execution "
            "only for trusted repositories"
        )
    if scored and any(not task.test_command for task in tasks):
        raise ValueError("Every scored task needs a behavioral test_command")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    repair_client = repair_client or get_repair_client(args.dry_run, args.model, args.provider)
    base = Path(args.repos_dir).resolve()
    with create_results_file(results_path) as handle:
        for task in tasks:
            repo_root = (base / task.repo).resolve()
            if not repo_root.is_relative_to(base) or repo_root == base:
                raise ValueError("task.repo must point inside repos_dir")
            with repo_snapshot(repo_root, task.commit) as snapshot:
                resolved_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=snapshot, text=True
                ).strip()
                if scored:
                    baseline = run_pytest_tests(snapshot, task.test_command, args.test_timeout)
                    if not baseline.passed:
                        raise ValueError(
                            f"Original baseline failed for {task_identifier(task)}: "
                            f"{baseline.stderr}"
                        )
                apply_mutation(snapshot, task)
                if scored:
                    mutant = run_pytest_tests(
                        snapshot, task.test_command, args.test_timeout, baseline.test_ids
                    )
                    if not mutant.completed or not mutant.failed_ids or mutant.exit_code != 1:
                        raise ValueError(
                            "Mutation must complete the original pytest tests "
                            "with assertion failures; "
                            f"{mutant.reason}"
                        )
                graph = CodebaseGraph.from_repo(snapshot)
                for query_type in query_types:
                    query = build_query(task, query_type)
                    for budget in budgets:
                        for method in methods:
                            oracle_location = args.oracle_location or method == "target_file"
                            assembled = assemble_context(
                                graph,
                                query,
                                method,
                                budget,
                                task.bug_location if oracle_location else None,
                                args.alpha,
                            )
                            prompt = build_prompt(task, query, assembled.context.render())
                            response = repair_client.generate_patch(prompt)
                            compiled = tests_passed = pass_at_1 = None
                            metadata = {
                                "mode": "dry_run" if args.dry_run else "skip_eval",
                                "evaluated": False,
                            }
                            if scored:
                                trial = snapshot.parent / "trial"
                                shutil.copytree(
                                    snapshot,
                                    trial,
                                    symlinks=True,
                                    ignore=shutil.ignore_patterns("__pycache__"),
                                )
                                try:
                                    evaluation = evaluate_patch(
                                        trial, task, response, args.test_timeout, baseline.test_ids
                                    )
                                finally:
                                    shutil.rmtree(trial)
                                compiled, tests_passed, pass_at_1 = (
                                    evaluation.compiled,
                                    evaluation.tests_passed,
                                    evaluation.pass_at_1,
                                )
                                metadata = {
                                    **evaluation.metadata,
                                    "evaluated": True,
                                    "mode": "scored",
                                    "exit_code": evaluation.exit_code,
                                    "stdout": evaluation.stdout,
                                    "stderr": evaluation.stderr,
                                    "baseline_exit_code": baseline.exit_code,
                                    "baseline_test_ids": sorted(baseline.test_ids),
                                    "mutant_exit_code": mutant.exit_code,
                                    "mutant_failed_test_ids": sorted(mutant.failed_ids),
                                }
                            result = BenchmarkResult(
                                repo=task.repo,
                                commit=resolved_commit,
                                task_id=task_identifier(task),
                                method=method,
                                budget=budget,
                                query_type=query_type,
                                pass_at_1=pass_at_1,
                                compiled=compiled,
                                tests_passed=tests_passed,
                                context_tokens=assembled.context.total_tokens,
                                selected_node_ids=assembled.selected_node_ids,
                                metadata={
                                    **metadata,
                                    "bug_location": task.bug_location,
                                    "mutation_type": task.mutation_type,
                                    "oracle_location": oracle_location,
                                    "oracle_query": query_type == "exact"
                                    or (query_type == "dev_report" and not task.prompt),
                                    "model": response.model,
                                    "model_metadata": response.metadata,
                                    "raw_response": response.raw_response,
                                    "alpha": args.alpha,
                                    "token_codec": "cl100k_base",
                                    "test_timeout_seconds": args.test_timeout,
                                    "prompt": asdict(prompt),
                                    "repair_patch": response.patch_text,
                                    "task_sha256": hashlib.sha256(
                                        json.dumps(asdict(task), sort_keys=True).encode()
                                    ).hexdigest(),
                                    "prompt_sha256": hashlib.sha256(
                                        json.dumps(asdict(prompt), sort_keys=True).encode()
                                    ).hexdigest(),
                                    "response_sha256": hashlib.sha256(
                                        response.patch_text.encode()
                                    ).hexdigest(),
                                },
                            )
                            handle.write(json.dumps(result.to_json()) + "\n")
                            handle.flush()
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
    parser.add_argument("--provider", choices=("openai", "ollama"), default="openai")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--allow-local-execution", action="store_true")
    parser.add_argument("--oracle-location", action="store_true")
    parser.add_argument("--test-timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_benchmark(args)


if __name__ == "__main__":
    main()
