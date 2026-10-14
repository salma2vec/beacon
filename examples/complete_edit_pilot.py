"""Complete only the unattempted cell of an interrupted frozen edit pilot.

Never retry a recorded failure. Original evidence remains immutable. This small
recovery executor verifies every original source hash before reconstruction.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from cegraph import CodebaseGraph, assemble_context
from cegraph.edits import EDIT_SCHEMA, EDIT_SYSTEM, ExactEditRepairClient
from cegraph.evaluation import apply_mutation, repo_snapshot, run_pytest_tests
from cegraph.llm import OllamaRepairClient, OllamaResponseError
from cegraph.models import RepairPrompt, task_identifier
from examples.edit_format_pilot import (
    CHALLENGE_PATH,
    CHALLENGES,
    MODELS,
    append_row,
    digest_bytes,
    migrate_test_hook,
    score_copy,
)
from examples.public_repo_benchmark import MANIFEST, ReferencePatchClient, qwen_counter
from paper.experiments.benchmark import build_prompt, build_query, load_tasks


def missing_cell(rows: list[dict], task_ids: list[str]) -> tuple[str, str, int, str]:
    expected = {
        (task, method, budget, fmt)
        for task in task_ids
        for method in ("bm25", "bca")
        for budget in (256, 1024)
        for fmt in ("diff", "exact_edit")
    }
    if len(task_ids) != 3 or len(expected) != 24 or len(rows) != 23:
        raise ValueError("Recovery requires exactly 23 recorded cells of the 24-cell pilot")
    observed = [(r["task_id"], r["method"], r["budget"], r["format"]) for r in rows]
    if len(set(observed)) != 23 or not set(observed) <= expected:
        raise ValueError("Duplicate or out-of-protocol cells cannot be recovered")
    if any(r["status"] not in {"completed", "error"} for r in rows):
        raise ValueError("Recorded cells must have an explicit terminal outcome")
    return (expected - set(observed)).pop()


def run(repo: Path, interrupted: Path, output: Path, tokenizer: Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("Freeze recovery executor before inference")
    if output.exists() or (interrupted / "summary.json").exists():
        raise ValueError("Use fresh output and an incomplete original pilot")
    protocol = json.loads((interrupted / "protocol.json").read_text())
    for path, expected in protocol["source_sha256"].items():
        if digest_bytes((root / path).read_bytes()) != expected:
            raise ValueError(f"Frozen experiment source changed: {path}")
    if digest_bytes(MANIFEST.read_bytes()) != protocol["manifest_sha256"]:
        raise ValueError("Frozen manifest changed")
    model = protocol["model"]
    if model not in MODELS or protocol["expected_digest"] != MODELS[model]:
        raise ValueError("Unexpected pinned model")
    if protocol["controls_only"] or any(
        protocol[k] != v
        for k, v in {
            "seed": 42,
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 512,
            "timeout_seconds": 60,
            "maximum_attempts": 24,
            "retries": 0,
        }.items()
    ):
        raise ValueError("Recovery only supports the unchanged frozen pilot parameters")
    rows = [json.loads(line) for line in (interrupted / "results.jsonl").read_text().splitlines()]
    tasks = load_tasks(MANIFEST)
    task_id, method, budget, fmt = missing_cell(rows, [task_identifier(t) for t in tasks])
    original = next(t for t in tasks if task_identifier(t) == task_id)
    task = replace(original, test_command=f"{original.test_command} {CHALLENGE_PATH}")
    target = task.bug_location.rsplit(":", 1)[0]
    output.mkdir(parents=True)
    provenance = {
        "kind": "one_unattempted_cell_recovery",
        "retries": 0,
        "reason": "Original run stopped before cell24 under its 2GiB disk-headroom gate.",
        "parent_protocol_sha256": digest_bytes((interrupted / "protocol.json").read_bytes()),
        "parent_results_sha256": digest_bytes((interrupted / "results.jsonl").read_bytes()),
        "original_source_head": protocol["source_head"],
        "recovery_source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "executor_sha256": digest_bytes(Path(__file__).read_bytes()),
        "cell": [task_id, method, budget, fmt],
        "complete": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    counter = qwen_counter(tokenizer)
    with repo_snapshot(repo, task.commit) as snapshot:
        migrate_test_hook(snapshot)
        challenge = snapshot / CHALLENGE_PATH
        challenge.write_text(CHALLENGES[target])
        baseline = run_pytest_tests(snapshot, task.test_command)
        apply_mutation(snapshot, task)
        mutant = run_pytest_tests(snapshot, task.test_command, required_test_ids=baseline.test_ids)
        challenge.unlink()
        reference = ReferencePatchClient(repo, original).generate_patch(RepairPrompt("", ""))
        restoration = score_copy(snapshot, task, reference, baseline.test_ids)
        control = {
            "task_id": task_id,
            "baseline": baseline.passed,
            "test_ids": sorted(baseline.test_ids),
            "mutant_completed": mutant.completed,
            "mutant_failed_ids": sorted(mutant.failed_ids),
            "restoration": asdict(restoration),
        }
        append_row(output / "controls.jsonl", control)
        old_controls = [
            json.loads(x) for x in (interrupted / "controls.jsonl").read_text().splitlines()
        ]
        previous = next(c for c in old_controls if c["task_id"] == task_id)
        if not (
            baseline.passed
            and mutant.completed
            and mutant.failed_ids
            and restoration.pass_at_1
            and sorted(baseline.test_ids) == previous["baseline_test_ids"]
        ):
            raise ValueError("Recovered task's scoring controls differ or fail")
        graph = CodebaseGraph.from_repo(snapshot)
        query = build_query(task, "dev_report")
        assembled = assemble_context(graph, query, method, budget)
        prompt = build_prompt(task, query, assembled.context.render())
        if fmt == "exact_edit":
            prompt = RepairPrompt(EDIT_SYSTEM, prompt.user)
        row = {
            "task_id": task_id,
            "method": method,
            "budget": budget,
            "format": fmt,
            "context_tokens": assembled.context.total_tokens,
            "selected_node_ids": assembled.selected_node_ids,
            "prompt": asdict(prompt),
            "prompt_sha256": digest_bytes(json.dumps(asdict(prompt), sort_keys=True).encode()),
            "pass_at_1": False,
            "status": "error",
        }
        if shutil.disk_usage(output).free < 2 * 1024**3:
            raise ValueError("Recovery retains the 2GiB disk-headroom gate")
        transport = OllamaRepairClient(
            model,
            timeout=60,
            prompt_token_counter=counter,
            response_schema=EDIT_SCHEMA if fmt == "exact_edit" else None,
        )
        if transport.digest != MODELS[model]:
            raise ValueError("Pinned model digest differs")
        try:
            client = (
                transport
                if fmt == "diff"
                else ExactEditRepairClient(transport, {target: (snapshot / target).read_text()})
            )
            response = client.generate_patch(prompt)
            row["response"] = asdict(response)
            evaluation = score_copy(snapshot, task, response, baseline.test_ids)
            row.update(
                evaluation=asdict(evaluation), pass_at_1=evaluation.pass_at_1, status="completed"
            )
        except (OSError, ValueError) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, OllamaResponseError):
                row["rejected_provider_response"] = exc.response
        append_row(output / "results.jsonl", row)
    combined = [*rows, row]
    summary = {
        "planned_cells": 24,
        "attempted_cells": len(combined),
        "recovered_cells": 1,
        "formats": {
            fmt: {
                "attempts": sum(r["format"] == fmt for r in combined),
                "passes": sum(r["format"] == fmt and r["pass_at_1"] for r in combined),
                "errors": sum(r["format"] == fmt and r["status"] == "error" for r in combined),
            }
            for fmt in ("diff", "exact_edit")
        },
        "complete": True,
        "limits": (
            "Exposed-task diagnosis across two process windows; no retrieval superiority claim."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    provenance["complete"] = True
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "interrupted", "output", "tokenizer"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--allow-local-execution", action="store_true")
    args = parser.parse_args()
    if not args.allow_local_execution:
        parser.error("Explicit local execution acknowledgement is required")
    print(json.dumps(run(args.repo, args.interrupted, args.output, args.tokenizer), indent=2))
