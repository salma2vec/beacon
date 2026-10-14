"""Frozen exposed-task diagnosis for compact context and visible-path edits.

This is not a heldout benchmark. It uses the three previously exposed Boltons
mutations and compares the existing BM25 rendering with the new grounded stack
at one strict 256-token budget. Generated code executes only with the explicit
local-execution acknowledgement; this is not a sandbox.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from cegraph import CodebaseGraph, assemble_context, grounded_edit_contract
from cegraph.edits import ExactEditRepairClient
from cegraph.evaluation import apply_mutation, evaluate_patch, repo_snapshot, run_pytest_tests
from cegraph.llm import OllamaRepairClient, OllamaResponseError
from cegraph.models import RepairPrompt, TaskRecord, task_identifier
from examples.edit_format_pilot import (
    CHALLENGE_PATH,
    CHALLENGES,
    HOOK_HASH,
    migrate_test_hook,
)
from examples.public_repo_benchmark import MANIFEST, ReferencePatchClient, qwen_counter
from paper.experiments.benchmark import build_prompt, build_query, load_tasks

MODEL = "qwen2.5-coder:1.5b"
MODEL_DIGEST = "d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668"
METHODS = ("bm25", "grounded_bm25")
BUDGET = 256


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def score_copy(
    snapshot: Path,
    task: TaskRecord,
    response: Any,
    test_ids: frozenset[str],
) -> Any:
    trial = snapshot.parent / "grounded-edit-trial"
    shutil.copytree(snapshot, trial, symlinks=True, ignore=shutil.ignore_patterns("__pycache__"))
    try:
        target = task.bug_location.rsplit(":", 1)[0]
        (trial / CHALLENGE_PATH).write_text(CHALLENGES[target], encoding="utf-8")
        return evaluate_patch(trial, task, response, 60, test_ids)
    finally:
        shutil.rmtree(trial)


def run(
    repo: Path,
    output: Path,
    tokenizer: Path | None,
    controls_only: bool,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if not controls_only and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip():
        raise ValueError("Commit protocol and source before model generation")
    if output.exists():
        raise ValueError("Output directory must not exist; keep previous evidence")
    output.mkdir(parents=True)
    counter = qwen_counter(tokenizer) if tokenizer else None
    if not controls_only and counter is None:
        raise ValueError("Model trials require the pinned native tokenizer")

    source_files = [
        "examples/grounded_edit_pilot.py",
        "examples/edit_format_pilot.py",
        "examples/public_repo_benchmark.py",
        "src/cegraph/edits.py",
        "src/cegraph/llm.py",
        "src/cegraph/evaluation.py",
        "src/cegraph/methods.py",
        "src/cegraph/graph.py",
        "src/cegraph/parser.py",
        "src/cegraph/index.py",
        "src/cegraph/models.py",
        "src/cegraph/tokens.py",
        "paper/experiments/benchmark.py",
    ]
    tasks = load_tasks(MANIFEST)
    protocol = {
        "kind": "exposed_task_grounded_edit_diagnosis",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {path: digest_bytes((root / path).read_bytes()) for path in source_files},
        "manifest_sha256": digest_bytes(MANIFEST.read_bytes()),
        "python": platform.python_version(),
        "versions": {
            **{package: version(package) for package in ("beacon", "pytest", "tiktoken")},
            "tokenizers": version("tokenizers") if counter else None,
        },
        "model": MODEL,
        "expected_digest": MODEL_DIGEST,
        "methods": list(METHODS),
        "budget": BUDGET,
        "states": ["mutant", "clean"],
        "attempt_order": "task_manifest_then_state_mutant_clean_then_method_list",
        "seed": 42,
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 512,
        "timeout_seconds": 60,
        "maximum_attempts": 12,
        "retries": 0,
        "hook_hash": HOOK_HASH,
        "challenge_sha256": {
            path: digest_bytes(source.encode()) for path, source in CHALLENGES.items()
        },
        "controls_only": controls_only,
        "frozen_gate": {
            "repair": "mutant patch passes original tests and independent behavior probes",
            "clean_damage": "a compiled clean-source patch fails either protected test set",
            "budget": "literal rendered context <= 256 cl100k tokens; native full prompt verified",
            "grounding": "schema path enum and compiler sources derive only from visible chunks",
        },
        "limits": [
            "Three previously exposed synthetic mutations; diagnostic only.",
            "One small model, one seed and no retries; no reliability or superiority claim.",
            "Compact rendering and query-symbol anchoring are engineering choices, "
            "not novelty claims.",
            "Local trusted host, not authenticated inference or an execution sandbox.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    rows: list[dict[str, Any]] = []
    for original_task in tasks:
        task = replace(original_task, test_command=f"{original_task.test_command} {CHALLENGE_PATH}")
        target = task.bug_location.rsplit(":", 1)[0]
        with repo_snapshot(repo, task.commit) as original:
            hook = migrate_test_hook(original)
            challenge = original / CHALLENGE_PATH
            challenge.write_text(CHALLENGES[target], encoding="utf-8")
            baseline = run_pytest_tests(original, task.test_command)
            challenge.unlink()
            mutant = original.parent / "grounded-mutant"
            shutil.copytree(original, mutant, symlinks=True)
            apply_mutation(mutant, task)
            challenge = mutant / CHALLENGE_PATH
            challenge.write_text(CHALLENGES[target], encoding="utf-8")
            failing = run_pytest_tests(
                mutant, task.test_command, required_test_ids=baseline.test_ids
            )
            challenge.unlink()
            reference = ReferencePatchClient(repo, original_task).generate_patch(
                RepairPrompt("", "")
            )
            restored = score_copy(mutant, task, reference, baseline.test_ids)
            control = {
                "task_id": task_identifier(task),
                "hook_migration": hook,
                "baseline_passed": baseline.passed,
                "mutant_completed": failing.completed,
                "mutant_failed_ids": sorted(failing.failed_ids),
                "reference_passed": restored.pass_at_1,
            }
            append_row(output / "controls.jsonl", control)
            if not (
                baseline.passed and failing.completed and failing.failed_ids and restored.pass_at_1
            ):
                raise ValueError("Behavioral controls failed; inspect controls.jsonl")

            try:
                for state, snapshot in (("mutant", mutant), ("clean", original)):
                    graph = CodebaseGraph.from_repo(snapshot)
                    query = build_query(task, "dev_report")
                    compact = assemble_context(graph, query, "compact_bm25", BUDGET)
                    grounded = assemble_context(graph, query, "grounded_bm25", BUDGET)
                    compact_equivalent = compact.context.render() == grounded.context.render()
                    for method in METHODS:
                        assembled = assemble_context(graph, query, method, BUDGET)
                        contract = grounded_edit_contract(graph, assembled.selected_node_ids)
                        prompt_base = build_prompt(task, query, assembled.context.render())
                        prompt = RepairPrompt(contract.system, prompt_base.user)
                        row: dict[str, Any] = {
                            "task_id": task_identifier(task),
                            "state": state,
                            "method": method,
                            "budget": BUDGET,
                            "context_tokens": assembled.context.total_tokens,
                            "context_sha256": digest_bytes(assembled.context.render().encode()),
                            "selected_node_ids": assembled.selected_node_ids,
                            "query_anchor_node_ids": assembled.metadata.get(
                                "query_anchor_node_ids", []
                            ),
                            "visible_paths": list(contract.sources),
                            "visible_source_sha256": contract.source_sha256,
                            "schema_sha256": digest_bytes(
                                json.dumps(contract.schema, sort_keys=True).encode()
                            ),
                            "compact_bm25_equivalent": compact_equivalent,
                            "prompt": asdict(prompt),
                            "prompt_sha256": digest_bytes(
                                json.dumps(asdict(prompt), sort_keys=True).encode()
                            ),
                            "status": "controls_only" if controls_only else "error",
                            "repair_passed": False,
                            "clean_damage": False,
                        }
                        if not controls_only:
                            try:
                                transport = OllamaRepairClient(
                                    MODEL,
                                    timeout=60,
                                    prompt_token_counter=counter,
                                    response_schema=contract.schema,
                                )
                                if transport.digest != MODEL_DIGEST:
                                    raise ValueError("Pilot model digest differs")
                                response = ExactEditRepairClient(
                                    transport, contract.sources
                                ).generate_patch(prompt)
                                row["response"] = asdict(response)
                                if state == "mutant":
                                    evaluation = score_copy(
                                        snapshot, task, response, baseline.test_ids
                                    )
                                    row["evaluation"] = asdict(evaluation)
                                    row["repair_passed"] = evaluation.pass_at_1
                                elif response.patch_text:
                                    evaluation = score_copy(
                                        snapshot, task, response, baseline.test_ids
                                    )
                                    row["evaluation"] = asdict(evaluation)
                                    row["clean_damage"] = bool(
                                        evaluation.metadata.get("patch_applied")
                                    ) and not evaluation.pass_at_1
                                row["status"] = "completed"
                            except (OSError, ValueError) as exc:
                                row["error"] = f"{type(exc).__name__}: {exc}"
                                if isinstance(exc, OllamaResponseError):
                                    row["rejected_provider_response"] = exc.response
                        append_row(output / "results.jsonl", row)
                        rows.append(row)
                        print(
                            f"{len(rows)}/12 {row['task_id']} {state}/{method}: "
                            f"repair={row['repair_passed']} damage={row['clean_damage']}",
                            flush=True,
                        )
            finally:
                shutil.rmtree(mutant)

    summary = {
        "controls_passed": len(tasks),
        "attempt_count": 0 if controls_only else len(rows),
        "planned_attempts": 12,
        "repairs": sum(row["repair_passed"] for row in rows),
        "clean_damage": sum(row["clean_damage"] for row in rows),
        "all_contexts_within_budget": all(row["context_tokens"] <= BUDGET for row in rows),
        "compact_grounded_equivalent_cells": sum(
            row["compact_bm25_equivalent"] for row in rows
        ),
        "interpretation": "Exposed-task diagnostic only; no retrieval superiority claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--allow-local-execution", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_local_execution:
        parser.error("Use --allow-local-execution for this trusted-local experiment")
    print(
        json.dumps(
            run(
                arguments.repo.resolve(),
                arguments.output,
                arguments.tokenizer_json,
                arguments.controls_only,
            )
        )
    )
