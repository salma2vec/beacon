"""Paired edit-format diagnosis on the already exposed Boltons pilot tasks.

This is not a heldout retrieval benchmark. Run --controls-only first. Generated
code executes locally only after explicit acknowledgement; this is not a sandbox.
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

from cegraph import CodebaseGraph, assemble_context
from cegraph.edits import EDIT_SCHEMA, EDIT_SYSTEM, ExactEditRepairClient
from cegraph.evaluation import apply_mutation, evaluate_patch, repo_snapshot, run_pytest_tests
from cegraph.llm import OllamaRepairClient, OllamaResponseError
from cegraph.models import RepairPrompt, TaskRecord, task_identifier
from examples.public_repo_benchmark import MANIFEST, ReferencePatchClient, qwen_counter
from paper.experiments.benchmark import build_prompt, build_query, load_tasks

MODELS = {
    "qwen2.5-coder:0.5b": "4ff64a7f502a08b7616edb8ca0a79eb1853fc363d842b7df4b46915d11a3fb09",
    "qwen2.5-coder:1.5b": "d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668",
}
HOOK_HASH = "c41d72fb69c1db601e8509929e65b758e24766268faa4f7797d458593ab12e80"
CHALLENGE_PATH = "tests/test_beacon_behavior.py"

# Independent behavior probes are installed only for evaluation, never indexing.
# They were written before generating any completion in this experiment.
CHALLENGES = {
    "boltons/mathutils.py": """from boltons.mathutils import clamp
def test_independent_clip_properties():
    for lower in [-100, -1, 0, 7]:
        for width in [0, 1, 3.5, 100]:
            upper = lower + width
            for x in [lower-2, lower, lower+width/2, upper, upper+2]:
                expected = lower if x < lower else upper if x > upper else x
                assert clamp(x, lower, upper) == expected
""",
    "boltons/iterutils.py": """from boltons.iterutils import pairwise
def test_independent_pairwise_properties():
    for size in range(12):
        values = list(range(-size, 0))
        assert list(pairwise(iter(values))) == list(zip(values, values[1:]))
        sentinel = object()
        expected = list(zip(values, values[1:] + [sentinel]))
        assert list(pairwise(iter(values), end=sentinel)) == expected
""",
    "boltons/strutils.py": """from boltons.strutils import indent
def test_independent_indent_properties():
    for text in ["a", "a\\nb", "\\na\\n\\nb", "", "a\\n", " "]:
        for margin in [">", "  ", ""]:
            lines = text.split("\\n") if text else []
            expected = "\\n".join(margin + line if line else line for line in lines)
            assert indent(text, margin) == expected
            expected_all = "\\n".join(margin + line for line in lines)
            assert indent(text, margin, key=lambda _: True) == expected_all
""",
}


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def migrate_test_hook(snapshot: Path) -> dict[str, str]:
    """Explicit pinned API migration for pytest9; no test assertion is changed."""
    path = snapshot / "tests/conftest.py"
    before = path.read_bytes()
    if digest_bytes(before) != HOOK_HASH:
        raise ValueError("Pinned Boltons collection-hook hash differs")
    after = before.replace(
        b"pytest_ignore_collect(path, config)", b"pytest_ignore_collect(collection_path, config)"
    ).replace(b"filename = path.basename", b"filename = collection_path.name")
    path.write_bytes(after)
    return {"before_sha256": digest_bytes(before), "after_sha256": digest_bytes(after)}


def append_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def score_copy(snapshot: Path, task: TaskRecord, response: Any, test_ids: frozenset[str]):
    trial = snapshot.parent / "edit-trial"
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
    model: str = "qwen2.5-coder:0.5b",
) -> dict[str, Any]:
    if model not in MODELS:
        raise ValueError("Choose a pinned pilot model")
    tasks = load_tasks(MANIFEST)
    root = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    if not controls_only and dirty.strip():
        raise ValueError("Commit protocol and source before model generation")
    if output.exists():
        raise ValueError("Output directory must not exist; keep previous evidence")
    output.mkdir(parents=True)
    counter = qwen_counter(tokenizer) if tokenizer else None
    if not controls_only and counter is None:
        raise ValueError("Model trials require the pinned native tokenizer")
    source_files = [
        "examples/edit_format_pilot.py",
        "examples/public_repo_benchmark.py",
        "src/cegraph/edits.py",
        "src/cegraph/llm.py",
        "src/cegraph/evaluation.py",
        "src/cegraph/methods.py",
        "src/cegraph/graph.py",
        "src/cegraph/parser.py",
        "src/cegraph/index.py",
        "src/cegraph/optimize.py",
        "src/cegraph/models.py",
        "src/cegraph/tokens.py",
        "paper/experiments/benchmark.py",
        "examples/results/edit-format-pilot/PROTOCOL.md",
    ]
    protocol = {
        "kind": "exposed_task_paired_edit_format_diagnosis",
        "source_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "source_sha256": {p: digest_bytes((root / p).read_bytes()) for p in source_files},
        "manifest_sha256": digest_bytes(MANIFEST.read_bytes()),
        "python": platform.python_version(),
        "versions": {
            **{p: version(p) for p in ("beacon", "pytest", "tiktoken")},
            "tokenizers": version("tokenizers") if counter else None,
        },
        "model": model,
        "expected_digest": MODELS[model],
        "methods": ["bm25", "bca"],
        "budgets": [256, 1024],
        "formats": ["diff", "exact_edit"],
        "seed": 42,
        "temperature": 0,
        "num_ctx": 4096,
        "num_predict": 512,
        "timeout_seconds": 60,
        "maximum_attempts": 24,
        "retries": 0,
        "edit_schema": EDIT_SCHEMA,
        "challenge_sha256": {p: digest_bytes(s.encode()) for p, s in CHALLENGES.items()},
        "controls_only": controls_only,
        "limits": [
            "Three previously exposed mutations; diagnostic, not heldout.",
            "Single small model, one seed; no retrieval superiority or generalization claim.",
            "Local trusted host, not authenticated inference or an execution sandbox.",
            "Explicit pytest9 collection-hook migration; original assertions unchanged.",
        ],
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    transports = {}
    if not controls_only:
        for fmt in ("diff", "exact_edit"):
            transport = OllamaRepairClient(
                model,
                timeout=60,
                prompt_token_counter=counter,
                response_schema=EDIT_SCHEMA if fmt == "exact_edit" else None,
            )
            if transport.digest != MODELS[model]:
                raise ValueError("Pilot model digest differs")
            transports[fmt] = transport
    rows = []
    for index, original_task in enumerate(tasks):
        task = replace(original_task, test_command=f"{original_task.test_command} {CHALLENGE_PATH}")
        target = task.bug_location.rsplit(":", 1)[0]
        with repo_snapshot(repo, task.commit) as snapshot:
            hook = migrate_test_hook(snapshot)
            challenge = snapshot / CHALLENGE_PATH
            challenge.write_text(CHALLENGES[target], encoding="utf-8")
            baseline = run_pytest_tests(snapshot, task.test_command)
            apply_mutation(snapshot, task)
            mutant = run_pytest_tests(
                snapshot, task.test_command, required_test_ids=baseline.test_ids
            )
            challenge.unlink()
            reference = ReferencePatchClient(repo, original_task).generate_patch(
                RepairPrompt("", "")
            )
            restored = score_copy(snapshot, task, reference, baseline.test_ids)
            control = {
                "task_id": task_identifier(task),
                "hook_migration": hook,
                "baseline_passed": baseline.passed,
                "baseline_test_ids": sorted(baseline.test_ids),
                "baseline_stdout": baseline.stdout,
                "baseline_stderr": baseline.stderr,
                "mutant_completed": mutant.completed,
                "mutant_failed_ids": sorted(mutant.failed_ids),
                "mutant_exit_code": mutant.exit_code,
                "restoration": asdict(restored),
            }
            append_row(output / "controls.jsonl", control)
            if not (
                baseline.passed and mutant.completed and mutant.failed_ids and restored.pass_at_1
            ):
                raise ValueError("Independent behavioral controls failed; inspect controls.jsonl")
            if controls_only:
                continue
            graph = CodebaseGraph.from_repo(snapshot)
            sources = {target: (snapshot / target).read_text(encoding="utf-8")}
            for budget in (256, 1024):
                for method in ("bm25", "bca"):
                    query = build_query(task, "dev_report")
                    assembled = assemble_context(graph, query, method, budget)
                    diff_prompt = build_prompt(task, query, assembled.context.render())
                    formats = ("diff", "exact_edit") if index % 2 == 0 else ("exact_edit", "diff")
                    for fmt in formats:
                        if shutil.disk_usage(output).free < 2 * 1024**3:
                            raise ValueError("Stop model experiment: less than2GiB disk headroom")
                        prompt = (
                            diff_prompt
                            if fmt == "diff"
                            else RepairPrompt(EDIT_SYSTEM, diff_prompt.user)
                        )
                        row = {
                            "task_id": task_identifier(task),
                            "method": method,
                            "budget": budget,
                            "format": fmt,
                            "context_tokens": assembled.context.total_tokens,
                            "selected_node_ids": assembled.selected_node_ids,
                            "prompt": asdict(prompt),
                            "prompt_sha256": digest_bytes(
                                json.dumps(asdict(prompt), sort_keys=True).encode()
                            ),
                            "pass_at_1": False,
                            "status": "error",
                        }
                        try:
                            client = (
                                transports[fmt]
                                if fmt == "diff"
                                else ExactEditRepairClient(transports[fmt], sources)
                            )
                            response = client.generate_patch(prompt)
                            row["response"] = asdict(response)
                            evaluation = score_copy(snapshot, task, response, baseline.test_ids)
                            row.update(
                                evaluation=asdict(evaluation),
                                pass_at_1=evaluation.pass_at_1,
                                status="completed",
                            )
                        except (OSError, ValueError) as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            if isinstance(exc, OllamaResponseError):
                                row["rejected_provider_response"] = exc.response
                        append_row(output / "results.jsonl", row)
                        rows.append(row)
                        print(
                            f"{len(rows)}/24 {task_identifier(task)} "
                            f"{method}/{budget}/{fmt}: {row['pass_at_1']}",
                            flush=True,
                        )
    summary = {
        "controls_passed": len(tasks),
        "attempt_count": len(rows),
        "formats": {
            fmt: {
                "attempts": sum(r["format"] == fmt for r in rows),
                "passes": sum(r["format"] == fmt and r["pass_at_1"] for r in rows),
                "errors": sum(r["format"] == fmt and r["status"] == "error" for r in rows),
            }
            for fmt in ("diff", "exact_edit")
        },
        "complete": controls_only or len(rows) == 24,
        "interpretation": "Diagnostic only; exposed tasks and one model, no generalization claim.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--model", choices=MODELS, default="qwen2.5-coder:0.5b")
    parser.add_argument("--controls-only", action="store_true")
    parser.add_argument("--allow-local-execution", action="store_true")
    args = parser.parse_args()
    if not args.allow_local_execution:
        parser.error("Use --allow-local-execution for this trusted-local experiment")
    print(
        json.dumps(
            run(
                args.repo.resolve(),
                args.output,
                args.tokenizer_json,
                args.controls_only,
                args.model,
            )
        )
    )
