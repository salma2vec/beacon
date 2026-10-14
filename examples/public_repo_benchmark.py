"""Known-repair controls or local-model trials on three pinned Boltons mutations.

The manifest is deliberately small and hand-selected; it is not a held-out
real-issue benchmark. Public tests and docstrings remain eligible context.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from cegraph import CodebaseGraph, assemble_context
from cegraph.evaluation import apply_mutation, repo_snapshot
from cegraph.llm import OllamaRepairClient
from cegraph.models import RepairPrompt, RepairResponse, TaskRecord
from paper.experiments.benchmark import build_parser, build_query, load_tasks, run_benchmark

MANIFEST = Path(__file__).parent / "tasks" / "boltons-25.0.0.jsonl"


class ReferencePatchClient:
    """Known original-source restoration: a scoring control, never a model result."""

    def __init__(self, repo: Path, task: TaskRecord) -> None:
        path, line = task.bug_location.rsplit(":", 1)
        original = subprocess.check_output(
            ["git", "show", f"{task.commit}:{path}"], cwd=repo, text=True
        ).splitlines(keepends=True)
        mutant = original.copy()
        mutant[int(line) - 1] = task.mutated_line + "\n"
        self.patch = "".join(difflib.unified_diff(mutant, original, f"a/{path}", f"b/{path}"))

    def generate_patch(self, prompt: RepairPrompt) -> RepairResponse:
        return RepairResponse(self.patch, self.patch, "known-original-restoration-control")


def qwen_counter(tokenizer_json: Path):
    """Match the pinned no-tools ChatML template using Qwen's native tokenizer."""
    from tokenizers import Tokenizer

    expected = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    if hashlib.sha256(tokenizer_json.read_bytes()).hexdigest() != expected:
        raise ValueError("Qwen native tokenizer checksum mismatch")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def count(prompt: RepairPrompt) -> int:
        rendered = (
            f"<|im_start|>system\n{prompt.system}<|im_end|>\n"
            f"<|im_start|>user\n{prompt.user}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    return count


def run(repo: Path, output: Path, model: str | None, tokenizer_json: Path | None = None) -> Path:
    tasks = load_tasks(MANIFEST)
    if repo.name != "boltons":
        raise ValueError("The pinned source checkout must be named boltons")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory to preserve previous trial records")
    output.mkdir(parents=True, exist_ok=True)
    if model and (model != "qwen2.5-coder:0.5b" or tokenizer_json is None):
        raise ValueError(
            "This pilot requires qwen2.5-coder:0.5b and --tokenizer-json for count checks"
        )
    counter = qwen_counter(tokenizer_json) if tokenizer_json else None
    client = OllamaRepairClient(model, prompt_token_counter=counter) if model else None
    all_rows = []
    for index, task in enumerate(tasks):
        folder = output / f"task-{index + 1}"
        folder.mkdir(exist_ok=True)
        task_file = folder / "task.jsonl"
        task_file.write_text(json.dumps(asdict(task)) + "\n")
        config = build_parser().parse_args(
            [
                "--tasks-file",
                str(task_file),
                "--repos-dir",
                str(repo.parent),
                "--budgets",
                "256,1024",
                "--methods",
                "no_retrieval,bm25,bca",
                "--query-types",
                "dev_report",
                "--output-dir",
                str(folder),
                "--allow-local-execution",
            ]
        )
        result_path = run_benchmark(config, client or ReferencePatchClient(repo, task))
        rows = [json.loads(line) for line in result_path.read_text().splitlines()]
        # Target presence is computed after retrieval; it is never a selector input.
        with repo_snapshot(repo, task.commit) as snapshot:
            apply_mutation(snapshot, task)
            graph = CodebaseGraph.from_repo(snapshot)
            path, line = task.bug_location.rsplit(":", 1)
            for row in rows:
                selected = [graph.nodes[node] for node in row["selected_node_ids"]]
                assembled = assemble_context(
                    graph, build_query(task, "dev_report"), row["method"], row["budget"]
                )
                row["metadata"]["target_span_retrieved"] = any(
                    node.file_path == path and node.lineno <= int(line) <= node.end_lineno
                    for node in selected
                )
                row["metadata"]["mutated_line_in_context"] = (
                    task.mutated_line.strip() in assembled.context.render()
                )
                row["metadata"]["graph"] = asdict(graph.summary())
                all_rows.append(row)
        print(
            f"Task {index + 1}: {sum(bool(r['pass_at_1']) for r in rows)}/{len(rows)} pass",
            flush=True,
        )
    path = output / "results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in all_rows))
    summary = {
        "kind": "local_model_mutation_pilot" if model else "real_repo_known_repair_control",
        "python": platform.python_version(),
        "versions": {p: version(p) for p in ("beacon", "pytest", "tiktoken")},
        "repo_url": "https://github.com/mahmoud/boltons",
        "commit": tasks[0].commit,
        "task_count": len(tasks),
        "attempt_count": len(all_rows),
        "native_tokenizer_sha256": (
            hashlib.sha256(tokenizer_json.read_bytes()).hexdigest() if tokenizer_json else None
        ),
        "conditions": [
            {
                "method": method,
                "budget": budget,
                "passes": sum(
                    bool(r["pass_at_1"])
                    for r in all_rows
                    if r["method"] == method and r["budget"] == budget
                ),
                "target_spans": sum(
                    r["metadata"]["target_span_retrieved"]
                    for r in all_rows
                    if r["method"] == method and r["budget"] == budget
                ),
                "tasks": len(tasks),
            }
            for method in ("no_retrieval", "bm25", "bca")
            for budget in (256, 1024)
        ],
        "limitations": [
            "Three hand-selected mutations in one repository; not natural or held-out issues.",
            "Function-named reports, public tests and docstrings can reveal intended behavior.",
            "No-retrieval prompts repeat across budgets and are not independent samples.",
            "One fixed-seed model completion per condition; no retries or iterative tools.",
            "Token budgets use cl100k_base, while Ollama also reports its native token counts.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", help="Already installed local Ollama model; omit for known-repair controls"
    )
    parser.add_argument("--allow-local-execution", action="store_true")
    parser.add_argument("--tokenizer-json", type=Path, help="Pinned Qwen native tokenizer JSON")
    args = parser.parse_args()
    if not args.allow_local_execution:
        parser.error("Use --allow-local-execution; generated code is not sandboxed")
    print(run(args.repo.resolve(), args.output, args.model, args.tokenizer_json))
