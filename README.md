# beacon

A small, reproducible laboratory for **code-context retrieval under token budgets**.
Beacon parses a Python repository into a symbol graph, selects context, and evaluates
single-line mutation repairs against a pinned checkout and behavioral tests.

## Run locally

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m examples.offline_benchmark --output /tmp/beacon-control
```

The offline example creates a tiny Git repository, proves its original tests pass,
introduces a failing mutation, and evaluates a known correct repair across retrieval
methods and budgets. It needs no API key. Its results are **harness controls**, not
LLM scores or evidence that one retrieval method is better.

## Library

```python
from pathlib import Path
from cegraph import CodebaseGraph, assemble_context

graph = CodebaseGraph.from_repo(Path("your_python_repo"))
result = assemble_context(graph, "Where is retry delay computed?", "bm25", budget=2000)
print(result.context.render())
assert result.context.total_tokens <= 2000
```

The budget includes every rendered context header and separator, measured using
`tiktoken`'s `cl100k_base`. It excludes the system prompt, issue query and model output;
it is not a promise about another model's tokenizer. Tiktoken may download its codec
on first use.

## Retrieval methods

| Method | Implementation |
| --- | --- |
| `no_retrieval` | Empty context control |
| `bm25` | BM25 ranking with ordered, budget-constrained inclusion |
| `keyword_map` | Lexical token-overlap ranking |
| `vector`, `embedding` | Aliases for deterministic 96-dimensional signed token hashing; **not pretrained semantic embeddings** |
| `bca`, `bca_d1`, `bca_d5` | Lexical seeds, graph closure, hashed similarity and greedy coverage selection; depths 2, 1, 5 |
| `bca_no_closure` | Seed-only ablation |
| `bca_no_scoring` | Fixed-seed shuffled graph candidates |
| `target_file` | Explicit oracle file-location control |

Nodes represent modules, classes, functions and methods. AST-derived edges represent
calls, imports, assignments, inheritance and decorators. Dynamic Python dispatch is
not fully resolved. Overlapping module and function snippets can consume budget;
this is a baseline implementation, not a learned retriever.

## Repair experiments

Supply a JSONL manifest with one task per row: `repo`, `commit`, `bug_location`
(`path.py:line`), `mutation_type`, `original_line`, `mutated_line`, `token_budget`,
and a `test_command`. Use an immutable commit hash and an argv-style test command,
for example `python -m pytest -q tests/test_retry.py`; shell pipelines are not supported.
The bundled `eval_tasks_full.jsonl` is intentionally empty: there is no released
large-scale validated task corpus or measured LLM benchmark in this repository.

```bash
python -m paper.experiments.benchmark \
  --tasks-file /path/to/tasks.jsonl --repos-dir /path/to/repos \
  --budgets 2000,4000 --methods no_retrieval,bm25,bca \
  --query-types vague --output-dir /tmp/beacon-dry --dry-run
```

Dry runs assemble context without invoking an API or executing repository code, and
write `null` scores. Reports exclude these rows. `--skip-eval` also produces null
scores but can still invoke a configured model; use `--dry-run` for offline work.

For scored experiments, omit `--dry-run` and explicitly add
`--allow-local-execution`. This runs the repository **and the generated repair** with
your user's filesystem and network permissions. Use a disposable external container
for untrusted code. Subprocesses have a credential-free environment, temporary home,
60-second default timeout and 1 MiB captured-output limit; these are not a sandbox.
The OpenAI adapter requires an API key and can incur API charges.

Each scored task must pass tests before mutation and fail them after mutation.
Retrieval sees the mutated pinned snapshot, never the caller's working tree. Every
repair starts from a fresh copy. Only the task's declared Python source file may be
patched; test-file, symlink, mode-change and escaping-path edits are rejected. A
missing test command, timeout, invalid patch or failing test cannot score as a pass.
Test command quality remains the experiment author's responsibility.

`exact` queries reveal the original line and are oracle controls. The fallback
`dev_report` does too; supply a real report in `prompt` to avoid this. `vague` queries
include the target filename but not the repair. BCA receives no oracle line unless
`--oracle-location` is supplied. Output metadata records these conditions, the actual
commit, task/prompt/response hashes, raw prompt/repair, model name, selection settings
and baseline/mutant results. Treat experiment outputs as source-bearing artifacts.

## Research context and limits

[Agent Retrieval Bench (Qin & Xie, 2026)](https://arxiv.org/abs/2607.24882)
separates repository retrieval from downstream repair and evaluates frozen commits
and no-gold controls. Beacon adopts frozen-snapshot evaluation; it does not reproduce
that benchmark or its selective-retrieval tests.
[SWE-bench (Jimenez et al.)](https://arxiv.org/abs/2310.06770) evaluates real repository
issues; Beacon currently studies controlled single-line mutations, a narrower task.

The posthoc router report is an **in-sample oracle upper bound**, not a trained router
or held-out performance claim. Comparisons need paired tasks, appropriate query
controls, a strong test suite and held-out repositories before research conclusions.
Runtime repair prompts remain in source because they are part of the reproducible
experiment. No performance ranking is claimed here.

MIT licensed. See [LICENSE](LICENSE).
