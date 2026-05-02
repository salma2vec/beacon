# beacon Design

## Goal

Build a reproducible benchmark and library for budgeted context assembly in LLM-based code repair, with Python-first graph extraction, explicit ablations, and benchmark-grade reporting.

## Repository shape

- `src/cegraph/` contains the reusable library.
- `paper/experiments/` contains orchestration and analysis scripts.
- `paper/results/` contains generated benchmark artifacts and checked-in examples.
- `tests/` covers parsing, graph building, mutation validity, budget enforcement, assembly, and reporting.
- `csrc/` contains optional native traversal acceleration.

## Core model

- Parse Python repositories into a directed knowledge graph over modules, classes, functions, and methods.
- Track `calls`, `imports`, `assigns_to`, `inherits_from`, and `decorates` edges.
- Index each node lexically and semantically.
- Assemble context under a token budget using multiple retrieval methods, including greedy submodular BCA.

## Benchmark flow

1. Load pinned tasks from JSONL.
2. Materialize or verify pinned target repository commits.
3. Assemble context for each `(task, method, budget, query_type)` cell.
4. Prompt an LLM to repair the bug using the assembled context.
5. Apply the patch in a disposable workspace.
6. Run syntax checks and repository tests.
7. Record pass@1 plus retrieval and budget metrics.

## Guardrails

- Token counting must use `tiktoken` `cl100k_base` for both assembly and prompt estimation.
- Invalid mutations are rejected during generation.
- Commits are pinned per task to prevent contamination.
- `target_file` is treated as an oracle ceiling and checked during reporting.

## Delivery strategy

- Bootstrap a clean nested repository and push immediately.
- Implement the core library and benchmark harness in parallel slices.
- Keep native acceleration optional and behind a Python fallback.
- Ship with tests, docs, and a reproducible CLI surface first; expand task coverage iteratively.

