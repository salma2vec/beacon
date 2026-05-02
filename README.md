# beacon

Budgeted context assembly for LLM-based code repair under strict token budgets.

## Why beacon

`beacon` is for measuring and improving how we choose code context before asking an LLM to repair a bug. It is opinionated about reproducibility, pinned target commits, and budget-aware assembly. It is not a generic agent framework, not a benchmark for arbitrary repositories out of the box, and not a promise that retrieval alone solves code repair.

## Architecture

```text
Python repository
    │
    ▼
AST parser + symbol extractor
    │
    ▼
cegraph knowledge graph
    │ nodes: module, class, function, method
    │ edges: calls, imports, assigns_to, inherits_from, decorates
    ▼
retrieval indexes
    ├─ lexical
    ├─ semantic
    └─ graph traversal
    │
    ▼
assembly methods
    ├─ bm25 / keyword / dense retrieval
    ├─ BCA ablations
    └─ greedy submodular selector
    │
    ▼
assembled context (<= token budget)
    │
    ▼
LLM repair
    │
    ▼
compile + test evaluation
    │
    ▼
benchmark reports / router studies / figures
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from pathlib import Path

from cegraph import CodebaseGraph

graph = CodebaseGraph.from_repo(Path("example_repo"))
summary = graph.summary()

print(summary.node_count)
print(summary.edge_count)
```

## API Reference

Core entrypoints live in `src/cegraph/`:

- `CodebaseGraph.from_repo(path: Path) -> CodebaseGraph`
- `CodebaseGraph.graph_neighbors(node_id: str, depth: int) -> list[tuple[str, int]]`
- `CodebaseGraph.semantic_neighbors(query_embedding: list[float], k: int) -> list[tuple[str, float]]`
- `CodebaseGraph.token_count(node_id: str) -> int`
- `assemble_context(...) -> AssembledContext`

## Benchmark

The benchmark runner will live in `paper/experiments/benchmark.py` and supports:

```bash
python paper/experiments/benchmark.py \
  --tasks-file paper/experiments/eval_tasks_full.jsonl \
  --repos-dir /path/to/repos \
  --budgets 2000,4000,8000,10000 \
  --methods no_retrieval,bm25,vector,embedding,keyword_map,bca_d1,bca,bca_d5,bca_no_closure,bca_no_scoring,target_file \
  --query-types exact,vague,dev_report \
  --output-dir paper/results/my_run
```

## Extending

- Add a retrieval or assembly method in `src/cegraph/methods.py`
- Add a scorer or optimizer in `src/cegraph/optimize.py`
- Add benchmark tasks in `paper/experiments/eval_tasks_full.jsonl`
- Add new target repositories by pinning commits and setup instructions

## Current scope

Current scope is Python source, controlled single-line mutations, and pass@1 evaluation under token budgets. It does not claim to solve multi-file repair generally, remove the need for strong prompts, or establish universal retrieval rankings outside the included task distributions.

## License

MIT

