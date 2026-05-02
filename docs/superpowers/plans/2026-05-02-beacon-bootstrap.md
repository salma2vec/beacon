# beacon Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first working end-to-end spine of beacon, then push it to GitHub and iterate from a feature branch.

**Architecture:** Start from a reproducible Python package skeleton, then add a typed graph model, retrieval methods, mutation and benchmark orchestration, and analysis/reporting scripts. Keep the optional native acceleration behind stable Python APIs so the benchmark remains runnable without compilation.

**Tech Stack:** Python 3.11, setuptools, pytest, tiktoken, openai, numpy, typer

---

## Chunk 1: Repository Bootstrap

### Task 1: Package and docs skeleton

**Files:**
- Create: `README.md`
- Create: `LICENSE`
- Create: `.gitignore`
- Create: `pyproject.toml`
- Create: `src/cegraph/__init__.py`
- Create: `docs/superpowers/specs/2026-05-02-beacon-design.md`
- Create: `docs/superpowers/plans/2026-05-02-beacon-bootstrap.md`

- [ ] **Step 1: Write the baseline metadata files**
- [ ] **Step 2: Initialize the nested git repository**
Run: `git init && git branch -M main`
Expected: empty local repository on `main`
- [ ] **Step 3: Stage and commit the bootstrap**
Run: `git add README.md LICENSE .gitignore pyproject.toml src docs && git commit -m "feat: bootstrap beacon repository"`
Expected: first bootstrap commit created
- [ ] **Step 4: Push `main` to the GitHub remote**
Run: `git push -u origin main`
Expected: remote default branch established

## Chunk 2: Core Library

### Task 2: Graph and indexing core

**Files:**
- Create: `src/cegraph/models.py`
- Create: `src/cegraph/parser.py`
- Create: `src/cegraph/graph.py`
- Create: `src/cegraph/index.py`
- Create: `src/cegraph/tokens.py`
- Create: `tests/test_graph.py`
- Create: `tests/test_index.py`

- [ ] **Step 1: Write failing graph construction tests**
- [ ] **Step 2: Implement typed models and parser**
- [ ] **Step 3: Implement graph traversal and token counting**
- [ ] **Step 4: Run `pytest tests/test_graph.py tests/test_index.py -v`**
- [ ] **Step 5: Commit**

## Chunk 3: Retrieval, Optimization, and Benchmarking

### Task 3: Assembly methods and optimizer

**Files:**
- Create: `src/cegraph/methods.py`
- Create: `src/cegraph/optimize.py`
- Create: `src/cegraph/llm.py`
- Create: `src/cegraph/evaluation.py`
- Create: `tests/test_methods.py`

- [ ] **Step 1: Write failing tests for budgeted selection**
- [ ] **Step 2: Implement the 11 methods and greedy BCA selector**
- [ ] **Step 3: Implement LLM and evaluation adapters**
- [ ] **Step 4: Run `pytest tests/test_methods.py -v`**
- [ ] **Step 5: Commit**

### Task 4: Benchmark and analysis scripts

**Files:**
- Create: `paper/experiments/benchmark.py`
- Create: `paper/experiments/merge_and_report.py`
- Create: `paper/experiments/posthoc_analysis.py`
- Create: `paper/experiments/router_ab.py`
- Create: `paper/experiments/generate_figures.py`
- Create: `paper/experiments/eval_tasks_full.jsonl`
- Create: `tests/test_benchmark_cli.py`

- [ ] **Step 1: Write failing CLI and report tests**
- [ ] **Step 2: Implement the benchmark and analysis commands**
- [ ] **Step 3: Add a checked-in starter task file schema**
- [ ] **Step 4: Run `pytest tests/test_benchmark_cli.py -v`**
- [ ] **Step 5: Commit**

## Chunk 4: Native Acceleration and Final Validation

### Task 5: Optional acceleration and release validation

**Files:**
- Create: `csrc/README.md`
- Create: `src/cegraph/native.py`
- Create: `tests/test_native_fallback.py`

- [ ] **Step 1: Implement a clean Python fallback API**
- [ ] **Step 2: Document the optional native path**
- [ ] **Step 3: Run `pytest -v`**
- [ ] **Step 4: Run `python -m compileall src paper/experiments tests`**
- [ ] **Step 5: Push the feature branch and open a draft PR**

