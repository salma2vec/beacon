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

The supported development install requires pytest 9.0.3 or newer. The archived
Boltons pilot below used an older, vulnerable pytest release for compatibility
with upstream hooks; its frozen environment is historical evidence, not the
recommended install.

A [real-repository pilot](examples/results/boltons-pilot/README.md) adds three
pinned Boltons mutations, upstream behavioral tests, 18 passing known-repair
controls and an actual local-model trial. It keeps model failures, prompt/response
records, target-span coverage and native tokenizer checks. This small, hand-selected
pilot is not a held-out real-issue benchmark or evidence of a retrieval advantage.

A [paired edit-format follow-up](examples/results/edit-format-pilot/README.md)
adds a strict exact-edit interface and records both failures and the first tested
model repair. Qwen 0.5B repaired 0/24 conditions; Qwen 1.5B repaired 1/24, including
0/12 diff and 1/12 exact-edit attempts. The successful clamp edit passed additional
behavior probes. This establishes a working path, not reliable repair or a BCA
advantage. A timeout and the separately completed disk-interrupted condition remain
in the evidence.

The [grounded exact-edit diagnostic](examples/results/grounded-edit-pilot/README.md)
adds two safeguards without changing the published baselines.
`grounded_bm25` reserves source for a symbol named verbatim in the developer report
and uses a compact header that keeps the repository path and qualified symbol.
Exact-edit schemas can then restrict `path` to non-test source files actually shown
in the rendered context. This selection reads no task location, mutation label or
known repair. The frozen run repaired 0/6 mutations and caused no observed damage
across six clean-source trials. It removed unauthorized-path failures but did not
solve exact-copy or repair reasoning errors. The compact BM25 and query-anchored
contexts were identical on this panel, so the result identifies no benefit from the
anchor and establishes neither retrieval superiority nor generalization.

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
| `compact_bm25` | BM25 order with compact, indexed-source-recoverable path/symbol headers |
| `grounded_bm25` | Compact BM25 with report-named symbols considered first; no task-label access |

Nodes represent modules, classes, functions and methods. AST-derived edges represent
calls, imports, assignments, inheritance and decorators. Dynamic Python dispatch is
not fully resolved. Overlapping module and function snippets can consume budget;
this is a baseline implementation, not a learned retriever.

## Repair experiments

Supply a JSONL manifest with one task per row: `repo`, `commit`, `bug_location`
(`path.py:line`), `mutation_type`, `original_line`, `mutated_line`, `token_budget`,
and a `test_command`. Use an immutable commit hash and a pytest command, for example
`python -m pytest -q tests/test_retry.py`. Scoring supports pytest only, using the
harness Python interpreter and installed pytest in isolated import mode. Custom
scripts, shell pipelines and alternate runners need a structured adapter and are
not supported for scoring. Install the target repository dependencies into the
harness environment. Plugin autoload and configured `addopts` are disabled; pass
test paths/node IDs and simple reporting flags (`-q`, `-v`, `--tb=short`) explicitly.
The bundled `eval_tasks_full.jsonl` is intentionally empty: there is no released
large-scale validated task corpus in this repository. The separate Boltons pilot
manifest is intentionally limited to three controlled regressions.

Mutation discovery enumerates all candidates within explicit resource limits and
raises an error before writing the output if any limit is exceeded. Defaults include
256 KiB per source file, 64 MiB of cumulative syntax-validation input, 10,000 syntax
checks and 10,000 task records. `python -m paper.experiments.discover_mutations --help`
lists all configurable limits. Successful task records preserve those limits in
their metadata; an incomplete prefix is never presented as a complete corpus.

```bash
python -m paper.experiments.benchmark \
  --tasks-file /path/to/tasks.jsonl --repos-dir /path/to/repos \
  --budgets 2000,4000 --methods no_retrieval,bm25,bca \
  --query-types vague --output-dir /tmp/beacon-dry --dry-run
```

Dry runs assemble context without invoking an API or executing repository code, and
write `null` scores. Reports exclude these rows. `--skip-eval` also produces null
scores but can still invoke a configured model; use `--dry-run` for offline work.
Report writers publish through temporary file handles and atomic replacement, so
existing output symlinks cannot redirect report bytes. Figure labels cannot contain
path separators or control characters. The run directory and its ancestors must
still be trusted against concurrent replacement.

For scored experiments, omit `--dry-run` and explicitly add
`--allow-local-execution`. This runs the repository **and the generated repair** with
your user's filesystem and network permissions. Use a disposable external container
for untrusted code. Subprocesses have a credential-free environment, temporary home,
60-second default timeout and 1 MiB captured-output limit; these are not a sandbox.
The OpenAI adapter requires an API key and can incur API charges.

For an already installed local Ollama model, use `--provider ollama --model <tag>`.
The adapter contacts only `127.0.0.1:11434`, rejects redirects and cloud-model routes,
checks the service-reported digest, and records native input/output token counts and stop
reason. It never installs a model or falls back to an API provider. Requests near
the configured context limit fail; truncated output cannot become a scored patch.
The local listener must be trusted. Plain loopback HTTP does not authenticate its
owner; another local process can impersonate Ollama and receive prompts. Digest
continuity and reported routing metadata are not independent model attestation.
Responses explicitly record `server_authenticated=false` and
`model_identity_source=local_service_self_report`. Use an isolated host or network
namespace for sensitive repositories; this adapter does not provide an authenticated
multi-user service boundary.
The public-repository pilot additionally verifies exact full-prompt native counts
using a pinned Qwen tokenizer. General Ollama use without a matching tokenizer
counter records `native_prompt_verified=false`; a count below capacity alone is
not a full truncation guarantee. Source execution still needs the same explicit
local-execution acknowledgement.

Each scored task must complete a nonempty pytest suite before mutation, then
complete the same suite with failures after mutation. A fresh JUnit report outside
the checkout records original test IDs. A repair only passes when all those test
IDs complete and pass, with no skips, errors, missing reports or inconsistent
counts. A zero exit status alone never establishes success; direct `sys.exit(0)`
and `os._exit(0)` bypasses are regression-tested.
Retrieval sees the mutated pinned snapshot, never the caller's working tree. Every
repair starts from a fresh copy. Only the task's declared Python source file may be
patched; conventional test filenames/directories, `conftest.py`, files explicitly
selected by the pytest command, named pytest hooks, local plugins registered by
literal `pytest_plugins` assignments in `conftest.py`, and files matching literal
root-level `python_files` collection patterns under selected test directories are
rejected, along with symlinks, mode changes and escaping paths. Syntax checks
compile in memory and never
write through repository-controlled bytecode-cache paths. A
missing test command, timeout, invalid patch or failing test cannot score as a pass.
Test quality remains the experiment author's responsibility. Task authors must
also exclude test helpers, plugins registered in test
modules and dynamically registered or aliased plugins from repair targets;
static checks cannot identify every verifier
component. A static import edge from a selected test is not enough to classify its
target: the same syntax imports ordinary application code and private assertion
helpers. Beacon therefore does not reject the full test import closure; an
independent operator-owned editable-source policy is still required for adversarial
manifests. These completion
checks are not tamper-proof: adversarial code in the pytest process can forge
reports or interfere with the runner. Strong adversarial evaluation requires an
external sandbox and independently protected test instrumentation.

For new format experiments, `grounded_line_span_contract` renders the selected
indexed nodes as full, numbered source lines with each file's indexed-text SHA-256.
`LineSpanEditRepairClient` accepts one inclusive
`path`/`start_line`/`end_line`/`replacement` edit only when the range is wholly
inside one rendered span and the compiler source still matches that hash. It also
records an explicit `{"abstain":true}` response without turning it into an error.
The earlier `path`/`search`/`replace` interface remains available for reproducing
the published pilots. Line-number prefixes and the line-span system prompt must be
included in any future frozen prompt-budget accounting; this interface has only
deterministic format controls so far, not repair-rate evidence.

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
