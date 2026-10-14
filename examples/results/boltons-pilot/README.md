# Pinned public-repository pilot

This 2026-10-14 run uses three hand-selected, single-line mutations in
[Boltons 25.0.0](https://github.com/mahmoud/boltons/tree/c23dbdadb6fecdf505eb4231559561913b84c13f),
commit `c23dbdadb6fecdf505eb4231559561913b84c13f`: `mathutils.clamp`,
`iterutils.pairwise` and `strutils.indent`. The original upstream pytest selections
complete 11, 1 and 14 tests respectively. Each mutation makes its original suite
fail, and restoring the original source passes all 18 known-repair controls.
The manifest is in `examples/tasks/boltons-25.0.0.jsonl`.

The real-repository control exposed and fixed a harness bug: stripping all trailing
whitespace corrupted valid unified diffs ending in a blank context line. A dedicated
regression test now preserves that line. Control results below use the corrected
evaluator, before the small-model pilot began.

## Actual model result

The pilot used [Qwen2.5-Coder 0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct),
an Apache-2.0 model, through Ollama 0.40.1 on an Apple M1 with 8 GB unified memory.
The [Ollama artifact](https://ollama.com/library/qwen2.5-coder:0.5b) was
`qwen2.5-coder:0.5b`, Q4_K_M, digest
`4ff64a7f502a08b7616edb8ca0a79eb1853fc363d842b7df4b46915d11a3fb09`.
Each condition received one completion at temperature 0, seed 42, a 4096-token native
context and a 512-token output cap. There were no retries, tools or human patch edits.

| Method | Context budget (cl100k_base) | Target source span retrieved | Valid passing repairs |
| --- | ---: | ---: | ---: |
| no retrieval | 256 | 0/3 | 0/3 |
| no retrieval | 1024 | 0/3 | 0/3 |
| BM25 | 256 | 2/3 | 0/3 |
| BM25 | 1024 | 3/3 | 0/3 |
| BCA | 256 | 0/3 | 0/3 |
| BCA | 1024 | 1/3 | 0/3 |

**No model-generated repair passed.** Ten responses reached the output cap and
were rejected as incomplete; the remaining eight stopped but did not supply an
applicable patch confined to the declared source file. These include prose-wrapped,
malformed and wrong-file diffs. The run diagnoses a weak model/patch-protocol
combination; it does not show whether better context improves repair quality.
Target-span coverage is a separate retrieval diagnostic and is not a repair score.

All 18 full native-prompt counts matched the server's `prompt_eval_count` exactly:
56–1149 tokens, comfortably below the 3584 tokens reserved for the prompt. Counts
come from Qwen's tokenizer at Hugging Face revision
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, tokenizer JSON SHA-256
`c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539`.
The retrieval budget counts rendered context only using `cl100k_base`; Qwen's
native count includes the system/query/chat template as well. Output lengths were
212–512 native tokens. Total server-reported request duration was 133.618 seconds,
including loading and prompt processing; it is not a controlled latency benchmark.

An earlier `llama3:latest` 8B attempt timed out before its first response while the
Mac ran short of memory/disk. It produced **zero completed attempts and no model
scores**. `llama3-infrastructure-failure.json` preserves that failed protocol;
the smaller-model results are a separate experiment, not substituted scores.

## Current harness control

The supported development install uses pytest 9.0.3 or newer. To validate the
current mutation/repair harness without downloading a model or executing an
external repository, run from a fresh checkout and use a new output directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m examples.offline_benchmark --output /tmp/beacon-current-control
```

This runs six known-repair controls on a generated local fixture. It does not
reproduce the Boltons or Qwen results above.

## Archived reproduction record

The original pilot used Python 3.11 and pytest 8.4.2 because this upstream
release's collection hook is incompatible with pytest 9. Pytest 8.4.2 is affected
by [CVE-2025-71176](https://github.com/pytest-dev/pytest/releases/tag/9.0.3).
Its frozen requirements and results are preserved unchanged. They do not satisfy
the current development requirements; use the original experiment checkout for
historical replay in an isolated disposable environment. A replay with patched
pytest requires a separately recorded upstream-hook compatibility change and
fresh controls; no such replay is claimed here.

The following commands record the original procedure, not the supported current
installation. Use a fresh output directory. The model runs
locally, but generated source still executes with your user's filesystem/network
permissions after the explicit acknowledgement. Use a disposable external sandbox
for untrusted evaluation.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev]' \
  -r examples/results/boltons-pilot/requirements.txt
git clone --branch 25.0.0 --depth 1 https://github.com/mahmoud/boltons.git /tmp/beacon-repos/boltons
git -C /tmp/beacon-repos/boltons rev-parse HEAD

# Known original-source restoration controls; no model required.
.venv/bin/python -m examples.public_repo_benchmark \
  --repo /tmp/beacon-repos/boltons --output /tmp/beacon-public-control \
  --allow-local-execution

# Explicitly obtain the small model and native tokenizer before the model run.
ollama pull qwen2.5-coder:0.5b
curl --fail -L \
  https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct/resolve/ea3f2471cf1b1f0db85067f1ef93848e38e88c25/tokenizer.json \
  -o /tmp/qwen-tokenizer.json
.venv/bin/python -m examples.public_repo_benchmark \
  --repo /tmp/beacon-repos/boltons --output /tmp/beacon-qwen-pilot \
  --model qwen2.5-coder:0.5b --tokenizer-json /tmp/qwen-tokenizer.json \
  --allow-local-execution
```

The pilot validates the tokenizer hash, checks the installed model digest before
each request, verifies exact native prompt counts, and refuses to overwrite an
existing run directory. Compare the recorded model digest to the value above;
mutable model tags can change. Cross-version or cross-hardware bitwise replay is
not promised.

## Artifacts and limits

`controls.jsonl.gz` and `qwen05.jsonl.gz` retain every condition, selected node IDs,
prompts, unedited model responses, patches, scores, model settings and test records.
Only temporary machine paths in runner stdout/stderr are replaced with
`<temporary-path>`; source and model response text are unchanged. The corresponding
summary JSON files retain counts and environment versions. `BOLTONS-LICENSE`
preserves the upstream BSD license for reproduced source context.

These are three purposively chosen mutations in one public repository. Reports
name functions; public tests/docstrings remain retrievable and may reveal intended
behavior. No original repair line or target location is passed to the selectors.
No-retrieval prompts repeat across budgets, so the 18 conditions are not 18
independent tasks. This is neither held-out natural-issue validation nor a reliable
method ranking. Stronger models, robust constrained patch generation, more tasks,
held-out repositories and independently protected evaluation remain necessary.
