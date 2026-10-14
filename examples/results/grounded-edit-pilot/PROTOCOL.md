# Grounded exact-edit diagnostic protocol

This experiment tests a narrow repair-path failure observed in the already exposed
Boltons tasks. Seven of twelve Qwen 1.5B exact-edit attempts named unauthorized
files. At the 256-token budget, the relevant `clamp` source used 242 tokens but the
old score-bearing header raised its rendered cost to 260, so BM25 skipped it.

The new stack makes two deterministic changes. First, context headers retain only
the exact repository path and symbol identifier. Second, exact-edit decoding gets a
`path` enum made from non-test source files actually visible in that context. The
compiler reads the corresponding normalized indexed source from the same immutable
graph snapshot, records its UTF-8 SHA-256 hash, and independently enforces path, unique
match, no-op and syntax checks. Neither change reads the task bug location, original
line, mutation operator, known repair or hidden behavior probes.

## Frozen matrix

- Tasks: the three existing exposed Boltons mutations, in manifest order.
- States: mutated source, then clean source.
- Methods: unchanged `bm25`, then `grounded_bm25`.
- Budget: 256 literal rendered `cl100k_base` tokens.
- Model: Qwen2.5-Coder 1.5B Q4_K_M at digest
  `d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`.
- Generation: temperature 0, seed 42, context 4096, output 512, timeout 60
  seconds, no retries; 12 total calls.
- Output: one exact `path`/`search`/`replace` object under the per-attempt schema.

Every task must first pass the original tests, fail after mutation, and pass after
the known restoration. Independent behavior probes are evaluation-only and absent
from retrieval. A mutant repair passes only when both protected test sets complete
and pass. Clean-code damage means a compiled edit makes either set fail. Invalid or
empty clean edits do not mutate source and count as no observed damage, while their
format failure remains visible.

The gate is at least one executed repair with no clean-code damage increase. All
prompts, responses, native counts, stop reasons, visible paths, source hashes and
failures are retained. Native prompt verification uses the pinned official Qwen
tokenizer. The three tasks were used in prior development, so any outcome remains a
diagnostic. `compact_bm25` is also computed as a representation control; if it and
`grounded_bm25` render identical evidence, no benefit is attributed to query-symbol
priority.

Primary design references opened before implementation:

- [Coverage.py measurement contexts](https://coverage.readthedocs.io/en/latest/contexts.html)
  provide per-test execution contexts for the later trace-based stage. This pilot
  does not yet claim or simulate trace evidence.
- [Pytest runtest protocol](https://docs.pytest.org/en/stable/reference/reference.html#test-running-runtest-hooks)
  documents the phases that future trusted observers must distinguish.
- [Spectrum-based fault-localization survey](https://arxiv.org/abs/1607.04347)
  establishes execution spectra as prior work; Beacon makes no novelty claim for
  Ochiai ranking or combining it with lexical evidence.

Run controls before any model call:

```bash
python -m examples.grounded_edit_pilot \
  --repo /path/to/pinned/boltons \
  --output /fresh/control-directory \
  --controls-only --allow-local-execution
```

For the frozen model trial, omit `--controls-only` and add
`--tokenizer-json /path/to/pinned-qwen-tokenizer.json`. The script rejects dirty
source, an existing output directory, a digest mismatch or a missing tokenizer.
