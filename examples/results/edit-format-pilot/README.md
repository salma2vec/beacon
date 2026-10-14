# Edit-format recovery experiments

These paired experiments separate output-format failure from model capacity on
three previously exposed Boltons mutations. They use supported pytest 9, unchanged
upstream assertions and additional evaluation-only behavior probes. The
[original protocol](PROTOCOL.md) was frozen before generation.

| Model | Unified diff | Exact edit | Compiled edits | Native prompt checks |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder 0.5B | 0/12 repairs | 0/12 repairs | 0/24 | 24/24 matched |
| Qwen2.5-Coder 1.5B | 0/12 repairs | 1/12 repairs | 2/24 | 23/23 received responses matched |

One 1.5B diff attempt timed out and stays in its 12-attempt denominator. Every
original/mutant/reference control passed. These are exposed-task diagnostics,
not a heldout benchmark or a ranking of retrieval methods.

## What changed

The new exact-edit compiler accepts one schema-constrained JSON object containing
`path`, `search` and `replace`. It rejects ambiguous matches, unauthorized paths,
no-op changes and invalid Python. It mechanically creates a diff for the existing
independent evaluator. It neither infers a correction nor retries a failed edit.

The 0.5B experiment at `b40598c` failed its diagnostic gate. All 12 JSON responses
stopped normally, but eight named an unauthorized path, two made no change, one
produced invalid Python and one failed exact matching. Six diff outputs truncated
and six used the wrong target. The [complete records](qwen05/) retain all failures.

The capacity comparator at `ea20ab0` changed only the model. The 1.5B model repaired
`clamp` with BM25 context at 1024 tokens, passing the original test suite and the
additional clipping-property probes. Another exact edit compiled but failed its
tests. The remaining exact edits failed path (7), exact matching (2), or Python
syntax (1) checks. Diff outputs included eight wrong paths, three truncations and
one timeout. Valid JSON alone did not establish a valid or correct edit.

This passes the narrow diagnostic gate: at least one genuine model repair and a
higher compiled-edit rate for exact edits than for diffs. One success in 12 exact
edits is not a reliable repair system. It provides no evidence of BCA superiority,
which still has no passing repair in this panel.

## Interrupted run and immutable recovery

The 1.5B process stopped before attempt 24 when disk space fell below its frozen
2 GiB floor. After unloading the model and removing two disposable validation
environments, the separately frozen executor at `58efad0` completed only the
unattempted condition. It verified all original source hashes and scoring controls.
None of the first 23 attempts was retried. The last condition also failed.

The [recovery specification](RECOVERY.md), [interrupted records](qwen15/interrupted/),
[recovery record](qwen15/recovery/) and [combined analysis](qwen15/analysis.json)
preserve both process windows. JSONL compression changes no content. Combined
uncompressed results SHA256:
`135e9b708c35ddf5a129f431de7d76e525297e2019d9ee7893b0a0f69057142f`.

## Reproduction and limits

The model is [Qwen2.5-Coder 1.5B](https://ollama.com/library/qwen2.5-coder:1.5b),
Apache 2.0, Q4_K_M, 986,062,089 bytes, reported digest
`d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`.
Its official tokenizer revision is
`2e1fd397ee46e1388853d2af2c993145b0f1098a`; the tokenizer JSON matches the verified
0.5B tokenizer. Both runs used temperature 0, seed 42, context 4096, output 512,
timeout 60 seconds, BM25/BCA and 256/1024 rendered context-token budgets, with no
retries. Actual prompts and full native counts are recorded. A timed-out call has
no received response and no verified native count.

```bash
python -m examples.edit_format_pilot --model qwen2.5-coder:1.5b \
  --repo /path/to/boltons --output /fresh/model-run-15b \
  --tokenizer-json /path/to/pinned-qwen-tokenizer.json \
  --allow-local-execution
```

The local HTTP service is trusted but unauthenticated; reported digests are
continuity metadata, not attestation. These small local models do not represent
current best repair models. Broad repair reliability and a retrieval improvement
still require untouched tasks, natural issues, heldout repositories, paired
same-budget baselines and independent behavior tests. The next hypothesis is to
select execution evidence that distinguishes candidate fault locations; it is
unproven, and compact lexical and fault-localization baselines remain necessary.
