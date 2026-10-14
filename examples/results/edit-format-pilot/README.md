# Edit-format recovery experiments

The original three-task local repair pilot returned0/18passing repairs. The
[first paired protocol](PROTOCOL.md) isolates formatting from retrieval with
supported pytest9 and additional evaluation-only behavior probes.

## Recorded0.5B result

At frozen source`b40598c`, all24trials completed and all full native prompt counts
matched. **Both formats returned0/12passing repairs.** None reached valid compiled
evaluation. Six diff responses truncated and six used the wrong target path.
All12JSON completions stopped normally, but eight chose an unauthorized path,
two made no change, one produced invalid Python and one failed exact matching.
Three original/mutant/reference control sets passed. The diagnostic gate failed.

The [complete protocol and records](qwen05/) retain actual prompts, raw outputs,
test identifiers and hashes. Compressed JSONL has no content changes. A valid JSON
shape did not establish a valid edit. These previously exposed cases do not rank
retrievers or establish generalization.

## Prospective capacity comparator

After inspecting the0.5Bfailures, freeze a new run of the **same24conditions**
with [Qwen2.5-Coder1.5B](https://ollama.com/library/qwen2.5-coder:1.5b),
Apache2.0, Q4_K_M,986,062,089bytes, reported digest
`d7372fd828518a4d38b1eb196c673c31a85f2ed302b3d1e406c4c2d1b64a0668`.
No temperature, sampling seed, context budget, timeout, output limit, task,
behavioral probe or acceptance criterion changes. The server is still trusted
local HTTP; reported digests are continuity metadata, not authentication.

The official tokenizer at revision
`2e1fd397ee46e1388853d2af2c993145b0f1098a` has the same verified tokenizer JSON
SHA256 as the0.5Bmodel. Actual prompt counts must still match at runtime.
This is a modest free local model download, not a cloud or paid inference run.
Stop before the next call if available disk falls below2GiB. No inference retry
or automatic fallback is permitted. This is a capacity diagnostic, not a
claim that this small model represents current best research models.

```bash
python -m examples.edit_format_pilot --model qwen2.5-coder:1.5b \
  --repo /path/to/boltons --output /fresh/model-run-15b \
  --tokenizer-json /path/to/pinned-qwen-tokenizer.json \
  --allow-local-execution
```

If a format starts producing real repairs, use it in the next **untouched-task**
evaluation. Do not turn repeated diagnosis on these three exposed cases into a
heldout success claim. A broader Beacon result still needs paired retrieval
comparisons with a usable repair interface and independent behavioral tests.
