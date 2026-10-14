# Paired edit-format diagnosis

Protocol version1,2026-10-14. This follow-up investigates the three **previously
exposed** Boltons mutations. It is not a heldout benchmark or a test of retrieval
superiority. Freeze source and this protocol before generating completions.

Compare the existing unified-diff interface against one schema-constrained JSON
edit with `path`, `search`, and `replace`. The edit compiler requires a unique
exact match in an authorized source file and valid changed Python. It never
chooses a correction, uses reference text, repairs mismatches or retries. The
same independent evaluator checks both arms. Formatting validity is separate
from behavioral correctness.

Use installed Qwen2.5-Coder0.5B, digest
`4ff64a7f502a08b7616edb8ca0a79eb1853fc363d842b7df4b46915d11a3fb09`,
temperature0, seed42, context4096, output512 and timeout60seconds. The pinned
native tokenizer must match every full actual prompt's reported count. Each
format has its own system instruction, recorded and hashed. Compare BM25/BCA
at256/1024 rendered context tokens:3tasks×2methods×2budgets×2formats=24calls.
Alternate format order by task. No retries or substituted seeds.

The original assertions remain unchanged. For supported pytest9, the runner
verifies the upstream collection hook's hash and mechanically migrates its
deprecated `path.basename` API to `collection_path.name`. Original and mutated
baselines plus reference restoration all run under that same migration. This is
not a bit-identical replay of the old pytest8.4.2 environment.

Additional finite behavior probes exercise clipping bounds, generator/sentinel
pairs, and empty/trailing/whitespace indentation. They are installed only during
evaluation and removed before graph construction; the model never sees them.
They provide extra checking on these exposed functions, not independent natural
issues. The first pre-inference control found an incorrect trailing-newline
expectation in the added probe. That failed control is retained privately; the
probe was corrected to the documented behavior before any model trial.

```bash
python -m examples.edit_format_pilot \
  --repo /path/to/boltons --output /fresh/controls \
  --controls-only --allow-local-execution
python -m examples.edit_format_pilot \
  --repo /path/to/boltons --output /fresh/model-run \
  --tokenizer-json /path/to/pinned-qwen-tokenizer.json \
  --allow-local-execution
```

The runner requires the optional `tokenizers` library for the pinned native
counter. It records source/manifest/probe hashes, protocol, full actual prompts,
unedited completions, native counts, rejected completion payloads where received,
patches, complete original test IDs and all errors. Output directories must be
new. Interrupted runs without a final summary are incomplete.

Report all24attempts, invalid/truncated/transport-failed rows, paired differences
and successful behaviors. A diagnostic improvement requires at least one model
repair and better valid-edit rate; it does not establish generalization. Further
work requires untouched tasks/repositories and identical repair interfaces for
the compared retrieval methods. Local generated code execution and plain HTTP
Ollama require a trusted host; neither is an authenticated sandbox.

The interface uses established [structured
outputs](https://docs.ollama.com/capabilities/structured-outputs). Future selection
of execution evidence must be compared with work such as
[PracRepair](https://arxiv.org/abs/2606.17612), and clean-code controls are motivated
by [observed damage from blind iterative repair](https://arxiv.org/abs/2609.10123).
No novelty claim is made for structured edits.
