# Completing the interrupted capacity comparator

The frozen `ea20ab0` capacity comparator recorded 23 of its 24 planned attempts.
It stopped before the last BCA / 1024-token / exact-edit condition for the indent
task because available disk fell below its prespecified 2 GiB floor. The first
23 outcomes include one successful clamp repair and one timeout. Neither is
discarded or repeated.

The recovery executor completes only that unattempted condition. Before calling
the model it verifies every original experiment source hash, the task manifest,
the model digest, the unchanged parameters, and a unique 23-cell partial matrix.
It reconstructs the same prompt from the pinned mutant and repeats the scoring
controls, requiring the original test identifiers. The disk floor remains in
force. An error from the remaining model call counts as its outcome, without a
retry. New execution provenance and results go into a fresh directory; the
interrupted records remain unchanged.

```bash
python -m examples.complete_edit_pilot --repo /path/to/boltons \
  --interrupted /path/to/interrupted-run --output /fresh/recovery-run \
  --tokenizer /path/to/pinned-qwen-tokenizer.json --allow-local-execution
```

The combined result spans two process windows. It remains an exposed-task
diagnostic, not a heldout comparison. A single validated repair shows that the
interface can work; it does not establish a reliable repair system or a retrieval
advantage. Subsequent research must use untouched tasks and retain failures.
