# Grounded exact-edit diagnostic: negative result

This frozen exposed-task diagnostic tested whether compact recoverable context and
schema-constrained visible edit paths could turn the earlier valid-format gain into
executed repairs. It did not: Qwen2.5-Coder 1.5B repaired **0/6** mutated conditions.
No retries or post-result matrix changes were made.

| Outcome | Existing BM25 | Grounded stack | Total |
| --- | ---: | ---: | ---: |
| Mutant repairs | 0/3 | 0/3 | 0/6 |
| Valid exact edits, clean + mutant | 2/6 | 0/6 | 2/12 |
| Observed clean damage | 0/3 | 0/3 | 0/6 |
| Target file visible, posthoc | 2/3 per state | 3/3 per state | — |

All 12 generations stopped normally and matched the pinned Qwen tokenizer count.
Rendered context stayed at or below the 256-token budget; full native prompts used
372–415 of the reserved 3,584 tokens. The visible-path enum eliminated unauthorized
path failures, but ten responses still failed before execution: five produced
invalid Python and five used a search string that was absent or non-unique. Of the
two compiled BM25 edits, the mutant edit failed the behavioral tests. The clean edit
changed `pairwise` to return the same size-2 window explicitly and passed both the
upstream tests and independent probes. “Zero observed damage” is limited to those
protected tests; it is not proof of semantic equivalence.

The compact BM25 control and query-anchored method rendered identical context on
all six unique task-state cases. The result therefore does not identify query-symbol
priority as a cause and gives no evidence of retrieval superiority. The main useful
result is narrower: grounded path decoding can remove one interface failure class,
but it does not solve exact-copy or repair reasoning failures for this model.

The [protocol](PROTOCOL.md), generated [run metadata](run/protocol.json),
[analysis](analysis.json), compressed [raw outcomes](run/results.jsonl.gz),
[controls](run/controls.jsonl.gz) and [summary](run/summary.json) retain the complete
negative result. Decompressing `results.jsonl.gz` gives SHA-256
`a1b997d4fefed3ffbb4e57048ee169f822fd376ca1afcc38f7eab3aa6893faa7`.

This implementation follows prior work only as motivation. ContextSniper describes
compact evidence packets with recoverable full source, while coverage.py documents
per-test execution contexts for the later trace-based stage. Beacon does not claim
novelty for compact rendering, query anchors, fault localization or their
combination, and these three reused synthetic tasks cannot support a general result.

The next gate is a location-and-format control followed by a frozen untouched panel
of natural issues with independent behavior tests. It must compare the same model,
representation and visible-path schema across retrievers, report invalid edits and
clean-code damage, and pass real repairs before any retrieval claim. These exposed
tasks will not be rerun or selected again to promote the result.

- [ContextSniper v6](https://arxiv.org/abs/2607.01916v6)
- [Coverage.py measurement contexts](https://coverage.readthedocs.io/en/latest/contexts.html)
- [Spectrum-based fault-localization survey](https://arxiv.org/abs/1607.04347)
