# QTA processor snapshot digest recomputation (2026-10-19)

## Purpose

Close one narrow provenance gap in the QTA prescore adapter: it previously
copied a syntactically valid `typed_batch_sha256` string from the caller without
recomputing it from the mapper's private CPU processor snapshot. A future
processor manifest should carry a digest derived from actual tensor bytes,
while avoiding persistence of the full video tensors.

## Change

`scripts/qta_prescore_processor_row_adapter.py` now requires the nonempty
`_processor_batch_snapshot` mapping returned by the mapper, computes
`nextgqa_atomic_batch_freeze.batch_sha256(snapshot)`, and rejects a missing,
unsupported, or digest-mismatched snapshot before projecting the digest into
the small row binding. The snapshot remains private and is omitted from the
projected row. The synthetic three-row adapter-composition fixture now computes
its expected row digests from each actual synthetic tensor mapping rather than
injecting unrelated hash strings.

This allows later, deterministic processor re-execution to compare its exact
typed batch digest with a compact frozen row binding; it does not yet implement
that production scorer comparison or prove that a specific batch reached a
model forward.

## Validation

Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_qta_prescore_processor_row_adapter \
  test_qta_synthetic_adapter_composition \
  test_qta_prescore_mapping_binding \
  test_qta_synthetic_cohort_preflight -v
```

Result: **30/30 tests passed**. The added tests reject a forged digest, a
mutated snapshot with a stale digest, a missing snapshot, and a recursive
malformed snapshot with a stable error. Existing
three-row structural composition still passes and continues to deny model load
and all scoring forwards. Python compilation and scoped `git diff --check`
also passed.

## Interpretation and limits

This verifies only that the adapter's output digest is derived from the exact
snapshot object it receives. A caller can still supply a fabricated mapping or
snapshot; source, review, processor configuration, and clock authenticity are
not established here. The processor-row digest is not yet derived in a
production source-bound runner, and no scorer currently enforces digest
equality immediately before every generation/forward. Phase receipts remain
non-authorizing. No media, model, GPU, or human review was used in this cycle.

## Faithfulness progress

- Direct claim-linked faithfulness: **0/3 reviewed, 0/3 scored**.
- Player-clock/decoder-frame correspondences: **0/15**.
- Stage-2 prediction-link judgments: **0/3**.
- Direct `ΔF`, uncertainty interval, and inferential test: **none**.
- This adapter check is provenance engineering, not a faithfulness result.

## Next action

Port the existing frozen-batch-to-model-call pattern into the future QTA scorer:
reprocess from source and pinned processor configuration, compare the actual
typed batch to this frozen digest, then have a single wrapper transfer and call
the model while recording the exact input digest for generation and each
condition forward. Validate tamper, row-order, transfer, and target-appended
input cases synthetically before any direct GPU run. Independently, the direct
experiment still needs completed human claim review, all 15 clock/frame
checks, frozen baseline `p/q`, and Stage-2 prediction-link judgments.
