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

## Follow-up: mapper-to-scorer batch composition (2026-10-19)

### Purpose and setup

The prior code bound a mapper snapshot but left untested whether it matched the
full batch reconstructed later by the scorer's batch builder. Added a
CPU-only, synthetic integration case in
`scripts/test_qta_synthetic_adapter_composition.py`: build a deterministic
three-row 3-condition bundle twice, pass first-build full batches through the
actual temporal mapper and processor-row adapter, then compare the resulting
typed digests against all nine second-build batches. Expected hashes for the
freeze come from the first build, not the second objects being frozen.

### Verified results

- Focused composition test: **5/5 passed**.
- Combined composition, batch-builder, and atomic-freezer suites: **32/32
  passed** in an independent primary rerun; Python compilation and scoped
  `git diff --check` passed.
- All three mapper snapshots matched the corresponding reprocessed full batch;
  all nine second-build condition batches matched first-build hashes and were
  frozen in the original row order.
- A changed second-build processor output was rejected as
  `exact_condition_batch_hash_mismatch` before the final callback. A reordered
  mapper row was rejected by the cohort join, and a reversed scorer row order
  was rejected before the freezer precheck.
- The successful fixture traversed all nine batch mappings, but reported
  `execution_authorization=not_granted`. No model/scorer was constructed and
  forwards remained zero. The fixture used only synthetic local inputs; no
  protected data, GPU, or external source was accessed.

### Interpretation and limitations

This closes a narrow deterministic composition gap between two synthetic CPU
processor builds and the mapper's captured full-batch snapshot. Typed SHA-256
comparison covers mapping structure and tensor dtype, shape, and bytes. The
fixture's precheck/final callbacks are trivial test callbacks, not authenticated
phase gates. It does not execute the production scorer or ModelCallBoundary,
prove real-video/media-source authenticity, establish human judgments, or
measure explanation faithfulness. The exact nine-condition freeze remains
non-authorizing by contract.

Faithfulness remains **0/3** claim-reviewed, **0/15** player-clock/decoder
correspondences, **0/3** prediction-link judgments, and **0/3** directly
scored. No direct `ΔF`, interval, or test exists.

### Next action

Retain this synthetic invariant as regression coverage. Continue only with
useful CPU-side handoff or phase-gate integration while human/trust inputs are
absent. Do not use a trivial synthetic callback to justify any real-data score;
once formal prerequisites are present, the first GPU run remains a bounded
three-row baseline-only smoke, followed by a separate Stage-2 decision before
intervention scoring.
