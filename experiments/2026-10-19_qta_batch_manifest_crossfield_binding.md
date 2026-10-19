# QTA processor-manifest cross-field binding (2026-10-19)

## Purpose

An independent audit found that the typed digest manifest verified the full
processor-batch digest but did not independently derive two row-level fields
from that same live batch: `input_ids_sha256` and `video_grid_thw`. The complete
batch digest could be correct while these copied metadata fields were
internally inconsistent with it.

## Change

The non-authorizing manifest builder now requires a mapping batch with one
integer `input_ids` row and one three-integer `video_grid_thw` row. It
canonicalizes the input IDs using the existing baseline-provenance JSON rules,
hashes them, and compares the result to the manifest row. It normalizes the
single-row grid and compares all three positive integer values to the pinned
row. Tensor/list inputs are supported without numeric coercion; booleans,
floating grid entries, missing values, malformed shapes, and metadata
mismatches fail closed.

The primary builder-backed synthetic manifest and phase-receipt fixture now
carry these metadata values inside their typed batches and derive the
corresponding row fields from those same values. No receipt schema, branch
count, treatment, GPU path, or scoring authorization changed.

## Verification

- `test_qta_processor_batch_digest_manifest_v1` and
  `test_qta_phase_freeze_receipts`: **60/60 passed**. Tests include changed
  input IDs, changed grids, missing/malformed metadata, bool-as-int rejection,
  list and CPU-tensor shapes, manifest/receipt linkage, source joins, and
  legacy/v2 receipt behavior.
- The implementation contains no model import or forward and performs no
  protected media access. No GPU was used.
- `py_compile` passed for the implementation and affected tests; scoped
  `git diff --check` passed.

## Interpretation and limits

The builder now checks that these two manifest fields agree with the live
typed batch supplied to it. Persisted-manifest validation still proves only
canonical-byte equality to caller-supplied expected row/protocol pins; it does
not reconstruct the historical tensor bytes. The phase-receipt verifier itself receives only the persisted manifest, not
live tensors. A hand-assembled canonical manifest with matching caller pins can
therefore pass that byte-level verifier even if its input-ID/grid metadata do
not describe a historical tensor batch; the new cross-field check applies only
when the candidate builder receives the live batch. The `prompt_sha256`,
processor, tokenizer, runtime, and sampler pins remain externally supplied and
their trust origin is not authenticated by this helper. It remains a CPU structural
consistency check and grants no model-load or scoring permission.

Direct faithfulness evidence remains **0/3** formal claim reviews, **0/15**
verified clock/frame correspondences, **0/3** prediction-link judgments, and
**0/3** direct scores. No direct effect estimate, confidence interval, or
inferential test exists. Rights status remains unverified; protected inputs and
derived outputs remain local.

## Next action

Continue fail-closed CPU validation of the QTA pre-baseline path and preserve
the separation between per-row processor batches and nine intervention
branches. Do not launch real-video scoring until independent review, timing,
prediction-link, external-root, and baseline gates exist together. No repeat of
closed human-free GPU proxy cohorts is justified by this metadata consistency
fix.
