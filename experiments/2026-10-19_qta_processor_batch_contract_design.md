# QTA processor-batch provenance contract design (2026-10-19)

## Purpose

Resolve the next design question from the processor-batch audit: must the
production system persist large raw tensor batches, or can it reproducibly
rebuild and hash them before loading model weights?

## Source review and decision

Inspected `nextgqa_atomic_batch_freeze.py`,
`nextgqa_claim_mask_scoring.py`, `qta_prebaseline_live_batch_binding.py`,
`qta_phase_freeze_receipts.py`, and the phase-1 bundle runbook. The existing
scorer path already supports an in-process batch-provenance contract:

- `batch_sha256()` hashes string-keyed mapping structure and exact tensor
  dtype, shape, and contiguous CPU bytes; it rejects nonfinite, quantized,
  complex, non-strided, or unsupported values.
- `freeze_all_batches()` creates CPU snapshots for all three rows × three
  fixed input conditions, verifies all nine expected hashes, and only then
  invokes a separate final-gate callback. It returns those exact objects in a
  `FrozenBatch`; it never grants execution authorization.
- `score_preflighted_triplets()` requires freezer-produced `FrozenBatch`
  objects, rehashes all nine before the first call, and checks each device
  transfer against the expected digest before its forward. It stops on a
  mismatch or nonfinite output and does not average a partial cohort.
- The signed-live-batch binding validates row order/token/source/video/batch
  digest agreement but explicitly remains non-authorizing.

**Design choice:** avoid persisting the tensor payload a second time. A future
signed processor-batch binding manifest can pin the deterministic recipe,
environment, record order, conditions, prompt/input binding, group geometry,
and all nine typed-batch digests. The scoring process must rebuild every batch
once from exact pinned source bytes, compare against that independently signed
manifest, retain the same CPU snapshots through model calls, and rehash at the
existing pre-forward boundaries. The detailed proposed contract is in
`docs/qta_live_batch_provenance_contract_v1_ja.md`.

This is a versioned design proposal only. The current Stage-1 role
`full_video_processor_batches.bin` remains opaque and unchanged; existing
signatures are not reinterpreted. The current callbacks/bridge do not prove
their trust-root origin and do not combine actual human judgments, all
15 clock/frame correspondences, baseline `p/q`, independent prediction-link
review, and a production scorer into an authenticated transaction.

## Verification

The existing synthetic in-process freeze/scorer/live-binding suites passed
**48/48** tests. They cover typed hash behavior, snapshot preservation,
expected-hash mismatch, row reorder, batch mutation, pre-forward blocking, and
all-or-none nine-call scoring. These are tests of existing structural
components, not a new implementation or an authorization result. No real
review input, actual video batch, model, or GPU was used in this design step.

## Faithfulness status and next action

Direct faithfulness remains 0/3 formal claim reviews, 0/15 verified
player-clock/decoder-frame correspondences, 0/3 prediction-link judgments, and
0/3 direct scores; there is no empirical `ΔF`, confidence interval, or test.
Rights remain unverified; protected materials and derived outputs stay local.

The additive pre-baseline receipt v2 integration is recorded in
`2026-10-19_qta_processor_batch_receipt_v2_integration.md`: legacy v1 remains
unchanged, while v2 binds the candidate manifest to receipt rows and Stage-1
source-video digests using explicit caller pins. The remaining implementation
work is a one-shot production caller that derives those expectations from
independently rooted artifacts, reconstructs and retains the exact batches,
and fails closed before model import. Keep real scoring blocked until the
independently rooted human, release, clock, prediction-link, and baseline
inputs are actually available and pass together.
