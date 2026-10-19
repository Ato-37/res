# QTA processor-batch contract reconciliation (2026-10-19)

## Purpose

Independently check whether the proposed processor-batch digest manifest
matches the actual QTA two-phase nine-branch transaction before implementing a
new signed schema.

## Source findings

Direct source inspection and an independent audit agree:

- `qta_phase_freeze_receipts.PHASE_BRANCHES["pre_baseline"]` is only
  `baseline_full` (three row-forwards); post-baseline contains the remaining
  eight branches per row. Its current `record_bindings` carry one
  `processor_batch_sha256` per row.
- `qta_nine_branch_plan.py` specifies one already-processed model input reused
  across baseline/target/sham × full/evidence-ablation/control-ablation. The
  intervention descriptors do not represent distinct processor batches.
- `nextgqa_atomic_batch_freeze.py` and `nextgqa_claim_mask_scoring.py` implement
  a different contract: three processor inputs per row (`full`,
  `claim_evidence_black`, `temporal_control_black`), nine frozen batches total.

The earlier design text incorrectly proposed mapping those three claim-mask
inputs onto the QTA nine intervention branches. That would confuse the
processor-input axis with the intervention axis and could change receipt
semantics or multiply the planned forwards. No code had implemented that
incorrect design. The design document now specifies three base batches for the
QTA cohort, reused across all nine branches, and explicitly separates the
legacy 3×3 claim-mask path.

## Verification and limits

Ran the same-source Phase-1/Phase-2 synthetic composition and phase-receipt
regression suites: the existing **42/42 passed**. Added a focused regression
test that asserts for all three synthetic rows that the receipt, baseline,
Phase-2 plan, and each of nine branch descriptors carry the same single
processor digest; the expanded composition suite passes **3/3**. These fixtures
use synthetic batch digests; they do not test the corrected proposed schema. No
protected video, human review, model, or GPU was used. No real batch manifest,
trusted caller pin, or production scorer was created.

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link judgments, and **0/3**
direct scores. No direct effect estimate, confidence interval, or test exists.

## Next action

The next implementation can add the versioned digest-manifest role to a new
pre-baseline receipt domain/schema while preserving v1 verification exactly.
Keep it non-authorizing and require a separately supplied expected artifact
hash. The production caller must remain blocked until independent
review/root/clock/prediction-link gates are genuinely present.
