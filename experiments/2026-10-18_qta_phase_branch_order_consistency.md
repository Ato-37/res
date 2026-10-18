# QTA phase branch vocabulary and order consistency (2026-10-18)

## Purpose

The nine-branch plan and the two-phase schedule represented the same
conditions with different labels: the plan uses `target_*` and
`*_evidence_ablation`/`*_control_ablation`, while the phase schedule used
`qta_*` and shortened suffixes. In addition, the schedule checker compared
branch names as a multiset, so a reordered schedule passed. This change aligns
the two CPU-only candidate descriptions and makes the frozen order explicit.

## Implementation

- `qta_nine_branch_plan.BRANCH_ORDER` is now the canonical vocabulary and
  order. Phase 1 contains only `baseline_full`; phase 2 is the exact remaining
  eight branches. The phase-freeze receipt candidate uses the same tuples.
- The phase schedule diagnostic requires exact branch sequence equality per
  row; reordering, omission, duplication, or adding `baseline_full` to phase 2
  blocks the entire fixed cohort. The 3-row/24-forward schedule counts remain
  shape-only and do not authorize execution.
- Tests assert the phase tuple matches the nine-branch plan and reject a
  permuted phase-2 row.

## Checks and limits

- Focused phase schedule, phase receipt, nine-branch scorer, and three-row
  aggregate tests: **57/57 passed**. Syntax compilation and `git diff --check`
  passed.
- Full repository scripts regression: **1,916/1,916 passed** in 60.583 s
  using `.venv` with CUDA hidden.
- Independent audit confirmed canonical branch ordering and the 3+24 count;
  focused independent tests passed **34/34**. It also confirmed that signed
  receipt verification still treats the nine-branch plan artifact as opaque:
  this patch does not validate the artifact's internal semantics or integrate
  the schedule diagnostic with a production execution gate.
- No protected data, review artifacts, model, or GPU were accessed. This only
  checks condition naming, order, row count, and existing structural plan
  tests. It does not authenticate that a `target` selection is actually the
  model's question attention or that group labels faithfully represent human
  video evidence; all supplied bytes remain unauthenticated and the candidate
  receipts remain non-authorizing.

## Faithfulness progress and next action

- Direct evaluation remains 0/3 claim reviews, 0/15 verified clock-to-decoder
  frame correspondences, 0/3 prediction-link judgments, and 0/3 direct scores.
  There is no empirical `DeltaF`, interval, inferential test, or multiplicity
  correction.
- This is auxiliary protocol consistency only; no model forward or metric was
  run. Next, finish independent audit and full CPU regression, then investigate
  an explicit row-level composition schema for signed phase artifacts without
  presenting it as a production gate. Actual scoring remains blocked on the
  human trust/review/capture evidence. Rights remain unverified; protected
  materials stay local.

## Source hashes

- `scripts/qta_phase_schedule_diagnostic.py`:
  `07920f2e82b69addee23577783d64da3ef887ccdafd911fe1d79e8b160c988b2`
- `scripts/qta_phase_freeze_receipts.py`:
  `4de3828aa22e230766f74d42788f76e6ba29014e5e78b05d3c601db67f6e11d5`
- `scripts/test_qta_phase_schedule_diagnostic.py`:
  `229ce4cd2470cb4702bc32aa8b922848fc450cc3d226dffcecc92c6f6c413ec5`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `3c89d6d2ce46abe6fb6aaedc638487035759930eaa5e02ca8c1694dd71bbb1db`
