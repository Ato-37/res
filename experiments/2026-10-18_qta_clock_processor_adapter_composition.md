# QTA clock/processor adapter composition and atomic failure counts (2026-10-18)

## Purpose

Close the next human-review-free test gap noted in the synthetic atomic input
adapter log: exercise actual clock-row and processor-row adapters together
through the three-record QTA structural binder. This is implementation
feasibility only; it neither authenticates the caller-provided fixture nor
measures explanation faithfulness.

## Implementation and finding

Added `scripts/test_qta_synthetic_adapter_composition.py`. It builds three
synthetic HTML5 clock-capture/calibration outputs through the existing clock
adapter and three processor rows through the existing processor adapter, then
passes those exact objects into `preflight_synthetic_qta_cohort` with the
existing fixed review/baseline rows. The valid composition passes 3/3
structural rows. A per-row processor-batch mismatch and a clock/video mismatch
each block the full requested cohort and emit no row hashes.

The integration first failed because the synthetic mapper fixture's fixed
calibration hash and 50 ms guard did not match the actual clock adapter's
calibration output and 10 ms guard. The test now derives the expected
calibration digest and processor guard from the clock adapter output. This
exposed an atomicity-reporting issue: the cohort wrapper returned 1 or 2
`verified_structural_rows` after a later-row mismatch, even while blocking the
cohort and suppressing row digests. Updated
`scripts/qta_synthetic_cohort_preflight.py` so every unsuccessful cohort
reports zero verified rows; its regression now checks this contract.

## Checks and limits

- New adapter composition tests: **3/3 passed**.
- The first focused composition + cohort run reported 20 passing executions,
  including 13 duplicated clock/processor test methods; those imports were
  corrected as described below. The corrected command is the 7-test run
  recorded below.
- `py_compile` and `git diff --check`: passed.
- An independent audit confirmed the fixed 3-row denominator, no partial row
  hashes on failure, and non-authorization. It also found that importing the
  clock/processor `TestCase` classes directly caused duplicate test execution
  during full discovery. Changed the composition test to use module aliases.
  The focused command now runs **7 tests** (3 composition + 4 cohort), all
  passing; the independent audit confirmed there are no duplicate IDs.
- After removing those duplicate test registrations, full `scripts/` unittest
  discovery passed **1,915/1,915** in 60.282 s using the repository `.venv`
  with CUDA hidden. An independent attempt using system `python3` did not pass
  because that interpreter lacked required dependencies; it is not counted as
  validation evidence.
- The successful-composition test checks that output JSON contains neither
  row tokens nor source-video hashes.
- These tests use synthetic capture/calibration metadata and synthetic mapper
  objects. The clock adapter explicitly marks correspondence unauthenticated;
  the cohort wrapper explicitly marks provenance/untrusted prediction-link
  semantics and keeps model/scoring authorization denied. No media, review
  material, checkpoint, model, or GPU was accessed.

## Faithfulness progress and next action

- `直接評価済み`: unchanged at 0/3 reviewed claims, 0/15 verified
  browser/player-clock to decoder-frame pairs, 0/3 prediction-link judgments,
  and 0/3 direct claim-linked scores. No real `ΔF`, interval, hypothesis test,
  or multiplicity adjustment exists.
- `補助診断のみ`: adapter composition and atomic-denominator behavior; no
  model forward or faithfulness metric.
- Source hashes for this reproducible change: `scripts/qta_synthetic_cohort_preflight.py`
  `4eb322fb737128c09233e7fed2021b5c9941fd0ca0e4fd336b118be90b215fe3`;
  `scripts/test_qta_synthetic_cohort_preflight.py`
  `6bfdc27be21457d095c1d6e512ff06690363e89f7667ed4b1d1eb6ec5df0543b`;
  `scripts/test_qta_synthetic_adapter_composition.py`
  `d8b81a08d6ab3d105941dad741d55dfc4f2f016822851fa2729b4614240500a5`.
- Syntax and whitespace checks: `PYTHONPATH=scripts .venv/bin/python -m py_compile scripts/qta_synthetic_cohort_preflight.py scripts/test_qta_synthetic_cohort_preflight.py scripts/test_qta_synthetic_adapter_composition.py` and `git diff --check`.
- Next, specify and independently audit canonical per-role schemas and row
  cross-links for phase-1 preflight and phase-2 branch-plan artifacts before
  implementing a non-authorizing composition verifier. The role-name byte
  receipts alone do not validate artifact meaning. Real scoring remains
  blocked by missing trusted role assignments, stage-1 review, 15 timing
  correspondences, and stage-2 judgments. Rights remain unverified; protected
  files stay local.
