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
- Combined clock-adapter, processor-adapter, composition, and cohort-wrapper
  suites: **20/20 passed** with
  `PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_synthetic_adapter_composition test_qta_synthetic_cohort_preflight -v`.
- `py_compile` and `git diff --check`: passed.
- An independent audit reran the clock, processor, composition, and cohort
  modules: 33 test executions passed (20 distinct methods; imported clock and
  processor test classes run twice in that command). The audit confirmed the
  fixed 3-row denominator, no partial row hashes on failure, and
  non-authorization. It also confirmed that fixtures are synthetic and do not
  exercise a real Qwen processor or browser capture.
- Full `scripts/` unittest discovery passed **1,928/1,928** in 60.708 s after
  the final privacy assertions. The CUDA devices were hidden from this
  CPU-only regression run.
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
  `575eaf40683668c5f85581bd6104cdf5e3f3a18cefb8ef4e99cca41621836e1a`.
- Syntax and whitespace checks: `PYTHONPATH=scripts .venv/bin/python -m py_compile scripts/qta_synthetic_cohort_preflight.py scripts/test_qta_synthetic_cohort_preflight.py scripts/test_qta_synthetic_adapter_composition.py` and `git diff --check`.
- Next, complete an independent audit and full CPU regression pass, publish
  only this code/test/log, then implement the non-authorizing composition
  contract that combines these row adapters with the signed phase receipts.
  Real scoring remains blocked by missing actual trusted role assignments,
  stage-1 review, 15 timing correspondences, and stage-2 judgments. Rights
  remain unverified; protected files stay local.
