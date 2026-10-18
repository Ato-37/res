# Stage-1 source-pin to phase-1 receipt composition (2026-10-19)

## Purpose

Check that the actual output shape of the synthetic stage-1 source-pin adapter composes with the phase-1 external cohort pin and signed-freeze verifier. Existing tests exercised the adapter and phase-freeze receipt independently using separate fixtures; this focused integration check closes that interface gap without using protected review data or authorizing model work.

## Setup

- Added `scripts/test_qta_stage1_phase1_composition.py` using the existing synthetic stage-1 adapter fixture and synthetic phase-freeze fixture.
- The test passes the adapter's serialized output into `build_external_cohort_pin_candidate`, binds those exact bytes into a phase-1 receipt, and verifies the signature and report.
- A negative case changes a source-row claim digest while reusing the previous external pin; receipt construction must reject the mismatch.
- All key generation and inputs are synthetic. No GPU, real review artifact, or model score is used.

## Verified results

- Focused composition smoke: **1/1 passed**; three ordered source rows composed with a valid signature, while the verifier reported zero authorized forwards and `scoring_authorized=false`.
- Relevant suites (`test_qta_stage1_phase1_composition.py`, `test_qta_stage1_source_pin_adapter.py`, and `test_qta_phase_freeze_receipts.py`): **102/102 passed** in 53.52 seconds.
- The negative source-row mutation was rejected with `external_cohort_pin_cohort_binding_mismatch`.

## Interpretation and limitations

This verifies a synthetic software interface from source-pin derivation through phase-1 byte/signature verification. It does not authenticate a real trust root or reviewer, establish semantic review correctness, prove clock/frame correspondence, validate prediction links, or authorize scoring. The phase receipt explicitly remains non-authorizing. This is reproducibility plumbing, not evidence of improved faithfulness.

## Faithfulness progress and next action

- Direct explanation-faithfulness scoring remains **0/3 claims**; no effect estimate, confidence interval, or hypothesis test was added.
- Formal claim review remains **0/3**, clock-to-decoder correspondence **0/15**, and prediction-link review **0/3**.
- Next, inspect whether the later phase-2 plan manifest has an equivalent test using the real adapter output or only manually constructed pins. If a missing composition test remains, establish it with synthetic inputs; do not change trust or review gates based on this test and do not launch a GPU score run.
