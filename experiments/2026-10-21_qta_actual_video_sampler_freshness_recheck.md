# QTA actual-video sampler freshness recheck — 2026-10-21

## Purpose

Refresh the local pre-review code inventory after recent changes, then verify that the fixed three-video cohort still decodes and joins to the actual QA source by video-byte identity. This is a data/readiness check, not a faithfulness score or execution authorization.

## Change and checks

The inventory builder's static import closure found one newly required local dependency, `qta_processor_batch_digest_manifest_v1.py`, missing from its registered code roles. Registered it and added it to the builder's required-role regression test. The first newly generated candidate correctly remained incomplete (96 roles, one missing closure slot). After the repair, a fresh candidate contained 97 roles and zero missing required slots, while remaining `candidate_only` and `execution_authorization=not_granted`.

The current candidate passed the CPU sampler on the fixed three local videos: 3/3 decoded, 32 frames sampled per video; decoded frame counts were 510, 1,140, and 1,258. Maximum grid deviations were 16.145, 16.145, and 16.355 ms. The actual-QA source preflight then joined the three sampler source digests to the three fixed QA rows: 3/3. The report kept all authorization, authentication, model-load, forward, and scoring fields false/zero.

Validation passed 21/21 inventory-builder tests, 100/100 pre-baseline and scoring-gate tests, and 38/38 inventory/sampler/source-join integration tests. The 100-test run uses synthetic fixtures and only verifies fail-closed wiring. No model was loaded and no GPU inference ran.

## Interpretation and limits

The change repairs stale code-closure accounting. The local real-video run establishes only current decoder/sampler feasibility and source-video byte identity for this fixed three-row candidate. It does not establish claim truth, human review, clock/frame correspondence, reviewer identity, sampler provenance, or faithfulness. The candidate remains pre-review and non-authorizing. Rights status remains unverified; all raw inputs and detailed receipts remain local.

Direct faithfulness progress is unchanged: **0/3 claims reviewed, 0/15 clock/frame correspondences, 0/3 Stage-2 prediction-link reviews, and 0/3 direct scores**. No direct effect estimate, confidence interval, or test exists. The GPU baseline is still blocked on trusted role/release inputs, completed Stage-1 semantic review, capture/frame review and calibration, frozen processor/prompt bindings, and a production pre-load transaction runner.

## Next action

Use the current local review handoff to complete the human Stage-1 claim judgments and clock/frame work. Once those authenticated inputs exist, implement and verify the production atomic pre-load transaction and run only the three authorized baseline forwards; then collect the separate Stage-2 prediction-link judgments before any treatment/control forwards. Do not treat this sampler/source join as a model result.
