# Claim-review clock export schema compatibility (2026-10-19)

## Purpose

Repair a handoff mismatch discovered during the independent review of a
one-file reviewer-export design. The current claim-review renderer emits
clock-observation payload schema 3, while the byte-preserving staging validator
accepted only schema 2. This made the renderer's current optional clock output
unimportable even when its integrity index matched.

## Change and verification

`stage_nextgqa_review_exports._validate_exports` now accepts clock observation
schema 2 (existing exports) and schema 3 (current renderer), while still
requiring the same purpose, raw-observation status, and
`execution_authorization=not_granted`. Staging copies the original payload bytes
unchanged; it does not normalize or rewrite either schema.

Synthetic tests cover byte-identical staging of schema 3 and rejection of
schema 4 before any output is written. The prior schema-2 case remains covered.
Focused suites passed **20/20** across the staging validator and claim-review
renderer. Python compilation and scoped `git diff --check` passed. No protected
review response, media, or human identity was opened or used.

## Interpretation and limits

This repairs an actual format mismatch and permits the current optional clock
payload to pass structural staging. It does not validate clock-to-decoder-frame
correspondence, establish reviewer identity, or make browser timestamps trusted.
The records remain explicitly unvalidated and non-authorizing.

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. No direct `ΔF`, confidence interval, or test exists. Rights
status remains unverified; all protected materials stay local.

## Next action

Continue building and independently testing a single-file container that wraps
each existing review payload and integrity index byte-for-byte, preserving
reviewer boundaries and any optional clock pair. Then test round-trip import
through this validator and the downstream custodian handoff audit using
synthetic fixtures only.
