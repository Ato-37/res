# Validation annotation-key-bias sequential cohort 3 (plan)

## Purpose and scope

The attempted held-out test-split continuation could not produce one unexposed
local candidate: all 85 locally available test video groups matched the
recorded exposure exclusion union. To continue a useful real-video diagnostic
without discarding those exclusions, this plan prospectively fixes a third
disjoint **validation** cohort under the same existing QA-annotation
key-bias-versus-hash-sham protocol. This is a sequential exploratory
continuation after two validation cohorts have already been observed; it is
not confirmatory and will not be called a new test split or pooled with those
cohorts for inference.

Primary per-video endpoint is unchanged:
`(gold - fixed baseline-best-foil margin under QA-annotation key bias) -
(same margin under equal-size disjoint hash-sham bias)`. QA intervals are
benchmark annotations, not claim-specific evidence. This estimates internal
answer-margin sensitivity only, not explanation faithfulness, semantic claim
support, or clock-calibrated grounding.

## Fixed cohort and run rules

- Freeze exactly **20** validation videos using the current runner's
  deterministic hash ordering, 32-frame decode, strict interval/timing
  checks, and disjoint-sham geometry. Exclude the 1,034-hash known-media union,
  latest 117-hash claim-exposure union, and all local validation/test split
  lock media. The exact private exclusion-manifest bytes are the cohort input.
- If fewer than 20 rows pass, stop before model load; do not reduce N, replace
  rows, or loosen exposure/timing/geometry criteria after seeing outcomes.
- If preflight passes, run the existing one-row/3-forward smoke and exact
  20-row/60-forward full job with the existing pinned model/protocol: Qwen
  2.5-VL-7B-Instruct, 4-bit weights/BF16 compute, layers 5-16, requested
  `+ln(2)` visual-key bias, 32 frames, baseline-fixed foil, and deterministic
  equal-size disjoint hash sham. Freeze source, lock, preflight and smoke
  receipts. No metric, dose, layer, or subgroup changes are allowed.
- Report the finite-cohort mean, median, range, sign counts and accuracy for
  each condition. Any t interval is descriptive only. Do not run a p-value or
  population inference and do not combine with the prior 40 rows as if
  independently sampled.
- Keep all detailed inputs and outputs local and owner-only; rights remain
  unverified. Human review and clock calibration remain explicitly false.

## Rationale and limits

The two completed validation cohorts had small positive annotation-bias minus
sham means (+0.0125 and +0.0250 nat), near-zero medians, and mixed per-row
directions. A third non-overlapping cohort is a useful stress test of whether
that descriptive pattern persists on additional fixed validation videos.
Because the same validation candidate process and prior outcome visibility
remain, the result is sequential exploratory evidence only. A positive result
cannot validate the interval or claim as semantic evidence; a negative result
cannot prove that all internal interventions fail.

## Outcome-blind feasibility result

The same mode-0600 exclusion manifest used for the attempted test-split
preflight was applied to validation. The fixed-N=20 preflight failed closed
before model loading with `requested_20_eligible_samples_but_found_0`. A
CPU-only census found 33 validation video groups in the current local
candidate archive, all 33 had video files, and all 33 media hashes matched
the private exposure exclusion set. Therefore no eligible third-cohort row
exists in the present local validation inventory. No lock, score, or GPU
forward was produced. Do not remove exclusions or lower the frozen N to force
a run.

This plan is closed for the current local video inventory. Obtain genuinely
new eligible video material or continue non-inference readiness work before
attempting another real-video cohort. The two prior validation cohorts remain
the only QA-annotation key-bias rows; direct claim-support faithfulness remains
unmeasured.
