# QTA review-contract alignment and sham-seed pin (2026-10-18)

## Purpose

Resolve whether the existing NExT-GQA review handoff clearly supports the
user-selected QTA internal-intervention study, and close a protocol-setting
drift gap in the all-three-row structural plan aggregator.

## Changes

- The review runbook now distinguishes the legacy claim-mask v2 draft from the
  current QTA v1 engineering draft. A review-only freeze must bind both exact
  protocol bytes and the outcome-blind QTA treatment-selection record.
- The runbook explicitly separates stage-1 claim/evidence interval review from
  stage-2 baseline-prediction/foil link adjudication. It records that current
  stage 2 adjudicates the claim against the union of intervals and does not
  connect individual claim clauses to individual intervals. Any score is
  therefore scoped to fixed-prediction dependence on the reviewed interval
  set under the specified intervention/ablation, not general explanation
  faithfulness or claim truth.
- The three-row plan aggregator now requires the exact QTA sham seed
  `qta-sham-seed-20261018-v1` from the protocol, even if all rows consistently
  provide another seed. This prevents a cohort-wide but unauthorized setting
  drift. A redundant temporal-group-width check was removed; the subsequent
  contiguous, disjoint-position invariant already captures the structural
  requirement without forcing groups to have equal sizes.

## Verification

- Focused aggregation and synthetic cohort-plan tests: **17/17 passed**.
- Complete `scripts` unittest discovery: **1,584/1,584 passed** in **35.944
  seconds**; `git diff --check` passed. The suite emitted expected diagnostics
  from negative-path fixtures and one Transformers deprecation warning.
- Independent protocol audit found the score formula consistent with the QTA
  draft. It highlighted unresolved construct limits: the selector reads
  question attention rather than claim content, stage 2 does not bind clauses
  to individual intervals, rank-distance controls do not ensure semantic or
  duration equivalence, and the metric is fixed single-token choice-log-odds
  dependence rather than complete answer probability or claim truth.
- The review runbook still requires a fresh exact-byte inventory, trusted root
  and signed role roster, signed review-only receipt, and witnessed pre-open
  handoff. None is created by this change.
- Independent audit caught an ambiguous later reference to “the same v2
  contract” before model loading; the wording now states that scoring freeze
  must bind the QTA draft and the legacy v2 review/export contract, while the
  legacy status field does not freeze QTA scoring.
- No human review, real-data score, GPU inference, empirical `DeltaF`, interval,
  or hypothesis test was produced.

## Interpretation and limitations

This is protocol/structural validation progress only. Direct faithfulness
evidence remains **0/3** stage-1 reviews, **0/3** stage-2 prediction-link
adjudications, **0/15** verified clock/frame correspondences, and **0/3**
scored records. Thus there is no empirical estimate of internal intervention
improvement. The independent audit supports the arithmetic implementation,
not semantic validity of reviews or efficacy. Review opening and production
scoring remain blocked by the existing trust, witness, human-review, timing,
and production-preflight gates. Rights status remains unverified; all protected
data and any derived review content remain local.

## Next action

Run the complete CPU regression suite and an independent audit of the exact
diff. Then continue implementing a QTA-specific stage-2 review receipt path and
production binding without opening private media or loading the model. Do not
start real scoring until the review-only and later score-freeze gates pass.
