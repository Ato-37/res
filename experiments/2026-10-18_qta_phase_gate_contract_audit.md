# QTA phase-gate contract audit (2026-10-18)

## Purpose

Resolve ordering and artifact-binding contradictions before implementing any
QTA runner that could load the model on reviewed real data. The intended primary
contrast remains the user-selected internal intervention on dependence of a
fixed baseline prediction on reviewed video evidence.

## Verified findings

- The QTA draft requires baseline prediction/foil `p/q` before stage-2
  prediction-link review, then requires stage-2 before treatment/ablation
  forwards. The current human-review runbook described one score freeze before
  model loading, so that ordering could not be satisfied.
- The QTA draft says all outcome branches use the same original full-video
  processor batch. The runbook had required “all nine exact processor batches”
  per record, which was inconsistent with the draft.
- The existing stage-2 receipt validator is structural only. It returns
  `reviewer_identity_verified=false`,
  `semantic_judgment_authenticated=false`, and
  `execution_authorization=not_granted`. No trusted-roster signature/adoption
  path for the stage-2 reviewer or independent receipt auditor exists.
- The runbook required a 32-frame endpoint-inclusive sampler, but the QTA draft
  did not define the sampling algorithm/source/configuration. An independent
  audit confirmed that sampler choice affects interval overlap and eligibility,
  so it must be frozen before reviewers inspect intervals.
- No production QTA score-freeze schema/verifier or baseline-only runner exists.
  Existing cohort plan aggregation remains structural and non-authorizing.

## Changes

Updated `docs/nextgqa_v6_human_review_runbook.md` to describe two prospective
phases: a pre-baseline gate limited to exactly three baseline forwards (one per
row), and a post-baseline gate limited to the remaining 24 forwards (eight per
row) after the stage-2 judgments. The runbook now binds one immutable
full-video processor batch per row and nine branch plans to that batch. It also
requires the sampler contract and source bytes to be frozen before review,
calls out missing stage-2 reviewer/auditor keys and signature verification, and
states that the phase schemas/runners are not implemented. The referenced
32-frame sampler specification is the endpoint-inclusive contract in
`docs/nextgqa_finite_claim_descriptive_protocol_v2.md`; this QTA binding still
needs a versioned protocol/reference hash before any review begins.

An independent protocol auditor inspected the QTA draft, runbook, and current
trust/receipt code and confirmed all four findings. A separate design review
independently inspected the release-roster verifier, stage-1 bundle audit,
clock/frame audit, stage-2 validator, plan aggregator, and current runners; it
confirmed there is no production phase receipt or score gate. Both reviews
emphasized that cryptographic key adoption cannot prove real-world identity or
semantic correctness.

## Verification and limits

- `git diff --check`: passed.
- No code, tests, media, model, or GPU execution changed or ran in this
  documentation-only cycle.
- The local candidate inventory now predates the edited runbook; it is not a
  freeze and must be rebuilt with a unique name before any future release.
- The current release-roster schema does not contain stage-2 reviewer/auditor
  roles. The documentation records this as an explicit no-go rather than
  suggesting the current roster can satisfy it.
- The proposed phase limits are a design contract only. They do not grant
  authorization and are not backed by a production verifier or runner.

## Faithfulness progress

- `直接評価済み`: none in this cycle. Current direct real-data score status
  remains 0/3; stage-1 reviews 0/3, stage-2 adjudications 0/3, verified
  player-clock/decoder correspondences 0/15.
- `補助診断のみ`: protocol ordering and source-code contract audit. No
  empirical effect estimate, uncertainty interval, hypothesis test, or
  multiplicity adjustment exists for this branch.
- Rights status remains unverified; all protected material remains local.

## Next action

Implement and test an additive detached-signature envelope for the stage-2
prediction-link receipt, rooted in a dedicated QTA scoring roster that is
independent of the package-opening roster. Then independently audit the
cryptographic role-separation and exact-byte binding. This still will not
authorize scoring; after that, implement the two non-overlapping phase
receipts and baseline-only/branch-only execution gates, including a strict
all-three-row preflight. Do not start a real GPU score until human review,
15/15 timing correspondence, sampler freeze, trust, and both phase gates pass.
