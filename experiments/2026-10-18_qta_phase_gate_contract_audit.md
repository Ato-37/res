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
  `execution_authorization=not_granted`. At the start of this audit, no
  trusted-roster signature/adoption path for the stage-2 reviewer existed.
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
calls out the stage-2 reviewer key and signature verification requirement, and
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

- Initial documentation pass: `git diff --check` passed; no code, tests, media,
  model, or GPU execution ran in that sub-cycle.
- The follow-up adds a distinct stage-2 reviewer to the release roster. It
  does not add a second human auditor signature or authenticate real-world
  identities.
- The proposed phase limits are a design contract only. They do not grant
  authorization and are not backed by a production verifier or runner.

## Follow-up implementation: stage-2 reviewer key adoption

Added the distinct `prediction_link_reviewer` role to the root-signed release
roster. It is required before the package-opening ceremony and is checked for
key reuse against the claim reviewer, frame reviewer, freeze owner, custodian,
and witness. Added `scripts/nextgqa_prediction_link_attestation.py`, which
reruns the existing stage-2 receipt validator and requires every row to carry
the self-reported `supports_p_over_q` judgment before allowing a rostered key to
sign the exact baseline-manifest and stage-2-receipt hashes. Verification
repeats structural checks and verifies the detached signature. The result
explicitly keeps human identity and semantic judgment unauthenticated and
execution unauthorized. There is no separate human auditor signature; the
independent verifier reruns structural validation from the raw bytes.

Five focused suites for prediction-link attestation, stage-2 structure,
release-order auditing, ceremony signing, and inventory closure passed
**50/50** after the CLI and edge-case additions. `py_compile` for the new
module passed. These tests use synthetic manifests/receipts and temporary
Ed25519 keys; they do not represent human review. They do not exercise the real
private cohort or authorize scoring. The private-file CLI requires 0600 inputs
and writes an exclusive 0600 output under a 0700 directory; an end-to-end
synthetic CLI test confirmed sign, verify, denied authorization, and
no-overwrite behavior.

The first post-change candidate inventory registered the attestation module
but failed its import-closure gate because the stage-2 validator was not
registered. The closure now binds both the validator and offline worksheet
renderer. A fresh private candidate inventory has **70/70** slots present and
zero missing slots; it remains candidate-only, `joint_freeze=false`,
`score_ready=false`, and `execution_authorization=not_granted`. The earlier
candidate is left untouched. It was regenerated after the final runbook edit;
the new JSON is mode 0600 in a mode-0700 directory. No model, video frames, or
GPU was used.

The independent auditor re-read the current role verifier, CLI, runbook, and
inventory closure; independently recomputed the registered code/public-source
hash inventory (47 code entries, no missing or stale hashes); and ran the
focused suite **50/50**. The primary agent also ran the full scripts unittest
suite: **1,593/1,593 passed** in **39.498 seconds**, exit code 0. It emitted
expected negative-path fixture messages and a Transformers video-processor
deprecation warning. `git diff --check` passed. The audit found no path that
grants model-load or score authorization.

## Faithfulness progress

- `直接評価済み`: none in this cycle. Current direct real-data score status
  remains 0/3; stage-1 reviews 0/3, stage-2 adjudications 0/3, verified
  player-clock/decoder correspondences 0/15.
- `補助診断のみ`: protocol ordering and source-code contract audit. No
  empirical effect estimate, uncertainty interval, hypothesis test, or
  multiplicity adjustment exists for this branch.
- Rights status remains unverified; all protected material remains local.

## Next action

Have an independent reviewer re-audit the stage-2 signature envelope, role
separation, and real import closure; repair any issues. Then implement the two
non-overlapping phase receipts and
baseline-only/branch-only execution gates, including a strict all-three-row
preflight. Do not start a real GPU score until human review, 15/15 timing
correspondence, sampler freeze, trust, and both phase gates pass.
