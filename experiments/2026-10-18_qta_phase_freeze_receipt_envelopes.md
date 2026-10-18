# QTA phase-freeze receipt envelopes (2026-10-18)

## Purpose

Build a human-review-free cryptographic byte-binding layer for the QTA
pre-baseline and post-baseline phases. This follows the non-authorizing
schedule-shape diagnostic and addresses receipt chaining only. It is not a
production score gate or evidence that any actual review took place.

## Implementation

Added `scripts/qta_phase_freeze_receipts.py`. The pre-baseline envelope binds
eight required artifact roles: stage-1 review/lineage, clock/frame evidence,
source-backed targets, prompt/answer-token bindings, sampler/model/runtime,
condition/scorer source closure, full-video processor batches, and the atomic
three-row preflight. It binds three opaque record tokens and processor-batch
hashes, fixes the one `baseline_full` branch per row, and records a three-forward
budget.

The post-baseline envelope binds the exact phase-1 receipt bytes, the baseline
`p/q` manifest, stage-2 prediction-link review, nine-branch plans, and scorer
source closure. Verification recursively re-verifies phase 1, requires phase-2
record and processor-batch bindings to match phase 1, and checks the phase-1
receipt hash. The freeze-owner signature is checked against the freeze-owner
key in the root-pinned release roster, whose relationship to the integrity
roster is verified by the existing release-roster validator. Phase receipts
store only artifact sizes and SHA-256 values, not artifact content.

Both phase reports always deny model load, scoring, and phase authorization;
`authorized_forwards` is always zero. The implementation verifies key
adoption and exact artifact bytes only. It does not authenticate people or
semantic judgments, independently obtain the trust-root pin, rerun the
upstream review/clock/model/source audits, or establish that a bundle contains
the named semantics. The artifact-role bundles and their use by a future
runner still require independent protocol review and integration tests.

An independent security audit found that positive forward-budget wording in
the signed object could be misread as permission and that row tokens allowed
readable IDs. The signed receipt schema now itself says
`signed_non_authorizing_candidate`, `execution_authorization=not_granted`,
`model_load_permitted=false`, `scoring_authorized=false`,
`phase_authorized=false`, and `authorized_forwards=0`; planned counts are
labelled `planned_forward_count`. Record tokens must be 64 lowercase hex
characters, and phase-1 verification rejects post-baseline-only arguments.
Tests assert the denial fields in both signed phase objects. The final
independent audit found no remaining schema or authorization blocker.

The token regex only enforces a shape; it cannot establish whether a token was
randomly generated or prevent linkage, and `freeze_id` remains caller-supplied
text. Operational inputs must use non-identifying random tokens and a
non-identifying freeze ID. No anonymity claim follows from these fields.

## Verification

Synthetic tests generate temporary Ed25519 keys and root-signed rosters. They
cover valid phase-1 and chained phase-2 receipts, artifact mutation, prior
receipt substitution, changed row/batch binding, wrong signing key, altered
receipt signature, missing artifact roles, wrong external root pin, and
absence of artifact payload strings from receipts, explicit denial inside both
signed objects, and rejection of readable record IDs. The focused suite passed
**12/12**. The full CPU repository suite passed **1,912/1,912** in 60.779
seconds, exit code 0; GPU visibility was disabled. `py_compile` and
`git diff --check` passed. No user keys, protected review data, media, model,
or GPU were used.

Source SHA-256:

- `scripts/qta_phase_freeze_receipts.py`:
  `6405154cf22e2697018e59a7bc92301fb57eb9caa438c2c6ab52deeaf8306413`
- `scripts/test_qta_phase_freeze_receipts.py`:
  `3b6d172e3a532eea81a34e16b2caf779039b9b54b5a2980fcc08ecf09b866c0b`

## Faithfulness progress and next action

- `直接評価済み`: unchanged at 0/3 independently reviewed claims, 0/15
  verified browser/player-clock to decoder-frame pairs, 0/3 prediction-link
  judgments, and 0/3 direct claim-linked scores. No direct `ΔF`, CI, hypothesis
  test, or multiplicity-adjusted result exists.
- `補助診断のみ`: synthetic signature and exact-byte binding behavior; no
  model inference or faithfulness metric.
- A fresh direct-study cohort is not justified by current local data. The
  previously used validation/test-associated cohorts are closed to outcome-
  driven reanalysis; source/event independence is unresolved.
- Next, independently audit the role/roster contract and all byte-binding
  claims, then integrate only with the all-row leaf-validator preflights. Do
  not open scoring until actual trusted role assignments, stage-1 review, all
  15 clock/frame correspondences, stage-2 judgments, and external trust-root
  pinning exist. Rights status remains unverified and protected files stay
  local.
