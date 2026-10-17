# 2026-10-17 — non-authorizing signature for completed NExT-GQA review exports

## Purpose

The v6 claim-review export structurally binds the policy, worksheet, review
index, and declared review fields, but its reviewer identity and decisions are
self-reported. Add an optional detached signature over a structurally valid
export without changing the frozen review page, policy, or JSON schema. This
supports later provenance checks; it does not establish semantic validity or
permit model execution.

## Implementation and setup

`scripts/nextgqa_completed_review_attestation.py` signs/verifies a
domain-separated Ed25519 envelope. It binds the exact worksheet, source-lineage
manifest, review JSON, review index, v6 policy fingerprint, pre-review receipt
digest, reviewer key ID, and an externally supplied reviewer public-key
fingerprint. Before signing or verification, it invokes the existing structural
handoff audit and requires its unvalidated, non-authorizing pass. The module
provides `sign` and `verify` CLI subcommands. It requires private input files
(0600), creates outputs without overwrite under private directories (0700),
and prints only sanitized status. It does not read real review/video files
unless a caller supplies them.

Tests use only generated synthetic review artifacts and temporary Ed25519 keys.
No protected candidate, real review output, existing private key, or GPU model
was used for this change.

## Verified checks

- Focused attestation tests: 12/12 passed, including private CLI sign/verify,
  output permissions, overwrite refusal without changing prior bytes, and
  symlinked private-input rejection.
- Full repository test discovery after CLI and final adversarial checks:
  1,337/1,337 passed in 25.4 seconds.
- Compilation and whitespace checks after CLI change passed.

Adversarial coverage includes exact-byte changes, stale policy/index bindings,
incomplete review rows, duplicate JSON keys, exponent-overflow nonfinite values,
bad lineage and reviewer-key pins, mismatched signing keypairs, and envelope
tampering. A successful test report explicitly keeps reviewer identity,
semantic review authentication, receipt verification, roster verification,
media-custody verification, and execution authorization false.

## Interpretation and limits

A valid signature means only that the holder of the supplied private key signed
or adopted those exact structurally valid bytes. This module does not itself
verify the pre-review receipt, root-signed reviewer roster, or key-to-person
mapping; expected receipt and key pins must come from a trusted external
workflow. The structural handoff audit does not prove that a person watched the
source video or that the decisions are true, and it does not perform the
separate on-disk media custody audit. The attestation is not a review receipt,
human-authentication result, score freeze, or scoring permission.

Direct generated-claim faithfulness remains 0/3 independently human-reviewed
and 0/3 scored, with no direct estimate, confidence interval, or test. The
reviewer, independent frame-reviewer, witnessed-release, and browser-to-decoder
correspondence gates remain open. Rights status remains unverified; all
protected data and derived review material remain local.

## Next action

Have an appropriately trusted reviewer/custodian workflow provide the external
receipt digest and reviewer key fingerprint, then validate this tool against a
real completed export only after the separately required witnessed review
release. Continue to deny scoring until all upstream human, timing, and atomic
score-freeze gates pass. Do not run another GPU claim-score job before then.
