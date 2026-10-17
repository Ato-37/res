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
- Full repository test discovery after CLI, inventory registration, and final
  handoff documentation: 1,337/1,337 passed in 25.7 seconds.
- Compilation and whitespace checks passed.

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

The human-review runbook now documents when to call the local `sign`/`verify`
commands and where their trust inputs must come from. The pre-review inventory
builder registers the attestation code as a required source artifact; a fresh
local inventory contains 62/62 roles with zero missing slots and remains
`candidate_only`, not frozen or score-ready. The inventory itself stays private.

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

## 21:53–22:00 JST — bind completed-review signature to the trusted role roster

An independent trust-boundary audit found that the first CLI version accepted
the reviewer key ID and fingerprint from its caller. It therefore proved
control of a selected key, not that the key had the `claim_reviewer` role. The
attestation now validates the root-signed three-role integrity roster against
an externally pinned root key, derives the reviewer ID and public key from its
`claim_reviewer` entry, and signs/verifies against that key. The body binds both
the exact roster hash and trust-root fingerprint. This is a new envelope v2;
the frozen v6 review exports and policy are unchanged.

Synthetic verification passed: 12 attestation tests and the 17 existing
integrity-receipt tests (29 total in the focused module invocation); full
repository discovery passed 1,354/1,354 tests in 27.9 seconds. Compilation and
`git diff --check` passed. A new private inventory at
`data/private_nextgqa/pre_review_inventory_v6_20261017_2158_01/` has 62 roles,
zero missing, candidate-only status, and modes 0700/0600. Its local SHA-256 is
`dc4b1d4f587703b3a46daa816a7bf438d23e5ec2d281a87f591a053f7f88fcbe`; the
registered attestation source is 17,389 bytes with SHA-256
`3987052283985f59daf271ff9627292ee9940b8f7f76410cca6b4186687f06e0`. The
inventory contains only derived byte hashes and remains local.

The independent code audit found no critical defect. The trust root still
requires a genuinely out-of-band fingerprint: a caller who supplies both a
self-chosen root and its matching fingerprint can create a self-signed roster.
The pre-review receipt digest remains a caller-supplied, format-checked value;
the receipt itself, roster-to-cohort link, witness ceremony, key-to-person
identity, actual video viewing, and semantic review are not verified by this
attestation. `reviewer_role_roster_verified=true` now means only that the
presented roster validates under the presented root pin; scoring remains
`not_granted` and all identity/semantic/media-custody flags remain false.

Faithfulness progress is unchanged: direct generated-claim evaluation is
0/3 independently reviewed and 0/3 scored, with no direct estimate, interval,
or hypothesis test. This is handoff integrity engineering, not faithfulness
evidence. No GPU run was justified. Next, independently close a real
trust-root/release-order and frame-correspondence gate, or document which
specific artifact is still absent before any real review or scoring attempt.

## 22:00 JST — recheck the real human-review prerequisites

Independent metadata-only inspections of the local release and clock-evidence
areas confirm that the trust ceremony cannot currently be instantiated with
real materials. No externally pinned root, root-signed integrity/release
rosters, freeze-owner receipt, witnessed release attestation, or complete
release evidence set is present. The latest candidate inventory is only a
candidate inventory; it is not a freeze or release receipt. These missing
artifacts require trusted human role assignment and a witnessed pre-opening
ceremony, so local synthetic testing cannot substitute.

The clock path has the 15 planned decoder PNGs and plan/provenance files, but no
actual browser screenshots, capture observations, screenshot receipts, crop
map/crops, frame-review export, or signed evidence manifest. Verified
correspondences therefore remain **0/15**. The screenshot helper requires an
operator to use the active Windows browser and foreground-window capture; a
prior validate-only or permission smoke does not count as a capture. The
runbook has been corrected to describe attestation v2's externally pinned
roster inputs and its still-unverified receipt-digest declaration.

Rebuilt private candidate inventory: 62 roles, zero missing, 0700/0600,
candidate-only; review, freeze, and score flags all
false, execution denied, rights unverified-local-only. Inventory SHA-256 is
`2ba107edfff577ac466fd16f664ac686dfafe4b1e9fa45ea2c72c9c608994cb0`; attestation
code hash is `3987052283985f59daf271ff9627292ee9940b8f7f76410cca6b4186687f06e0`.

Direct generated-claim faithfulness remains **0/3 independently reviewed,
0/3 scored**, with no direct estimate/CI/test. Fifteen decoder anchors are
prepared but there are zero verified screenshot correspondences. No GPU run
was justified because neither real human review nor its release/capture
prerequisites exist. The exact external blocker is the absence of trusted
human role/key setup and witnessed release, followed by interactive browser
capture and independent visual frame-pair review. Continue only preparation
that can reduce those gates; do not call the synthetic checks research evidence.

## 22:04 JST — audit whether v2 measures explanation faithfulness

An independent source-and-method audit checked the v2 prompt, score, and
intervention against the thesis construct. The QA scoring prompt contains the
question and options but omits the generated claim. The claim selects which
video frames are black-masked. The outcome is a gold-versus-best-foil QA
next-token margin. Therefore, even after every human and score gate passes,
v2 can support only a finite three-record description of claim-aligned visual
input dependence. It cannot establish whether the model used the explanation
text, whether the explanation reflects internal reasoning, or whether the
explanation is faithful/true. A positive contrast also can come from a gold
score change, a condition-specific best-foil change, or both; black fill can
create OOD inputs and the control does not match semantic information.

This construct distinction agrees with primary ACL work: Atanasova et al.
propose NLE tests that intervene on counterfactual reasons absent from the
explanation or reconstruct inputs from reasons in the generated explanation
([ACL 2023 paper](https://aclanthology.org/2023.acl-short.25/)). Their later
discussion separates causal output dependence from whether explanation text
represents useful predictive factors ([ACL 2024 paper](https://aclanthology.org/2024.acl-short.49/)). This supports narrowing the current v2 result label; it does not itself prove a replacement video metric.

No v2 outcomes have been inspected, so this is a pre-outcome construct-validity
correction. Keep v2 unchanged only as a claim-aligned visual-evidence
dependence feasibility study, or prepare a new pre-outcome protocol if the
thesis requires a stronger explanation-faithfulness claim. The closed head
search remains closed. The concrete v3 estimand is unresolved because the
thesis target (textual rationale dependence, visual evidence attribution, or
both) cannot be inferred from the current QA-margin endpoint.

Faithfulness status: direct generated-explanation faithfulness **not measured**;
review 0/3, scoring 0/3, no effect estimate/interval/test. Real release artifacts
and actual frame correspondences remain absent (0/15). This is a design audit,
not an outcome. Next, establish which explanation-faithfulness construct should
be the thesis endpoint before any score outcomes exist; continue no-go on GPU
claim-mask scoring until human/trust/capture gates and production preflight pass.
