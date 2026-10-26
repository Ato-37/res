# NExT-GQA human-review role assignment intake

日本語版は
[`nextgqa_human_review_role_assignment_template_ja.md`](nextgqa_human_review_role_assignment_template_ja.md)を参照。

Fill this sheet locally before opening any review package. Use a stable
reviewer identifier and a local public-key file path; do not put private-key
material, raw video, claim text, answers, or personal contact details here.
The external root fingerprint must come from a separately trusted channel,
not from the package being reviewed.

## Trusted root

- Root authority ID: `[fill locally]`
- Root public-key path (regular mode-0600 file): `[fill locally]`
- Independently obtained root DER SHA-256 fingerprint: `[64 lowercase hex]`
- How the fingerprint was verified out of band: `[record channel/procedure]`

## Role roster

| Role | Person/reviewer ID | Key ID | Public-key path (mode 0600) | Independent of |
|---|---|---|---|---|
| Pre-outcome freeze owner | `[fill]` | `[fill]` | `[fill]` | claim generation, outcome scoring, reviewers, capture operator |
| Custodian | `[fill; may alias freeze owner]` | `[fill; owner key only if same person]` | `[fill]` | claim generation, outcome scoring, claim reviewer |
| Capture operator | `[fill]` | `[fill]` | `[fill]` | freeze owner/custodian, claim reviewer, frame-pair reviewer, prediction-link reviewer, witness |
| Stage-1 claim reviewer | `[fill]` | `[fill]` | `[fill]` | claim generation, scoring, custody, capture |
| Frame-pair reviewer | `[fill]` | `[fill]` | `[fill]` | capture, custody, pair preparation, scoring |
| Stage-2 prediction-link reviewer | `[fill]` | `[fill]` | `[fill]` | stage-1 reviewer, generation, scoring, custody, freeze owner, witness |
| Release witness | `[fill]` | `[fill]` | `[fill]` | every other role; independently root-pinned |

The root authority must sign the integrity and release rosters only after it
has established the person-to-key assignments through its independent channel.
Only the freeze-owner/custodian alias may reuse a role key. The release witness
uses a unique identity and key. The prediction-link reviewer must have a key
distinct from every other release-roster role.

## Before review begins

- [ ] Freeze owner has selected the exact review artifact inventory and frozen
  protocol/sampler, with all relevant hashes recorded before anyone views the
  package or outcomes.
- [ ] Custody and role rosters are signed by the trusted root; the root pin is
  independently verified.
- [ ] Freeze owner has signed a `review_only` receipt. It grants no model load,
  scoring, or experiment execution.
- [ ] Witness has personally observed the unopened package handoff from freeze
  owner to custodian and signed a fresh ceremony statement.
- [ ] Release-order audit passes immediately before the package is opened.
- [ ] Reviewers have acknowledged the runbook's disclosure and blinding rules.

## Review completion checklist

- [ ] Stage 1: 3/3 claim-support judgments signed against the exact frozen
  export bytes.
- [ ] Capture: 15/15 planned browser screenshot / decoder-frame captures have
  raw clock events, private screenshots, video-only crops, and capture receipts.
- [ ] Frame pairs: an independent reviewer has signed all 15 match/mismatch/
  ambiguous decisions. Any missing, mismatched, or ambiguous pair blocks the
  cohort.
- [ ] Stage 2: 3/3 separate prediction-link judgments signed against the exact
  post-baseline worksheet bytes.
- [ ] All structural, signature, source-lineage, clock calibration, and cohort
  audits pass before any scoring phase is considered.

This sheet is an intake aid only. Completing it does not authenticate any
identity, open a package, freeze a cohort, or authorize model scoring. Follow
the current procedure in `nextgqa_v6_human_review_runbook.md`; its production
score-freeze gates remain required.
