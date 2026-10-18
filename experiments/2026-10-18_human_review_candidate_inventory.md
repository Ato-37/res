# Human-review candidate inventory checkpoint

## Purpose

Refresh the non-authorizing byte inventory required before any reviewer opens
the local NExT-GQA review package. This operation hashes bytes only; it does
not render the HTML, decode videos, inspect review answers, freeze a protocol,
or authorize model execution.

## Setup

- Date: 2026-10-18 (Asia/Tokyo).
- Builder: `scripts/build_nextgqa_pre_review_byte_inventory.py`.
- Private output: `data/private_nextgqa/pre_review_inventory_candidate_20261018_171830/pre_review_candidate_byte_inventory.json`.
- The output remains local under the mode-0700 private directory; file mode is
  0600. Its SHA-256 is
  `f32fb74dfe3a9e7333085dc54d9605f696ab503e288ed64cda92cea8c231415d`.

## Verified results

The builder reported `candidate_only`, **80** inventory artifacts, and **0**
missing required slots. An independent structural check reproduced 80 unique
roles, valid SHA-256 formatting for every artifact, and the explicit flags
`review_started=false`, `joint_freeze=false`, `score_ready=false`, and
`execution_authorization=not_granted`. Rights remain unverified and local-only.

Local readiness tests passed: 16 release-order tests, 4 ceremony-signing
tests, 6 prediction-link-attestation tests, and 20 inventory-builder tests.
The release-order test's sample invocation intentionally failed closed with
`trust_root_key_file_invalid`; it granted neither package opening nor model
load. This matches the absence of a current externally pinned trust root and
signed role rosters in the review workspace.

## Interpretation and limits

The current candidate bytes are structurally inventoried. This does not prove
reviewer identities, role independence, witnessed custody, review completion,
frame correspondence, prediction linkage, or scoring readiness. No review
package was opened and no human judgments were created.

## Next action

To start human review, the local process still needs an independently trusted
root public key and separately verified fingerprint; seven assigned role/key
bindings with required independence; root-signed integrity/release rosters; a
freeze-owner-signed `review_only` receipt over a fresh inventory; and a release
witness's observed unopened handoff followed by a passing release-order audit.
Until these inputs exist, continue human-free, non-authorizing experiments and
software validation only. Do not report any human-review count above 0/3
claims, 0/15 verified frame pairs, or 0/3 prediction-link judgments.

## Refreshed candidate inventory

At 2026-10-18 19:42 JST, the hash-only builder was rerun after the local source
tree had changed. It wrote
`data/private_nextgqa/pre_review_inventory_candidate_20261018_1942/pre_review_candidate_byte_inventory.json`
with 80 unique roles, 0 missing required slots, `status=candidate_only`,
`review_started=false`, `joint_freeze=false`, `score_ready=false`,
`execution_authorization=not_granted`, and `rights_status=unverified_local_only`.
The artifact is local-only (directory mode 0700, file mode 0600); SHA-256:
`8a34ee37f02e00b06189026300c109cf570950e90cbefcb986d44f7137ef2c9f`.
This refresh only hashes registered bytes and static source imports. It does
not establish reviewer/key assignments or permit anyone to open the package;
the independent root pin, signed role rosters, review-only freeze receipt,
observed handoff, and release-witness audit remain absent.

At 2026-10-18 20:10 JST the hash-only builder was rerun after the source and
experiment log updates. It again reported 80 unique roles, 0 missing required
slots, `candidate_only`, `review_started=false`, `joint_freeze=false`,
`score_ready=false`, `execution_authorization=not_granted`, and
`rights_status=unverified_local_only`. The private inventory is
`data/private_nextgqa/pre_review_inventory_candidate_20261018_2010/pre_review_candidate_byte_inventory.json`,
directory mode 0700/file mode 0600, SHA-256
`7128af12e41e04f0171fe7a8614d3194b54836acecb511d4cfee1612419e122d`.
The identical hash to the 19:53 candidate indicates no change among the
builder's registered inventory inputs; it is not evidence that all repository
files were frozen. No reviewer opened the package, and the trust-root, signed
role roster, review-only receipt, and witnessed handoff are still missing.
