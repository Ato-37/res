# Bind the capture operator in the witnessed review release (2026-10-18)

## Purpose

Close the role-identity gap for human-operated browser capture before the
review-only release. Bind the capture operator's identity key in the
root-signed release roster and state the operator's evidence duties without
opening review materials or creating a freeze.

## Implementation

- Added `capture_operator` to the release roster schema. Its key ID and public
  key must be distinct from the freeze owner, custodian, claim reviewer,
  frame-pair reviewer, prediction-link reviewer, and release witness. The
  existing freeze-owner/custodian same-key exception remains the only allowed
  alias.
- The ceremony-signing CLI now requires the capture-operator public key when
  creating a release roster and validates it through the release auditor.
- Updated the runbook to require an out-of-band identity-to-key binding, define
  the operator's 15-anchor capture evidence, separate capture from visual
  match decisions, and identify the frame-pair reviewer as the signer of the
  final evidence manifest. The runbook states that the current unsigned viewer
  export is not that signed evidence and that its integration remains
  unimplemented and unaudited.
- Documented that the capture-operator identity/key and all human evidence
  gates remain pending.

## Verification

- Focused release-order and ceremony-signing suites: **20/20 passed**. Tests
  cover capture-key reuse against every roster role, acceptance of the
  owner/custodian alias when the capture key remains distinct, signing a valid
  synthetic roster, and rejection of a reused capture key through the CLI.
- `py_compile` for the auditor, signer, and focused tests: passed.
- Scoped `git diff --check`: passed.
- Source SHA-256 values at verification:
  - `scripts/audit_nextgqa_review_release_order.py`: `da9dcc6cc325ceb9a143b6bb828b9cd0cafebb53c95f071ec4d19d69587b680d`
  - `scripts/nextgqa_review_ceremony_signing.py`: `6fdc73cc4b95a0475d31d918cf4fcc2784407cc398853b2b844ff8e94ab89bc9`
  - `scripts/test_audit_nextgqa_review_release_order.py`: `1a747b44631af33f11625dc8f016754f4bb1bb023eda84cbfcbd20fbd127efbd`
  - `scripts/test_nextgqa_review_ceremony_signing.py`: `3fd0953f6589cc330575ff1fdf1530a667d7bf7c36fad3c1363ba0f07e9de778`
  - `docs/nextgqa_v6_human_review_runbook.md`: `5b9ad2b5c8b4d56553ab856715066cfac6ad6fb32c5bd1b85724d5116e06a2b7`

## Interpretation and limitations

This is a code/schema and runbook change only. All checks use synthetic keys
and temporary synthetic bytes. No protected package, HTML, video, inventory,
receipt, freeze, or human response was opened, built, signed, or changed. No
real identities or signatures were created. Rights status remains unverified.

The release roster can now pin who is assigned to capture, but the operator
still has to perform the real browser interaction and preserve the exact
capture artifacts. The operator key currently identifies the assigned role
only; no detached operator-signed capture attestation or verifier exists. The
final signed evidence manifest and its bridge from the
unsigned blinded-pair viewer export remain unimplemented and unaudited. The
release gate, actual review, capture, and pair-review gates remain unsatisfied.

## Next action

After an external authority pins the root and the real participants establish
their identity-to-key mappings, rebuild the current candidate inventory after
these source changes, root-sign both rosters, and complete the freeze-owner and
witnessed handoff. Do not open the package before the exact-byte release-order
audit passes. Then perform claim review and capture only under the runbook's
separate human roles and private-local handling rules.

## Follow-up: refreshed non-authorizing inventory (2026-10-18)

To advance the local handoff without opening review contents, the hash-only
inventory builder was run against the current paths recorded in the existing
private source-provenance note:

- package: `data/private_nextgqa/reviewer_package_v6_20261018_0243`
- staged media: `data/private_nextgqa/review_media_v6_20261017_0635`
- standalone HTML: `data/private_nextgqa/claim_review_v6_20261018_0243.html`

The builder reported 77 inventory slots, zero missing slots,
`candidate_only`, and `execution_authorization=not_granted`. The new private
inventory is mode `0700`; its JSON and source-provenance sidecar are mode
`0600`. Inventory SHA-256 is
`1487935e5633802ea32a3d7ab0ad0196d55675aa607e0ea13fc2c39338505963`; sidecar
SHA-256 is
`181d91a62911ceef395b07b777a6d68e4a04d298b86e5947d0f4c3312616c4d2`. The
sidecar binds those selected paths to the inventory digest and keeps review,
freeze, score, and execution flags false. This hash-only build does not verify
human identity, trust root, key custody, or the witness ceremony. No protected
HTML or video content was opened. Rights remain unverified and everything
remains local.

The next human-dependent step remains unchanged: obtain and independently pin
the authority root, assign real people and distinct keys for all seven roles,
and complete the root-signed roster, freeze-owner receipt, witnessed unopened
handoff, and final release-order audit. The user was asked for role pseudonyms;
no assignments have yet been received.

## Follow-up: current-source inventory refresh (2026-10-18)

Rebuilt the hash-only candidate inventory using the current package, media,
HTML, runbook, and code inputs after the latest source changes. It contains
**80 artifact slots**, with **0 missing required slots**. The private directory
is mode `0700` and its JSON is `0600`. The receipt remains
`candidate_only`, `review_completed=false`, `joint_freeze=false`,
`score_ready=false`, and `execution_authorization=not_granted`. No review HTML,
video, claim, answer, or annotation content was opened. Rights remain
unverified; the inventory stays local.

An independent gate audit confirmed that this refresh does not clear the
human-dependent prerequisites: independently pinned root key/fingerprint,
identity-to-key assignments and root-signed role rosters, a freeze-owner
review-only receipt, witnessed unopened owner-to-custodian handoff, fresh
ceremony ID, and a passing immediate pre-open release-order audit. Until
those inputs exist, no reviewer should open the package. Continue source and
synthetic-test work that does not depend on the package.

After the QTA sidecar dtype-binding repair changed the source set, the
inventory was rebuilt again at a fresh unique local path. The refreshed
candidate has the same **80 present / 0 missing** counts and the same
non-authorizing flags, with mode `0700` for its directory and `0600` for the
inventory JSON. This latest candidate supersedes the immediately previous
candidate for any future freeze, but neither can be used as freeze evidence.

## Follow-up: blinded-export-to-signed-evidence bridge completed

The earlier statement that the bridge from the unsigned blinded-pair viewer
export was unimplemented is superseded by the current schema-v2 path in
`scripts/build_nextgqa_clock_evidence_manifest.py`. The builder consumes the
exact package/export bytes, joins every blinded pair ID through the
custodian-only identity map to its capture token and anchor, checks each
screenshot crop and decoder-frame hash against the signed capture/operator
artifacts and decoder crosswalk, requires all 15 reviewer rows to attest crop
purity and `match`, then creates the frame-pair reviewer attestation for the
reviewer key. The downstream `scripts/audit_nextgqa_clock_frame_evidence.py`
recomputes the v2 signature and exact artifact joins. A synthetic end-to-end
builder/auditor test is enabled; the builder, operator-attestation, and
evidence-auditor suites pass **79/79**, and full repository discovery passed
**2,013/2,013** in the recorded checkpoint. An independent code audit found
no material bypass in this non-authorizing adapter.

This closes the software bridge only. It does not create real capture evidence,
verify crop purity or visual matches, authenticate a human identity, establish
the independently trusted root, or authorize opening/scoring. The human-review
gates remain blocked on actual root/key assignments, signed review-only
custody, witnessed handoff, and human completion of claim, 15 frame-pair, and
prediction-link reviews. The current candidate inventory also needs a fresh
source-hash rebuild immediately before any future freeze.

## Follow-up: frame-reviewer role-key enforcement and fresh inventory

The schema-v2 frame evidence builder and auditor now require the signing key
to match the `frame_reviewer` public key in the root-signed release roster.
They record the roster key ID and DER fingerprint and verify the role-key
join again downstream. Synthetic tests pass for a valid frame-reviewer key
and reject a capture-operator key. This binds the key to a roster role only;
it does not independently establish the external root fingerprint, human
identity, or truthful visual review. The code change and its full-suite
verification are recorded in
`experiments/2026-10-18_frame_reviewer_role_key_binding.md`.

After updating the inventory builder's source closure for the current
repository state, rebuilt a local hash-only candidate at
`data/private_nextgqa/pre_review_inventory_candidate_20261018_2306/`.
It contains **81 artifact slots, 0 missing required slots**; the directory is
mode `0700` and inventory JSON is `0600`, SHA-256
`3ce6c966e9d0d5820bc3cd59f91af3d8c0036f7f9c33fce44063759c044c08ac`.
It remains `candidate_only`, with `review_completed=false`,
`score_ready=false`, and `execution_authorization=not_granted`. It is not a
freeze receipt; rebuild immediately before any future freeze. No protected
review content was opened.
