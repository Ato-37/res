# NExT-GQA v6 local human-review runbook

Status: **preparation only**. The local byte inventory is a candidate receipt,
not a freeze, review, or authorization. Use two distinct freeze stages: a
review-only release freeze before any reviewer opens the package, followed by
a score freeze after review and clock evidence are complete. A review-only
receipt never authorizes model loading or scoring.

## Current local package

The private offline package is stored under the local private-data root. It
contains one HTML page and three opaque-token-named videos. Keep it and all
exports on the local machine; do not upload, commit, email, or otherwise share
the package or derived material. An earlier hash-only candidate inventory is
also stored under the local private-data root. It recorded 50 present roles
and no missing slots, but predates later changes to the review-opening gate
code and does not bind the current runner bytes. Treat it only as a historical
candidate snapshot, not current freeze evidence. Before any review-only release
freeze, rebuild a unique candidate inventory with the current inventory builder.
That inventory must bind this runbook, the inventory builder itself, the
integrity-receipt verifier, and the release-order auditor along with the other
listed inputs. It still has
`joint_freeze=false`, `review_completed=false`, `score_ready=false`, and
`execution_authorization=not_granted`.

The latest local candidate inventory is ephemeral and private; it is not named
or pinned here. Any earlier inventory count (including later candidate-only
counts) is historical and must not be used for release. Immediately before a
new freeze, rebuild the inventory from the current builder and verify the
exact resulting bytes. A clean inventory proves byte integrity only.

## Before first opening: readiness checklist

Do these steps in order. Stop at the first unmet condition; no one opens the
HTML page or videos before the release witness has observed the complete
pre-open handoff and the independent auditor passes immediately before opening.

1. Obtain the root public key and its DER fingerprint through a separate
   trusted channel. Independently confirm that the fingerprint belongs to the
   intended authority; a key and fingerprint delivered together are not a
   trust anchor.
2. Have that authority sign the current role roster. The roster binds roles
   to key IDs and public keys; it does not authenticate people. Through the
   separate trusted channel, independently establish and locally record which
   real person controls each key. Assign distinct keys to the freeze owner,
   capture operator, claim reviewer, frame-pair reviewer, prediction-link
   reviewer, and release witness. Name the custodian too; the owner may also be
   custodian, but no reviewer or capture operator may fill either role, and
   every role key must be distinct except the owner-custodian alias. The witness
   must be separate from every other role.
   Resolve and pin the identity-to-key mapping before any review content is
   opened.
3. Rebuild a unique candidate inventory using the current builder. Verify its
   owner-only modes, exact input hashes, source closure, and current runbook
   and code bytes. The freeze owner signs the review-only receipt over this
   exact inventory. Keep the receipt, rosters, keys, and reports local.
4. Independently verify the root-signed rosters and freeze-owner signature
   against the out-of-band root pin. Confirm the receipt is explicitly
   review-only and still denies model loading, scoring, and execution.
5. Before anyone opens the package, the release witness personally checks the
   unopened state and exact frozen bytes and receipt, then observes custody
   pass from the owner to the custodian. If one person fills both roles, the
   witness must still observe and record that person's custody acceptance
   after the freeze is established. The witness signs a fresh ceremony ID
   only after observing that order. A detached signature created without this
   witnessed event is not acceptable evidence.
6. Immediately before first opening, run
   `scripts/audit_nextgqa_review_release_order.py` against the exact current
   receipt, both rosters, inventory, detached witness attestation, complete
   artifact set, independent root pin, and ceremony ID. Pass an
   `--artifact ROLE=PATH` argument for every inventoried artifact. Require the
   witnessed release-order status and
   `package_opening_permitted=true`; this permits only the offline human-review
   step. If any artifact changed, any identity/role is unresolved, the handoff
   was not witnessed, or the auditor fails, stop and repeat the freeze and
   ceremony from the beginning.

Passing this checklist is a procedural precondition for human review, not
proof of identity, semantic review quality, actual viewing, or scoring
authorization. The claim reviewer, capture operator, and independent
frame-pair reviewer must then complete the separate steps below. Scoring stays
denied until both sequential score freezes pass.

## Review-only release freeze before anyone views a video

The legacy claim-mask v2 protocol is a draft and is not frozen for scoring.
The current thesis candidate is separately described in
`docs/qwen_question_attention_faithfulness_protocol_v1_draft.md`; it too is an
engineering draft, not preregistered, frozen, or authorized for real-data
scoring. Before any reviewer opens the package, the review-only release freeze
must bind the exact bytes of the QTA draft and the selected-treatment record
`experiments/2026-10-18_qta_treatment_selection.md`, alongside the legacy v2
review/export contract used to collect the claim and interval annotations.
This makes clear which draft estimand those annotations are intended to inform;
it does not freeze either scoring protocol or authorize scoring. The QTA draft
specifies question-attention selection (layers 5–16, `k=max(1,ceil(0.10*T))`,
earlier-index tie break), `ln(2)` answer-query bias, a fixed baseline-full
prediction/foil pair, evidence/control key ablations, and a deterministic sham
seed. A review-only freeze must hash-bind those exact draft and selection-log
bytes; the concrete cohort plan, branch tensors, and executable scorer still
require later score-freeze validation.

Stage 1 review records whether the entire claim is supported by visible video
evidence and every sufficient occurrence interval. It does not establish that
the claim distinguishes the model's later baseline prediction `p` from foil
`q`. Under the QTA draft, a separate blind stage-2 prediction-link adjudication
must therefore occur after baseline `p/q` is frozen and before any treatment or
ablation branch. The stage-2 judgment currently applies to the claim and the
union of its reviewed intervals; it does not map a particular claim clause to
each interval. Consequently, any resulting score is only fixed-`p/q` score
dependence on that reviewed interval set under the stated intervention and
ablation, not a general measure of explanation faithfulness or claim truth.
The current review package does not collect stage-2 judgments and does not
authorize them or scoring.

The release freeze must also bind the exact v6 claim/cohort inputs,
source and media lineage/custody, staged media, reviewer page and renderer,
review policy and disclosure, capture helper/policy, 15-anchor plan and its
provenance, crop and pair-review procedure, thresholds, and the designated
reviewers' public-key hashes. The QTA draft now defines a fixed 32-frame
endpoint-inclusive sampler, Python round-to-nearest/ties-to-even index rule,
50 ms uniform-grid tolerance, PyAV two-pass source/full-PTS integrity and
sampled-pixel checks, and Qwen processor metadata contract. Preserve each
video's maximum observed grid deviation and require actual-PTS and
processor-implied uniform-grid group assignments to match for every interval;
any mismatch fails the row before scoring. Before review begins, hash-bind
these exact draft and implementation bytes in the review-only inventory. Do not
choose or tune the sampler after viewing review intervals. The inventory pins design bytes but
does not authorize scoring. Freeze the exact bytes before anyone opens the
page. An independently established trusted
roster/key root must bind the freeze-owner identity and signing key as well as
the reviewer identities and keys; a key hash supplied only by the receipt
being verified is not a trust anchor. A signed local timestamp is only a
machine-clock assertion, so record the witnessed release order separately if
no trusted timestamp service is used.

No real review-only release receipt, independent roster/key pin, or witnessed
handoff currently exists; the verifier is only an implementation, not evidence.
The capture-operator identity/key assignment is also not established, and no
real capture evidence or independent frame-pair review has been completed.
Do not open the package until the exact receipt and its independently
established trust-root pin have been checked against the frozen bytes, and
release order is supported by either trusted timestamp evidence or a witnessed
procedure.

`scripts/nextgqa_pre_review_integrity_receipt.py` provides an integrity-only
candidate check for an externally pinned root, its signed role roster, the
freeze-owner signature, and the exact bytes in the candidate inventory. Its
successful status is `signature_valid_order_unverified`; it always reports
`package_opening_permitted=false`, `model_load_permitted=false`,
`score_ready=false`, and `execution_authorization=not_granted`. This helper is
not a release receipt and does not satisfy this opening gate. Do not create a
real receipt until the roster/trust-root pin is established, and do not open
the page based on its output.

The helper's integrity-roster schema covers only the freeze owner, claim
reviewer, and frame-pair reviewer. The release roster additionally binds the
custodian, capture operator, prediction-link reviewer, and release witness.
The capture operator must be a named real person with a separately root-pinned
key and may not share a key with any other role except that the
owner-custodian alias is the sole permitted role-key reuse in the roster.
The detached stage-2 adoption verifier reruns structural checks against raw
baseline and receipt bytes but does not provide a second human auditor
signature. The release roster requires the prediction-link key to differ from
the claim reviewer, freeze owner, custodian, frame-pair reviewer, and release
witness. Stage-2 signatures attest adoption of exact bytes; they cannot
establish real-world identity or semantic correctness. The attestation source
and CLI pass focused synthetic tests and independent code audit, but are not
integrated into a phase score gate; therefore the self-reported receipt still
cannot pass the score gate. The custodian role is not independently
bound by this integrity-only receipt. If the freeze owner also acts as
custodian, that assignment still needs to be recorded in the eventual witnessed
release procedure; if a separate custodian is assigned, the actual release
roster must bind that identity and key. The helper cannot infer or verify
either arrangement.

For the separate witnessed handoff, `scripts/audit_nextgqa_review_release_order.py`
rechecks the integrity receipt and every current inventory-bound artifact,
then verifies a second root-signed roster with explicit freeze-owner,
custodian, capture-operator, claim reviewer, frame-pair reviewer,
prediction-link reviewer, and release-witness roles. The custodian may share
the freeze owner's key; the capture-operator key must be distinct from the
owner/custodian, claim reviewer, frame-pair reviewer, prediction-link reviewer,
and witness. The witness must have a separate key and key ID from all other
roles.
The witness signature binds the exact receipt, both rosters,
inventory, artifact bindings, and a fresh ceremony ID, and asserts that the
freeze/package handoff occurred before first opening while review had not
started. Supply the ceremony ID independently to the verifier and do not reuse
it intentionally. The report's `review_release_order_witness_attestation_valid`
status verifies the signature and statement bytes only: it does not prove the
real-world event, human identity, or chronology, and it has no one-shot replay
registry or file-access enforcement. Its `package_opening_permitted=true` is a
procedural permission for the witnessed offline review step only. It never
permits model loading or scoring. Do not open the package based solely on the
older integrity-only verifier.

`scripts/nextgqa_review_ceremony_signing.py` can create the root-signed
integrity/release rosters and the detached witness statement from supplied
keys and exact local files. It never creates keys or confirms a person's
identity. For either roster command, supply the root key pair, an independently
obtained root DER fingerprint, a fresh roster ID, and one
`--role-key ROLE=KEY_ID=PUBLIC_KEY_PEM` argument per role. For
`sign-release-roster`, also supply the already signed `--integrity-roster`.
After the freeze owner has established the exact inventory and signed integrity
receipt, the witness may run `sign-witness` with the independently pinned
`--root-public-key` and `--root-public-der-sha256`, both rosters, inventory,
receipt, one `--artifact ROLE=PATH` argument for every inventoried artifact,
the witness private key, and a fresh `--ceremony-id`. The witness must
explicitly pass `--witness-confirms-unopened-preopen-handoff` only after
personally observing that statement to be true. All inputs must be regular
0600 files and output must be a new file under an existing 0700 directory.
The CLI verifies the signatures, receipt, and exact artifact bytes before
signing; its successful output still does not prove identity or that the
ceremony occurred. Run the independent release-order auditor immediately
before opening the page. Keep all signing material private and local.

The release witness must be independently trusted through the out-of-band
root pin and must personally observe the release ceremony: exact frozen bytes
and signed receipt are established while the page is unopened, then the owner
hands the package to the custodian before the first opening. A detached
signature cannot repair a ceremony that did not occur. Because the gate cannot
prevent someone opening readable files outside the procedure, record the
witnessed handoff locally and run the verifier immediately before opening.
Keep all keys, receipts, attestations, and output reports local and private.

The review-only receipt must say `review_only`, keep
`protocol_v2_status=design_draft_not_frozen`, `score_ready=false`,
`model_load_permitted=false`, and `execution_authorization=not_granted`. It
permits only the frozen human annotation and capture/review work. It must not
be accepted by any scoring runner. If the claim text, cohort, review policy,
estimand, metric, conditions, eligibility/failure rule, or mask/control rule
changes, stop and version the study; affected reviews must be repeated before
scoring. Scoring requires two sequential phases because baseline `p/q` does
not exist until after the first three baseline forwards, while independent
prediction-link review cannot occur before those predictions are shown to a
reviewer.

The **pre-baseline gate** must bind completed stage-1 review exports,
audited clock/frame evidence, all three source-backed targets, exact prompt
prefixes and answer-token IDs, the QTA sampler and model/runtime configuration,
condition-builder/scorer code, one immutable unmodified full-video processor
batch per record, and an all-three-row atomic preflight. A signed,
independently verified phase receipt may authorize only three baseline forwards
total (one per row); it must not authorize treatment or ablation forwards. This
gate and baseline-only runner do not currently exist.

After baseline `p/q` is produced, bind the baseline manifest and the
independently attested stage-2 prediction-link review. The **post-baseline
gate** must additionally bind the exact nine branch plans per row (including
the common processor-batch hash, selected and sham token positions, hook
specification, and condition labels), scorer code, and accepted stage-2
judgments. A separate signed, independently verified phase receipt may
authorize only the remaining 24 forwards total (eight per row). The
same exact processor batch is reused for all nine branches; there are not nine
separately processed or masked processor batches. Neither freeze is currently
implemented as a production score-freeze receipt, and neither may be inferred
from a review-only receipt or structural artifacts alone. The phase-1 receipt
must bind the QTA draft, frozen sampler, and legacy v2 review/export contract
before baseline model loading; phase 2 must bind that receipt plus all baseline
and stage-2 artifacts before treatment forwards. These phase schemas, signature
verification, and runners do not yet exist. The legacy
`protocol_v2_status` field records only the annotation/export contract status;
it does not identify or freeze QTA scoring. No production score-freeze
receipt currently exists.

Assign people to these roles before opening the package. The freeze owner may
also serve as custodian, but must not serve as a reviewer or capture operator.
Use the [role-assignment intake template](nextgqa_human_review_role_assignment_template.md)
to collect reviewer IDs, key IDs and local public-key paths without placing
private keys or review content in the worksheet.

1. **Pre-outcome freeze owner:** records the exact artifact hashes, protocol,
   sampler, capture policy/helper, thresholds, and reviewer-key hashes before
   any human viewing or outcome access. Keeps the signed freeze receipt local.
2. **Custodian:** verifies source lineage and media bytes, holds the private
   token crosswalk, and validates returned files. Must not generate claims,
   score outcomes, or serve as the claim reviewer.
3. **Capture operator:** independently operates the actual review browser and
   capture helper for all 15 preplanned anchors. Produces the raw clock events,
   screenshots, OS capture receipts, and video-only crops under private local
   storage. The operator records captures but does not judge screenshot/frame
   matches; the frame-pair reviewer must be a different person and key.
4. **Claim reviewer:** independent of claim generation, scoring, and custody.
   Receives only the frozen offline page and token-named videos. Must not see
   dataset identifiers/indexes, the crosswalk, gold answers, seed intervals,
   outcome scores, or external answer sources.
5. **Frame-pair reviewer:** independent of capture, custody, pair preparation,
   and scoring. After capture artifacts are prepared, reviews all 15 blinded
   screenshot-crop/decoder-frame pairs and records match, mismatch, or
   ambiguous decisions. The reviewer signs the final evidence manifest with
   the frame-reviewer key whose hash was pinned before review; an unsigned
   blinded-viewer export alone does not satisfy the signed evidence gate.
6. **Prediction-link reviewer:** independent of the stage-1 claim reviewer,
   generation, scoring, and custody. The release-roster key is separate and
   root-pinned before the first package opening. The reviewer sees only the
   post-baseline blinded worksheet. The verification code must rerun the
   structural check over exact raw bytes; it does not prove semantics.
7. **Release witness:** independently trusted through the external root pin.
   Personally verifies the exact unopened package and signed review-only
   receipt, observes transfer from freeze owner to custodian, and signs the
   fresh ceremony attestation only after that event.

The root authority signs both rosters after the identity-to-key assignments
have been independently established through the separate trusted channel. The
freeze owner signs the review-only receipt over the exact current inventory;
the release witness signs the witnessed handoff statement. After stage-1
review, the claim reviewer signs the exact exported review bytes for the
stage-1 structural gate. The capture operator's evidence consists of the raw
capture artifacts and receipts; the operator does not sign match decisions.
The rostered operator key can sign and verify a **candidate-byte** attestation
with `scripts/nextgqa_capture_operator_attestation.py`; the private filesystem
wrapper is `scripts/run_nextgqa_capture_operator_attestation.py`. The candidate
binds the exact 15-anchor plan and observation, capture receipt, screenshot,
crop-map, crosswalk, and crop bytes to the capture-role key in the root-signed
release roster. Its signature establishes key adoption of those byte hashes
only. It does not establish the signer's identity, that capture occurred, that
a screenshot matches a decoder frame, or that a crop is video-only.

The frame-pair reviewer can create a schema-v2 candidate with
`scripts/build_nextgqa_clock_evidence_manifest.py`. The builder rechecks the
operator attestation against an exact private screenshot/crop snapshot and
signs the joined candidate. The downstream
`scripts/audit_nextgqa_clock_frame_evidence.py` accepts v1 and v2; for v2 it
revalidates the operator signature and byte joins from the supplied rosters,
trust-root pin, screenshot/crop map, and exact artifact snapshots. This is
synthetic-tested integration only. The caller-supplied root/reviewer pins are
not independently authenticated by these commands, and the audit does not
establish human identity, capture-process truth, visual match, crop purity, or
clock calibration. All authorization fields remain denied. The package is not
ready to open: actual root trust, role assignments, signed receipts, witnessed
handoff, and human reviews remain outstanding. The unsigned blinded-pair viewer
export alone still does not satisfy the signed evidence gate.

If no independent people can fill these roles, leave the gate blocked; do not
substitute an agent-generated label or an operator self-attestation.

## Claim review

After the freeze owner releases the package under the review-only receipt, the
claim reviewer reads and
acknowledges the full export disclosure before viewing. The reviewer may stop
without viewing if they do not accept its terms. Otherwise, they open the local
`index.html` offline and review all three complete videos at normal speed with
audio muted and text/caption tracks disabled. Visible pixels are the only
permitted evidence; burned-in text cannot support a claim. The reviewer records
whether every material factual component is visibly established and marks
**every distinct sufficient occurrence** as its own positive-duration interval. Do not merge
disjoint occurrences, invent an interval, select only a convenient occurrence,
or replace a failed item. Keep all three items in the denominator.

The page's review JSON, integrity index, and optional clock JSON/index are
self-reported evidence. Save them in the private local custodian area with
files mode 0600 in directories mode 0700; do not publish them. The custodian checks exact token coverage,
package/media hashes, and source mapping with the existing handoff audit. A
structural pass does not authenticate reviewer identity or semantic support.

The detached reviewer-key attestation v2 is produced separately from the page
export by `scripts/nextgqa_completed_review_attestation.py`. It is not required
to save a local draft review, but it is required by the current
`scripts/qta_stage1_bundle_audit.py` for a passing stage-1 bundle audit. Without
a valid attestation, keep the pre-load gate blocked. Only after a real review
export exists and the custodian's structural audit passes, the claim reviewer
may sign the exact frozen worksheet, lineage, review JSON, and review index
using the `sign` subcommand. Supply the integrity roster, root public key, and
root DER fingerprint obtained through separate trusted channels; the CLI
validates the root signature and derives the claim-reviewer key from that
roster. The caller still supplies the pre-review receipt digest, which is only
bound as a declaration: this CLI does not verify the receipt or its connection
to the roster/cohort. The root fingerprint must not come from the same
untrusted handoff as the root key. Signing also requires the claim-reviewer
private/public key pair and checks that it matches the roster entry. The CLI
writes a new 0600 file under an existing 0700 directory and refuses to
overwrite. Keep the attestation, roster, keys, receipt, and all source files
private. The matching `verify` command independently rechecks the root-signed
roster, derived reviewer key, detached signature, and structural handoff audit.
It proves only that the rostered key signed/adopted those exact bytes under the
presented root pin; key-to-person identity, actual viewing, source-media
custody, semantic truth, and score authorization remain unverified/denied. This
step does not replace the separate clock/frame evidence audit or later score
freeze.

## Browser-clock and frame correspondence

Using the exact staged media and the frozen plan, perform all 15 planned
beginning/interior/end anchor observations in the actual review browser. The
OS helper prompts with the one-based record slot and anchor ID, without
printing opaque review tokens or planned PTS to the console. Follow the
already-opened review rows in order, position the player, record each HTML5
clock event, and trigger capture. The helper drains previously queued hotkeys,
rejects timestamp ambiguity at millisecond resolution and stale events
(including a duplicate press during the preceding capture), and actually times
out after 60 seconds without a valid hotkey. It captures the
foreground window. Produce a video-only crop for every capture
and retain the original screenshot privately. The crop coordinates and hash
alone do not establish crop purity. The separate frame-pair reviewer must
inspect every blinded crop/decoder pair. Missing, mismatched, or ambiguous
pairs block the whole cohort and calibration. Then build and run the signed
evidence audit and the predeclared fit/held-out calibration audit; record
empirical residuals without describing them as uncertainty bounds.

## Route back into the experiment

Once the review exports and clock evidence exist, retain the original bytes,
recompute hashes, and run the existing structural handoff and signed clock
evidence audits. Resolve every audit failure before constructing masks. A real
score still requires a production atomic preflight that binds all three rows,
the pre-review frozen 32-frame sampler, actual source target and prompt/token
IDs, one exact full-video processor batch per row, and the pre-baseline phase
receipt before baseline model loading. For every row, the calibrated 32-frame
sample must overlap at least one reviewed
evidence interval and admit an equal-count set of distinct non-evidence control
ranks under the frozen distance/tie rule. Zero evidence overlap or too few
controls blocks the full cohort before model loading. Any other missing/failed
row likewise blocks all nine forwards; never score a surviving subset.

The planned result is a finite three-item descriptive contrast only. Even a
complete score does not establish that the model used the evidence or that the
generated explanation is true. Keep per-item scores and all QA/video-derived
material local; publish no protected derivatives.
