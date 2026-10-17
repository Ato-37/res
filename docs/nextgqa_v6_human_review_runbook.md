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

## Review-only release freeze before anyone views a video

The current v2 protocol is a draft and is not frozen for scoring. Its exact
bytes and its already stated finite-cohort estimand, three-record denominator,
conditions, 32-frame sampler, mask/control rule, and failure rule must still be
hash-bound in the review-only release freeze. This records which fixed study
contract the annotations serve; it does not mark v2 frozen, score-ready, or
authorized. The release freeze must also bind the exact v6 claim/cohort inputs,
source and media lineage/custody, staged media, reviewer page and renderer,
review policy and disclosure, capture helper/policy, 15-anchor plan and its
provenance, crop and pair-review procedure, thresholds, and the designated
reviewers' public-key hashes. The v2 sampler rule is pinned at this stage, but
the concrete scoring sampler/configuration and executable scorer are not thereby
approved; bind and validate those in the later score freeze. Freeze the exact
bytes before anyone opens the page. An independently established trusted
roster/key root must bind the freeze-owner identity and signing key as well as
the reviewer identities and keys; a key hash supplied only by the receipt
being verified is not a trust anchor. A signed local timestamp is only a
machine-clock assertion, so record the witnessed release order separately if
no trusted timestamp service is used.

No real review-only release receipt, independent roster/key pin, or witnessed
handoff currently exists; the verifier is only an implementation, not evidence.
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

The helper's current roster schema covers only the freeze owner, claim
reviewer, and frame-pair reviewer. The custodian role is not independently
bound by this integrity-only receipt. If the freeze owner also acts as
custodian, that assignment still needs to be recorded in the eventual witnessed
release procedure; if a separate custodian is assigned, the actual release
roster must bind that identity and key. The helper cannot infer or verify
either arrangement.

For the separate witnessed handoff, `scripts/audit_nextgqa_review_release_order.py`
rechecks the integrity receipt and every current inventory-bound artifact,
then verifies a second root-signed roster with explicit freeze-owner,
custodian, two reviewer, and release-witness roles. The custodian may share the
freeze owner's key; the witness must have a separate key and key ID from all
other roles. The witness signature binds the exact receipt, both rosters,
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
receipt, the witness may run `sign-witness` with both rosters, inventory,
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
scoring. The later score freeze must bind the completed review exports, audited
clock/frame evidence, all three source-backed targets, exact prompt prefixes and
answer-token IDs, sampler/config/code, model checkpoint/configuration,
tokenizer/processor, precision, condition builder/scorer code, all nine exact
processor batches, and a production all-three-row atomic preflight bound to the
same v2 contract before model loading. No production score-freeze receipt
currently exists.

Assign people to these roles before opening the package. The freeze owner may
also serve as custodian, but must not serve as either reviewer.

1. **Pre-outcome freeze owner:** records the exact artifact hashes, protocol,
   sampler, capture policy/helper, thresholds, and reviewer-key hashes before
   any human viewing or outcome access. Keeps the signed freeze receipt local.
2. **Custodian:** verifies source lineage and media bytes, holds the private
   token crosswalk, and validates returned files. Must not generate claims,
   score outcomes, or serve as the claim reviewer.
3. **Claim reviewer:** independent of claim generation, scoring, and custody.
   Receives only the frozen offline page and token-named videos. Must not see
   dataset identifiers/indexes, the crosswalk, gold answers, seed intervals,
   outcome scores, or external answer sources.
4. **Frame-pair reviewer:** independent of capture, custody, pair preparation,
   and scoring. After capture artifacts are prepared, reviews all 15 blinded
   screenshot-crop/decoder-frame pairs and records match, mismatch, or
   ambiguous decisions.

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

An optional detached reviewer-key attestation v2 is available in
`scripts/nextgqa_completed_review_attestation.py`. Only after a real review
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
OS helper needs a human to position the player and trigger each capture; it
captures the foreground window. Produce a video-only crop for every capture
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
the exact 32-frame sampler, actual source target and prompt/token IDs, all nine
processor batches, and the pre-outcome joint freeze before model loading. For
every row, the calibrated 32-frame sample must overlap at least one reviewed
evidence interval and admit an equal-count set of distinct non-evidence control
ranks under the frozen distance/tie rule. Zero evidence overlap or too few
controls blocks the full cohort before model loading. Any other missing/failed
row likewise blocks all nine forwards; never score a surviving subset.

The planned result is a finite three-item descriptive contrast only. Even a
complete score does not establish that the model used the evidence or that the
generated explanation is true. Keep per-item scores and all QA/video-derived
material local; publish no protected derivatives.
