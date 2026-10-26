# Claim-support frame review handoff (2026-10-21)

## Purpose

Prepare a low-friction local review of whether each frozen generated claim is
supported by evidence in the model's own sampled video frames. This is a data
collection handoff, not an efficacy experiment. The intended thesis direction
remains internal intervention to improve explanation faithfulness; evidence
grounding is one possible mechanism and is not the only thesis contribution.

## Frozen inputs and provenance

The builder uses the three frozen test-split packets already referenced by
`run_test_split_internal_bias_frame_mean_interaction.PACKETS`, joins each row to
its baseline and frame-mean interaction records, and checks the locked cohort
assignment and source-video SHA-256. It decodes the model's 16 sampled frames,
verifies their sampled RGB pixels and relative presentation times against the
packet, and resolves Japanese question/claim text through the existing private
translation artifact. It refuses an existing output path.

The generated local artifact is:

- `data/private_nextgqa/claim_support_frame_review_test_20261021_05.html`
- 2,363,616 bytes, mode `0600`
- SHA-256 `7a6a93a567acc358a4d22dc1444f7355003b7d5cb43146ce789f6fad4a74e8ba`
- Three frozen items, 16 model-input frames per item (48 embedded JPEGs total)

The corresponding NExT-GQA val/test CSVs do include QA-level annotated
evidence intervals: all 3,358 validation and 5,553 test QA rows join to
`gsub` spans. These annotations support question/answer grounding; they do not
establish that an individual generated claim is supported by those spans. The
aggregate coverage and invalid-duration boundary audit is recorded in
`2026-10-21_nextgqa_native_interval_coverage_audit.md`.

The HTML and response JSON remain local. Rights status is unverified; this
workflow does not publish protected media, claims, questions, or responses.

## Interface and response checks

Each reviewer enters a nickname, answers the same three items, and can hand the
page to the next person without restarting it. `supports` and `contradicts`
require selecting at least one numbered frame; `not_visible_in_model_frames`
and `cannot_judge` require no selection. The final button downloads one combined
JSON. The inspector binds the response to the exact artifact fingerprint and
checks item order, answer schema, frame indices, source-manifest structure,
and disclosures. The private source manifest includes generator, locked
  assignment, pair-filter, translation, video-decoder source, per-item
  packet/video/baseline/interaction, and row token digests. These hashes improve
  later traceability but are unsigned: they
are not cryptographic attestation against someone who rewrites the artifact
and recomputes its embedded fingerprint. Duplicate
nicknames are accepted because nicknames are not identity credentials; the
inspector explicitly reports that identity and independence are unverified.
The UI revokes/hides an earlier download link when advancing to the next
reviewer, avoiding a stale link being mistaken for the latest combined export.

The report describes the images as reduced-size model-input frames. This
annotation concerns the 16 sampled frames only: absence there is not evidence
that an event is absent from the full video. Self-reported choices do not prove
that the reviewer watched the video or acted independently. Human confirmation
against the actual artifact, video, and review protocol is still needed before
using any label as evidence.

## Verified results

- Unit suite: `PYTHONPATH=scripts .venv/bin/python -m unittest -v test_build_test_split_claim_support_frame_review` — 8/8 passed after provenance checks were added.
- `py_compile` passed for builder, inspector, and unit test.
- Scoped `git diff --check` passed.
- Independent artifact-structure check verified mode `0600`, the embedded
  fingerprint, exactly three unique item IDs, 48 decodable embedded JPEGs,
  absence of external HTTP(S) resource URLs, the network-denying CSP, and no
  embedded gold-answer field. It also recomputed the source-manifest digest,
  checked its generator digest against the current builder, and passed the
  generated inline JavaScript through `node --check`.
- Synthetic inspector checks accepted a complete artifact-bound response and
  rejected a malformed decision. These are software/schema checks only; no
  human response was created.
- The independent audit found issues in the earlier draft (a stale download
  link could remain visible, and malformed values could make validation throw).
  The current source clears the old download link on handoff and rejects malformed
  decision types. The nickname behavior and inspector are aligned as described
  above. The first audit also found absent source digests; v04 added packet/video/
  baseline/interaction and cohort digests. v05 added translation/decoder digests.
  Independent re-audit compared these declared hashes against the current local
  source and confirmed they match. The inspector validates manifest structure
  and response binding; it does not independently hash source files. The
  manifest is unsigned and the fingerprint can be recomputed, so neither proves
  artifact authenticity or actual viewing.
- Focused unit suite passed 8/8. Chrome headless E2E passed for two synthetic
  reviewers × three items, including final Blob payload inspection and
  revocation of the prior export when advancing. The combined JSON passed the
  response inspector. Filesystem download is **not verified**: Chrome headless
  did not write a JSON file into the temporary download profile. No human
  response was created.

No current-cohort review response was received and no faithfulness score was
measured. Current direct faithfulness progress remains 0/3 item scores; the
broader prior inventory remains 0/25 formal claim-support reviews. The two
Downloads JSON files from 2026-10-19 belong to different artifacts and cannot be
joined to these items. No GPU experiment was run in this cycle: the existing
GPU pilots already tested generic QA-key steering on automated-window proxies,
with small, heterogeneous, non-confirmatory effects. Repeating the same proxy
would not answer whether claim-evidence-specific steering improves faithfulness;
that requires either validated support frames or a new frozen, discriminating
proxy design. This is a measurement gap, not a conclusion that internal
intervention cannot improve faithfulness.

## Reproduction

The protected artifact is already built locally. To create a fresh copy, use a
new non-existing name, for example:

```bash
.venv/bin/python scripts/build_test_split_claim_support_frame_review.py \
  --output data/private_nextgqa/claim_support_frame_review_test_20261021_06.html
```

Run its focused checks with:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest -v test_build_test_split_claim_support_frame_review
.venv/bin/python scripts/test_claim_support_frame_review_browser_e2e.py
```

## Next action

The independent source/artifact audit is complete. Open this local artifact for
reviewers if collecting frame labels. Keep any response local and inspect it with
`scripts/inspect_claim_support_frame_review_response.py` against the exact HTML
before joining any labels. Once validated labels exist, continue with the
pre-specified intervention comparison and matched-sham analysis. Until then,
continue independent experiment design/code validation that can discriminate
whether internal steering improves claim evidence dependence, output stability,
and matched-sham specificity; do not present a proxy-only result as direct
faithfulness evidence. A clean worktree validation of the public source handoff
also found that the builder's helper modules are untracked on the fetched
repository branch. The four-file scoped publication was not pushed rather than
publishing a builder that fails outside this dirty local tree. Do not resume the
closed head/layer/alpha search.

## Launcher alignment follow-up (2026-10-21)

The Japanese launcher previously opened the older temporal-order validation
page, while the latest claim-support frame artifact existed separately. Updated
the launcher and Japanese runbook to open
`data/private_nextgqa/claim_support_frame_review_test_20261021_06.html` and to
describe its actual workflow: nickname per reviewer, three claim/video items,
16 sampled model-input frames per item, supporting/contradicting/not-visible/
cannot-judge decision, supporting-frame selection, and one combined JSON export.
The runbook now gives the matching response inspector command and states that
this preview-level check does not evaluate the whole video or authenticate
reviewers. The exact artifact remains mode 0600, SHA-256
`309739d3aacb39ea46745c41893fbc2736661e421886e228922f1690faa52715`; the
launcher and source link both resolve to that file. This corrects the handoff
target but does not create any human response: response count and direct
claim-support count remain 0/3. It does not change the separate formal gate,
which remains blocked on trusted reviewer/release evidence, frame-time review,
prediction-link judgments, and the production pre-load transaction.

Verification: `test_build_test_split_claim_support_frame_review` passed 8/8;
the current-artifact Windows Chrome E2E passed using synthetic judgments for
two reviewers × three items, and its export was validated by the real response
inspector. The normal WSL sandbox could not start Windows interop
(`UtilBindVsockAnyPort`), so the same narrowly scoped E2E was run with
individual elevated access. A static check confirmed launcher refresh and
fallback link resolve to the exact local artifact and its SHA-256/mode remain
unchanged. No human response or model/GPU score was created. The Japanese
runbook now gives the matching open and inspection commands; private videos
and review outputs remain local.

## Filesystem download verification (2026-10-21)

The prior Chrome E2E verified the final JSON Blob but returned
`browser_filesystem_download_verified=false` without testing the downloaded
file. Extended it to poll Chrome's temporary download directory, parse both the
one-reviewer intermediate export and the final two-reviewer export, and
compare the final downloaded JSON to the in-page Blob payload. The CDP
download-event counters remained zero, so file creation—not CDP event delivery—
is the evidence for successful download. The current exact artifact passed in
Windows Chrome: 2 synthetic reviewers × 3 items; both exports were
structurally valid; the final file matched the Blob and the current artifact
fingerprint; stale Blob revocation passed. The inspector still correctly
reports semantic support and reviewer identity/independence as unvalidated.
Temporary Chrome profile/downloads were removed. The final E2E summary reports
`browser_filesystem_download_verified=true`. No actual human response or GPU
inference was created. This supersedes the earlier “download not verified”
status for this E2E harness, while real human response collection remains
0/3.

A follow-up no-content scan of the private review-response folder, current
private response-named JSON files, and Downloads found **0/52** readable
candidate JSON files bound to the current frame-review artifact fingerprint.
The 52 are candidate files across those locations, not 52 reviewer responses.
No names, free text, judgments, or raw item IDs were printed. Human review and
direct faithfulness remain **0/3**.

## Stale launcher repair (2026-10-21)

Independent handoff audit found that the older general workspace URL still
redirected to the superseded v14 temporal-order page, despite the canonical
launcher and runbook pointing to the current claim-support artifact. Updated
the compatibility page's redirect and fallback link to
`claim_support_frame_review_test_20261021_06.html`, with Japanese text that
matches the current nickname → three items → sampled frames → combined JSON
workflow and its exploratory scope. This removes an incorrect entry point but
does not add reviewer responses: current-artifact-bound responses remain 0/3
across 52 inspected candidate files. The valid human review handoff remains
available; actual human review is still pending, and no judgment is inferred.

Verification: HTML inspection confirms both refresh and fallback use the current
artifact path; the exact target remains mode 0600 with the handoff-log SHA-256.

## Native QA-interval overlap check (2026-10-21)

The three exact frozen review-packet rows each contain QA-level reference
intervals and 16 model-input sample timestamps. A local aggregate-only check
found four reference intervals total; every item has at least one sampled
frame time inside its QA interval, with 15 of 48 sample times overlapping the
interval union. The reviewed artifact intentionally does not highlight these
frames: the intervals are QA-level references, not claim-specific support
labels, and pre-highlighting them would bias the reviewer. This makes the
artifact practically reviewable on the model-input timeline, but it does not
validate that any generated claim is supported. Independent receipt/source
audit is pending; the aggregate dataset coverage audit is in
`2026-10-21_nextgqa_native_interval_coverage_audit.md`.
