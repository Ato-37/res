# Exposure-screened NExT-GQA claim-generation cohort 11 (2026-10-21)

## Purpose

Generate candidate natural-video explanations on a newly frozen, locally
exposure-screened three-video cohort. The GPU job prepares possible review
items only; generated claim fluency and CPU format validation do not establish
claim truth or faithfulness.

## Frozen cohort and execution

The initial cohort10 lock was created before generation. Its one-video GPU
smoke completed 1/1 generation with one structured claim in 17.2 seconds and
7.49 GiB peak allocated VRAM. That smoke video was added to the local exposure
scan, so it was excluded before selecting the full cohort below.

The new cohort11 lock was then frozen from the same outcome-blind test pool.
It applies 93 source-pool media exclusions plus 161 unique local
claim-generation/scoring exposure hashes (including the latest union manifest
of 117 hashes from 240 source files); the combined private exclusion list has
176 hashes. The lock selects 3 unique video/question rows after scanning 7
eligible candidates. All three exact rows were independently re-decoded on
CPU: 3/3 passed the 16-frame uniform-timing check, zero decode failures. A
separate CPU reconstruction reproduced the lock's cohort digest and ordered
media/QID assignments. This establishes non-overlap against the currently
recorded local exposure files only, not complete model or historical
freshness.

The generation job used Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
16 frames per video, deterministic decoding, and the explicit `v2_short`
prompt. The RTX 5070 Ti run processed all 3 frozen rows; elapsed time including
model setup was 20.7 seconds and peak allocated VRAM was 7.72 GiB. No raw
question, answer, claim, path, frame or protected media was published.

The private output
`data/private_nextgqa/claim_generation_cohort11_gpu_full_20261021_01.json`
is mode `0600`, SHA-256
`bfec8c2bad107784437e7d546b8e0bacc6dd5ec9465a3b42e520cf3505132fe0`. Its
cohort digest, requested/effective rows, ordered media and QID hashes, selected
pair-filter digest, private exclusion set and mode were independently checked
against lock11; all checks passed. The independent receipt audit is pending.

## Local review handoff built from the packet

Added `scripts/build_generated_claim_support_frame_review.py` to consume an
exact generated packet, lock, selected-pair filter and exclusion manifest.
It checks private-file permissions and bindings, ordered media/QID/pair
hashes, zero overlap against the frozen exclusions, exact decoded source
video and 16 sampled timestamps, then displays only valid claims. It retains
the failed row as explicit attrition in the embedded provenance and does not
replace it. The existing renderer and inspector now support one to three
review items while preserving the legacy three-item artifact schema.

The resulting private artifact is
`data/private_nextgqa/claim_support_frame_review_cohort11_20261021_02.html`,
mode `0600`, SHA-256
`43a182ced456858aa7684853aa889d42c719c446261a6b1205db61f918c5b5fa`. It
contains the two format-valid claims and 16 sampled frame thumbnails per
claim; the one failed generation row is not shown and is recorded as
unreviewable attrition. Question/claim text remains in the original English.
The page presents the established Japanese review instructions, requires a
nickname per reviewer, and exports a combined local JSON. No answer or gold
label is displayed.

The focused generation-packet and review suites passed **21/21 tests**, Python
compilation passed, and `git diff --check` passed. The real two-item artifact
then passed the Windows Chrome browser E2E with two synthetic reviewers,
including frame-choice validation, combined JSON blob and filesystem download,
exact artifact-fingerprint inspection, and revocation of the earlier export.
The E2E used only synthetic answers; it does not count as human review or
semantic validation. Independent source audit of this new builder/inspector
path is pending.

## Observed result

| Measure | Result |
|---|---:|
| Frozen / generated rows | 3 / 3 |
| Standalone structured claims | 2 / 3 |
| Predefined claim-shape failures | 1 (`claim_sentence_shape_invalid`) |
| Non-empty generation responses | 3 / 3 |
| Faithfulness scores / effect estimate | 0 / none |

The runner returned `partial`, as required by the 2/3 structured count. No row
was replaced after seeing this failure. The two valid-format claims are
candidate text only; their visual correctness and support are unknown. The
single malformed-format row remains an attrition case and is not silently
dropped from the requested count.

## Interpretation and limits

This is a successful real-video GPU generation-path demonstration with
one-third format attrition, not a semantic or faithfulness experiment. It
establishes that the pinned local model processed the frozen three-row cohort
and produced two strings satisfying the current standalone-sentence contract.
The strings need human review against their sampled frames; the QA-level
`gsub` intervals cannot label the generated claims. No intervention or
baseline/sham score was computed. The separate capture-authorized treatment
gate remains closed, and the number of successful valid claims is only two.
Dataset/model rights remain unverified; all artifacts stay local.

Direct faithfulness remains: claim-support reviews 0/3 on the current artifact,
clock/frame correspondence 0/15, prediction-link review 0/3, and direct
intervention score 0/3. This run changes none of these counts.

## Next action

Have an independent auditor finish the packet-to-lock/provenance and
builder/inspector review. The variable-count handoff now exists for the two
valid claims; keep it explicitly pilot-scale and unscored. The user declined
the previous review round, so no judgments are inferred. Do not run internal
faithfulness scoring until claim review and the separate capture-authorization
requirements exist; do not tune or replace rows from this cohort. The packet
does not independently bind the prompt suffix digest or checkpoint shard
hashes, so a future runner version should add those pins before any
confirmatory claim-generation workflow.

## 2026-10-21 builder hardening after independent audit

The independent code audit found that the output argument used a lexical
prefix check, which could accept `data/private_nextgqa/../../...` and reach
outside the private directory. The auditor did not perform such a write, and
no evidence indicates the existing artifact escaped the private directory.
The builder now resolves and confines the destination under the resolved
private root, rejects symlink path components, requires an existing private
parent directory, and reconciles packet status and valid structured-row count
before decoding frames. The response inspector now checks the embedded item
shape (including all 16 JPEG data URLs), record-index range, and uniqueness.

Observed checks: new path-boundary tests and existing renderer/inspector tests
passed (14/14); Python compilation and `git diff --check` passed. Rebuilding
from the same frozen inputs produced byte-identical HTML SHA-256
`43a182ced456858aa7684853aa889d42c719c446261a6b1205db61f918c5b5fa`, mode
0600, with 2 reviewable rows and 1 explicit attrition row. A post-fix Windows
Chrome E2E attempt could not start because the WSL-to-Windows PowerShell
transport failed (`UtilBindVsockAnyPort`); the earlier E2E pass predates these
hardening changes. The independent re-audit is pending.

The follow-up audit identified a moderate missing join: the builder did not
force the frozen lock's selected pair-filter digest to equal the supplied
pair-filter file. It now enforces that digest, lock schema/status, packet-lock
split agreement, and strict integer counts (so booleans cannot masquerade as
0/1). New unit tests cover those bindings and count validation. The final
focused suite passes 16/16, `py_compile` and scoped `git diff --check` pass,
and a second independent build again produced the same HTML digest at
`claim_support_frame_review_cohort11_20261021_04.html` (mode 0600). Final
independent review is pending. The failed Windows E2E transport remains the
only unrerun check; no human judgments were entered.

Interpretation: this repairs an artifact-generation safety defect and raises
confidence in local handoff integrity; it provides no new evidence about claim
correctness, evidence grounding, prediction dependence, or faithfulness.

## Sampled-frame sequence review control (2026-10-21)

Independent review-instrument audit noted that the original artifact presented
only a static contact sheet. That can make action order or continuation harder
to judge. The Japanese page now has a chronological slideshow with previous,
next, play/pause, and seek controls, synchronized timestamp/frame number, and
highlighting of the matching selectable thumbnail. Its disclosure explicitly
says this is a slideshow of the 16 sampled model-input frames, not continuous
video playback. The original per-frame checkboxes and response schema are
unchanged; advancing to another item stops playback.

The updated cohort11 artifact is
`data/private_nextgqa/claim_support_frame_review_cohort11_20261021_05.html`,
mode 0600, SHA-256
`d0c8aa5d7f9ba434902d36a73ed2a3fbb86b9951c780833494a267fcdb78db25`. It is
still bound to the same 3-row GPU packet, exposes only the 2 format-valid
claims, and records the 1 failed format row as attrition. Unit tests passed
17/17, Python compilation and Node inline-JavaScript syntax checks passed. The
exact artifact passed the Windows Chrome E2E: previous/next and timed
play/pause operated, two synthetic reviewers completed both items, the
combined JSON passed the inspector and matched the browser download, and the
previous export URL was revoked.

This improves the capacity to review temporal claims but does not validate
the sampled frames as a faithful representation of the full event, nor prove
claim truth, reviewer independence, or faithfulness. Actual review remains
0/2 eligible items for this packet; direct prediction-link and intervention
scores remain 0/3. Rights remain unverified and all protected files remain
local. A separate source audit of the new slideshow transition/control code
is requested.
