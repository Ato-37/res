# Claim-review handoff: temporal sampling and evidence-derived inventory (2026-10-21)

## Purpose

Improve the local claim-support review instrument so temporal/action claims can
be inspected in sample order, and replace manually declared progress counts
with a privacy-minimal exact-artifact response inventory. Separately assess
whether one more real-video GPU run is justified while no direct human review
is available.

## Instrument and implementation

The generated-claim artifact now presents the same 16 model-input samples with
previous/next, play/pause, seek, synchronized timestamp, and a highlighted
matching selectable thumbnail. The Japanese UI states this is a slideshow of
sampled frames, not continuous playback or the full video. The selected-frame
indices and response JSON schema are unchanged; item transitions stop the
timer. A separate inventory tool validates each private HTML fingerprint,
scans requested local JSON roots, matches exports only by exact fingerprint,
deduplicates byte-identical response files and byte-identical HTML copies, and
prints only counts and hashes. It explicitly does not authenticate reviewers,
validate semantic support, inspect signed custody/release bundles, or inventory
direct intervention scores.

## Verified results

The current two-item cohort11 handoff is
`data/private_nextgqa/claim_support_frame_review_cohort11_20261021_07.html`,
mode 0600, SHA-256
`2986df0e531d6affe1cd13763e63827b464bcb68fa7d3b8e390d59fda43919d1`. It is
bound to the existing 3-video GPU generation packet; two claims are displayed
and one fixed format failure is attrition. The canonical three-item artifact
remains `claim_support_frame_review_test_20261021_06.html`, SHA-256
`309739d3aacb39ea46745c41893fbc2736661e421886e228922f1690faa52715`.

The response inventory was independently rerun over 1,163 JSON candidates in
`data/private_nextgqa` and the user's Downloads directory against these two
exact artifacts. It found **0** bound response files, 0 invalid bound files,
0 self-reported reviewer entries, and 0 item decisions. These counts do not
prove that no other kind of feedback exists; they establish only that the
scanned candidate JSON files contain no response bound to these two current
fingerprints.

Later current-state recheck on 2026-10-21 scanned 1,164 JSON candidates over
the same roots and exact artifact fingerprints. It again found 0 bound or
invalid responses, 0 reviewer entries, and 0 item decisions for both artifacts.
This is only a local structural inventory; semantic review and reviewer
identity remain unverified.

The previously supplied exploratory response
`exploratory_video_review_2026-10-19T09-49-47-676Z.json` was structurally
inspectable only against its declared legacy v14 quick-review artifact. A
separate exact-content audit compared its question/claim bindings with the
current three-item `_06` artifact and the current two-item `_07` artifact; it
found **0/3** and **0/2** exact question-plus-claim matches, respectively, and
zero stable item-ID matches. Therefore its three respondents' ratings remain
informal observations about the legacy items and cannot be credited to either
current formal handoff. No review or faithfulness count changes.

This exact-match audit also located the old three question/claim pairs in the
local v6 sentence-claim mapping and the original exploratory generation
packet; each maps to one source media digest and the response's three stable
item IDs match the legacy package's review-token IDs. The inspector reports
three self-reported respondents and nine item ratings. Per-item descriptive
counts (no inference, no reviewer authentication) were:

| Legacy item | Clarity | Answer fit | Evidence sufficiency |
|---|---|---|---|
| 1 | unclear 2/3 | fits 1/3; differs 1/3; cannot tell 1/3 | insufficient 2/3; cannot tell 1/3 |
| 2 | unclear 2/3 | differs 2/3; fits 1/3 | sufficient 1/3; insufficient 1/3; cannot tell 1/3 |
| 3 | clear 3/3 | fits 3/3 | insufficient 2/3; sufficient 1/3 |

These coarse self-reports agree with the user's item-level feedback that the
second claim did not fit, while the third seemed correct but incomplete. They
are useful construct feedback for a future claim-generation protocol: retain
the event details needed to answer the question and allow a short sequence of
actions when one action omits a relevant outcome. This is not a retroactive
revision of those claims, a score for the current artifacts, or an estimate of
faithfulness; current formal counts remain unchanged.

Focused renderer, builder, and inventory tests passed **30/30**; Python
compilation, inline JavaScript syntax check, and scoped `git diff --check`
passed. Windows Chrome E2E passed for both the updated two-item artifact and
the legacy three-item artifact. On the new artifact it verified sample
frame stepping, timed slideshow, timer stop on item advance, two synthetic
reviewers, response inspection, combined JSON download, and stale export
revocation. The legacy page verified its prior three-item workflow. All
answers were synthetic; no human review was created. An independent code and
artifact audit found no critical issue after the duplicate-artifact inventory
handling was fixed.

## GPU decision and faithfulness status

GPU resources were available at the preflight (`RTX 5070 Ti`, 15,284 MiB free
VRAM, approximately 13.8 GiB WSL memory available). An additional two-row
original-video versus constant-gray likelihood run was considered on the new
cohort11 claims. Independent audit found it technically eligible as an
exploratory run (the lock preceded generation, selected media overlap with the
recorded prior exposure union was 0, and the third row's format attrition was
fixed before scoring). It was nevertheless rejected because it repeats the
already closed gray-replacement proxy estimand and would not measure claim
truth, claim-specific support, or intervention-driven faithfulness. No GPU
model load or forward was started in this cycle.

Direct faithfulness remains **0/3** on the active formal cohort, with **0/15**
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct intervention scores. For the cohort11 generated-claim handoff, eligible
human review is 0/2. The UI and inventory improvements change none of those
counts. The first missing research gate remains actual Stage-1 human review
and trusted custody/release evidence; clock/frame evidence and Stage-2
prediction-link review follow. The user's exploratory-review waiver does not
create judgments or remove formal authentication gates. Dataset/model rights
remain unverified; all protected data and artifacts remain local.

## Next action

When a real response bound to the exact artifact arrives, inspect it with
`scripts/inspect_claim_support_frame_review_response.py` and separately
validate the signed Stage-1 custody/release bundle. Until then, do not treat
the generated claims as semantically supported or score the closed gray
proxy again. Continue only non-duplicative measurement and fail-closed
production-runner readiness work that does not manufacture human evidence.
