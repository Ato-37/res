# Simplified local quick-check artifact (2026-10-19)

## Purpose

Make the three-item exploratory video check easy to start. A reviewer should enter a nickname, then answer a broad impression question for each embedded video without selecting separate files, locating timestamps, or writing a note. This is a usability change for a lightweight check, not a complete hallucination annotation protocol.

## Change and setup

- Updated `scripts/build_nextgqa_quick_review_artifact.py` to generate a new local-only v4 artifact. The earlier v3 artifact remains unchanged.
- The new flow asks for a nickname and then presents three Japanese questions with the video already embedded. The reviewer chooses one of: “だいたい合っている”, “気になる違いがある”, or “よくわからない”. Timestamp, free-text, and optional English fields were removed.
- Kept local JSON download, opaque item IDs for mapping answers back to the fixed items, and a clear notice that these informal impressions cannot establish hallucination status, faithfulness, or an experiment result.
- Updated the Japanese launcher, runbook command, and response-inspector default to the v4 filename. Video/question data and answer JSON remain local and are not committed.

## Checks and observed outcome

- Builder and response-inspector tests: **16/16 passed**.
- Python compilation and scoped `git diff --check`: passed.
- Generated `data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v4.html` from the existing three-item local package. File size is 30,649,305 bytes and permissions are `0600`.
- Structural checks confirmed embedded video data, no file input, nickname-before-question flow, no timestamp or free-text fields, and exactly the three broad response options.
- No human responses were collected or analyzed in this change.

## Interpretation and limitations

The artifact now matches a short impression check: entering a nickname is the only setup before the first question, and no per-video file handling is needed. The labels capture coarse impressions only. They do not establish whether a claim is wholly true, identify all hallucinations, provide evidence intervals, validate reviewer identity or independence, or authorize model scoring. These same three items are in the formal candidate set; anyone who sees them here should not later serve as a blinded reviewer for those items. Rights status remains unverified, and the artifact must stay local.

## Faithfulness progress and next action

- Direct explanation-faithfulness scoring remains **0/3 claims**; this UI change adds no score, effect estimate, confidence interval, or test.
- Formal claim review remains **0/3**, player-clock/decoder-frame correspondence **0/15**, and prediction-link review **0/3**. No new response JSON was produced by this implementation.
- Next, a reviewer can open the v4 artifact, answer the three broad questions, and save the response JSON locally. Treat it only as an exploratory impression check. Continue the independent CPU-side receipt integration audit separately; do not start an unreviewed GPU score run from this artifact.

```bash
xdg-open ./data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v4.html
```

## Follow-up wording adjustment (2026-10-19)

The user clarified that this is only a lightweight check and should not ask reviewers to make a complete hallucination judgment. Updated the prompt to ask whether the explanation generally matches the video, with plain choices: “はい、だいたい合っている”, “違うところがある気がする”, and “わからない”. The nickname remains the only setup before the first embedded video/question. The page explicitly describes this as a preliminary impression check, not a strict hallucination test or performance evaluation. No file picker, timestamp, or free-text entry was added.

- Updated the private local artifact in place; it remains mode `0600` and must not be published.
- Builder tests: **9/9 passed**. `git diff --check` passed for the source/test changes.
- No reviewer response was collected. Direct faithfulness scoring remains **0/3**; this UI wording change adds no efficacy evidence.

## Current builder browser E2E retest (2026-10-19)

The first E2E attempt in the normal sandbox could not invoke Windows
PowerShell/Chrome because WSL interop returned a vsock error; this was an
environment denial, not a page-flow result. Re-running the same bounded test
with one-command elevated execution passed **1/1**. It built a synthetic
three-video fixture, opened it in an isolated temporary Chrome profile, entered
a synthetic nickname, traversed all three questions, verified there was no
file picker and the informal-only notice was present, and confirmed the Save
button was not activated. The exact temporary browser profile was cleaned up.
The protected v4 artifact, videos, and human responses were not opened by this
test, so it verifies the production builder's flow rather than playback of the
protected media. Builder/inspector/importer tests passed **25/25** in the same
work period. Direct faithfulness remains **0/3** scored; this usability and
engineering check adds no human or model evidence.

## WSL open-command check (2026-10-19)

The current Ubuntu environment does not have `xdg-open`, so the runbook's
original launch command was not usable here. Opening the same private HTML via
`powershell.exe` and Windows `Start-Process` succeeded. The Japanese runbook now
documents that WSL command first and retains `xdg-open` for Linux desktops. The
HTML stayed local with mode `0600`; this only launches the exploratory quick
check and does not create a response or formal review evidence.

## Browser E2E recheck (2026-10-19)

Re-ran the synthetic Windows Chrome harness against the current production
builder. The test failed at its final exact-profile cleanup assertion
(`exact Chrome process/profile cleanup failed`) before surfacing the page-flow
result, so this invocation is **inconclusive** for the nickname-to-question
interaction. The run generated only a temporary synthetic artifact/video and
did not open the protected v4 artifact or save any response. Static builder and
response-inspector tests remain the verified checks for the current version;
the earlier recorded browser run is unchanged. Do not treat this failed
harness invocation as a fresh E2E pass.

## Latest current-builder verification (2026-10-19)

Re-ran `scripts/test_quick_review_artifact_browser_e2e.py` against the current
builder. The normal sandbox again failed only at its exact Windows Chrome
profile cleanup check because WSL interop is unavailable there. The same
synthetic-only test with individually elevated Windows interop passed **1/1**
in 1.98 seconds. It verified the nickname-first transition, all three
questions, and completion state without opening protected media or clicking
Save. This verifies the builder flow, not playback of the private v4 artifact
or human review.

The builder, response inspector, and private importer suites passed **25/25**.
A local, redacted scan found **10** response files, **1** unique byte content,
and **0** files valid for the current v4 artifact; all ten were rejected for
item-ID/judgment mapping. These are synthetic fixtures, not reviewer
observations. No response was changed or removed. Current formal evidence
remains **0/3** claim reviews, **0/15** clock/frame correspondences, **0/3**
prediction-link reviews, and **0/3** direct scores. No GPU model workload or
faithfulness estimate was run in this checkpoint.

## Embedded media verification on current private v4 artifact (2026-10-19)

Read the current owner-only HTML and checked its three embedded video payloads
locally with `ffprobe`, using temporary owner-only files that were removed on
exit. All **3/3** streams were positive-duration, video-only H.264. The HTML
remains mode `0600`. No question/claim text or media details were emitted, and
no video was opened in a player or displayed as part of this check. This
confirms container/codec readability for the current artifact, not playback in
the reviewer's browser/device or human viewing. No review response or
faithfulness score was created; formal direct evidence remains unchanged.
