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
