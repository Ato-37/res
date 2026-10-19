# One-click sequential reviewer handoff (2026-10-19)

## Purpose

Remove the extra nickname form between reviewers in the local, informal
three-item video impression check. After one person finishes, one click should
start the next person's first item; the last person should be able to save all
responses as one JSON. This is a usability change, not a change to the formal
claim/frame review protocol or its evidence requirements.

## Implementation

The first participant still enters a nickname. After completion, **次の人へ**
starts the next participant's first question immediately and assigns an ordered
local label (`参加者2`, `参加者3`, ...). Each completed participant contributes
three answers to the existing schema-v4 `respondents` array. The existing
**全員分を保存** action downloads one JSON containing all completed
participants. Japanese launcher/runbook instructions now describe the one-click
handoff. The new HTML is private/local and is deliberately not committed.

The page keeps separate choices for (1) whether the explanation's claim matches
the video and (2) whether the explanation provides enough basis to answer the
question. These are descriptive impressions only; they do not establish
faithfulness or resolve the reviewed examples.

## Checks and observed results

- Builder, response-inspector, and private-importer suites: **28/28 passed**.
- Windows Chrome E2E on synthetic items/video: **1/1 passed**. It completed two
  sequential participants, confirmed the second participant starts at item 1
  after one click under the automatic `参加者2` label, confirmed no radio choice
  carries over, downloaded one JSON to disk, and validated schema-v4 structure,
  reviewer order, and all six ordered judgments.
- Inline JavaScript `node --check`: passed.
- Scoped `git diff --check`: passed.
- The protected review HTML was not opened by the E2E; the browser used only its
  generated synthetic fixture. The real-data local artifact was generated with
  mode `0600`; it remains local and is excluded from Git.

## Interpretation and limitations

This verifies the software handoff and combined local download, not human
review quality. Nicknames and assigned participant labels are not identity or
independence evidence. Responses stay in tab memory until download, so closing
or reloading early loses unsaved work. The artifact remains an informal
impression check and must not be counted as a formal faithfulness result.

## Next action

Use the updated local artifact if additional informal reviewers are desired.
Import any downloaded JSON through the existing private importer. Keep formal
claim/frame review and direct faithfulness scoring at their existing gates.
Current direct faithfulness evidence remains **0/3** formal claim reviews,
**0/15** verified clock/frame correspondences, **0/3** prediction-link reviews,
and **0/3** direct scores; no direct `ΔF` estimate exists.
