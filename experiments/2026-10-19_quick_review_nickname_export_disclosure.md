# Nickname disclosure on the lightweight review artifact (2026-10-19)

## Purpose

Keep the three-question local impression check easy to complete while making
clear that the optional downloaded response contains the nickname. This is a
usability/privacy disclosure change, not a hallucination or faithfulness test.

## Change

The completion screen now says the saved JSON includes the entered nickname,
asks reviewers not to use a real name, and asks them not to share or publish the
file. The opening screen now accurately says answers are not sent externally
and that saving a JSON download is optional at the end. The flow remains
nickname → “質問へ” → three embedded-video questions;
there is no file picker, timestamp entry, or detailed hallucination taxonomy.
The private artifact was regenerated at
`data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v4.html` with mode
0600. The protected HTML and video content remain local and are not included in
this log or any publication.

## Verification

- Builder and response-inspector tests: **17/17 passed**.
- Synthetic-only Windows Chrome browser E2E: **1/1 passed**. It exercised the
  nickname-first three-question flow, reached the completion screen, did not
  click Save, did not open the protected artifact, and created no response.
- Current generated artifact contains the new warning and retains mode 0600.

## Interpretation and limits

This reduces the chance that a reviewer unintentionally shares a nickname in
the optional JSON. It does not establish that anyone completed the check or
that a selected impression is correct. No response was collected or analyzed.
Rights status remains unverified; protected materials remain local.

## Faithfulness progress and next action

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. There is no real-data `ΔF`, confidence interval, or test. The
informal UI still cannot supply formal Stage-1 evidence. Next, continue
source-bound synthetic handoff integration and retain the reviewer's local JSON
only as exploratory feedback unless the separate formal review protocol is
completed.
