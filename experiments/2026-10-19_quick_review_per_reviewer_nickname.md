# Per-reviewer nickname in sequential quick review (2026-10-19)

## Purpose

Make the local informal review flow identify each respondent with their own
nickname while retaining one combined final JSON download.

## Implementation

The previous handoff automatically labeled later respondents as `参加者2`,
`参加者3`, and so on. In v10, **次の人へ** returns to the nickname form and
labels it with the next reviewer number. That person enters a nickname and
presses **質問へ** to begin. A **前の画面に戻る** control lets the group return
to the completed-review screen without starting another respondent. Each
respondent's nickname remains attached to their own answers in the combined
JSON; the page warns that names are included and advises against real names.
The runbook and Japanese launcher now describe the same flow. The previous v9
artifact remains unchanged.

## Verification

- Builder, inspector, and private-importer unit suites passed **31/31** after
  the update. Python compilation and `git diff --check` passed.
- The individually elevated synthetic Windows Chrome E2E passed **1/1** in
  **1.933 s**. Two synthetic reviewers each entered a different nickname,
  completed all three questions, and saved one combined schema-v5 JSON. The
  harness also verified that backing out of the next-person form returns to
  the completion screen without losing the first response. It checked
  respondent order, each person's nickname, all 18 rubric judgments, and item
  ordering. It used generated synthetic items/videos only; it did not open the
  protected review page or create a human response.
- The local v10 artifact is mode `0600`, size 30,654,163 bytes, SHA-256
  `ba5ae731b4574041528e7a1e7413daee923c2731ec1453cd61996f8cefcf5808`.

## Limits and research status

This is a usability and software-handoff improvement, not a new review or
faithfulness measurement. It does not verify reviewer identity, independence,
or whether each person actually watched each video. Direct faithfulness remains
**0/3 formal claim reviews, 0/15 verified clock/frame correspondences, 0/3
prediction-link reviews, and 0/3 direct scores**; there is no direct `DeltaF`,
confidence interval, or test. The page remains a simple informal impression
check and is not a substitute for the formal review ceremony.
