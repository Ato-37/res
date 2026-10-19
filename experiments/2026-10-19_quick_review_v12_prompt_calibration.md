# Quick-review v12 prompt calibration (2026-10-19)

## Purpose

Correct two issues found in the independent review of v11 before handing the
new wording to participants: question 2 could mean topical relevance rather
than answer correctness, and the prior guide gave a one-sided example that
could prime an “insufficient evidence” response.

## Change

- Question 2 now asks whether the answer is correct when compared with the
  video. Question 3 separately asks whether the explanation presents the
  video events that support that answer, including order when needed.
- The guide names these as separate dimensions without exemplifying either a
  positive or negative rating.
- The answer categories, schema v5, sequential-review flow, and three stable
  item IDs remain unchanged. The v11 file remains local for iteration history;
  the launcher and runbook now open v12.

## Provenance

Local output:
`data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v12.html`
(30,654,208 bytes, mode `0600`, SHA-256
`dc0dd58fbbc8300b4a1ed44925890c62e18c1ad31ac598589bfb877527ae73e2`).
Protected media and HTML stay local and are not committed.

Source SHA-256:

- Builder: `5f26ca2f37a0327f42d754cdabcc99c3a3b2d414c9cb20832287e18eebe06094`
- Builder tests: `a74568090d60905e66fca201700066270c589fadd44d16e17dc0d909080ccdf4`
- Browser E2E: `38c75e06cc99b0923eac028bb4797b16b3fec50dbcbb607c979f13f033547a19`
- Response inspector: `613cf3cd248e26bbaf0b1490b71d95ecac736157d4fcb5dc223ac9da29554fe0`

## Verification

- Builder/inspector/importer CPU suites passed **32/32**; Python compilation
  and scoped whitespace checks passed.
- The imported three-person exploratory JSON validated against the v12 item
  IDs, reproducing the same 9-response aggregate without displaying names or
  free text.
- Synthetic Windows Chrome E2E passed **1/1** in 2.18 seconds. It checks the
  revised neutral guide, three sequential nicknames/reviews, blank and cleared
  state for reviewer 3, and one combined schema-v5 JSON with 3 respondents,
  9 item records, and 27 rubric judgments.
- Independent audits of the prompt distinction, item-ID/schema compatibility,
  launcher/runbook paths, and three-reviewer E2E source found no remaining
  material regression. The auditor independently reran the CPU suites; the
  Chrome E2E pass above was run by the primary agent.
- This is software verification only. No new human review was run.

## Interpretation and limits

The prior exploratory feedback motivated the wording change, but cannot be
used as an unbiased evaluation of the revised wording. A fresh informal
review is needed to determine whether users now distinguish answer correctness
from evidence sufficiency and whether the question wording is clear. The
three-person feedback is selected and descriptive, not a formal faithfulness
measure.

Formal evidence remains **0/3** claim reviews, **0/15** verified clock/frame
correspondences, **0/3** prediction-link reviews, and **0/3** direct scores.
No direct effect estimate, confidence interval, or test exists.

## Next action

An independent translation audit confirmed that item 1's Japanese wording
faithfully follows its local source and found no answer cue. The clarity risk
appears to come from the source question's broad scene locator; adding an
item-specific gloss would change the stimulus. Before any handoff, v12 was
superseded by v13 with a general no-guess instruction for unclear question or
video scene. Preserve the current categories unless a new schema version and
backwards-compatible validation are implemented. Do not use these informal
responses to authorize direct GPU scoring.
