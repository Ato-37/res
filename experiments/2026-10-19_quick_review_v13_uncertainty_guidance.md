# Quick-review v13 uncertainty guidance (2026-10-19)

## Purpose

The three-person feedback found two of the three item-1 clarity judgments
unclear. An independent translation audit verified the local Japanese
translation against the source; it found the source question's scene locator
was broad, not mistranslated. Adding an item-specific gloss could change the
question stimulus, so this iteration adds only a general uncertainty rule.

## Change

- Added common instructions: when the question meaning or video scene is
  unclear, do not guess; mark question clarity as unclear and dependent answer
  correctness/evidence judgments as cannot tell.
- Kept the item text verbatim, the v12 distinction between video-answer
  correctness and explanation evidence, all answer categories, schema v5, and
  the same three opaque item IDs.
- Updated the local launcher, runbook, and inspector default to v13. Earlier
  artifacts remain local and unchanged.

## Provenance

Local output:
`data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v13.html`
(30,654,097 bytes, mode `0600`, SHA-256
`9abb9b8d647c67bddf49a728aaead44efe062ffdb0a929f14fd0b9a529913ebb`). The
artifact embeds protected local media and is not committed.

Source SHA-256:

- Builder: `5098e312bfd0c4961fde0dd273e5647afddf6ce4266289153fd71953c3b76255`
- Builder tests: `9ea6dd319a59a98f798f4f7c6ede5a98791ef1c3831733a06e52e2056065e3b6`
- Browser E2E: `08a5e5b23dd70b2e459bf94abeed44064772cc22c52e52afb0d6bae35aa0332a`
- Response inspector: `6945322de582efdcdcfaaaef38924f3bac53aacca290e92a0eee4679bc64a407`

## Verification

- Builder/inspector/importer CPU suites passed **32/32**; Python compilation
  and `git diff --check` passed.
- The existing 3-person exploratory response validated against v13, with the
  same item IDs and unchanged 9-response aggregate. Nicknames and free text
  were not printed.
- Final synthetic Windows Chrome E2E passed **1/1** in 2.19 seconds. It checked
  the uncertainty instruction, Q1/Q2/Q3 wording, the three sequential
  reviewers, cleared reviewer-3 state, and one combined schema-v5 download.
  Reviewer 3 selected “unclear” for Q1 and “cannot tell” for Q2/Q3 on the first
  item; the downloaded JSON retained those abstentions while the other two
  items had ordinary synthetic choices.
- The first abstention-path assertion incorrectly required all three of
  reviewer 3's items to be abstentions; the fixture intentionally set only the
  first item that way. After narrowing the assertion to that item, the Chrome
  E2E passed as reported above.
- The independent source audit found the v13 item wording/defaults/docs aligned
  and the uncertainty guide non-item-specific. The E2E-source audit found the
  three-reviewer export checks present and confirmed that the final test sets
  only reviewer 3's first item to unclear/cannot-tell, then checks those exact
  values in the downloaded JSON. The actual Chrome E2E was run by the primary
  agent, not by the auditors.
- No new human review was run; browser validation used only synthetic rows and
  a generated synthetic video.

## Interpretation and limits

The item-specific question is unchanged, which preserves the evaluated
stimulus. The added general instruction may help reviewers abstain when either
the wording or clip location is unclear, but only a fresh reviewer can show
whether it improves comprehension or calibration. Prior feedback is selected
and informal; it is not an unbiased evaluation of v13.

Formal faithfulness remains **0/3** claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. No direct effect estimate, confidence interval, or test exists.

## Next action

Use v13 for another informal check and compare the same three judgment axes
descriptively. Do not use these exploratory responses as formal labels or as
authorization for direct GPU scoring.
