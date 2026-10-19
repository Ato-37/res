# Quick-review v11 prompt clarity and three-person E2E (2026-10-19)

## Purpose

Respond to the three-person exploratory review (3 reviewers, 9 item responses):
item 1 and 2 clarity skewed unclear, item 2 answer-fit was mostly negative,
and item 3 answer-fit was positive for all reviewers while evidence
sufficiency was insufficient for two. Improve rubric instructions without
changing judgment categories or converting these informal responses into
formal evidence.

## Changes

- Built a new local-only v11 artifact; v10 remains intact. Question 1 now asks
  whether a reader can tell what to answer. Question 2 directly asks whether
  the explanation's answer addresses what the question asks. A short guide
  explicitly separates answer-fit from evidence sufficiency and says that an
  answer can fit while necessary events or their order are still missing.
  Question 3 asks whether the explanation gives enough of the video events
  supporting the answer, including event order when needed.
- Kept the three existing choices per dimension and schema v5. Item IDs remain
  stable so the already-collected exploratory JSON still validates. Updated the
  local launcher/runbook and response inspector's default artifact path.
- Extended the synthetic Windows Chrome E2E from two to three consecutive
  reviewers, matching the number in the supplied exploratory JSON. It checks
  the new prompt guidance, clean nickname/choice state for reviewer 3, reviewer
  order, one downloaded schema-v5 JSON, exact 3x3 item ordering, and all 27
  categorical judgments.

## Provenance

Local output:
`data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v11.html`
(30,653,850 bytes, mode `0600`, SHA-256
`b0abf922c12965b68615f8745ad4bda0e368dcb6624321ad9fda1a0a3c2504b5`). The
artifact and protected source media remain local and are not committed.

Source SHA-256:

- Builder: `fc83275e040e9a26d465d7febbcdbca19733a515236f95640f3c47ca60ab29fc`
- Builder tests: `ab62fcb3f51b94133b8d5db210e1b61a3f2e4c01fa9db25f4a550dc9a2a8bd59`
- Browser E2E: `0471903fb21c35f89dbbf7df38c5756543354f2167e2cfffe78324696a67e2f9`
- Response inspector: `6db19e96d7357eb184e77ca58236b7332ad9ed44c5ccc95f91a2f8e5d0f9f9dd`

## Verification

- Builder/inspector/importer CPU suites passed **32/32**; Python compilation
  and scoped `git diff --check` passed.
- The previously collected 3-person JSON revalidated against v11 with the
  same item IDs and same 9 response counts.
- Synthetic Windows Chrome E2E passed **1/1** in 2.19 seconds, including the
  three-person flow and combined download. The initial E2E attempt failed
  because its new clarity-text assertion looked in the choices rather than
  the question section. The assertion was corrected to inspect the rendered
  question section; the rerun passed.
- The Chrome E2E used synthetic records and a generated tiny synthetic video;
  the protected artifact, real videos, and human response were not opened.

## Interpretation and limits

This verifies the revised wording renders and the three-reviewer export still
works in a synthetic browser. It does not establish that independent reviewers
will interpret the new wording consistently; that needs another human check.
The supplied responses remain informal, selected-item observations. Formal
faithfulness remains **0/3** claim reviews, **0/15** verified clock/frame
correspondences, **0/3** prediction-link reviews, and **0/3** direct scores.
No direct effect estimate, confidence interval, or test exists.

## Next action

An independent audit found that v11's question 2 could be read as relevance
rather than answer correctness, and that its worked example foregrounded an
insufficient-evidence judgment. The v11 file was retained locally but not
handed to reviewers. These issues are corrected in the follow-up v12 entry:
question 2 now asks whether the answer is correct against the video, and the
guidance explains the two axes neutrally. Do not treat the earlier responses as
formal judgments or use them to unlock a direct model-scoring run.
