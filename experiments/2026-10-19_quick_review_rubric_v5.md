# Quick-review rubric v5 and sequential handoff — 2026-10-19

## Purpose

Use the user's exploratory response and follow-up to separate three questions
that the previous form conflated: whether the question is clear, whether the
explanation answers it correctly, and whether the explanation names enough
video events to support that answer. Preserve the one-click sequential
reviewer handoff and one combined local JSON export.

## Evidence and change

The Downloads JSON was validated against the existing opaque item IDs. It
contains one respondent and three complete answers. The old schema records
two broad matches and one reported difference, but has no separate question-
clarity or evidence-sufficiency field. The user's accompanying qualitative
notes were: item 1's setup was unclear; item 2's answer was wrong for the
question; item 3's answer seemed right, but the explanation omitted a decisive
sequence of events. These are one person's exploratory observations. The
three item-level comments were not retroactively encoded into the older JSON.

The local-only v9 artifact now asks, for each item:

1. Is the question's situation and condition clear?
2. Does the explanation answer the question correctly?
3. Does the explanation cite events in the video as support for its answer?

The form retains the nickname-first start, one-button transition to the next
reviewer, automatic participant labels after reviewer 1, per-review reset,
and one schema-v5 JSON download. The inspector/importer accept old schemas
v2–v4 and new v5. CLI output is aggregate-only by default; `--show-items`
opts into item-level categorical outputs. The Japanese launcher and runbook
now point to v9 and describe the three axes and combined export.

The generated artifact is a local mode-0600 HTML file under
`data/private_nextgqa/`, 30,652,473 bytes. It embeds protected local media and
must stay out of Git/publication; its file digest is kept out of this log.

## Verification

- Builder, response-inspector, and private-importer suites: **31/31 passed**.
- Synthetic Windows Chrome E2E: **1/1 passed**. It used generated fixture
  videos, completed two sequential reviewers, checked that the second
  reviewer's first question had reset inputs, and downloaded one combined
  schema-v5 JSON containing two respondents and all 18 categorical judgments.
- Python compilation and scoped `git diff --check`: passed.
- The production v9 HTML was used to revalidate the existing old-format
  response: 1 respondent, 3 answers, 2 broad matches and 1 difference. The
  response contains no question-clarity or evidence-sufficiency labels, as
  expected. No new human response was collected in this cycle.
- An independent code audit found and helped repair a malformed-v5-input edge
  case: list/dict values previously escaped as `TypeError`; the validator now
  rejects non-string axis values as the expected `ReviewError`. The same audit
  confirmed v2–v5 compatibility, separate axis counts, reset behavior, and
  formal scoring denial. The audit also found item-level CLI output was
  enabled by default; it is now opt-in via `--show-items`.

## Interpretation and limitations

The new dimensions make the next informal calibration easier to interpret and
directly address the user's comments. The single person's older broad
judgments cannot be treated as independent review, ground-truth correctness,
evidence sufficiency, or a faithfulness measure. The synthetic E2E proves the
software flow and export structure only, not real-video playback or reviewer
agreement. Rights status remains unverified; the media and response stay
local.

## Faithfulness progress

- **Direct faithfulness evidence: not increased.** Formal claim reviews remain
  0/3, verified player-clock/decoder-frame correspondences 0/15,
  prediction-link reviews 0/3, and direct scores 0/3. No direct `ΔF`, interval,
  or test exists.
- **Auxiliary diagnostics only:** one informal respondent supplied three
  broad judgments; software tests cover the v5 schema and combined sequential
  export. Neither is a measured faithfulness intervention.
- **GPU:** no model load or forward was run. The ordinary sandbox reports
  `torch.cuda.is_available() == False`; that visibility alone does not prove
  host GPU unavailability. More importantly, no valid direct-score run is
  currently eligible because the trusted formal review/release and 15/15
  timing/frame evidence remain absent. Closed proxy cohorts were not repeated.
- **Next:** continue the formal independent review/release and timing/frame
  handoff. After its gates pass, the first prespecified GPU action remains
  exactly three untreated baseline forwards, followed by frozen `p/q` and
  separate Stage-2 prediction-link review before any treatment.
