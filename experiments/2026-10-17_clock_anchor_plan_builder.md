# Candidate clock anchor plan builder — 2026-10-17

## Purpose

Implement a deterministic, outcome-blind selection of candidate HTML5 clock
anchors from the exact staged-video decoder PTS timelines. This prepares the
manual capture workflow; it does not measure explanation faithfulness.

## Setup

The candidate schedule selects nearest decoded frames to 5%, 25%, 50%, 75%,
and 95% of the first-to-last decoded PTS span, with ties resolved toward the
earlier display-order frame. The 5%, 50%, and 95% points form the fit set; the
other two are held out. The builder validates worksheet, lineage, custody, and
staged media hashes, then hashes and decodes the same in-memory media bytes.
Outputs are local-only, exclusive-create, mode 0600, and explicitly
`execution_authorization=not_granted`.

## Verified results

- Focused tests: 9/9 passed, covering deterministic ties, disjoint fit and
  held-out points, short/duplicate timelines, duplicate selected frames,
  zero time-base denominator rejection, private output modes, pair-write
  collision rollback, and exact input/plan/dependency-byte provenance hashes.
- Full repository script suite after final source changes: 1150/1150 passed
  in 14.629 seconds.
- The protected local builder path processed the existing three staged videos
  and wrote a 0600 candidate plan plus provenance sidecar, containing three
  records and fifteen anchor points. Independent metadata-only audit verified
  exact plan/source/input hash bindings and private permissions. The artifacts
  remain under `data/private_nextgqa/`; no video, token, timestamp, digest, or
  score is published.
- Independent audit verified the same-byte hash/decode repair, sanitized
  invalid time-base handling, omission of private digests from stdout, and the
  limited meaning of the downstream `predeclared_frozen` status.

## Interpretation and limitations

This is a successful data-path preparation result, not a browser-clock
calibration or exact compositor-to-decoder-frame correspondence. No browser
capture, human evidence audit, claim scoring, or model forward was performed.
The v2 protocol remains a design draft, human review remains 0/3, and direct
generated-claim faithfulness remains 0/3 scored. Thus there is no treatment
effect, confidence interval, or significance test to report. The user has
authorized local protected-data operations; rights status remains unverified
and all protected outputs stay local.

The plan schema's `predeclared_frozen` label freezes only this outcome-blind
anchor schedule. The full cohort, claim, target, sampler, and scoring protocol
have not been jointly frozen. The builder writes a private provenance sidecar
with input, plan, dependent code, PTS-sequence, decoder, and runtime hashes;
the joint-freeze receipt remains null. Windows Edge 155 exists. WSL-to-Windows
interop fails in the ordinary sandbox; individual elevation can read the
private page path and query the browser version. A dedicated Edge process was
started with the local review-page argument. No human review or browser
capture occurred, and no GPU work was attempted.

## Next action

Build and test a strict evidence reader for screenshot-to-decoder-frame
correspondence and establish a real reviewer-browser capture route. In
parallel, keep the review handoff ready for the independent human reviewer.
Once the joint protocol freeze, reviewed claims, browser evidence, and
production preflight pass, run the minimum prespecified all-record GPU
experiment; do not substitute the older gold-interval pilot or reopen head
search while these direct gates remain actionable.
