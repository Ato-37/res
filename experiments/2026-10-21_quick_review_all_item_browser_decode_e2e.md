# Quick-review all-item browser decode E2E (2026-10-21)

## Purpose

Verify that the local Japanese quick-review page can load and decode a video
after each of its three item transitions. The prior browser harness checked
video readiness only for the first item, so a later broken media URL could
pass the flow test unnoticed.

## Change and setup

- Extended `scripts/test_quick_review_artifact_browser_e2e.py` to wait for
  metadata, a nonempty active media source, and a decoded frame at each item
  before injecting synthetic judgments.
- Used Windows Chrome via the existing per-command WSL interop path. The
  synthetic run created temporary generated clips. The actual-page run opened
  the already-existing local validation-cohort review artifact; judgments and
  the downloaded response were synthetic and stayed in a temporary profile.
- No model was loaded, no response from a human was created, and no protected
  file was published. Dataset and model rights remain unverified.

## Verified results

- Synthetic three-item page: browser E2E **1/1 passed**.
- Existing local generated-claim page: browser E2E **1/1 passed**. All three
  item slots passed browser metadata and decoded-frame readiness checks; the
  three-reviewer schema-v6 export flow and artifact-fingerprint match also
  passed. The harness returned only fixed pass/fail information and did not
  print page text, judgments, or media values.
- The saved response was not imported or used as human evidence. UI decode
  does not establish that any claim is true, that a human agrees, or that
  decoded frames correspond semantically to displayed text.

## Interpretation and limits

This closes a browser-playback gap in the review handoff: all three item slots
were loaded by the local Chrome runtime, not just the first. It does not
authenticate the user’s browser, establish reviewer identity or independence,
or validate claim truth. The test fixture for the synthetic page reuses one
generated clip across slots; only the actual local artifact run checks its
three embedded media payloads at each slot. This remains a workflow check,
not a faithfulness experiment.

Direct faithfulness evidence remains **0/3** reviewed claims, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. There is no direct `DeltaF`, interval, or statistical test.

## Next action

Collect a human review bound to the exact generated-claim artifact and complete
the separately required reviewer/release and 15-row clock/frame gates. Then
finish the pre-load transaction runner before attempting the three-row GPU
baseline. Keep the local review response and media out of Git.
