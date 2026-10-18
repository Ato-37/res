# Clock capture hotkey queue guard (2026-10-18)

## Purpose

An independent source review found a race in the Windows screenshot helper: a
second global hotkey posted during one screenshot could remain queued and be
consumed as the next planned anchor. A full 15-row receipt could therefore
misbind a screenshot while retaining complete token/anchor coverage. Repair
this capture-only risk without opening the review page or capturing a screen.

## Change

- Require the candidate anchor plan to retain the expected
  `predeclared_frozen` plan status as well as its existing non-authorizing
  execution field.
- Show only the ordinal record slot and anchor ID in the interactive prompt;
  opaque token and planned decoder PTS remain in the private output mapping and
  are not written to the console.
- Drain queued `WM_HOTKEY` messages before arming each anchor. Poll only the
  registered hotkey using `PeekMessage`, with a 20 ms sleep and an actual
  60-second timeout. Reject a message whose unsigned modulo-2^32 timestamp
  delta from the arm tick is zero (millisecond ambiguity) or exceeds 60 seconds.
  Any failure follows existing cleanup of the exclusively created output
  directory.
- Update the capture evidence guide and human runbook to describe this
  behavior and its limits.

## Verification

- Windows PowerShell language parser: **passed** on the modified helper. This
  is syntax validation only; it does not execute Win32 message handling.
- Capture helper SHA-256: `4d2c463f68b7a66ec7cb3f8318105a1fa6ba307abacb0036ea78e83a22bcd652`;
  source-contract test SHA-256:
  `5ca66f28be2ff516a9dcdaac5ccd274ddea2b4b54c3b44c4ac5aae5753215302`.
- Focused capture source-contract, clock evidence, calibration-input, and crop
  suites: **54/54 passed**.
- Complete `scripts/` unittest discovery: **1,652/1,652 passed** in **39.278 s**.
  Expected negative-path CLI output and existing Transformers deprecation
  warnings appeared; test process exited successfully.
- Independent source audit confirms ordinary duplicate-during-capture hotkeys
  are drained or rejected before the following anchor can consume them, with
  wrap-safe unsigned tick arithmetic. The audit notes a conservative false
  reject for a valid same-millisecond press or scheduler jitter at the 60 s
  boundary. Retrying is safe; the output directory is removed on failure.
- The test suite is partly source-marker checks and a Python contract model;
  it does not run PowerShell or Windows APIs. Actual Windows/WSL UNC integration
  with a synthetic window remains unverified.

## Limits and faithfulness progress

No review page, video, browser, screenshot, or decoder image was opened. No
human decision, clock correspondence, calibration, GPU forward, or score was
produced. This is engineering-only progress.

`未実施`: direct claim-linked faithfulness remains 0/3 independently reviewed,
0/15 player-clock/decoder correspondences, 0/3 stage-2 prediction-link
judgments, and 0/3 QTA scores. There is no empirical `ΔF`, confidence interval,
test, failure estimate, or multiplicity adjustment. The QTA treatment is the
pinned Qwen2.5-VL internal visual-key intervention on the candidate NExT-GQA
cohort, but no faithfulness metric was measured in this change. Human/trust,
capture/pair review, calibration, and production phase gates remain unresolved;
rights status remains unverified and protected material stays local.

Next: after the witnessed review-only release gate is satisfied, run a
synthetic-window Windows/WSL integration check before any real capture. Then
complete actual stage-1 claim review and 15/15 frame correspondences and
calibration. Do not treat this helper fix as permission to open or score the
candidate cohort.
