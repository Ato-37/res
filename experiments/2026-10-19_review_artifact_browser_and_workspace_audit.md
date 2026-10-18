# Human-review artifact browser and integration audit (2026-10-19)

## Purpose

Verify the simplified nickname-to-question artifact in a real browser, then
check whether the current repository contains the previously described
single-page formal audit workspace. Keep this UI smoke separate from direct
faithfulness evidence.

## Setup and verified observations

- Opened a temporary, mode-0600 copy of
  `data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v4.html` in
  Windows Chrome headless through the WSL file share. The test injected only a
  synthetic nickname and three `cannot_tell` choices; it did not save a response
  or inspect video content.
- The browser check passed: the nickname screen advanced to the first question,
  all three embedded videos reached loaded metadata, the three question steps
  advanced in order, the completion screen appeared, and no file input existed.
  The temporary copy was removed at process exit. No response JSON was created.
- Independent source inspection found that the tracked
  `docs/nextgqa_human_review_workspace_ja.html` is currently a quick-check
  launcher. It contains no Stage-1 receipt verification, Stage-2 v3 baseline
  verification, audit report generation, or stale-state clearing. Existing
  component tests cover the verification APIs separately, but not their
  integration in a shared formal-review browser artifact.
- The protected local directory contains no quick-review response JSON matching
  the expected response naming patterns. The current runbook correctly points
  to the nickname flow with `xdg-open` and distinguishes it from formal review.

## Interpretation and limits

The simple artifact is operational in Chrome and matches the requested short
preliminary check. This establishes only browser/UI behavior on the generated
artifact; it establishes no claim truth, reviewer identity, independent review,
frame/clock correspondence, prediction-link judgment, or model faithfulness.

The repository's prior notes describing a unified formal workspace are not
evidence that such an artifact exists in the current worktree. The formal
Stage-1/Stage-2 verifier components remain available, but the browser integration
and browser E2E described in the prediction-link log are still outstanding.
This is a software-delivery gap; it does not justify bypassing any real-human or
trusted-root requirements.

## Faithfulness progress

- Direct generated-explanation faithfulness: **0/3 scored**; no direct effect
  estimate, uncertainty interval, or test.
- Formal claim review: **0/3**; browser-clock/decoder-frame correspondence:
  **0/15**; prediction-link judgment: **0/3**.
- The Chrome interaction used synthetic answers and did not produce a response
  artifact. It is a usability check only.
- No new GPU run was justified: there is no reviewed cohort or valid handoff,
  and existing human-free proxy runs are already complete.

## Next action

Build or recover a separate formal-review browser workspace that invokes the
existing Stage-1 and Stage-2 verifiers in one interface, then test success,
tampering, state clearing, and persistent non-authorization with synthetic
fixtures in Chrome. Keep the nickname quick-check launcher separate. Do not
begin direct scoring until the real role roster, out-of-band trust-root pin,
witnessed handoff, human reviews, and frame/clock evidence are supplied and
verified.
