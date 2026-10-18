# Synthetic quick-review browser E2E (2026-10-19)

## Purpose

Exercise the v4 nickname-first, three-question flow in Windows Chrome through a
WSL UNC file path. The test is synthetic-only and must not open the protected
review artifact, save a response, or expose page content in its output.

## Setup

Added `scripts/test_quick_review_artifact_browser_e2e.py`. It creates three
synthetic records and one tiny generated H.264 frame under a unique private temp
directory, then uses `build_nextgqa_quick_review_artifact.build()` to generate a
v4-style HTML fixture. The builder's stdout is suppressed. The test translates
the fixture and Windows Chrome paths with `wslpath`, launches Chrome with a
unique profile, and keeps CDP on Windows `127.0.0.1` with an exact allocated
port. HTTP(S) requests are denied through CDP, Chrome background networking is
disabled, and a synthetic fetch is checked against the page CSP. The driver
enters a fixed synthetic nickname, advances through all three choices, checks
the final completion/save UI, and never clicks Save. Browser output is reduced
to fixed status codes; the page text, fixture bytes, and synthetic responses
are not printed or saved. A timeout cleanup helper uses a unique run ID and the
exact Chrome PID recorded for that profile.

## Verification

- `py_compile` passed for the new harness; scoped `git diff --check` passed.
- The initial elevated browser attempt after adding the final checks failed on
  repeated top-level JavaScript `const` declarations across CDP evaluations.
  Each synthetic choice is now evaluated in a fresh IIFE scope. A cleanup
  helper parse issue was also fixed; timeout cleanup verifies the exact
  per-run profile in the recorded process command line and the Chrome image
  path before terminating it.
- The final individually elevated Windows Chrome run passed **1/1** with the
  no-file-picker and informal-only assertions enabled. The combined browser and
  builder suites passed **10/10**. The test verified nickname-first, three
  sequential synthetic choices, completion UI, no file input, visible
  informal-only notice, CSP `connect-src 'none'`, a rejected synthetic external
  fetch, and no response/save state change. The Windows temp directory had
  **0** matching profiles and **0** Chrome processes after the run; the WSL
  synthetic fixture directory was removed. The ordinary sandbox cannot invoke
  Windows interop (`UtilBindVsockAnyPort:317: socket failed 1`), so this E2E
  requires individual elevation in this environment.
- The protected real artifact was not opened or read. No human response or
  response file was created. The temporary profile and synthetic fixture are
  removed by the harness.

## Interpretation and limits

The successful browser run verifies the flow on a generated synthetic fixture,
not the protected artifact's rendering or its video playback. It provides no
human judgment, claim-validity result, identity evidence, or faithfulness score.
Direct NExT-GQA faithfulness remains **0/3** claim reviews, **0/15** clock/frame
correspondences, **0/3** prediction-link reviews, and **0/3** direct scores; no
direct `ΔF`, interval, or test exists. Rights status is unverified; protected
artifacts remain local.

## Next action

Keep the informal quick-check flow separate from the formal claim/frame and
prediction-link review. The next direct-evidence step remains the external
trusted-root/role roster and witnessed review-only handoff; no direct GPU score
is permitted until the signed human reviews, all frame correspondences, and
separate baseline/prediction-link gates exist. Do not count this UI E2E as
faithfulness progress.

## Current-checkout recheck (2026-10-19, 07:40 JST)

The response-inspector tests passed 8/8. A repeat browser E2E under the normal
Codex sandbox reached an exact temporary-profile cleanup error, so that run is
not counted as a pass. Re-running the synthetic-only browser test with the
per-command Windows interop access required by this WSL environment passed
1/1 in 1.87 seconds. A separate read-only Windows check found **0** matching
test profiles and **0** Chrome processes afterward. The test again exercised
only generated synthetic HTML/video, entered a synthetic nickname and three
choices, and did not click Save or open the protected review artifact. No
review response or faithfulness evidence was produced.
