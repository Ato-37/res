# Candidate screenshot-to-decoder evidence reader — 2026-10-17

## Purpose

Close the code-level gap between raw HTML5 clock events and independently
reviewable screenshots / decoder frames for the three-item, five-anchor
candidate plan. The reader must fail closed on incomplete, ambiguous,
modified, or mismatched artifacts while remaining non-authorizing.

## Setup

The private reader requires the candidate plan and provenance sidecar, exact
raw observation payload and integrity index, signed evidence manifest, a
separately pinned Ed25519 public key, exact staged media files, and screenshot
files. It reuses the strict clock observation converter, decodes each exact
staged video through the repository's single-threaded PyAV decoder, checks
planned relative PTS and RGB frame hashes, recomputes screenshot and viewport
crop hashes, checks per-anchor event binding, and verifies a signed independent
reader declaration. Output includes counts and gate status only, not tokens,
paths, image content, or hashes.

## Verified results

- Focused synthetic tests: 12/12 passed after the capture helper integration
  audit. They cover all 15 plan/event/media/
  screenshot/crop/frame joins, missing/duplicate anchors, ambiguous review,
  screenshot and media tampering, decoder PTS/crop mismatches, screenshot/event
  timestamp skew, signature/key mismatch, and a signed false `match` assertion
  to demonstrate that the tool does not perform visual matching.
- All videos, screenshots, Ed25519 keys, and signatures were generated inside
  temporary synthetic test directories. No protected media or real browser
  screenshot was used. No GPU, model, or scoring call occurred.
- The output's strongest status is
  `candidate_signed_evidence_bytes_consistent`; authorization remains
  `not_granted` and the candidate provenance still says not jointly frozen.
- Independent source audit found and helped close a schema mismatch: the
  Windows helper emits three uint32 tick fields; the reader now checks their
  types/ranges, modulo wraparound arithmetic, and the 250 ms limit. Added
  rejection and wraparound tests.
- Windows PowerShell `ValidateOnly` completed successfully against the pinned
  2026-10-17 0542 candidate plan and sidecar, exiting before output directory
  creation, hotkey registration, or screen capture. This does not exercise
  the WSL UNC ACL or the capture path.

## Interpretation and limitations

This closes a structural byte-join and signature-checking implementation gap,
not the real evidence gate. The signature only authenticates the public key
when that key hash is independently frozen. The screenshot, crop geometry,
timestamps, and `match` decision still depend on signed declarations. The
reader does not prove that the image came from the Edge compositor, establish
the signer's identity/independence, or compare screenshot pixels with decoder
pixels. The 250 ms screenshot/event threshold is a candidate design value and
must be frozen before use.

Direct generated-claim faithfulness remains 0/3 human-reviewed and 0/3 scored;
there is no treatment estimate or uncertainty interval. The live review page
is open, but no human response, screenshot evidence, clock calibration, or
model forward has been recorded. The GPU is available, but the all-record
production atomic preflight and upstream evidence gates remain unmet. Local
protected-data use is user-authorized; rights status remains unverified and
protected artifacts stay local.

## Next action

Perform a bounded local capture-path smoke only after the capture method,
reviewer key, and 250 ms tolerance are jointly frozen and an independent audit
confirms the Windows ACL behavior on the WSL UNC target. Then complete all
3×5 human frame reviews, bind signed evidence to exact media and observations,
and satisfy the all-record production atomic preflight. Only then start the
nine-forward GPU feasibility run. No actual-data model command is valid yet
because there is no real clock observation or screenshot evidence bundle.
