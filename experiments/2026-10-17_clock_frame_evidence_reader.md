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

## Follow-up — align the human-review export contract before review

The independent audit of the prior reviewer handoff confirmed a v5 policy /
renderer mismatch: policy v5 promised only tokens, decisions, and intervals,
while the page exported additional reviewer, environment, timestamp, duration,
rationale, and optional clock fields. Added a new, hash-pinned v6 policy and
kept v5 unchanged. The v6 policy enumerates the actual payload and index field
sets, states which values are self-reported, asks for a pseudonym rather than
a legal name, and requires local private retention of all output. The review
page now makes the matching disclosure and the policy contract is checked
against renderer disclosure in tests.

Rebuilt a fresh, explicitly reconstructed three-item candidate worksheet,
claim-input manifest/sidecar, staged-media/custody bundle, and isolated review
package under `data/private_nextgqa/` using v6. The package builder reports
three records and four files; all outputs are local and scoring remains
`not_granted`. Focused policy, bundle, renderer, assignment-planner, custody
auditor, and diagnostic-target tests pass 93/93 before the final disclosure
sync test, then the policy/render/bundle subset passes 32/32 with the sync test
included. One initial run exposed a stale test string after removing the v5
deviation notice; that test was corrected and the rerun passed.

The independent SSS reviewer also found a low-probability race in Windows
capture cleanup: an output directory could be created by another process
between existence check and creation, then be removed on failure. Changed the
helper to use the Win32 exclusive `CreateDirectory` result and only clean up a
directory it successfully created. This new native path has not yet been
Windows-validated; the earlier ValidateOnly path does not exercise it.

This work makes the candidate page conform to the newly declared export
contract and prepares a fresh review handoff, but does not establish any human
review. Direct claim-faithfulness remains 0/3 human-reviewed and 0/3 scored;
there is no effect estimate, CI, or test. The clock-frame reader remains
candidate-only, no actual screenshot/clock calibration exists, and the
production atomic scoring path is still absent. The candidate cohort was
reconstructed and is exploratory, not a recovered historical or confirmatory
cohort. Protected data and derived artifacts remain local; rights status is
unverified.

Next: validate that a trusted reviewer can complete this handoff and preserve
the private export, then complete independent decoder-frame correspondence and
the all-record production atomic preflight before the nine-forward GPU
feasibility run. The capture helper permission smoke is not evidence of a real
browser capture or frame correspondence.

## Follow-up — validate Windows-to-WSL private modes and pin repair

The first attempt to apply a Windows DACL with `Set-Acl` on the WSL UNC path
failed (Windows error code 1). A separate probe showed default Windows-created
files/directories under the WSL share were Linux modes 0644/0755, which the
review reader correctly rejects. Updated the helper to set POSIX modes through
`wsl.exe chmod` and read them back from Linux; it also requires the private
parent to be mode 0700. The non-capture `PermissionSmokeOnly` branch passed:
directory 0700 and file 0600 were verified, then the helper removed its empty
probe. The latest `ValidateOnly` branch also passed and returned before any
output directory or capture was created. These checks establish mode handling
for the tested WSL bridge only; no screenshot has been taken.

The full 1167-test suite initially had 11 failures, all in synthetic v2
preflight tests. Updating the v2 protocol document to point to the v6 review
policy changed its pinned SHA-256; the preflight fixture still held the old
protocol hash and failed closed with `protocol_v2_identity_mismatch`. Updated
that fixture pin to the current protocol digest and added an exact assertion
that the v6 top-level review export contract includes every declared field,
including `export_disclosure_acknowledged`. A separate audit initially alleged
that this field was missing from policy v6; a byte-level recheck showed it is
already present in policy and its hash matches the generated package, so that
finding was withdrawn. The exact-field assertion prevents this class of
contract drift. After the protocol pin repair, the full 1167-test suite passed.

The generated 0635 v6 package remains the current candidate; its policy hash is
`1e04d9c001b1249bd2d04d9f43730902c45341543d0a7f1aa6badf4b40dbff91`. An
independent audit checked worksheet/sidecar/custody/HTML policy-hash alignment
and private modes (directory 0700, files 0600). This does not certify the
reviewer, the correctness of human judgments, or the package's eligibility for
confirmatory inference. Direct claim-faithfulness remains 0/3 reviewed, 0/3
scored, with no effect estimate, CI, or test; rights status remains unverified
and protected artifacts remain local.
