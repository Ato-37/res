# Blinded frame-pair browser E2E (2026-10-21 JST, local)

## Purpose

Verify the full offline human-review UI path that unit tests alone could not
establish: decode all embedded pairs, enforce completion before export, save
the JSON from a real browser, and pass that exact download through the
production byte-level auditor.

## Setup and provenance

- Test: `scripts/test_blinded_frame_pair_review_browser_e2e.py`.
- Browser: installed Windows Chrome launched from the existing WSL checkout
  with an isolated temporary profile. WSL/Windows interop was blocked in the
  normal sandbox (`UtilBindVsockAnyPort`, socket failure); the exact test ran
  under individual elevated execution.
- Data: 15 wholly synthetic PNG pairs (30 embedded images, each 16×12 pixels),
  synthetic plan and manifest, and the fixed 15-pair review UI. No local dataset,
  protected review page, response JSON, model, or GPU was accessed.
- The browser test blocked HTTP/HTTPS requests, required all 30 images to
  decode, and entered a synthetic nickname. Export remained disabled after
  14/15 rows; it also remained disabled after all 15 rows until the independence
  checkbox was supplied. Chrome then downloaded `blinded_frame_pair_review.json`; the test
  checked schema, package digest, all 15 `match` decisions and purity
  attestations, then passed the exact bytes to `audit_export_bytes`.
- Observed result: browser E2E **1/1 passed** in 2.296 seconds; the exact
  downloaded export was structurally accepted with 15 matches and
  `execution_authorization=not_granted`. Browser temporary profile cleanup was
  also asserted by the E2E.
- Related builder, renderer, prediction-link and validator unit suites:
  **37/37 passed**. The broader release-order/ceremony/frame-pair/baseline suite
  passed **86/86** earlier in this cycle. `git diff --check` passed.
- Source SHA-256 at this checkpoint:
  - E2E: updated after the final gate assertions; see the current source hash
    recorded at the end of this log.
  - builder: `14d173bd6e733feddddf0fc6429eb8258fb9bbfa4ed57d3a7490f0c784295991`

## Integration recheck

The initial next-action note above assumed that the browser export still needed
to be connected to the clock/frame evidence package. Re-reading the current
source showed that this bridge is already implemented: the evidence builder
pins the pair package, custodian identity map, export and reviewer HTML; calls
`audit_export_bytes`; and the role-specific frame reviewer signs the resulting
evidence manifest. The independent auditor revalidates those exact bytes and
the rostered reviewer key. This was not inferred from the browser test alone.

The direct integration/race/attestation suites
(`test_build_nextgqa_clock_evidence_manifest`,
`test_audit_nextgqa_clock_frame_evidence`, and
`test_nextgqa_capture_operator_attestation`) passed **81/81** in 26.282 seconds.
These tests cover tampering, symlink/path races, incorrect role key, missing or
ambiguous pairs, exact package joins, and non-authorizing signed candidate
construction with synthetic fixtures. No production code change was needed for
this bridge in this turn.

## Interpretation and limits

This demonstrates that a reviewer can use the 15-pair offline browser artifact,
that incomplete responses stay blocked, and that the browser-generated file
round-trips through its exact-byte structural auditor. Synthetic identical
image pairs test software wiring only. The run does not establish image/video
correspondence, reviewer identity or independence, honest viewing, semantic
correctness, clock calibration, trust-root authenticity, scoring permission,
or explanation faithfulness. No real participant review or direct model score
occurred. Rights status for the protected dataset/model remains unverified;
this E2E used synthetic pixels only.

## Faithfulness status and next action

Direct faithfulness remains **0/3 claims reviewed, 0/15 clock/frame
correspondences, 0/3 prediction-link judgments, and 0/3 direct scores**. There
is no direct effect estimate or inferential result. The experiment is
engineering validation of one handoff component, not an efficacy result.

The software handoff path is exercised from browser input through the exact-byte
export auditor, and the separate evidence builder/auditor composition passes
its synthetic integration suite. The next real-world action is for the
assigned human roles to use the frozen artifacts, produce the frame-pair export
and role-signed evidence, and provide a root fingerprint independently verified
out of band. That external trust/person input remains absent, so the pre-score
experiment stays blocked; test keys and synthetic reviews cannot fill it.

Final E2E source SHA-256 after checking both the 14/15 and missing-independence
denials: `ad957cd8fc2a4d6f87d8b15b77d88d21742fed177dcb91eb6171b3ed84af2e48`.
