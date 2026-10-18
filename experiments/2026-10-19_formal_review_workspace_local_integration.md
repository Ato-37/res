# Local formal-review workspace integration (2026-10-19)

## Purpose

Provide a separate, localhost-only browser surface for Stage-1 bundle audit and
Stage-2 strict v3 prediction-link receipt/attestation display, preserving the
nickname quick-check as a separate artifact.

## Setup

Added `scripts/nextgqa_formal_review_workspace.py`, its synthetic-only focused
test, and Japanese runbook. Stage-1 uses the existing secure
`audit_qta_stage1_bundle_cli._load_bundle_dir` and `qta_stage1_bundle_audit`
interfaces. Stage-2 uses `qta_phase2_plan_manifest.validate_baseline_pq_manifest`,
`nextgqa_prediction_link_review.validate_prediction_link_receipt`, and
`nextgqa_prediction_link_attestation.verify_attestation`. The exact Stage-1
`pre_review_receipt` bytes hash binds Stage-2. Context pins are now externally
fixed by the required `--expected-stage2-context-sha256`, not trusted from
their own contents.

## Verification

Observed: `py_compile` passed for the controller and focused test;
`scripts/test_nextgqa_formal_review_workspace.py` passed 6/6 synthetic tests;
CLI `--help` succeeded. `git diff --check` succeeded for these paths. An
individually elevated, loopback-only integration check used synthetic Stage-1
and Stage-2 bundles to request the page, verify both stages, reset the session,
and confirm `authorized_forwards=0`; all assertions passed. No protected input,
real human review, video, or GPU job was used. The server binds
only to IPv4 loopback and uses per-launch
token routes, Host/Origin checks, no-store, CSP, referrer/frame/CORP/COOP and
permissions headers. Reset acquires the controller RLock. Stage changes clear
the corresponding stage and downstream state.

## Limits

The tests do not establish real identity, actual viewing, semantic validity,
clock/frame correspondence, authorization, or production readiness. The
loopback HTTP check exercised the handler and API but did not launch a browser;
browser E2E, reset/verification race, and mutation during audit remain untested.
Rights status remains unverified. Protected artifacts remain local.

## Next action

Perform a browser E2E and add reset/verification race coverage before any human
uses real bundles. The workspace is explicitly non-authorizing and does not
launch a scorer.

## Follow-up: primary HTTP verification and audit (2026-10-19)

The primary agent independently ran a localhost HTTP integration using synthetic
bundles after the normal sandbox had rejected socket binding. The run checked
HTML policy headers, state retrieval, Stage-1 and Stage-2 verification, reset,
and persistent non-authorization; all passed. An independent audit found no
remaining critical/high issue after adding the external exact-byte context pin,
locking reset, and response security headers. It identified browser E2E,
reset-race and mid-audit filesystem mutation testing as remaining engineering
coverage. Formal workspace instructions are now linked from the Japanese
launcher. None of this advances direct faithfulness evidence.

## Launch URL usability follow-up

The controller prints the exact tokenized `http://127.0.0.1:<port>/.../` URL before calling the browser opener. Opener exceptions are reported while the manual URL remains available. A synthetic test verifies URL construction and loopback-only host selection; it does not establish browser launch or HTTP E2E.
