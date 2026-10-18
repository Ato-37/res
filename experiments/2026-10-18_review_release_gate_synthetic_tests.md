# Human-review release-gate synthetic test checkpoint (2026-10-18)

## Purpose and setup

Verify the local review-opening controls without opening the protected review
package or pretending that human attestations exist. CPU-only unit tests use
synthetic keys, rosters, receipts, and artifact bytes. No private package,
video, signed human receipt, model, CUDA, or dataset content was read.

## Result

Command:

    PYTHONPATH=scripts .venv/bin/python -m unittest test_nextgqa_review_ceremony_signing test_audit_nextgqa_review_release_order test_render_nextgqa_claim_review_html test_nextgqa_prediction_link_attestation -v

Result: **35/35 tests passed**. The tests cover exact-artifact binding,
role/key separation, owner-custodian alias constraints, independent witness,
tamper/replay rejection, private output modes, and the rule that passing the
release audit permits only procedural package opening and never model scoring.
The renderer and detached review attestation also fail closed on incomplete or
tampered synthetic inputs.

This validates software behavior on synthetic fixtures only. It does not
establish any real person's identity, any independently trusted root pin,
actual review completion, an observed unopened handoff, or frame-clock
correspondence.

## Current human-review status

Direct review remains **0/3 claim judgments**, **0/15 verified browser-clock to
decoder-frame pairs**, **0/3 prediction-link judgments**, and **0/3 direct
scores**. The local role-assignment sheet remains an unfilled intake template.
No root fingerprint or signed release materials have been verified. Therefore
no reviewer should open the package, and claim-linked scoring remains blocked.
Rights status remains unverified; protected data stays local.

## Next action

Continue independent CPU-only contract work while the human authority and
participants establish the root fingerprint and role-key assignments through
their separate trusted channel. Once those inputs exist, rebuild a unique
current inventory and perform the witnessed unopened handoff and immediate
release-order audit before any reviewer opens the package.
