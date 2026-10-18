# QTA phase-1 cohort record binding — 2026-10-19

## Purpose

Close a pre-baseline identity-binding gap in the signed phase-1 freeze receipt.
The receipt previously required three unique opaque SHA-256 record tokens, but
did not require those tokens to be the ordered hashes of the review tokens in
the exact Stage-1 artifact already covered by the external cohort pin. A
reordered or substituted set could therefore be signed and only be rejected by
the later phase-2 manifest validation.

## Change and setup

`scripts/qta_phase_freeze_receipts.py` now derives the ordered expected record
tokens from the validated `expected_cohort_tokens` in the pinned
`stage1_review_and_lineage` bytes and checks exact equality during both
pre-baseline receipt creation and verification. Phase-2 receipt behavior is
unchanged. The implementation uses synthetic fixtures only; no protected review
material, model, or GPU was used.

`scripts/test_qta_phase_freeze_receipts.py` now covers a reordered list rejected
at creation and a substituted list rejected by verification even when the
synthetic freeze-owner signature is recomputed. Existing unrelated local edits
in that test file were preserved.

## Verified results

The focused phase-freeze suite passed **40/40** tests. This includes synthetic
creation, signature, tamper, authorization-denial, phase-2 linkage, and the new
pre-baseline cohort-order cases. `py_compile` and `git diff --check` passed for
the scoped files. The tests do not exercise a protected cohort or authorize any
model forward.

## Interpretation and limitations

This closes one structural mismatch window before any future baseline run; it
does not establish reviewer identity, claim validity, player-clock/decoder-frame
correspondence, baseline `p/q`, or Stage-2 prediction linkage. The freeze helper
still does not prove that each `processor_batch_sha256` comes from a parsed
per-row full-video-batch artifact. The signed roster, trusted-root fingerprint,
review-only handoff, and human reviews remain absent. Rights status remains
unverified; local-only data handling continues.

Direct faithfulness remains **0/3** claim reviews, **0/15** clock/frame
correspondences, **0/3** prediction-link judgments, and **0/3** direct scores;
there is no direct `ΔF`, confidence interval, or hypothesis test. This code
change is protocol engineering, not faithfulness evidence. No GPU run was
justified because the independent human and provenance prerequisites remain
closed. The closed proxy/head cohorts were not repeated.

## Next action

Audit the remaining pre-baseline processor-batch provenance binding and the
actual claim-mapped scorer call path. Any new verifier must remain fail-closed
and non-authorizing until the required independent human evidence exists. Do not
run a real-data model forward or count the informal nickname-first impression
artifact as formal evidence.
