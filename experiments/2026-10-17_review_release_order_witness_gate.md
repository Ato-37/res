# Review-only witnessed release-order gate

## Purpose

Close the procedural release-order gap between an integrity-checked candidate
freeze and independent human review, without authorizing model loading or
scoring. The earlier integrity verifier remains unchanged and non-authorizing.

## Implementation

Added `scripts/audit_nextgqa_review_release_order.py`. It re-verifies the
externally pinned Ed25519 root, original integrity roster and freeze-owner
signature, current inventory bytes, and every inventory-bound artifact. A
second root-signed roster explicitly binds the freeze owner, custodian, claim
reviewer, frame reviewer, and release witness. The custodian may share the
freeze owner's key; the witness must be distinct from all other roles. The
detached witness signature binds the exact receipt, both rosters, inventory,
artifact bindings, and a caller-pinned ceremony ID, with fixed statements about
the package remaining unopened through the witnessed handoff.

The verifier returns only a procedural review-opening permission after these
signatures and byte bindings validate. It does not validate real-world human
identity or prove actual chronology, prevent out-of-band file access, track
ceremony-ID uniqueness/replay, permit model loading, or authorize scoring.
Local timestamps are not used as order evidence. No real key, package, or
attestation was created or inspected.

## Verification

The new synthetic suite passed **12/12** and the existing integrity-receipt
suite passed **17/17**. Primary-agent rerun reproduced **29/29**; compilation
passed. Independent audit found the role separation, root and byte bindings,
private-file controls, and non-score-authorizing output fail closed. It found
and prompted wording changes so success is named
`review_release_order_witness_attestation_valid` and reports that replay
uniqueness is not tracked. The expected negative CLI permission test prints a
sanitized blocked status and exits 2 without writing output.

Source SHA-256:

- Verifier: `e6d5b5361ca16dcf1fa42f7f5c81b409b43c261df0b2305fe3a48bb86bc3bc0b`
- Tests: `c4bd5045c78b0a259e041f0aa052324ee03fdd91eeca7f843edff38ed0ae8fd0`

## Faithfulness progress and limits

This is workflow/engineering progress only. Direct generated-claim
faithfulness remains **0/3 human-reviewed and 0/3 scored**, with no direct
effect, interval, or statistical test. There is still no real trusted roster,
freeze, witness attestation, review export, screenshot/frame-pair review, or
clock calibration. A valid signature verifies the witness's statement, not
the truth of the ceremony. A fresh ceremony ID is required operationally, but
the verifier has no consumed-ID registry. Do not treat its output as a
technical access-control mechanism or as permission to score. Rights status
remains unverified; all real review artifacts remain local.

## Next action

An out-of-band trusted authority must establish the public-key root and actual
role identities. Then a distinct witness must observe the freeze and owner to
custodian handoff before first opening; run the verifier against the exact
current bytes immediately before opening. Complete the independent claim and
frame-pair reviews and real clock calibration. Only after a separate score
freeze and production all-three-row atomic preflight may the 9-forward real
video run proceed. No head search or repeated gold-interval GPU pilot is
justified while these direct-evidence steps remain available.
