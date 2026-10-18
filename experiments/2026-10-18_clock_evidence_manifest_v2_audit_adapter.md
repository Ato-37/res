# Clock evidence manifest v2 audit adapter (2026-10-18)

## Purpose

The frame-evidence builder was extended to bind a root-rostered capture-operator
attestation, but the existing evidence auditor accepted only manifest schema v1.
This work connects the new candidate manifest to the existing byte, screenshot,
and decoder-frame audit path without granting review or scoring authority.

## Implementation

`scripts/audit_nextgqa_clock_frame_evidence.py` now accepts the legacy v1 schema
with its original signature domain and the v2 schema with a distinct domain.
The v2 path requires the operator attestation, crop map/crosswalk, rosters,
trust-root key/pin, and crop directory. It verifies the operator signature using
the exact screenshot snapshot already checked by the auditor and an exact,
private crop-directory snapshot; hashes and identities are rechecked after the
audit. V2 denial fields are checked explicitly. The CLI exposes the additional
inputs, while v1 rejects them and v2 fails closed when any are missing.

The previously skipped builder-to-auditor integration test is enabled. It uses
synthetic video, screenshot/crop bytes, and throwaway Ed25519 keys to run the
entire v2 join and audit path. It also rejects an altered operator signature,
missing operator audit inputs, and schema downgrade. An independent code audit
found no material bypass in the snapshot/root/signature adapter.

The local pre-review inventory's static import closure initially became
incomplete after the auditor began importing the capture-operator verifier.
Registered that module as a dedicated source role, confirmed its local imports
resolve through existing roles, and rebuilt a fresh private hash-only
inventory: **81/81 slots, 0 missing**. It remains `candidate_only`, with review,
joint-freeze, score, and execution flags false. The inventory itself remains
local and unpublished.

## Verification

The focused builder, capture-operator-attestation, and frame-auditor suites pass
**79/79**. The v2 end-to-end test is no longer skipped. `py_compile` and scoped
`git diff --check` pass. The full repository discovery passes **2,013/2,013**
tests in **84.213 seconds**. The candidate inventory builder's focused suite
also passes **69/69** in combination with the evidence-manifest tests. No
protected review HTML, video, answers, or annotations were opened, and no GPU
work was needed.

## Limits and interpretation

This is synthetic protocol integration only. A valid signature proves that the
private key matching the supplied root-rostered capture-operator key adopted
the exact candidate bytes. It does not establish human identity, actual capture
process truth, crop purity, or player-to-decoder frame correspondence. The
frame-reviewer key remains caller-hash-pinned rather than jointly frozen. All
identity, process-authentication, correspondence-review, model-load, scoring,
and execution-authority fields remain false or `not_granted`. Rights status is
unverified; no protected artifacts or derivatives were published.

Direct faithfulness remains **0/3** claim reviews, **0/15** verified
player/browser-to-decoder correspondences, **0/3** prediction-link judgments,
and **0/3** direct scores; no real `ΔF`, confidence interval, or test exists.

## Next action

Run the full repository test suite and inspect the integrated diff. Refresh the
private hash-only candidate inventory after source changes, without opening the
review package. Continue independent human-free protocol validation while the
root pin, role assignments, witnessed handoff, and human judgments are pending.
