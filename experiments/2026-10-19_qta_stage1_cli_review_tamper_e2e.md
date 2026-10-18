# Stage-1 review tamper through bundle CLI (2026-10-19)

## Purpose

Verify that a review export changed after signing is rejected by the complete
private Stage-1 bundle CLI path, rather than only by an isolated receipt
validator. Keep all fixtures synthetic and preserve the explicit non-authorizing
contract.

## Setup

The existing synthetic signed Stage-1 fixture is written to an owner-only bundle
using the same manifest/source loader used by the CLI. The new negative case
appends one byte to the review export, points the bundle manifest to those
changed bytes, and invokes `audit_qta_stage1_bundle_cli.main` with the fixture's
independent expected pins. No real review material, protected media, model, or
GPU is used. The associated phase-1 composition fixture now derives its record
IDs from SHA-256 of the actual adapter-provided opaque review tokens, matching
the production identity rule.

## Verified results

Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_audit_qta_stage1_bundle_cli \
  test_qta_stage1_bundle_audit \
  test_qta_stage1_phase1_composition -q
```

Result: **74/74 tests passed**. The unmodified signed bundle remains a
structural-only pass with model loading, score readiness, and execution denied.
The changed review bytes produce CLI exit status 2 and a sanitized `blocked`
report with `stage1_audit_contract_invalid`; no opaque review token is emitted.
The phase-1 composition fixture passes after its records are bound to the
adapter's ordered Stage-1 token cohort.

## Interpretation and limits

This verifies that the bundle loader and CLI do not accept review bytes that
changed after the signed fixture was assembled. Synthetic signatures prove only
the code path with generated test keys. They do not establish a real trust root,
reviewer identity, semantic judgment, actual human viewing, clock/frame
correspondence, or production score authorization. The result is review-handoff
integrity coverage, not direct faithfulness evidence.

Direct progress remains 0/3 formal claim reviews, 0/15 player-clock/decoder
correspondences, 0/3 prediction-link judgments, and 0/3 direct scores. No direct
`DeltaF`, interval, or inferential test exists. Existing human-free GPU proxy
cohorts remain closed; no GPU run was made.

## Next action

Continue integrating and auditing the non-authorizing review-to-phase receipts
using synthetic inputs. The immediate operational path remains the opened
nickname-first impression artifact; a response, if saved, must stay separate
from formal review and must not be used to satisfy Stage-1 gates. Direct scoring
still requires eligible independent human review, timing correspondence,
trusted external cohort/root pins, prediction-link review, and a production
source-bound runner.
