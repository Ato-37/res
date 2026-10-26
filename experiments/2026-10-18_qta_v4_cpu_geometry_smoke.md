# QTA v4 one-video CPU geometry smoke

## Purpose

Verify the source-bound v4 annotation-union geometry path on one deterministic
evaluation video before requesting an independent review of the full CPU census
and scorer. This is an implementation smoke, not a model result.

## Setup

- Protocol: `nextgqa_test_associated_qta_annotation_union_v4`; frozen design is
  documented in `2026-10-18_qta_v4_union_treatment.md`.
- Command:

  ```bash
  PYTHONPATH=scripts .venv/bin/python scripts/run_nextgqa_qta_v4_union_geometry_census.py --n 1 --census-output-name nextgqa_tlvs_qta_v4_union_geometry_smoke_local_20261018_08.json --user-authorized-local-only
  ```

- Output: owner-only local receipt
  `data/private_nextgqa/nextgqa_tlvs_qta_v4_union_geometry_smoke_local_20261018_08.json`.
- Executed source snapshot: owner-only local archive
  `data/private_nextgqa/nextgqa_tlvs_qta_v4_source_snapshot_local_20261018_02.tar.gz`,
  SHA-256 `f65e21d2ac39a4a65ff6b59516ceb4d8b63de9d344f9c0646ff673a8a2de41e7`.

## Verified result

The receipt reports `smoke_only_not_census`, 1 requested/completed video, five
candidate QIDs attempted, five geometry-eligible QIDs, and 10.41 seconds. The
receipt is mode 0600 and all 24 pinned source hashes match the current files.
The executed census source SHA-256 is
`e1c001b24aee5e04fff0981d994fafdcae34657c678734c1513e57cce1b874db`; scorer
source SHA-256 is
`dff757b45ec314aa9a6f45e4b71c00fdb2e6ee172852e01211e51db2c7c94401`.

Focused validation passed: 38 unit tests, Python compilation, import-closure
check (no unbound direct local imports), and `git diff --check`.

## Interpretation and limits

Only the one-video geometry path ran. No full 147-video census or plan was
produced; no model weights, CUDA, forward pass, or score were used. The result
does not establish eligibility across the cohort, annotation validity, model
behavior, faithfulness, freshness, or confirmatory evidence. Rights remain
unverified and all data/receipts remain local.

## Next action

Obtain independent source/protocol audit. If it clears, run the all-or-none
147-video CPU census and emit the plan only if every frozen evaluation video
has at least one eligible QID. Do not run GPU before that gate and the separate
full-cohort CPU preflight plus one-row four-forward GPU smoke.
