# QTA v4 receipt hardening checkpoint

## Changes

The four-forward GPU smoke validator now rejects bools in integer count fields,
requires the exact four branch names and exact integer branch counters, checks
that the sole record is an object bound to the frozen first plan row, and binds
the smoke receipt's census digest to the full CPU preflight. The checkpoint byte
audit remains an exact equality gate against a fresh CPU audit before a full run
can accept the smoke receipt or load model weights.

The prior-exposure receipt validator now binds the exact split-plan byte hash,
ordered row-binding digests, and ordered media hashes. Denominator fields require
actual integers, so JSON booleans cannot satisfy zero or nonzero counts.

## Verification

Focused CPU tests passed: 13 tests across the v4 scorer gates and prior-exposure
audit, including bool substitutions, missing/mismatched census hash, branch
counter tampering, checkpoint audit mismatch, plan-hash mismatch, and ordered
row-binding mismatch. Python compilation passed for the modified runtime files.

A fresh one-video CPU geometry smoke completed in 10.41 seconds: one video,
five candidate QIDs attempted, all five eligible, zero failures. This is only a
smoke check, not a 147-video feasibility census. Its private receipt is
`data/private_nextgqa/nextgqa_tlvs_qta_v4_union_geometry_smoke_local_20261018_10.json`
(mode 0600, SHA-256 `407f4f08f7d86c66e4831aec5b57242b409585a5312724e5fdb6b5ceb9d6f56a`).
All 24 runtime source hashes in the receipt match the current source files. The
executed source archive is
`data/private_nextgqa/nextgqa_tlvs_qta_v4_source_snapshot_local_20261018_04.tar.gz`
(mode 0600, SHA-256 `cd2b20b71c99d28ce65a4c375568d3a999ebe8c357bc800bec8f1372c245fbd8`).
The matching v2 input freeze is local-only `_17`.

## Limits and next action

No GPU, model weights, forward pass, or score was used. This does not establish
full-cohort feasibility, model behavior, annotation validity, or faithfulness.
Keep all receipts and the source archive local. Obtain independent review of
the validator changes and smoke provenance before any full CPU census. Do not
start GPU scoring until all upstream cohort/preflight/smoke gates are separately
reviewed.
