# 2026-10-19 QTA same-batch phase-1 transaction preflight

## Purpose

Close a composition gap before the planned three-row untreated baseline: verify
the Stage-1 source/review bytes, signed phase-1 receipt, and processor batches
in one CPU-only call, while ensuring the successful result cannot authorize a
model load, forward pass, or faithfulness score.

## Setup

- Implementation: `scripts/qta_prebaseline_transaction_preflight.py`
- Tests: `scripts/test_qta_prebaseline_transaction_preflight.py`
- Inputs in tests are synthetic fixtures from the existing Stage-1 pin and
  phase-freeze receipt tests; no protected video or real participant response
  was used.
- The preflight snapshots each input mapping, requires the diagnostic source
  map to equal the QA map, binds review-role bytes to the Stage-1 snapshot,
  derives source pins from those exact bytes, and requires the signed Stage-1
  artifact to equal the canonical derived pins. It then validates the signed
  receipt and passes the same tuple of the same `FrozenBatch` objects to the
  source/review diagnostic, rehashing their tensors afterward.
- Independent audit: `adapter_audit` reviewed the latest implementation and
  found no remaining cross-cohort composition path. It verified the source,
  review, receipt, row-order, and batch-reference bindings. An optional
  defense-in-depth note about single-read row metadata was applied.

## Verified results

- Targeted CPU test invocation completed **82 tests, all passed** in 38.464 s:

  ```bash
  PYTHONPATH=scripts .venv/bin/python -m unittest \
    test_qta_prebaseline_transaction_preflight \
    test_qta_actual_prebaseline_diagnostic \
    test_qta_stage1_source_pin_adapter -v
  ```

- `py_compile` passed for both new Python files.
- `git diff --check` passed.
- Implementation SHA-256: `90e23c08de35acfd56b2529c49260218df1c17bd11bdc977e9d48c3dfa82b6f0`.
- Test SHA-256: `dfd25f7eaee8cca98d8982647ba078f532804324201a1b8ccdbebf2aff92c8f9`.
- Successful synthetic preflight reports three structural bindings only;
  trust-root origin, identity, semantic review, and clock/frame correspondence
  remain unauthenticated. `model_load_permitted=false`,
  `scoring_authorized=false`, and `authorized_forwards=0`.

## Interpretation and limits

This removes a code-composition gap in the prebaseline structural checks. It is
not a real-data result, participant audit, trust-root verification, semantic
claim judgment, temporal alignment result, model inference, or faithfulness
estimate. No direct faithfulness evidence was added. The informal three-item
browser check remains a broad impression exercise; its currently available
responses are still unusable for the current item mapping, so valid response
count remains zero.

## Next action

Integrate this preflight into a runnable transaction wrapper and validate the
input handoff end-to-end. Keep it non-authorizing until the external root and
formal review gates are independently satisfied. Then perform the prespecified
three untreated baseline forwards, freeze p/q, and request the Stage-2
prediction-link review before any treatment inference. Do not reopen the closed
head-search branch.
