# QTA actual candidate input bridge (2026-10-19)

## Purpose

Add a CPU-only composition layer that derives the three ordered processor-row
candidate pins from the existing actual-media byte preflight, Stage-1 source pin
adapter, review-row adapter, and per-token clock-calibration adapter. The bridge
must fail when any required source, review, or clock-calibration artifact is
missing or does not join, while keeping the output explicitly non-authorizing.

## Implementation and setup

Added `scripts/qta_actual_candidate_input_bridge.py` and its focused tests. The
bridge reruns the upstream source preflight and Stage-1 audit adapters, compares
the shared CSV/lineage/protocol/config bytes, independently binds private QA and
claim rows by opaque review token, verifies the reviewed interval digest, joins
sampler rows to Stage-1 source hashes, and requires one clock-calibration/capture
pair plus processor/tokenizer/duration pins per token. It returns the expected
row pins and sampler rows required by the existing processor-row candidate
builder. QA and claim text are processed in memory and never included in the
bridge return value or this log.

The bridge does not establish reviewer identity or semantic correctness and
does not authenticate browser-clock-to-decoder-frame correspondence. Existing
Stage-1 and clock adapters explicitly return those checks as false; the bridge
preserves them as false and will not promote calibration presence into a clock
correspondence pass. Candidate construction here cannot authorize inference.
Rights status also remains unverified.

## Verified checks

- Focused command: `../.venv/bin/python -m unittest -v test_qta_actual_candidate_input_bridge`
  from `scripts/` — 16 tests passed (9 bridge tests plus 7 inherited
  Stage-1-adapter fixture tests).
- Coverage includes ordered 3-row candidate pins, source byte mismatch, token
  cohort mismatch, missing clock rows, unsupported review decision, clock
  adapter falsely claiming verified correspondence, interval splice rejection,
  and synthetic claim-text/token splices rejected by the private QA/claim
  cross-binding helper. The latter test runs the real Stage-1 source pin adapter
  and actual QA binder on the repository's synthetic fixture. A separate
  negative mutates only the Stage-1 question digest, observes the real QA binder
  return `bound_unvalidated`, then confirms the bridge rejects the mismatch.
- End-to-end CPU composition passes the bridge's `expected_rows` and
  `sampler_rows` directly into `build_actual_processor_row_candidate`, then
  validates with `verify_actual_processor_row_candidate`. It used three
  synthetic PIL RGB clips and a fake tokenizer/processor (three processor
  preparation calls; zero model forwards) and preserved all denial flags.
- No protected local QA/media files were opened. Tests use synthetic fixtures;
  composition-control tests stub the upstream validators, while the private
  QA/claim cross-bind test uses the existing synthetic Stage-1/QA fixture and
  invokes the real binder.
- `py_compile` passed for the bridge and focused test module; `git diff --check`
  passed for the three owned files.
- No GPU, model, model forward, inference, or faithfulness scoring was run.

## Interpretation and limits

This closes an input-composition gap: expected source hashes and sampler rows
can now be derived together from local audited artifacts rather than assembled
from unrelated fixtures. It does not verify that a reviewer truly watched or
semantically judged the video, that a clock observer identity is trusted, or
that browser timestamps map to decoder frames. It is not a pre-load gate,
scorer-ready artifact, model execution record, or faithfulness result.

Current faithfulness result count from this work: **0 scored examples, 0 model
forwards, 0 effect estimates**. Prior head confirmation remains below its
preregistered practical threshold (79/79, −0.0756 nat, 95% t CI
[−0.1578, +0.0065]); this bridge does not alter that result or reopen head,
layer, or alpha search.

## Next action

Have the primary independently inspect the bridge and its cross-binding logic.
Before any real-data processor construction or inference, complete the separate
human review, trusted reviewer/key, clock-frame correspondence, custody/release,
processor-batch, and baseline protocol gates. Any downstream caller must check
its own validated authorization gate; this candidate bridge grants none.
