# Rendered Stage-1 review export to Phase-1 composition (2026-10-19)

## Purpose

Close the synthetic integration seam between the actual formal claim-review
page export and the existing Stage-1 source-pin/Phase-1 receipt path. Confirm
that the exact exported support decisions and reviewed intervals feed the
ordered three-row source pins used by Phase-1, and that a post-signature review
byte change is rejected.

## Setup

Extended `scripts/test_qta_stage1_phase1_composition.py` using the existing
synthetic Stage-1 audit, reviewer-key, actual-QA-source, and Phase-1 receipt
fixtures. It reuses the Node VM export helper in
`scripts/test_audit_qta_stage1_bundle_cli.py` to execute the real rendered
review and index-download handlers. The resulting exact bytes are signed by
the synthetic rostered reviewer key, passed to `derive_stage1_source_pins`,
then composed with the existing external cohort pin and signed Phase-1
pre-baseline receipt. A negative branch appends one byte to the signed review
JSON and requires source-pin derivation to reject it.

## Verified results

- Command: `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_qta_stage1_phase1_composition -v`.
- Result: **50/50 tests passed** (including the existing fixture test suites
  exercised by this composition module).
- The new integration case passed with **3/3 ordered source rows**. Each
  `reviewed_intervals_sha256` matched the digest of the corresponding exact
  rendered-export intervals; support decisions matched row-for-row.
- Explicit composition assertions independently recomputed SHA-256 of the raw
  rendered review and index downloads and matched the adapter's source-byte
  digests and review-payload digest.
- The exported review/index bytes were adopted by a synthetic reviewer
  attestation. The resulting source pins composed into a valid signed Phase-1
  receipt with matching artifact bytes and source rows.
- Phase-1 remained non-authorizing: `scoring_authorized=false`,
  `model_load_permitted=false`, `authorized_forwards=0`, and
  `execution_authorization=not_granted`.
- Appending one byte to the exported review after signing was rejected by
  `derive_stage1_source_pins` with `stage1_audit_not_complete`.

## Interpretation and limitations

This is a CPU-only synthetic software-composition check. It verifies export
schema/bytes through review-row adaptation, interval-digest derivation,
external cohort pinning, and Phase-1 signature verification without manual
review-schema translation. Node VM execution is not a real-browser download or
human-viewing test. Generated keys, media, reviewer identity, QA records, and
answers are synthetic; this does not authenticate real reviewer identity,
semantic support, source/media custody, or review correctness. Phase-1 remains
explicitly non-authorizing. No protected input, GPU, or model forward was used.

Formal claim review, clock/frame correspondence, Stage-2 prediction-link
judgments, and direct faithfulness scores remain **0/3**, **0/15**, **0/3**,
and **0/3**, respectively. This engineering test does not change those counts
or establish an empirical effect, interval, or significance test. Rights
status remains unverified; protected artifacts must stay local.

## Next action

Keep the composition as synthetic regression coverage. Actual review requires
an eligible independent reviewer and the separately established real trust,
release, custody, and witnessed-order evidence. Do not treat this fixture or
the informal nickname check as formal review evidence.
