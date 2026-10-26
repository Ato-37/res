# QTA v4 selected-media pin repair

## Trigger and repair

An independent primary read found that the v4 scorer attempted to read
`media_sha256` from `build_evaluation_items()` output, although that helper
keeps the media digest in the frozen base-plan row. This raised `KeyError`
before selected-item validation. Refactored the scorer to retrieve exactly one
digest from the frozen plan, validate its format, and bind the actual media
bytes and selected-row digest to that value. Added a regression test where the
derived item intentionally has no `media_sha256` field, plus duplicate/missing
frozen-plan pin cases.

The earlier canonical v4 `_01` census/plan files are retained unchanged, but
their source provenance predates this repair and they must not authorize a
scorer run. The next frozen outputs are explicitly versioned `_02`; the census
CLI rejects `_01` names for a full run. No existing receipt was overwritten.

## Verification

Forty focused CPU tests passed, including media-pin, exact output-version,
output-collision, processor/data smoke binding, and the 588-forward denominator
gates. Python compilation and whitespace checks passed.

The updated one-video CPU geometry smoke used:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_nextgqa_qta_v4_union_geometry_census.py --n 1 --census-output-name nextgqa_tlvs_qta_v4_union_geometry_smoke_local_20261018_09.json --user-authorized-local-only
```

It completed 1/1 video and 5/5 candidate QIDs, with all five geometry-eligible,
in 10.85 seconds. Its owner-only receipt is mode 0600 and its 24 source hashes
match current files. The exact source snapshot is
`data/private_nextgqa/nextgqa_tlvs_qta_v4_source_snapshot_local_20261018_03.tar.gz`,
SHA-256 `d93d8c99eed317553882805101f6f4364283136f1976a792e77aff141f8f0367`.

## Limits and next action

This smoke does not establish 147-video feasibility, annotation validity,
model behavior, or faithfulness. No GPU, weights, or scores were used. The
`_02` full census/plan has not run. Request an independent audit of the media
pin repair, output-version guard, tests, and updated source snapshot; only after
that audit should the all-or-none CPU census be run. Full-cohort CPU geometry
preflight and one-row four-forward GPU smoke remain prerequisites to any full
GPU score.
