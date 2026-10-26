# QTA v4 full-cohort CPU geometry preflight (2026-10-18)

> Historical run `_02`. The later full audited run is `_03`, documented in
> [the `_03` preflight log](2026-10-18_qta_v4_cpu_geometry_preflight_03.md).

## Purpose and setup

Recompute processor/actual-PTS geometry for the exact 147 rows selected by the
source-bound v4 union-geometry census, before any model load or score. The
command was:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_nextgqa_qta_v4_union_scorer.py \
  --mode preflight \
  --output-name nextgqa_tlvs_qta_v4_cpu_geometry_preflight_local_20261018_02.json \
  --user-authorized-local-only
```

This used the existing `.venv`, local media, and Qwen processor only. It did
not load model weights, call CUDA, or compute a score.

## Verified result

The runner completed **147/147** planned rows and wrote a non-authorizing
owner-only receipt under `data/private_nextgqa/` (mode 0600; parent mode 0700).
Primary-agent verification independently reloaded the frozen plan and
processor, matched all ordered video/QID/geometry/row-binding records, and
recomputed source, data, census, plan, processor, and library provenance. All
147 rows matched the frozen plan. No model score or forward-count fields are
present in the receipt. The focused v4 CPU tests passed **13/13**;
`py_compile` and `git diff --check` passed.

The receipt does not store elapsed time, so no exact runtime is reported here.

## Interpretation and limits

This passes the geometry-only preflight for the exploratory annotation-union
proxy. It does not validate reviewed explanation evidence, prior exposure
completeness, source-event independence, treatment efficacy, or faithfulness.
The cohort remains test-associated and processor-geometry-conditioned. Rights
remain unverified; the receipt and source media remain local.

## Next action

After independent audit of this receipt and the scorer's complete gates, run
the bounded one-row four-forward GPU engineering smoke. A passing smoke must
bind the same plan, data, source, processor, and first-row geometry before the
147-row / 588-forward exploratory score can be considered.
