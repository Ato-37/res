# Six-frame head-resolution diagnostic (2026-10-19)

## Question and frozen scope

The four-frame run had two temporal bins, for which the pre-existing strict
sink rule (`sink_fraction < 0.5`) necessarily excludes every head. Six sampled
frames are the smallest tested input that should yield three bins, making that
same fixed rule nontrivial. This one-pass, post-selection exploration asks
whether the fixed attention-shape ranking produces any eligible heads at that
minimum resolution. It does not reopen the closed faithfulness test or select
a head for intervention.

## Frozen procedure

- Use the same ordered 61 rows from `data/nextgqa_samples_big.json`, local pinned
  Qwen2.5-VL-7B-Instruct revision, 4-bit weights, BF16 compute, eager attention,
  and `max_pixels=200704`.
- Run exactly one 6-frame pass using the existing uniform endpoint-inclusive
  sampler. Require the processor to report three temporal bins; abort if its
  geometry differs.
- Keep the ranking rule unchanged: mean final-query video mass, chord elbow
  (`min_keep=10`), sink fraction `<0.5`, then ascending temporal entropy, top
  20. Do not lower/raise the sink threshold, remove the sink filter, or choose
  a “best” resolution after observing outputs.
- Report effective/failure counts, 784-head inventory, eligible survivor count,
  and top-20 overlaps against existing 8/16-frame rankings only if 6-frame
  survivors exist. The old 8/16-frame artifacts omit failed-row indices, so
  those overlaps are descriptive and not guaranteed to compare identical
  success intersections.
- Smoke on three rows first; then run all 61 once. Store output only in the
  local owner-only private directory. No video-derived per-head metrics are
  published.

## Interpretation limit

This can test whether the fixed filter is mathematically usable at three bins
on this fixed cohort. It cannot establish faithfulness, grounding, prediction
dependence, causal relevance, or an intervention candidate. Treat the result as
the final coarse-resolution diagnostic; no further threshold/resolution search
will follow from its observed head ranking.
