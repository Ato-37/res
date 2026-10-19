# Four-frame head-resolution diagnostic (2026-10-19)

## Question and frozen scope

The existing label-free attention-shape scans compare 8 and 16 sampled frames.
Add one 4-frame condition to determine whether the same fixed ranking rule is
especially sensitive at this coarser temporal resolution. This is a post-selection
exploration prompted by the observed 8/16 differences. It does not reopen the
closed head-faithfulness test, select a treatment head, use labels, or measure
faithfulness / grounding / prediction dependence.

## Frozen procedure

- Use all 61 rows, in order, from `data/nextgqa_samples_big.json`; do not alter
  questions, model revision, decoding, or video preprocessing. Use the same local
  Qwen2.5-VL-7B-Instruct revision, 4-bit weights, BF16 compute, eager attention,
  `max_pixels=200704`, and the existing uniform endpoint-inclusive frame sampler.
- Add exactly one 4-frame pass. Require and record observed temporal bins; abort
  if model processor geometry is not divisible or does not yield at least two bins.
- Use the existing fixed heuristic unchanged: mean video-token attention mass,
  chord elbow with `min_keep=10`, sink fraction `<0.5`, then ascending temporal
  entropy, top 20. No tuning after the result.
- Primary description: top-20 Jaccard of the 4-frame ranking against the already
  recorded 8-frame and 16-frame rankings. The old summaries do not retain failed
  row indices, so these comparisons are not guaranteed to use an identical
  successful-row intersection; report that limitation. Also report 4-frame
  effective/failure counts and all 784 head aggregate metrics for audit.
- Save only local owner-only derived output under `data/private_nextgqa/`; never
  publish the video-derived per-head results. No p-values or confidence intervals.

## Decision rule

This is a single additional resolution point. It can describe whether the existing
heuristic's selected set changes under coarse sampling on this fixed cohort. It
cannot establish that any head is causally relevant, faithful, or suitable for
intervention. Do not pick a winner or launch more head tuning from the observed
overlaps; return to the direct-faithfulness/review path afterward.
