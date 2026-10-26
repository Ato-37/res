# Head query-position × temporal-resolution exploration protocol (2026-10-19)

## Question and status

The immediately preceding 16-frame exploratory scan found almost disjoint
top-20 head sets for the final generation-prefix query and mean question-body
query. The earlier 8/16-frame scan at the generation-prefix query also showed
limited top-20 overlap. This follow-up asks whether query-position sensitivity
persists at eight sampled frames. It reuses the already collected 16-frame
arrays and adds only one 8-frame GPU pass. It is post-selection exploratory,
not a new confirmatory search or faithfulness test.

## Frozen design

- Use all 61 rows from `data/nextgqa_samples_big.json` in manifest order, same
  pinned Qwen2.5-VL-7B-Instruct revision, 4-bit weights, BF16 compute, eager
  attention, and `max_pixels=200704`.
- Run exactly one additional 8-frame pass, yielding four strata when paired
  with the existing 16-frame artifacts: two query definitions (final input
  query; equal mean over exact question-body subtokens) × two resolutions
  (4 or 8 temporal bins).
- Use the already fixed per-query attention summary, chord elbow (`min_keep=10`),
  sink threshold `<0.5`, entropy ordering, and top 20. Do not tune any setting
  after this run. Do not choose a “best” stratum or head for confirmatory use.
- Report each resolution/query top-20 Jaccard, and descriptive 2×2 contrasts:
  query-position Jaccard at each resolution, within-query resolution Jaccard,
  and the difference between the two query-position Jaccards. These are
  descriptive only; no p-values, confidence intervals, or causal claims.
- Before comparing resolutions, intersect `sample_index` across the full 8- and
  16-frame metric artifacts and recompute *all four* rankings on exactly this
  shared video set. Do not compare rankings computed on different successful
  cohorts. The 16-frame antecedent is fixed by full-summary SHA-256
  `9efd7d6b57bdbea210093b4c4c590b1f31ec0fe208056f62507f8be1b3dc41c2` and
  full-metrics SHA-256
  `11735b2fb92b8c0aa8db7c72bffd74e1cccf17f94c7492f14b3a23dc33d71d82`.
- CPU-preflight all rows before model loading. Exclude only the fixed frame
  decode failures. Any other preflight or GPU inference error aborts before
  artifact writing. Keep all outputs owner-only under `data/private_nextgqa/`.

## Limits and next action

The four summaries use the same fixed, previously inspected cohort, so this is
exploratory and post-selection. Attention query rows correspond to different
causal prefix lengths. A stable ranking across these settings would show only
heuristic robustness, not temporal grounding or faithfulness. A difference
does not show that any head is causally relevant to the model answer. Do not
resume the closed head-faithfulness search; use this measurement to decide
whether heuristic head selection is too setting-sensitive to justify further
head compute.
