# Head query-position × temporal-resolution exploration (2026-10-19)

## Purpose and status

This post-selection follow-up tests whether the attention-shape ranking's
query-position sensitivity persists at two temporal resolutions. It reuses the
already collected 16-frame arrays and adds an 8-frame GPU scan. It does not
reopen the closed confirmatory head-faithfulness branch and does not measure
faithfulness.

## Frozen protocol and execution

- Protocol: `2026-10-19_head_query_by_temporal_resolution_protocol.md`, SHA-256
  `e9b6c70c22e7163081ee4c62769a1db168aa60ac04cc0d9d3d199daa35836fb9`.
- Antecedent 16-frame full result, fixed by protocol: summary SHA-256
  `9efd7d6b57bdbea210093b4c4c590b1f31ec0fe208056f62507f8be1b3dc41c2`,
  metrics SHA-256
  `11735b2fb92b8c0aa8db7c72bffd74e1cccf17f94c7492f14b3a23dc33d71d82`.
- 8-frame model, precision, frame sampling, and manifest matched the 16-frame
  run: Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, local-only, 4-bit weights,
  BF16 compute, eager attention, `max_pixels=200704`, all 61 manifest rows.
- The CPU preflight allowed only the same fixed frame-decode errors. GPU
  inference was fail-fast. The final 8-frame run completed 58/61 rows with
  three frame-decode failures, 52.812 seconds, and 8.527 GiB peak allocated
  VRAM. All 58 successful row indices matched the 16-frame artifact exactly.
- A 12-row engineering smoke passed (12/12; 12.723 seconds; 8.515 GiB peak).
  It was excluded from analysis.
- The 8-frame arrays were run twice because the first summary recorded the
  previous query-position protocol hash due a hard-coded protocol path. The
  runner now takes an explicit `--protocol-file`; the second full-run summary
  records the correct protocol file and hash. The metric NPZ hashes from both
  8-frame runs are identical, confirming the measured arrays reproduced.
  The first run's summary is retained locally for provenance but is superseded
  and is not used by the final comparison.

## Results

The CPU combiner verified the 16-frame input hashes, 8-frame protocol file/hash,
model revision and data manifest, then re-ranked all four query × resolution
conditions on the exact 58-row intersection. With the pre-fixed chord elbow,
sink threshold, entropy ranking, and top 20:

| Contrast | Shared top-20 heads / union | Jaccard |
|---|---:|---:|
| Query-position change at 8 frames | 0 / 36 | 0.0000 |
| Query-position change at 16 frames | 1 / 39 | 0.0256 |
| 8 vs 16 frames, generation-prefix query | 9 / 27 | 0.3333 |
| 8 vs 16 frames, question-body mean query | 11 / 29 | 0.3793 |

The descriptive difference between the 16-frame and 8-frame query-position
Jaccards is 0.0256. This is not an inferential interaction estimate. It shows
that, on the fixed cohort, top-20 membership changed nearly completely between
the two query definitions at both resolutions; temporal resolution also
changed membership, with moderate overlap. It does not establish which ranking
is better or whether any head uses evidence faithfully.

## Verification and provenance

Focused tests passed 16/16, Python compilation passed, and the CPU combiner
requires the expected 16-frame SHA-256s and the exact 8-frame protocol hash.
An independent audit reproduced all four survivor counts, ordered top-20 sets,
contrasts, the 58-row matched cohort, hashes, and private file modes.

Owner-only local artifacts:

- 8-frame full summary: `head_query_8frame_full_20261019_02.json`, SHA-256
  `691a07b00660fd745a01c6f9fa71f35b7d762eb15f92c1abccbe44f670e47002`.
- 8-frame per-video metrics: `head_query_8frame_full_20261019_02.npz`, SHA-256
  `152ee17eeb2762297e2612d211edd02be6cd05fe4fb2be830159b88cee77103f`.
- Four-condition comparison:
  `head_query_temporal_resolution_comparison_20261019_03.json`, SHA-256
  `a45093d042aadf4a381c1c8511b15280a4ef385ccff50817932501e29072c6ac`.
  Analyzer source SHA-256:
  `1b2d0bbf06da012eb1e898454bb3731714dc24867f903d8e39335f2a754baee5`.

All are mode `0600` in the mode-`0700` private directory. Raw videos and
per-video/per-head metrics remain local. Rights status is unverified; nothing
was published.

## Interpretation and next action

These results show that this attention-shape selection heuristic is highly
dependent on query-position definition and somewhat dependent on temporal
resolution. They do not justify selecting a head for causal intervention.
Treat heuristic head selection as too setting-sensitive for further compute
unless a separate, outcome-independent rationale is established. Prioritize
the independent intervention-based faithfulness path and completion of human
review; direct-faithfulness evidence remains 0/3 claim reviews, 0/15
frame-clock correspondences, 0/3 prediction-link reviews, and 0/3 direct
scores.
