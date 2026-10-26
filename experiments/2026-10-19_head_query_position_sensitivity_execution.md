# Head query-position sensitivity exploration (2026-10-19)

## Purpose and status

Executed the frozen post-selection exploratory protocol in
`2026-10-19_head_query_position_sensitivity_protocol.md`. It tests whether a
head ranking based on attention shape depends on whether attention is read at
the final generation-prefix query or averaged across the question-body
subtokens. This is not a test of head faithfulness, grounding, prediction
dependence, or explanation quality. The closed preregistered head-faithfulness
result remains negative and was not reopened.

## Setup

- Manifest: `data/nextgqa_samples_big.json`, 61 rows, manifest order.
- Model: Qwen2.5-VL-7B-Instruct, pinned revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; local-only; 4-bit weights,
  BF16 compute, eager attention; 16 sampled frames and eight temporal bins;
  `max_pixels=200704`.
- Hardware/software observed in the private run summary: RTX 5070 Ti, CUDA
  12.8, PyTorch 2.11.0+cu128, Transformers 5.18.0, tokenizers 0.23.2,
  qwen-vl-utils 0.0.14, OpenCV 5.0.0, Pillow 12.3.0, NumPy 2.5.2.
- CPU processor/span preflight preceded model loading. Three rows failed only
  with the fixed frame-decode errors; the remaining 58 rows ran on GPU. Any
  inference exception would have aborted before artifact writing.
- The ranking rule was fixed before this run: mean video mass, chord elbow
  with `min_keep=10`, sink fraction `<0.5`, then ascending temporal entropy;
  top 20. Primary descriptive comparisons: top-20 Jaccard and average-tie
  Spearman entropy ranks among shared non-sink survivors. No inferential test
  or interval was planned.
- A 12-row engineering smoke passed 12/12 with no failures (41.13 s,
  15.229 GiB peak allocated VRAM). It is not included in the full-run result.

## Verified results

Full GPU execution requested all 61 rows and completed 58; 3/61 were excluded
by the predeclared frame-decode preflight. There were 61 unique media items.
Elapsed inference time was 173.409 s and peak allocated VRAM was 15.252 GiB.

| Query summary | Chord-elbow heads | After sink filter |
|---|---:|---:|
| Final generation-prefix query | 105 | 28 |
| Mean over question-body subtoken queries | 155 | 76 |

The two top-20 lists shared 1 head (union 39; Jaccard 0.02564). There were 11
heads surviving both filters; average-tie Spearman correlation of their mean
temporal-entropy rankings was 0.98182. Independent NumPy recalculation from
the private per-video metric arrays reproduced both estimates and the
58/3 coverage counts exactly. The top-20 overlap is only descriptive and
post-selection; its very small value indicates strong sensitivity of the
selected top-20 set to query-position definition on this fixed cohort.

## Provenance and privacy

Private artifacts are owner-only (`0600`) under the owner-only
`data/private_nextgqa/` directory:

- `head_query_position_full_20261019_01.json`, SHA-256
  `9efd7d6b57bdbea210093b4c4c590b1f31ec0fe208056f62507f8be1b3dc41c2`
- `head_query_position_full_20261019_01.npz`, SHA-256
  `11735b2fb92b8c0aa8db7c72bffd74e1cccf17f94c7492f14b3a23dc33d71d82`

The JSON records source-manifest, protocol, runner, helper, base-runner and
artifact hashes. Raw videos and derived per-head/video arrays remain local and
are not published. Rights status is unverified.

Focused CPU tests passed 12/12, Python compilation passed, and the independent
audit verified the fail-fast path and the corrected post-stack validation.
One initial smoke attempt exposed a shape-validator call placed at the wrong
dimension; it aborted before writing results. The call was removed, tests were
rerun, and the subsequent 12-row smoke and full run passed.

## Interpretation and limitations

The attention-shape heuristic's top-20 is not robust to these two query
definitions on this cohort. The high entropy-rank correlation among the small
intersection (11 heads) does not repair the low set overlap and does not imply
faithful evidence use. Question-body query rows occur at different causal
prefix lengths, so their mean is not equivalent to one fully-read-question
state. The 3 frame-decode exclusions limit the result to 58/61; the cohort is
small, fixed, and previously inspected. No p-value, confidence interval, or
confirmatory claim is justified.

## Next action

Do not select a confirmatory head from these rankings or resume head/layer/
alpha tuning. Any further head exploration needs a separately frozen,
scientifically motivated question and must remain explicitly exploratory.
Prioritize the independent internal-intervention faithfulness path and human
review completion; current direct-faithfulness review counts remain 0/3 claim,
0/15 frame-clock correspondences, 0/3 prediction-link, and 0/3 direct scores.
