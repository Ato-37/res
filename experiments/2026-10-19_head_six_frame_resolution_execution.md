# Six-frame head-resolution diagnostic execution (2026-10-19)

## Purpose and setup

Executed the frozen diagnostic in `2026-10-19_head_six_frame_resolution_protocol.md`.
The 4-frame run had shown that the fixed `<0.5` sink rule cannot admit any
head with two temporal bins. Six frames are the minimum attempted resolution
with three bins, where the rule can retain a head. This remains post-selection,
label-free attention-shape exploration; it is not a faithfulness test and does
not reopen the closed confirmation.

- Manifest: `data/nextgqa_samples_big.json`, 61 ordered rows; manifest and
  protocol hashes are recorded in the private run artifact.
- Model: local Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights, BF16 compute,
  eager attention, `max_pixels=200704`.
- Three-row GPU smoke: 3/3 success, exactly 3 temporal bins, 4.695 s model
  inference, 7.461 GiB peak allocated VRAM.
- Full run: requested 61, completed 58; three rows had sampled-frame read
  failures. Inference was 44.704 s, peak allocated VRAM 7.469 GiB.

## Verified observations

The output contains all **784** head summaries. Recalculation from its stored
aggregates reproduced the chord-elbow cutoff (**94**), non-sink survivor count
(**8**), and all eight ranked pairs. The observed sink fractions ranged from
0.3448 to 1.0. The fixed rule therefore yields eight candidates at three bins,
compared with zero candidates at two bins because of the mathematical lower
bound of 0.5 for two categories.

Against the prior full-run top-20 lists, the descriptive overlaps were:

| Comparison | Shared heads | Union | Jaccard |
|---|---:|---:|---:|
| 6-frame vs 8-frame | 8 | 16 | 0.500 |
| 6-frame vs 16-frame | 4 | 24 | 0.167 |

The earlier 8/16 artifacts omitted failed-row indices. The runs have identical
requested manifests and the old summaries report the same three sanitized
frame-read error strings, but the exact shared success intersection cannot be
verified; these Jaccards are descriptive and may reflect cohort differences as
well as resolution. The 6-frame top-20 contains only eight heads because only
eight pass the fixed filters. No p-values or confidence intervals are
appropriate, and no head is selected for treatment.

Independent CPU verification checked the protocol/manifest/runner/helper hashes,
mode-0600 private artifact and directory permissions, 61 = 58 + 3 accounting,
all 784 unique heads, and exact elbow/sink/rank recomputation. Private artifact:
`data/private_nextgqa/head_six_frame_full_20261019_01.json`, SHA-256
`796cd7925f9036340d9e8f612af22c44cf3d7dcb525265ea9ea09021867dd04e`.
Per-head rows and failed row indices remain local; rights status remains
unverified.

## Interpretation and faithfulness status

The unchanged heuristic can produce a nonempty set at three bins, but only
eight heads pass its filters, and overlap with prior full-run rankings is
limited. Because of the success-cohort caveat and post-selection, this is not a
robustness confirmation. It supplies no evidence that those heads track video
evidence, affect predictions faithfully, or improve an intervention.

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link judgments, and **0/3**
direct scores. There is no direct `ΔF`, confidence interval, or test. The
preregistered head intervention remains negative and closed: 79/79, −0.0756
nat, 95% t CI [−0.1578, +0.0065], below its +0.25 nat practical threshold.

## Next action

Stop this coarse-resolution head-search branch. Do not tune the sink cutoff,
resolution, or layers from these observations. Continue the direct-faithfulness
implementation/review work; run another head study only for a separately fixed
question with an independent evaluation cohort or label source.
