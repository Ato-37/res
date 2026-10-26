# NOAA annotation-time matched-group QA margin sensitivity (2026-10-22)

## Purpose and status

After the post-hoc alignment audit found QTA-selected groups hit listed QA
evidence timestamps in 2/6 rows versus 3/6 for a disjoint hash sham, test a
different but still exploratory question: with a baseline answer preference
held fixed, does the same internal visual-key bias change its restricted-choice
margin more when applied to annotation-mapped time groups than to equally sized,
temporally nearby groups outside those annotations?

This experiment measures forced QA prediction sensitivity to timestamp-selected
video groups. It does not measure generated explanation support or direct
faithfulness. The annotation timestamps are question-evidence points with mixed
review provenance, not reviewer-verified claim support.

## Frozen setup

- Data: same six local NOAA QA rows and three videos as the earlier pilot. No
  gold answer file or correctness field was opened. Annotation audit files were
  read locally and their content hashes were recorded.
- Model: pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights, BF16 compute,
  eager attention. Eight sampled frames, processor grid groups, and choice-token
  bindings used the existing runner's checks.
- For each row, mapped every listed timestamp to the nearest of the eight
  sampled frames and then to its adjacent-frame temporal group. The number of
  distinct evidence groups was one in five rows and two in one row.
- The control selected the same number of non-evidence groups, minimizing the
  sum of absolute temporal-group distances after sorting; ties were resolved by
  a deterministic SHA-256 ranking keyed by the opaque row token. Key counts
  were equal within each row (264 per arm for one group, 528 per arm for two).
- Baseline's top option and strongest foil were frozen before intervention.
  The score was the same pair's next-token log-probability margin. The paired
  estimand was `margin(evidence-group bias) - margin(matched-control bias)`;
  both arms added `ln(2)` to visual keys in layers 5–16. Baseline, evidence,
  and control required three forwards per row.
- CPU preflight and smoke: 6/6 source/annotation joins passed; 15/15 focused
  unit tests passed. One-row smoke completed 3/3 forwards in 51.428 s. Full
  run completed 18/18 forwards in 132.148 s on RTX 5070 Ti, CUDA 12.8; peak
  allocated VRAM was 6,846,158,848 bytes. Rights status remains unverified;
  all data and receipts are local-only.

## Observed results

The paired evidence-minus-control margin changes were `−0.125, 0, −0.125, 0,
−0.125, 0` nat, giving a six-question unweighted mean of `−0.06250` nat. One
row was positive by approximately `6e-8` nat (numerical zero at this scale),
four were negative, and one was zero. The five one-group rows had matched
controls one group away; the two-group row had evidence groups `{1,2}` and its
only feasible equal-size disjoint control `{0,3}`, with paired difference
approximately zero. Four of six rows showed no separation or a lower margin
under evidence-group bias than matched-control bias; no robust positive effect
appeared in this finite sample.

Annotation-provenance descriptive means were `−0.08333` nat for the three rows
with two annotator passes plus a blind visual auditor, and `−0.04167` nat for
the three independent AI blind-answer-review rows. They are not comparable
strata: annotation method is confounded with video, with only one video per
method. Equal-video cluster means were approximately `−0.08333`, `0`, and
`−0.06250` nat. No confidence interval, hypothesis test, or population claim
was computed.

Private full receipt:
`datasets/noaa_qa_pilot_20261011/private/noaa_annotation_matched_group_margin_full_20261022_01.json`
(SHA-256 `a18a42eea71163b87a2cd17ad561a660812033a34bf6f7d4d2d0ae9d9bc8c1eb`,
mode 0600). Smoke receipt:
`datasets/noaa_qa_pilot_20261011/private/noaa_annotation_matched_group_margin_smoke_20261022_01.json`
(SHA-256 `687f5aa194317c6a06ebc32c4f773f226dec9189edba305a97a38277385f96b6`,
mode 0600). Runner SHA-256 recorded in the run is
`112bdf07e7e21aa1300d5342532032147781b410e98957b984d7d9e49be411da`.

## Interpretation and limits

The result does not support the hypothesis that attention bias toward these
annotation-mapped QA time groups increases this fixed answer margin more than
the position-matched control. The mean contrast was negative and most row-level
contrasts were zero or negative. Given the small post-hoc cohort and coarse
temporal mapping, this is only a local descriptive observation, not evidence of
no effect in general.

The intervention imposes attention bias at the final answer-scoring query; it
does not establish natural model reliance. Question wording can reveal the
relevant time. Nearest-frame grouping can misclassify intervals, controls may
contain unannotated event evidence, and temporal proximity does not match visual
salience or semantics. One two-group row has only one possible control set, at
the opposite ends of the four-group sequence. Annotation methods differ by
video. The cohort is reused and outcome-exposed, so this is exploratory and not
confirmatory. Even a positive result would not validate explanation claims or
constitute direct faithfulness.

Direct faithfulness remains unchanged at 0/25 formal claim-support reviews and
0/3 current claim reviews, 0/15 verified frame-time pairs, 0/3 prediction-link
decisions, and 0/3 direct intervention scores. Human review was not performed.

## Next action

Do not tune the intervention on these six rows or present the result as a
faithfulness effect. The nearest useful direct-evidence step remains a current
artifact response that identifies claim-specific support intervals; absent
that input, use this run only as a small exploratory analysis of QA-time
prediction sensitivity. Seek a genuinely independent cohort with independently
reviewed support intervals before any confirmatory efficacy claim.
