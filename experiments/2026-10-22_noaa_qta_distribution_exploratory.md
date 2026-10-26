# NOAA QTA answer-distribution experiment (2026-10-22)

## Purpose

Measure whether the fixed question-attention QTA intervention changes the
restricted five-option next-token distribution on the six locally available NOAA
QA pilot rows, and compare it with an equal-budget disjoint hash sham. This is a
new exploratory endpoint to test model-output sensitivity after nearby generic
steering diagnostics proved heterogeneous. It does not test explanation
faithfulness or grounding.

## Frozen setup and provenance

- Dataset: six locally available question rows across three source videos
  (3/1/2 rows per video). Only the three `samples.json` manifests and video
  media were read. Gold answers, human-review materials, and audio were not
  read or used.
- Model: pinned `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit weights, BF16 compute,
  eager attention; 8 sampled frames per question; same fixed five-choice
  next-token scoring for every condition.
- Conditions: baseline; QTA adds `ln(2)` key bias at layers 5–16 to the
  highest-scoring `ceil(0.10*T)` temporal group selected from untreated question
  attention; sham applies the same key budget to a deterministic disjoint
  hash-selected temporal group. The sham is not position- or evidence-overlap-
  matched.
- Completed GPU work: 18/18 forwards (baseline/QTA/sham for each of 6 rows),
  6/6 rows, 0 failures; elapsed 139.034 s. RTX 5070 Ti, CUDA build 12.8,
  peak allocated VRAM 9,289,169,920 bytes. A preceding 1-row/3-forward smoke
  completed in 52.626 s. All outputs remain local, mode 0600; rights status is
  unverified.
- Full receipt:
  `datasets/noaa_qa_pilot_20261011/private/noaa_qta_distribution_full_20261022_01.json`
  (SHA-256 `a3a9105dc050f3b085ad723a555ba8d9d739120a946c16f41a8c7e2bbaf99f5f`).
  Current-code smoke receipt:
  `datasets/noaa_qa_pilot_20261011/private/noaa_qta_distribution_smoke_20261022_03.json`
  (SHA-256 `691d77516560255b39cb80396e54130088ebfddf5335aa638b6593b1753577c9`).
  Both files are private and not for publication.
- Runner:
  `scripts/run_noaa_qta_distribution_pilot.py`; run-recorded protocol and
  source hashes bind the full run to the checked smoke and implementation.
  Five focused unit tests and byte-compilation passed after repairing a smoke
  receipt schema mismatch discovered before the full run.

## Observed results

| Contrast | Mean total variation distance | Median | Range |
|---|---:|---:|---:|
| QTA vs baseline | 0.007412 | 0.005045 | [0.000360, 0.024532] |
| Sham vs baseline | 0.000830 | 0.000085 | [0.00000001, 0.003276] |
| QTA vs sham | 0.007547 | 0.004673 | [0.000529, 0.026708] |

TVD is computed between within-row softmax distributions over the five fixed
answer-option tokens; it ranges from 0 (identical) to 1 (disjoint). QTA and sham
changed the baseline top-choice token on 0/6 rows each. All 18 saved distributions
sum to 1 within numerical tolerance. Equal-video-weighted means were 0.005549
(QTA vs baseline), 0.000591 (sham vs baseline), and 0.005677 (QTA vs sham).
The video-level values are descriptive for only three clusters and repeat the
same six question rows.

## Interpretation and limitations

On this small local pilot, fixed QTA changed the restricted option distribution
more than the chosen disjoint sham on average, but did not change any argmax
choice. This is a model-output sensitivity observation, not evidence that QTA
attended to the correct visual evidence. The five-option renormalized endpoint
does not measure full-sequence answer quality, QA accuracy, explanation truth,
claim/frame support, or direct faithfulness. Only three videos were sampled;
there is no confidence interval, hypothesis test, or population claim. The
sham controls temporal-key budget but not evidence overlap or position. Human
review was not performed; this run does not supply review labels or authorize a
direct-faithfulness score.

A post-hoc grounding-proxy audit mapped the selected temporal groups to the
annotation-audit evidence timestamps: QTA hit 2/6 rows versus 3/6 for its
equal-budget disjoint sham, against an observed group-count-adjusted uniform
random expectation of 1.75/6. The annotation methods differ between videos and
the mapping is coarse; this does not establish an advantage for either
condition. See the separate [annotated-time alignment audit](2026-10-22_noaa_qta_annotation_time_alignment.md)
for method, source hashes, and limitations.

Direct faithfulness remains at 0/25 formal support reviews. The current artifact
remains at 0/3 claim reviews, 0/15 frame-time pairs, 0/3 prediction-link
decisions, and 0/3 direct intervention scores; no direct `DeltaF` or uncertainty
interval exists. This experiment adds a real-video GPU sensitivity result only.

## Next action

Do not label the observed TVD difference a faithfulness improvement. The next
useful experiment should freeze reviewer-supported claim and frame anchors, then
compare baseline, support-key intervention, and a review-eligible matched sham
on a preregistered held-out subset. Until such labels exist, use this result only
to motivate a clearly scoped exploratory thesis section on internal
intervention-induced answer-distribution sensitivity, with the three-video
limitation explicit.
