# Positive-control result: suppression of annotation keys, N=149 locked val cohort (2026-10-22/23)

Plan (written before any control run): [2026-10-22_annotation_key_bias_positive_control_plan.md](2026-10-22_annotation_key_bias_positive_control_plan.md).
Label-free; scope = answer-margin dependence on the QA time annotation, not explanation faithfulness.

## Provenance
Same 149 videos: the new preflight lock is byte-identical (SHA-256 prefix 470f310c...) to the +ln2/dose-response lock.
Runner change before the control: sign-capable bias in the mask audit (three `> 0` checks -> nonzero, unit-tested 22/22).
Two runs, -ln16 and -ln256 (additive pre-softmax attention-score bias on annotation key groups at the final query position,
layers 5-16, fp32 readout), each 149/149 rows, 447/447 forwards, 0 failures; smokes 3/3; finished 14:32Z and 15:27Z.
Summary: `results/annotation_key_bias_positive_control_valE149_summary_20261022_02.json` (analysis script
`scripts/analyze_annotation_key_bias_dose_response.py`, Holm over 2 doses, TOST +/-0.025).

## Results (annotation-minus-sham margin contrast, nat; negative = suppression lowers margin more than sham)
| bias | mean | SD | 95% t CI | Holm p | TOST p | annot-minus-baseline | sham-minus-baseline | choice flips |
|---|---|---|---|---|---|---|---|---|
| -ln16 (-2.77) | -0.0045 | 0.0905 | [-0.0191, +0.0102] | 0.545 | 0.0032 | -0.0019 | +0.0026 | 0 / 0 |
| -ln256 (-5.55) | +0.0092 | 0.0820 | [-0.0041, +0.0225] | 0.345 | 0.0101 | +0.0134 | +0.0042 | 0 / 0 |

## Decision (prespecified rule)
The control PASSES only if a dose shows a Holm-significant NEGATIVE mean contrast with |mean| >= 0.025. Neither does; both
doses are equivalent to zero within +/-0.025 (TOST p < 0.05/2). **The positive control FAILED.** Per the plan, the
pipeline has no demonstrated sensitivity, so the positive-bias nulls (ln2, ln4, ln8, ln16) are **uninformative about
whether the model depends on the annotated interval**; they only show that this manipulation does not move the margin.

## Interpretation and limits
- Even removing ~99.6% of the attention weight on the annotated frames' keys (x1/256) at the final query position in
  layers 5-16 leaves the answer margin unchanged on average and flips no predicted choice. The tested lever is weak.
- Hypothesis (not tested here): at the final query position in layers 5-16 the video information has already been routed
  into earlier text positions or other layers, so a last-position key-score edit has little leverage; input-level frame
  masking (earlier grounding-mask probe, n=37 baseline-correct, selective drop about +0.33 nat) does move the margin.
  A multi-query / all-layer or input-level control would be needed to test this; it is a new design and must be
  prespecified separately.
- Same cohort reused across all runs; delta_min=0.025 is my choice; one model, one dataset, one split; no pooling.
- Faithfulness status: 直接評価 `未実施`; auxiliary label-free measurement only.

## Next action
Prespecify and build an intervention with more leverage (all post-video query positions, optionally more layers), with its own
suppression control first, before any positive-bias test.
