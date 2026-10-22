# All-layer suppression control result (2026-10-23)

Plan (written before the run): [2026-10-23_annotation_key_bias_all_layer_control_plan.md](2026-10-23_annotation_key_bias_all_layer_control_plan.md).
Label-free; scope = answer-margin dependence on the QA time annotation, not explanation faithfulness.

## Provenance
Same locked N=149 cohort/lock/preflight as the layers 5-16 control. Bias -ln256 (-5.545) at the final query position in all
28 decoder layers (`--layers all`; receipt `layers_1_based` has 28 entries). Smoke 3/3 (mask audit confirmed only the intended
cells changed in every layer), full 149/149 rows, 447/447 forwards, 0 failures; finished 16:28Z.
Summary: `results/annotation_key_bias_all_layer_control_valE149_summary_20261023_01.json`.

## Result
annotation-minus-sham mean +0.0033 nat (SD 0.0924, 95% t CI [-0.0116, +0.0183], two-sided p = 0.66, TOST +/-0.025 p = 0.0024);
annotation-minus-baseline +0.0115, sham-minus-baseline +0.0082; predicted-choice flips 0 / 0. Prespecified pass rule (negative mean,
p < 0.05, |mean| >= 0.025): **FAILED**. Per plan, Step 2 (positive +ln4/+ln16, all layers) is NOT run.

## Interpretation and limits
- Removing ~99.6% of the attention weight on the annotated frames' keys at the final query row, in every layer, does not lower
  the answer margin relative to the equal-count sham and changes no predicted choice. Last-position key edits are inert for this
  design at any tested depth; all earlier positive-bias nulls stay uninformative about dependence on the annotated interval.
- Hypothesis (untested): answer-relevant video information reaches the final token through the residual stream of the other text
  query rows, which this manipulation leaves untouched; steering only the last row cannot remove it. A multi-query variant
  (all post-video query rows) tests that and needs a hash-bound runner change, so it is planned separately.
- Same cohort reused; delta_min=0.025 is my choice; one model/dataset/split. Faithfulness status: 直接評価 `未実施`.
