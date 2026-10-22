# Multi-query suppression control result: the control PASSES (2026-10-23)

Plan (written before the run): [2026-10-23_annotation_key_bias_multiquery_control_plan.md](2026-10-23_annotation_key_bias_multiquery_control_plan.md).
Label-free; scope = answer-margin dependence on the QA time annotation, not explanation faithfulness.

## Provenance
Same locked N=149 cohort (lock SHA-256 prefix 470f310c..., byte-identical to earlier runs). Base runner gained
`--query-scope post_video` (rows = every position after the last video token; 66 rows in the audited smoke) and a multi-row mask
audit (all-and-only rows x annotation-key cells changed; 28 layers; 133,056 changed cells in the smoke row, 2016 keys x 66 rows).
Unit tests 34/34. Bias -ln256, layers 1-28, fp32 readout; smoke 3/3; full 149/149 rows, 447/447 forwards, 0 failures; finished
17:40Z. Summary: `results/annotation_key_bias_multiquery_control_valE149_summary_20261023_02.json`.

## Result (annotation-minus-sham margin contrast, nat; negative = suppressing the annotated keys lowers the margin more)
- mean -0.515 (SD 1.886), 95% t CI [-0.820, -0.210], two-sided t p = 0.0011; sign-flip p = 0.00075; Wilcoxon p = 0.014;
  bootstrap 95% CI [-0.827, -0.220]; 10%-trimmed mean -0.328; median -0.089; 69 positive / 80 negative; range [-10.2, +4.2].
- annotation-minus-baseline -0.332, sham-minus-baseline +0.183; predicted-choice flips relative to baseline 13 (annotation) vs
  6 (sham); correct answers 109 baseline, 98 annotation-suppressed, 113 sham-suppressed.
- Prespecified rule (negative, p < 0.05, |mean| >= 0.025): **PASSED**. The pipeline is sensitive when the intervention covers all
  post-video query rows; Step 2 (+ln4, +ln16, same scope/layers, Holm over 2) was launched per plan.

## Interpretation and limits
- Suppressing the annotated frames' keys at every post-video row (all layers) lowers the gold-minus-foil margin more than
  suppressing an equal-count disjoint sham: the model's answer margin depends on the annotated interval's keys through the
  text rows, not only through the final row. This supports the earlier hypothesis (last-row-only edits had no leverage).
- The effect is heterogeneous and heavy-tailed (median -0.09 vs mean -0.52; trimmed mean -0.33): a minority of videos change a
  lot. Both conditions are drastic (x1/256 on ~2000 keys x 66 rows x 28 layers) and out of distribution; sham suppression slightly
  RAISES the margin (+0.18), so part of the contrast is sham-side.
- delta_min = 0.025 is my choice; same cohort reused; one model, dataset and split; not pooled with earlier runs.
- This is a suppression (necessity-type) dependence result on an annotation-defined interval, NOT explanation faithfulness and
  NOT a steering-improves-faithfulness result. Faithfulness status: 直接評価 `未実施`; 補助診断のみ.

## Next action
Step 2 (positive bias +ln4, +ln16, multi-query, all layers; Holm; TOST at +/-0.025) is running; result log to follow.
