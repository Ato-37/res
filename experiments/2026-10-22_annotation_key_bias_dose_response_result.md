# Dose-response result: QA-annotation key bias, N=149 locked val cohort (2026-10-22)

Plan (written before any dose run): [2026-10-22_annotation_key_bias_dose_response_plan.md](2026-10-22_annotation_key_bias_dose_response_plan.md).
Label-free; scope = answer-margin dependence on the QA time annotation, not explanation faithfulness. Same 149 videos,
lock and preflight as the +ln2 run; fp32 readout; only the additive attention-score bias on annotation key groups changed.

## Provenance
Three runs (ln4, ln8, ln16), each 149/149 rows, 447/447 forwards, 0 failures, identical cohort order (checked by the
analyzer); smoke 3/3 each; completed 11:34Z, 12:29Z, 13:24Z. Summary (no identifiers):
`results/annotation_key_bias_dose_response_valE149_summary_20261022_01.json`; analysis script
`scripts/analyze_annotation_key_bias_dose_response.py` (Holm across 3 doses; TOST at +/-0.025 nat, equivalence needs p < 0.05/3).

## Results (annotation-minus-sham margin contrast, nat)
| dose | mean | 95% t CI | Holm p | TOST p | annot-minus-baseline | sham-minus-baseline | choice flips |
|---|---|---|---|---|---|---|---|
| ln2 (earlier run) | -0.0006 | [-0.0137, +0.0124] | (single test p=0.92) | 1.5e-4 | -0.0028 | -0.0022 | 0 |
| ln4 | -0.0035 | [-0.0153, +0.0082] | 1.00 | 2.1e-4 | -0.0006 | +0.0030 | 0 / 0 |
| ln8 | -0.0079 | [-0.0220, +0.0061] | 0.80 | 9.0e-3 | -0.0028 | +0.0051 | 0 / 0 |
| ln16 | -0.0049 | [-0.0221, +0.0123] | 1.00 | 1.1e-2 | +0.0042 | +0.0091 | 0 / 0 |
(ln2 shown for reference only; not pooled; its p is not part of the Holm family.)

## Interpretation and limits
- No dose has a Holm-significant effect, and every dose is statistically equivalent to zero within +/-0.025 nat (TOST
  p < 0.0167). Up to an 16-fold (ln16) additive attention-score bias the annotation-key intervention is not more selective than
  the matched sham on the answer margin.
- The intervention barely changes the output at all: annotation-minus-baseline stays within +/-0.005 nat and no predicted
  choice flips in any condition. So the null at these doses cannot be read as "the model does not depend on the annotated
  interval"; it shows the tested manipulation (bias at the final query position, layers 5-16) has almost no leverage. Whether
  the pipeline can detect any effect is the question the prespecified suppression positive control answers
  ([plan](2026-10-22_annotation_key_bias_positive_control_plan.md)).
- delta_min=0.025 nat is my choice; the cohort is reused across doses (paired) and previously analysed at +ln2. SDs grow with
  dose (0.073 -> 0.106), consistent with more noise rather than directed signal.
- Faithfulness status: 直接評価 `未実施`; auxiliary label-free dependence measurement only.

## Next action
Implement and run the suppression positive control (-ln16, -ln256) per its plan: sign-capable bias audit, new preflight,
unit tests, then smoke/full.
