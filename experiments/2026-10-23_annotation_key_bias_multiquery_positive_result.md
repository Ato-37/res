# Multi-query positive key bias (+ln4, +ln16), N=149 locked val cohort (2026-10-23)

Plan (written before any run): [2026-10-23_annotation_key_bias_multiquery_control_plan.md](2026-10-23_annotation_key_bias_multiquery_control_plan.md) (Step 2; the
Step-1 suppression control passed). Label-free; scope = answer-margin dependence on the QA time annotation, not explanation faithfulness.

## Provenance
Same 149 videos/lock as the control; layers 1-28, all post-video query rows, fp32 readout, 4-bit/BF16 compute; runs ln4 and ln16, each
149/149 rows, 447/447 forwards, 0 failures, smokes 3/3. A first launch of these runs was interrupted by a PC bugcheck (nvlddmkm) and the
re-launch was blocked by the WSL RAM limit (default 50% of 31 GB; raised to 22 GB through `.wslconfig`); the completed receipts are
the `_06` runs (finished 21:10Z). Summary: `results/annotation_key_bias_multiquery_positive_valE149_summary_20261023_06.json`
(analysis `scripts/analyze_annotation_key_bias_dose_response.py`, Holm over 2 doses, TOST +/-0.025; the script verified identical scope/layers/precision).

## Results (annotation-minus-sham margin contrast, nat; positive = bias helps annotated keys more than sham keys)
| dose | mean | SD | 95% t CI | raw p | Holm p | TOST p | annot-minus-baseline | sham-minus-baseline | flips (annot / sham) | +/- rows |
|---|---|---|---|---|---|---|---|---|---|---|
| +ln4 | +0.207 | 1.311 | [-0.005, +0.420] | 0.055 | 0.055 | 0.95 | -0.089 | -0.296 | 11 / 9 | 84 / 65 |
| +ln16 | +0.449 | 2.434 | [+0.055, +0.843] | 0.026 | 0.052 | 0.98 | -0.365 | -0.814 | 28 / 26 | 88 / 61 |

## Interpretation and limits
- Prespecified rule: no dose is significant after Holm (ln16 narrowly misses: 0.052), and neither is equivalent to zero. The result is
  inconclusive for selectivity; the direction is positive and grows with dose (+0.21 -> +0.45).
- The contrast is NOT produced by annotated keys gaining: annotation-minus-baseline is negative (-0.09, -0.37) and the sham arm falls
  further (-0.30, -0.81). Biasing keys on non-annotated frames hurts the margin more than biasing the annotated frames, so the
  "selectivity" is a smaller degradation, not an improvement over the unmodified model. No dose improves the margin over baseline.
- Predicted-choice flips are many (28 / 26 of 149 at ln16) and nearly equal across arms: large bias at all post-video rows in all layers
  perturbs the model broadly. SDs are large (1.3-2.4 nat) and heavy-tailed. delta_min=0.025 is my choice.
- Combined with the passed suppression control (annotated keys matter) this gives: removing annotated-key attention lowers the margin, adding
  bias to them does not improve it (annotation-minus-baseline <= 0 at both doses). This remains label-free dependence on an annotation-defined
  interval, not explanation faithfulness; same cohort reused across runs; fresh-cohort replication is prespecified and queued.
- Faithfulness status: 直接評価 `未実施`; 補助診断のみ.
