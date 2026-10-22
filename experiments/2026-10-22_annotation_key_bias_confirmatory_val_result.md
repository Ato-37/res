# Confirmatory result: QA-annotation key bias (+ln2) vs equal-count sham, NExT-GQA val, N=149 (2026-10-22)

Plan (fixed before selection/outcomes, with two pre-outcome deviations recorded there):
[2026-10-22_annotation_key_bias_confirmatory_val160_plan.md](2026-10-22_annotation_key_bias_confirmatory_val160_plan.md).
Label-free: no human review used. Scope: dependence of the five-choice gold-minus-baseline-best-foil margin on the QA time
annotation. This is NOT explanation faithfulness.

## Setup / provenance (verified)
Qwen2.5-VL-7B-Instruct (cc594898...), 4-bit weights, BF16 compute, fp32 lm_head readout, eager attention, layers 5-16,
+ln(2) key bias on annotation temporal groups vs equal-count disjoint hash sham, 32 frames, seed 20261021. Cohort: val
candidates ranked 41-200, screened outcome-blind for sham geometry (6 of 155 ineligible) -> requested 149 = effective 149,
447/447 forwards, 0 failures, 0 replacements; 0 media overlap with the two earlier val cohorts or the 1,034-item exposure
union; receipt SHA-256 57e3fbf2...; run chain screen 09:32Z -> preflight 09:45Z -> smoke 09:57Z -> full 10:38Z (after the
plan, N and rule were written). Independent recomputation from the receipt reproduced n, mean, SD, t interval, p, contrast =
annotation-minus-baseline minus sham-minus-baseline (max abs diff 0), unique media 149.

## Result (prespecified primary)
| quantity | value |
|---|---|
| mean annotation-minus-sham margin | -0.00064 nat |
| SD / median | 0.0805 / -0.0018 nat |
| 95% t interval | [-0.0137, +0.0124] nat |
| two-sided t-test | p = 0.923 (sign-flip p = 0.925) |
| TOST vs +/-0.025 nat | p = 0.00015 (90% CI [-0.0115, +0.0103]) |
| sign counts | 73 positive / 76 negative |
| verdict (prespecified rule) | no_meaningful_effect_equivalence |
Secondary, descriptive only: annotation-minus-baseline mean -0.0028 (SD 0.081); sham-minus-baseline mean -0.0022
(SD 0.087); correct answers 109/149 in each of baseline, annotation-bias and sham (no predicted choice changed).

## Interpretation and limits
- Within this model, +ln2 bias, layers 5-16, fp32 readout, NExT-GQA val videos whose QA annotation covers at most half the
  clip, the annotation-key bias does not change the answer margin more than the matched sham: the effect is statistically
  equivalent to zero within +/-0.025 nat. delta_min was my choice, not an externally justified minimum; a smaller effect than
  0.025 is not excluded (the CI half-width is ~0.013).
- The intervention also barely moves margins relative to baseline (-0.003 nat), so a weak dose is a live explanation for the
  null; this cohort cannot separate "no selectivity" from "intervention too weak". No dose or layer was tuned on these data.
- Annotation intervals have no independent clock calibration to PTS; the 6 excluded rows restrict the population; one model,
  one dataset, one split. No claim about explanation faithfulness or about test-split / other models.
- bf16 and fp32 readouts are not pooled; the earlier exposed cohorts are not part of this estimate.
- Faithfulness status: 直接評価 `未実施`; this is an auxiliary label-free dependence measurement (補助診断のみ).

## Next action
Prespecified dose-response check on a fresh cohort (see the follow-up plan) to test whether a stronger bias produces
selectivity, so the null can be attributed to the mechanism rather than to dose. Reproduce analysis:
`.venv/bin/python scripts/analyze_annotation_key_bias_confirmatory.py --receipt data/private_nextgqa/annotation_key_bias_valE149_full_20261022_01.json --expected-n 149 --delta-min 0.025 --output results/<new>.json`

## Addendum (2026-10-23): positive control failed
The prespecified suppression positive control (-ln16, -ln256, same 149 videos) found no effect either
([result](2026-10-22_annotation_key_bias_positive_control_result.md)), and ln4/ln8/ln16 positive-bias runs were also null
([result](2026-10-22_annotation_key_bias_dose_response_result.md)). The "no meaningful effect (equivalence)" verdict above
therefore describes an inert manipulation at the final query position; it must not be read as evidence that the answer
margin does not depend on the annotated interval.
