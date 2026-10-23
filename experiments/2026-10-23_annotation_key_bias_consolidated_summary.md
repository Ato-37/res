# Consolidated summary: QA-annotation key bias on the answer margin (label-free), two fresh val cohorts (2026-10-23)

Scope: Qwen2.5-VL-7B-Instruct (4-bit weights, BF16 compute, eager attention, fp32 `lm_head` readout), NExT-GQA val videos whose QA time annotation covers at most
half of 16 temporal groups, five-choice gold-minus-baseline-best-foil log-prob margin, additive attention-score bias on the annotated (or an equal-count disjoint hash
sham) video-key groups at all post-video query rows in all 28 layers. Cohort 1 N=149, cohort 2 N=196 (no media overlap, outcome-blind hash-ranked, structural screens
only). All runs prespecified in dated plans (with pre-outcome deviations recorded), 100% completion, 0 failures. This is a label-free dependence measurement on an
annotation-defined interval; it is NOT explanation faithfulness and not generalized beyond this model/dataset/split/intervention.

## Numbers (nat; 95% t CI; recomputed from receipts, contrast arithmetic max abs diff 0)
| run | contrast annot-sham | annot-minus-baseline | sham-minus-baseline |
|---|---|---|---|
| suppression -ln256, cohort 1 | -0.515 [-0.820,-0.210] | -0.332 [-0.558,-0.105] | +0.183 [+0.017,+0.350] |
| suppression -ln256, cohort 2 (replication) | -0.235 [-0.449,-0.021] | +0.002 [-0.170,+0.173] | +0.237 [+0.116,+0.358] |
| +ln4, cohort 1 | +0.207 [-0.005,+0.420] | -0.089 [-0.278,+0.101] | -0.296 [-0.471,-0.121] |
| +ln4, cohort 2 | +0.124 [-0.071,+0.318] | -0.097 [-0.247,+0.054] | -0.221 [-0.369,-0.072] |
| +ln16, cohort 1 | +0.449 [+0.055,+0.843] | -0.365 [-0.706,-0.024] | -0.814 [-1.190,-0.437] |
| +ln16, cohort 2 | +0.304 [-0.011,+0.619] | -0.065 [-0.357,+0.227] | -0.369 [-0.656,-0.082] |
Prespecified decisions: suppression control passed in both cohorts (negative, p<0.05, |mean|>=0.025: p=0.001 and p=0.031). Positive steering: Holm over 2 doses was not
significant in either cohort (cohort 1: p_Holm 0.055/0.052; cohort 2: 0.21/0.118) and no dose was equivalent to zero. Secondary component rows are descriptive.

## What the data support
1. Applying the intervention to all post-video rows gives it leverage (the last-row-only version had none; both of its suppression controls failed).
2. The contrast annotation-minus-sham has the same SIGN in both cohorts for every condition (negative under suppression, positive under steering); only the point
   estimates replicate in sign, because the cohort-2 intervals for +ln4 and +ln16 include 0 (and the cohort-1 +ln16 interval excludes 0 only before Holm correction: p_Holm
   0.052). Exploratory Welch tests find no evidence that the contrast differs between cohorts (p = 0.14, 0.57, 0.57; no multiplicity adjustment).
3. In 5 of 6 runs the contrast is driven mainly by the SHAM arm: perturbing keys of randomly chosen non-annotated frames moves the margin in a consistent direction in both
   cohorts (suppression raises it, steering lowers it, monotone in dose; all six sham-minus-baseline 95% intervals exclude 0). The exception is cohort-1 suppression, where the
   annotation arm contributes about 64% of the contrast (annotation-minus-baseline -0.332, CI excludes 0); in cohort 2 the same arm is +0.002. That annotation-arm difference
   between cohorts is itself nominally significant (exploratory Welch p = 0.021, Mann-Whitney 0.072).
4. No run shows steering toward the annotated interval improving the margin over the unmodified model (annotation-minus-baseline <= 0 in all four steering runs; the cohort-1
   +ln16 value, -0.365 [-0.706,-0.024], is a nominal decrease, not a null). The positive steering contrasts are a descriptive consequence of the sham arm being damaged more,
   not a causal finding.

## What they do not support / limits
- No claim that the model "depends on" the annotated interval beyond the sham-relative contrast; no explanation-faithfulness claim; no claim for other models, datasets, splits,
  doses or layer/row scopes. Suppression/steering at x1/256 or +ln16 on ~2000 keys x ~66 rows x 28 layers is out of distribution.
- Fragility: the cohort-2 suppression "pass" met the prespecified t criterion (p = 0.031, sign-flip 0.031) but a rank test does not (Wilcoxon p = 0.127; 10%-trimmed mean
  -0.143; median -0.035) and its annotation arm is null. Treat the suppression replication as weak. Cohort-1 +ln4 Wilcoxon p = 0.032 vs t 0.055.
- Heavy tails: medians of the contrast are near 0 only for suppression (-0.09, -0.035) and cohort-2 +ln4; for steering they are +0.23 and +0.49 (cohort 1) and +0.32 (cohort 2
  +ln16). SD 1.3-2.4. Sign-flip p-values agree with t-tests to ~0.001 in all six runs.
- The sham is not a pure null control: disjoint hash-sham frames differ from the annotated frames in position and content, so the contrast mixes annotation and frame-position/
  content effects (diagnostic on cohort 1: mean temporal-group position annotation 7.44 vs sham 7.77, paired p = 0.53; contrast uncorrelated with that difference and with the
  baseline margin).
- delta_min = 0.025 nat is my choice. Intervals in the table are not multiplicity-adjusted (Holm was applied to the two steering doses per cohort in the prespecified
  decisions). Annotation intervals have no clock calibration to decoded-frame PTS; the sham-geometry screen restricts the population (8 of 353 candidates excluded: 6 of 155
  and 2 of 198). Cohort 1 also informed the design of later steps. Selection was outcome-blind (hash-ranked, structural screens only) per the plans and run logs.
- Earlier last-row bias nulls (+ln2..+ln16) and the all-layer last-row control stay uninformative (their positive control failed).

## Independent audit and SSS assessment (2026-10-23)
An independent auditor recomputed all six table rows, Holm/TOST values, completion/forward counts and zero media overlap from the raw receipts and confirmed the numbers; its
seven wording objections (sham-driven only in 5 of 6 runs, median claims, Wilcoxon fragility, multiplicity flag, nominal annotation-arm decrease, sign-only replication,
sham-not-a-pure-null) are incorporated above. The primary agent separately recomputed the contrasts from per-condition margins (max abs diff 0). Assessment: **SSS for the
scoped measurement statements above** (verified provenance and counts, independently reproduced estimates, audited implementation, alternative explanations tested or stated,
claims limited to this model/dataset/intervention with uncertainty). **Not SSS, and not claimed, for any faithfulness statement**: no human-attested claim-level evidence exists,
the suppression replication is weak, and the sham is not a clean null.

## Faithfulness progress report
直接評価: 未実施 (no human-attested claim-level faithfulness measurement). 補助診断のみ: the label-free dependence/steering measurements above. Next prespecified step that would
advance faithfulness evidence: a claim-level endpoint with the same intervention (claim likelihood under annotation-matched vs sham perturbation) once the human-review
requirement for validation is settled by the user.
