# Evidence-focusing Stage 2 (test split, N=80): result of the first confirmation cohort (2026-10-23)

Plan: [2026-10-23_evidence_focusing_steering_plan.md](2026-10-23_evidence_focusing_steering_plan.md). Per the revised plan (recorded before this outcome was seen) the **final confirmatory decision is the pooled analysis of Stage 2 + Stage 2b**;
this note reports Stage 2 alone (and the earlier Stage-2-alone rule for transparency). Label-free; scope = oracle evidence-interval attention cropping, answer margin; not explanation faithfulness.

## Provenance
NExT-GQA test split, outcome-blind hash-ranked candidates, excluding the 80 videos of the pathway-knockout replication; only 82 candidates passed the freezer's structural screens and 2 more lacked a disjoint equal-count sham -> N = 80 (requested = effective), 240/240 forwards,
0 failures. Keep mode, bias -ln256 (the dose chosen on the development cohort), all 28 layers, all post-video query rows, fp32 readout, 4-bit/BF16, seed 20261021. Summary (no IDs): `results/evidence_focusing_stage2_testE80_summary_20261023_50.json`.

## Results (nat; mean [95% t CI]; video = unit)
| quantity | Stage 2 (test, N=80) | development (val, N=123, for reference) |
|---|---|---|
| keepAnn - keepRand (primary) | +0.177 [-0.340, +0.694]; t p=0.50, sign-flip p=0.50, Wilcoxon p=0.40; median +0.291 | +0.659 [+0.225, +1.094] |
| keepAnn - baseline | **+0.636** [+0.116, +1.156]; t p=0.017, Wilcoxon p=0.025; median +0.293 | +0.564 |
| keepRand - baseline | **+0.459** [-0.019, +0.937]; t p=0.060, Wilcoxon p=0.044; median +0.450 | -0.095 |
| correct answers (baseline / keepAnn / keepRand) | 50 / 57 / 54 of 80 | 79 / 86 / 72 of 123 |
Prespecified Stage-2-alone rule (t, sign-flip and Wilcoxon all < 0.05 and mean >= 0.3): **not met** (mean 0.177, p=0.50); TOST at +/-0.3 nat: p=0.32 (not equivalent) -> inconclusive.
Dev vs test contrast means differ (Welch p=0.16, not significant).

## Interpretation and limits
- The annotated-keep arm again improves the margin over the unmodified model by a similar amount (+0.64 here vs +0.56 in development), but **the equal-count random-keep arm also improved on the test cohort** (+0.46; it was neutral in development), so the
  difference between the two is small and not distinguishable from zero at N=80. This suggests that most of the margin gain comes from removing visual context (keeping a few temporal groups) rather than from keeping the annotated interval specifically;
  the evidence-specific part (+0.18, interval spanning -0.34 to +0.69) remains open.
- The development estimate (+0.66) was selected as the best of two doses on the same data used to form the hypothesis and is likely inflated (winner's curse). Heavy tails (SD 2.3).
- Power: with N=80 and SD 2.3 the test has low power for effects of 0.3-0.6 nat; Stage 2b (N expected about 250, independent) is running next and the pooled Stage 2 + 2b analysis is the decision.
- Faithfulness status: 直接評価 `未実施`.
