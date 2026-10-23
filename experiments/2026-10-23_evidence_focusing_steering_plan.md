# Prespecified plan: evidence-focusing (keep-annotation) attention steering (written 2026-10-23, before any run of this design)

## Origin of the hypothesis (post hoc, so exploratory first)
In both multi-query cohorts, suppressing the keys of randomly chosen non-annotated frames RAISED the answer margin (+0.18 and +0.24 nat, both intervals exclude 0) and steering
toward them lowered it (monotone in dose), while the annotated frames' own keys were roughly neutral. Hypothesis H: **suppressing the attention to everything except the annotated
(evidence) interval improves the answer margin** (attention-level evidence cropping, an oracle upper bound for evidence-focused steering), more than keeping an equal-count random
subset. Because H was formed after seeing the data, the design is split into an exploratory development stage and a confirmatory stage on a different split.

## Intervention ("keep" mode)
For each video the temporal groups are G (16). Annotation set A (<= 8 groups, as before) and the equal-count disjoint hash sham S. Arms: baseline (no edit); KEEP-ANNOTATION =
additive bias b (b < 0) on the keys of all groups NOT in A, at all post-video query rows in all 28 layers; KEEP-RANDOM = the same bias on all groups NOT in S. Both arms edit the same number
of keys and rows. Model, readout (fp32), 32 frames, seed, margin definition unchanged. Contrast per video: keep-annotation minus keep-random margin (primary); keep-annotation minus
baseline (secondary, "does evidence cropping beat the unmodified model").

## Stage 1: development (exploratory), val split
- Cohort: remaining outcome-blind hash-ranked val candidates not used before (ranks > 400 plus none of the 400 already extracted), with the same structural screens (sham geometry).
  N = number eligible. Doses: b = -ln4 and b = -ln256. One run each, no re-runs.
- Gate (fixed now): proceed to Stage 2 only if for at least one dose the primary contrast mean is >= delta_min = 0.3 nat (justification: experiments/2026-10-23_practical_threshold_justification.md).
  If neither dose reaches 0.3 nat the hypothesis is dropped, the result is reported as negative, and no confirmation run is made. The dose for Stage 2 = the one with the larger
  primary mean (ties: the milder dose).

## Stage 2: confirmation, test split
- Cohort: NExT-GQA test split, outcome-blind hash-ranked candidates not exposed before (exposure scan as in the val builder), same structural screens; N = all eligible among the first 300 ranked
  (target >= 200). Single run at the chosen dose, one prespecified analysis.
- Primary: two-sided one-sample t-test of the primary contrast against 0 (alpha 0.05) with the mean >= 0.3 nat -> "practically meaningful improvement over random keep". TOST at +/-0.3 nat if not
  significant. Secondary (descriptive): contrast versus baseline, prediction flips, accuracy per arm. Rank test and sign-flip reported as robustness; a result is called confirmed only if the
  t-test, sign-flip test and the Wilcoxon test all give p < 0.05 with mean >= 0.3 nat.
- Reporting regardless of outcome: N requested/effective, failures, CIs, medians/trimmed means, effect decomposition per arm. Scope: answer-margin effect of oracle evidence-interval attention
  cropping; the annotation is used at inference (oracle), so this is an upper-bound mechanism test, not a deployable method and not explanation faithfulness.
