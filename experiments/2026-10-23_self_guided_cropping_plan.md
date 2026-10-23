# Prespecified plan: self-guided evidence cropping (SGC) as an annotation-free steering method (written 2026-10-23, before any SGC run)

## Motivation and link to the base paper
Map the Flow (Kim, Kim & Han, ICLR 2026) showed that video information reaches the answer through video -> text-row attention in the middle layers. WP0 replicated this on Qwen2.5-VL-7B (strongest video->text window centred at layer 14, i.e. layers 10-18).
The keep-mode experiments showed that attention-level cropping to the annotated temporal groups (oracle, uses the NExT-GQA annotation at inference) raised the answer margin over the unmodified model in two cohorts (+0.56 dev, +0.64 test N=80), but the random-keep control
also improved on the test cohort, so the evidence-specific part is unresolved (Stage 2b is running). SGC asks the deployable question suggested by the base paper's pathways: **can the model's own video->text attention in the identified layers choose which
temporal groups to keep, without any annotation, and does this beat a random choice of the same budget?** Label-free; scope = answer margin on NExT-GQA, not explanation faithfulness.

## Method (all parameters fixed now; no tuning)
- Score: from the unmodified forward, hooks on layers 10-18 read the eager attention weights; score s_g of temporal group g = mean over those layers, all heads and all text rows T (rows between the last video token and the final position) of the attention mass
  that each row puts on the keys of group g. (Layer band, rows and averaging are fixed from WP0's pathway result, not from any NExT-GQA outcome of this plan.)
- Cropping: additive bias -ln256 on the keys of all groups that are NOT kept, at all post-video rows (T and the last position), in all 28 layers (the keep mode of the evidence-focusing plan; same hooks, fp32 readout).
- Arms per video (6 forwards): baseline; O = oracle keep (annotation groups A, |A| groups); R = random keep (the equal-count disjoint hash sham of A, |A| groups); S = self-guided keep (top-|A| groups by s_g; matched budget);
  S4 = self-guided keep of the top-4 groups (annotation-free budget); R4 = random keep of 4 groups (hash-ranked from all groups, not conditioned on A).
- Localization diagnostic (label-free, no intervention): overlap of the top-|A| self-guided groups with A, versus the hypergeometric chance level.

## Cohorts (fixed now)
- Primary: NExT-GQA val, all outcome-blind hash-ranked media (525 extracted; same structural and sham screens as before; videos with |A| > 8 groups are excluded because R needs a disjoint equal-count set). N = number eligible (expected ~350-450). These videos were used by earlier
  key-bias endpoints; the SGC design is fixed from the WP0 test-split pathway profile and the earlier keep-mode logic, not from SGC outcomes; exposure is a limitation.
- Replication: the Stage 2b test cohort (NExT-GQA test candidates ranked 501-990, same screens); run after the primary and analysed once with the same rules.

## Prespecified decisions
- Primary pair (Holm over 2, effect tests on the t-test): P1 = S - R (matched budget, margin in nat), P2 = S4 - R4 (annotation-free budget). "Supported" for a contrast iff t, sign-flip and Wilcoxon p < 0.05 and mean >= 0.3 nat (delta_min justified in
  experiments/2026-10-23_practical_threshold_justification.md), with Holm p < 0.05 for the t-test; otherwise TOST at +/-0.3 nat or inconclusive.
- Reference contrasts (descriptive): O - R (is there any evidence-specific gain to find?), S - O (distance to the oracle), each arm minus baseline, prediction flips, correct counts.
- Positive-control logic: S - R can only show that selection matters if O - R is itself >= 0.3 nat in the cohort; if O - R < 0.3, a null for S - R is reported as "no detectable selection effect even for the oracle".
- Replication: same rules on the test cohort; a result is called replicated only if P1 or P2 (whichever was supported on val) is supported there with mean > 0 and p < 0.05.
- Reporting regardless of outcome: requested/effective N, exclusions, means with 95% intervals, medians and trimmed means, Wilcoxon, localization hit rates, heavy-tail caveats.

## Limits
Oracle count |A| leaks the budget in S and R (not in S4/R4); attention is a routing signal, not a faithful attribution; extreme OOD cropping (-ln256 on ~12 of 16 groups, all layers); one model, one dataset; val exposure; NExT-GQA annotation intervals are uncalibrated.
