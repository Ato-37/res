# Prespecification: annotation-free multi-crop ensemble (exploratory finding on SGC val, confirmation on SGC test) (written 2026-10-26, before the SGC test receipt was analysed)

## Origin (exploratory, SGC val N=508, post hoc; not a confirmatory result)
Averaging the option probabilities of the unmodified forward and the two annotation-free crop arms (selfguided4 = top-4 groups by attention, random4 = 4 hash-random groups; neither uses the annotation) changed accuracy 0.679 -> 0.697 (wrong->correct 22, correct->wrong 13, exact McNemar p = 0.18, bootstrap 95% CI of the difference [-0.006, +0.041]); S4+R4 only: 0.699 (32 vs 22, p = 0.22). The ensemble's top-option probability separated correct from wrong answers better than the baseline's (AUROC 0.767 vs 0.740; target = baseline correctness). Agreement of predictions across crops as a signal: AUROC 0.561 (weak). Whole-video-blocked contrastive decoding (WP1 data, N=112): accuracy 0.696 -> 0.714 at best alpha (0.5), 0.688 at alpha 1, worse in the all-layer version; no consistent gain. None of these is significant; they are the reason this prespecification exists, not evidence.

## Confirmation (single analysis, no tuning, uses the SGC test receipt that is already running; zero extra forwards)
- Data: the SGC test cohort (NExT-GQA test, Stage 2b cohort, N = number recorded, 278 expected).
- Ensemble: mean of the option probabilities of baseline, selfguided4 and random4 (equal weights, fixed now).
- Primary: accuracy of the ensemble minus accuracy of the baseline, exact two-sided McNemar test, alpha 0.05, effect called supported only if p < 0.05 and the accuracy difference >= +0.02 (the val point estimate; chosen as the smallest gain I consider useful).
- Secondary (descriptive, Holm not applied): Brier score and AUROC (target = ensemble correctness and baseline correctness) of the ensemble top-option probability versus the baseline's; risk-coverage is not used (no novelty).
- Power statement: with about 278 videos and the val discordance rate (about 7% of videos), only ~19 discordant pairs are expected, so a +2-point effect is NOT detectable; a non-significant result will be reported as underpowered, not as "no effect". A pooled val+test estimate is reported descriptively, not as the confirmation.
- Compute note: the ensemble costs 3 forwards per video versus 1; the fair compute-matched control (3 random 4-crop views, no baseline) is a follow-up GPU experiment, queued only if the test result is directionally positive.
- Scope: dataset-answer accuracy; not explanation faithfulness.
