# Prespecified plan: option-order permutation ensemble (error reduction; written 2026-10-26, before any PERM run)

## Aim and origin
Main aim of the correction branch (user, 2026-10-26): reduce wrong answers without the annotation. Exploratory val readings (SGC val N=508, post hoc, not confirmatory): (i) the gain of the crop ensemble sits in the least-confident videos (confidence terciles: accuracy 0.453 -> 0.500, 0.692 -> 0.698, 0.893 -> 0.893; the most confident third is unchanged) and ensembling only when the baseline top probability is below 0.6 (18% of videos, 1.36x cost) gave 0.699, the same as ensembling everything (0.697); (ii) a mild letter bias of the original order (predicted letter shares A-E 0.19, 0.17, 0.21, 0.23, 0.19 versus gold 0.20, 0.21, 0.18, 0.21, 0.19). Averaging predictions over permuted option orders is a standard debiasing of multiple-choice answers and needs no attention edit, no annotation and no gold label.

## Design (fixed now; annotation-free)
- Per video: the original forced-choice forward and 4 cyclic shifts of the option order (shift j shows original option (p+j)%5 at position p); option letter log-probs mapped back to the original options (original option o sits at position (o-j)%5). The identity prompt rebuilt through the same code is compared with the baseline batch (flag recorded). fp32 readout, 4-bit/BF16, 32 frames, same cohort machinery as SGC/CEN. 5 forwards per video.
- Cohorts (both evaluated separately, no pooling): NExT-GQA val, first 300 of the 508 sham-eligible media (the SGC val cohort), and NExT-GQA test, the Stage 2b / SGC-test cohort (N=278). Baseline answers on both cohorts are already known from earlier receipts; the new quantity is the permuted predictions.
- Primary (per cohort): P5 = mean option probability over the 5 orders versus the original-order answer, exact two-sided McNemar; supported iff p < 0.05 and accuracy difference >= +0.02 (the smallest gain judged useful). Replicated iff supported in both cohorts.
- Secondary (descriptive): gated P5 (ensemble only when the original-order top probability < 0.6; threshold fixed here, taken from the val reading above, not tuned further), 3-order and geometric-mean variants, single-shift accuracies, letter distributions. The same fixed gate (0.6) is added to the crop-ensemble analysis (variant A gated) for CEN.
- Power: with ~7-9% discordant videos, N=278 gives about 20-25 discordant pairs; a non-significant result is reported as underpowered, not as no effect.

## Limits
Dataset-answer accuracy only (not explanation faithfulness); 5x compute; one model/dataset; shifts are cyclic (every option visits every position once), not random permutations; the gate threshold was suggested by exploratory val data.

## Workload
About (300 + 278) x 5 = 2,900 forwards (plus one extra frame decode per video). Runner scripts/run_option_permutation.py (not yet run on GPU; smoke n=3 first), analysis scripts/analyze_option_permutation.py (+ test).
