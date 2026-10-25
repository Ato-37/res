# Prespecified plan: annotation-free random-crop ensemble, compute-matched confirmation (written 2026-10-26, before any CEN run)

## Origin
Exploratory SGC val (N=508): averaging the option probabilities of baseline + selfguided4 + random4 raised accuracy 0.679 -> 0.697 (p=0.18). The prespecified single confirmation on the SGC test receipt (experiments/2026-10-26_crop_ensemble_prespec.md; N=278): 0.755 -> 0.799 (+0.043; wrong->correct 19, correct->wrong 7; exact McNemar p=0.029), which meets that rule (p<0.05 and >= +0.02). Open issues: (i) the ensemble costs 3 forwards versus 1, (ii) it is not known whether the gain comes from crop diversity or from the particular selfguided4 arm, (iii) how it scales with the number of crops, (iv) the test and val effects differ (+4.3 vs +1.8 points).

## Design (fixed now; annotation-free; the annotation, the attention scores and the gold label are never used by the method)
- Per video: unmodified forward and K = 6 random 4-group keep crops (additive bias -ln256 on the keys of all non-kept groups, all post-video rows, all 28 layers; groups chosen by SHA-256 ranking with per-crop seeds "cen-crop-v1-0..5"). 7 forwards per video. fp32 readout, 4-bit/BF16, 32 frames, same cohort machinery as SGC.
- Cohorts: (1) NExT-GQA test, the Stage 2b / SGC-test cohort (N=278); (2) NExT-GQA val, the first 300 of the 508 sham-eligible media (SGC val cohort). Both were used for the earlier ensemble result with different arms; this is a new measurement with new random crops, not a re-analysis.
- Primary (per cohort, evaluated separately, no pooling): ensemble A = mean option probabilities of baseline + crops 0 and 1 (3 forwards, compute-matched to the earlier ensemble), accuracy versus the baseline answer, exact two-sided McNemar; supported iff p < 0.05 and accuracy difference >= +0.02. "Replicated" iff supported in both cohorts.
- Descriptive: B crops 0-2 only (3 forwards, no baseline), C baseline + all 6 crops, D all 6 crops only, single-crop accuracies, accuracy as a function of the number of crops added to the baseline. Selective-prediction follow-up with the ensemble confidence is handled by scripts/analyze_selective_prediction.py on the SGC receipts (separate prespecification) and can be repeated on these receipts without new forwards.
- Power: with discordance about 7-9% the test cohort has only ~20-25 discordant pairs; a non-significant result is reported as underpowered, not as no effect.

## Limits
Dataset-answer accuracy only (not explanation faithfulness); 6 forwards extra per video (an accuracy-for-compute trade); one model and dataset; cohort heterogeneity (baseline accuracy 0.68 val vs 0.755 test); crop size 4/16 groups chosen from the earlier arms, not tuned here.

## Workload
About (278 + 300) x 7 = 4,050 forwards, about 6-7 h. Runner scripts/run_crop_ensemble.py (not yet run on GPU; smoke n=3 first), analysis scripts/analyze_crop_ensemble.py (+ test).
