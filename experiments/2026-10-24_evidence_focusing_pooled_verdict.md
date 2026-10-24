# Evidence-focusing pooled confirmation (Stage 2 + Stage 2b): result (2026-10-24)

Prespecified rule (experiments/2026-10-23_evidence_focusing_steering_plan.md): pooled keep-annotation minus keep-random margin, video = unit, confirmed iff t, sign-flip and Wilcoxon all p < 0.05 and mean >= 0.3 nat.

## Verified (results/evidence_focusing_pooled_stage2_2b_summary_20261023_70.json, results/evidence_focusing_stage2b_testE278_summary_20261023_70.json; Stage 2b receipt status completed, 278/278)
| cohort | N | mean (nat) | 95% t CI | t p | Wilcoxon p | decision |
|---|---|---|---|---|---|---|
| Stage 2 | 80 | +0.177 | [-0.340, +0.694] | 0.50 | 0.40 | inconclusive (alone) |
| Stage 2b | 278 | +0.480 | [+0.204, +0.755] | 0.0007 | 0.0056 | meets rule alone |
| **pooled** | 358 | **+0.412** | [+0.170, +0.654] | 0.0009 (sign-flip 0.0008) | 0.0048 | **confirmed (prespecified)** |

Stage 2b descriptive: annotation - baseline +0.640 nat, sham - baseline +0.160 nat; correctness flips vs baseline 48 (annotation) vs 49 (sham).

## Interpretation (inference)
- The oracle evidence-interval keep beats an equal-count random keep by about +0.4 nat in the answer margin; the effect is statistically confirmed and above delta_min. The lower CI bound (+0.17) is below 0.3, so the size of the effect is uncertain; cohort means differ (0.18 vs 0.48; heterogeneity reported in the summary).
- The margin gain is not matched by accuracy: flips toward/away from correctness are about equal for annotation and sham keep in Stage 2b (48 vs 49 flips; direction not decomposed here). So the confirmed effect is on margin, not shown on accuracy.
- The annotation is used at inference (oracle): an upper bound for evidence-focused steering, not a deployable correction, and not explanation faithfulness (直接評価: 未実施 for this branch; 補助診断のみ).
- Unlocks stage B (annotation-free localization): SGC val running (tag 20261024_81), CSL queued.
