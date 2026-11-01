# 2026-10-31 WP2-TVB batch B (independent replication; Action Sequence videos not in batch A)

Same design/analysis as `..._result_batchA.md`, analysed once, NOT pooled with A. Receipt (local-only) `data/private_nextgqa/tvb_window_full_batchB_20261031.json`; derived `results/tvbench_window_expansion_batchB_summary_20261031.json`.
Counts: 126 requested (all qualifying videos not in A), 126 recorded, 4 excluded, 122 analysed, **83 baseline-correct**; 4059 forwards, 2.06 h.
## Primary (eff = video->text layers 6-20): evidence-coverage coefficient at equal blocked amount
**-21.7 pp, 95 % CI [-60.3, +17.9]**; t p=0.284, sign-flip p=0.282, Wilcoxon p=0.061; TOST(+-6.7) p=0.77. Rule not met, not equivalent => **inconclusive** (same decision as batch A: A -20.4 [-53.7, +12.6]).
Blocked amount (per full video): eff -27.8 [-64.3, +10.4] (CI includes 0 here; A -31.1 [-56.4, -7.2]); all layers -29.2 [-65.8, +9.0]; non-effective layers -1.2 [-5.1, +2.6]. Whole video: -37.3 % (eff), -2.6 % (non-eff), -38.3 % (all).
Group-0 sensitivity: indicator **-13.5 pp [-21.6, -5.2]** (eff), -14.7 [-23.8, -5.4] (all), -0.6 [-1.5, +0.2] (non-eff) — replicates batch A (-8.5 [-15.3, -1.9]); with it, the evidence-coverage coefficient shrinks to -5.6 [-37.1, +28.1].
## Interpretation
Both Action Sequence batches: the nominal evidence-coverage effect is large in point estimate but not distinguishable from zero, a window that contains the first temporal group lowers the answer probability by ~9-14 pp at equal amount (replicated), the effect lives in layers 6-20 (non-effective layers ~0). Action Localization (53 baseline-correct): no group-0 effect, coverage ~0. Limits as before; this is dependence on a temporal position, not explanation faithfulness.
Note: WP1-TVB batch B (margin rule MET, evidence - sham -12.5 %p) and WP2 batch B are not contradictory: WP1's contrast uses a hash-sham that need not avoid group 0, so a group-0 position effect can enter the evidence-vs-sham difference; this is a hypothesis, not tested.
