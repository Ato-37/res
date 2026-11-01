# 2026-10-31 WP3-TVB batch B result (frame removal vs masking; disjoint 126 videos, not pooled with A)

Same runner/analysis as `..._result_A.md`, analysed once. Receipt (local-only) `data/private_nextgqa/tvb_frameremoval_full_B_20261031.json`; derived `results/tvbench_frame_removal_B_summary_20261031.json`.
Counts: 126 requested, 126 recorded, 55 excluded, 43 baseline-correct (the same 43 videos as WP1-TVB batch B masking: paired).
## Primary (remove evidence frames - remove equal-count sham frames, %p)
Mean **-18.06 %p, 95 % CI [-31.99, -5.77]**, d_z -0.41; paired t p=0.010, sign-flip p=0.009, Wilcoxon p=0.027; evidence removal hurt more in 30/43 videos = 69.8 % (Wilson 54.9-81.4 %, sign test p=0.014). remove_evidence mean -22.11 %, remove_sham -4.05 %. (Single batch, uncorrected for the two batches; Holm over A and B would give p about 0.02 for B and 0.37 for A.)
## Masking vs removal (same 43 videos)
Masking contrast -12.53 [-23.97, -1.39] (p=0.039); removal - masking difference -5.52 %p ([-15.06, +3.79] for masking minus removal = +5.52), per-video correlation r=0.71. Removal gives an evidence-specific effect at least as large as masking in this batch, so the masking effect here is not an OOD artefact of masking alone.
## Group 0 (exploratory)
Evidence containing group 0 (n=5): -62.5 %p; not containing (n=38): -12.2 %p (an evidence effect remains without group 0 in this batch; in batch A it did not: +1.95).
## Interpretation
Batch A (removal: -5.0 [-16.3, +5.5], null) and batch B (-18.1 [-32.0, -5.8], significant) differ as masking did (A -5.7, B -12.5); in both batches removal and masking agree (r=0.76, 0.71; no significant difference). Heterogeneity between batches (A vs B) is not tested separately here and the two are not pooled. The consistent picture: masking is not merely an OOD artefact, the evidence-specific effect is of moderate size and variable across samples; evidence starting at the video beginning gives the largest effects. Limits: 43 baseline-correct videos, 2-option task, frame removal shifts the later timeline and shortens the video, exploratory group-0 split (n=5), not explanation faithfulness.
