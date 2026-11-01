# 2026-10-31 WP3-TVB batch A result (frame removal vs masking, Action Sequence, Qwen2.5-VL-7B 4-bit)

Plan: `2026-10-31_tvbench_frame_removal_plan.md` (prespecified). Runner `scripts/run_tvbench_frame_removal.py`; analysis `scripts/analyze_frame_removal.py` (run once). Receipt (local-only) `data/private_nextgqa/tvb_frameremoval_full_A_20261031.json`; derived `results/tvbench_frame_removal_A_summary_20261031.json`.
Counts: 200 requested, 200 recorded, 89 excluded (no disjoint sham, as in WP1-TVB), 64 baseline-correct (the same 64 videos as WP1-TVB batch A masking: paired), 333 forwards, 12.6 min.
## Primary: contrast c = %p change(remove evidence frames) - %p change(remove an equal number of sham frames)
Mean **-5.02 %p, 95 % CI [-16.29, +5.46]**; paired t p=0.369, sign-flip p=0.370, Wilcoxon p=0.422; d_z -0.11; removing the evidence hurt more in 32/64 videos = 50 % (Wilson 38-62 %, sign test p=1.0). remove_evidence mean -7.82 %, remove_sham mean -2.80 %.
Not distinguishable from 0 (same as masking on this batch: -5.67 [-14.1, +2.2], p=0.183).
## Masking vs removal on the same 64 videos
Per-video contrasts correlate r=0.758; masking minus removal -0.64 %p [-7.76, +6.22] (no difference). Hence the masking contrast is not explained by an OOD artefact of masking alone on this batch: an in-distribution intervention gives the same (null-ish) answer.
## Group 0 (exploratory, from stored indices)
Contrast when the evidence contains temporal group 0 (n=16): **-25.9 %p**; otherwise (n=48): +1.95 %p. (Masking, same videos: -18.8 % vs about -1.3 %.) The apparent evidence effect lives in videos whose annotated interval starts at the beginning of the video, in both interventions. Possible reasons (hypotheses, not tested): early frames carry information about the first action/context; attention-sink-like role of the first tokens; the question may ask about the earlier event.
## Interpretation
Frame removal and masking agree: the average evidence-specific effect on this batch is small and not significant, and when present it is concentrated in evidence starting at group 0. This supports reading the masking result as dependence on the early part of the video rather than an OOD artefact. Limits: 64 baseline-correct videos, 2-option task, removal shifts the later timeline and shortens the video, exploratory group-0 split (16 videos), not explanation faithfulness. Batch B (126 videos, disjoint) is running; it is analysed separately.
