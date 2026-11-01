# 2026-10-31 WP3-TVB: frame removal vs attention masking (OOD check) — prespecified plan

Status: PLAN, written before any WP3 outcome. Runner `scripts/run_tvbench_frame_removal.py`; analysis `scripts/analyze_frame_removal.py` (written before the full run).

## Why
Attention masking (WP1/WP2) creates a state the model never saw in training. The drop of the answer probability after masking the annotated interval could reflect distribution shift instead of lost information (the amount-effect in WP2 cannot separate the two). Removing the frames from the input and re-encoding a shorter video gives the model an ordinary (in-distribution) video that lacks the evidence frames.
## Design
- Model, readout, cohort ranking (`tvb-ev-v1`), sham construction (`choose_disjoint_hash_sham`, seed `tvbench-annotation-key-sham-v1`) identical to WP1-TVB, so the evidence/sham temporal groups of each video are the same as in the masking run (paired comparison by `media_sha256`).
- 32 sampled frames = 16 temporal groups of 2 frames (Qwen2.5-VL temporal patch 2). Arms per video: `full` (baseline); `remove_evidence` (delete the 2 frames of every annotated evidence group); `remove_sham` (delete the 2 frames of every sham group; same count k). Frame timing metadata is kept at the effective fps; the shortened video simply has fewer frames.
- Batch A = the first 200 hash-ranked videos (the WP1-TVB batch A cohort); batch B = the 126 disjoint videos (`--skip 200`), analysed separately, never pooled.
## Analysis (one primary, video = unit)
Baseline-correct videos; per-video contrast c = (%p change under remove_evidence) - (%p change under remove_sham), %p = relative change of the predicted-option probability. Report mean with bootstrap 95 % CI (10k, seed 0), paired t, sign-flip and Wilcoxon p, the proportion of videos where removing the evidence hurts more than removing the sham (Wilson 95 % CI, sign test), Cohen d_z. No pass/fail threshold; the question is "is c different from 0, and how large".
Secondary (descriptive): (a) the same contrast under masking (from the WP1-TVB receipt of the same videos) and the paired difference c_mask - c_remove with CI; (b) correlation of per-video c_mask and c_remove; (c) c by whether the evidence contains temporal group 0.
Interpretation rule (stated before running): if removal shows an evidence-specific drop of similar size to masking, the masking result is not explained by OOD alone; if removal ~0 while masking is negative, masking likely over-states dependence (OOD); if both ~0 the dependence on the interval is not established. Caveat for any reading: removing frames also shifts all later frames' timeline and shortens the video (a different, not "cleaner", intervention); equal-count sham removal controls for shortening.
## Cost
3 forwards + 3 video preparations per video; est. 10-15 s/video => batch A ~0.8 h, batch B ~0.5 h plus ~10 min cohort decoding each.
## Limits
2-option Action Sequence only; videos with no disjoint sham are excluded as in WP1-TVB (~45 %); 4-bit model; removal changes the frame timeline; not explanation faithfulness.
