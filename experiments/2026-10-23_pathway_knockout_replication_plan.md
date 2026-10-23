# Prespecified plan: pathway attention-knockout replication of Map the Flow on Qwen2.5-VL-7B (WP0), written 2026-10-23 before any run

Base paper: Kim, Kim & Han, *Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs*, ICLR 2026 (arXiv:2510.13251). It reports, for LLaVA-NeXT-7B-Video-FT and three other VideoLLMs
on TVBench/TOMATO, that time reasoning uses (i) cross-frame attention among video tokens in early-to-middle layers, (ii) video-to-question(-and-options) flow in middle layers, (iii) question-to-last flow
in middle-to-late layers, while video-to-last edges matter little. Qwen2.5-VL is not among the evaluated models. This plan asks whether the same pathway profile appears in Qwen2.5-VL-7B-Instruct on NExT-GQA.
Label-free, no human review. Scope: attention-pathway dependence of the predicted answer; not explanation faithfulness.

## Intervention and measurement (fixed now)
- Token groups: V = video tokens (grouped into temporal groups of 2 frames), T = text rows between the last video token and the final position (vision-end marker, question, options, generation prompt), L = final position.
- Edge classes (blocked cell = query row <- key column): `cross_frame` (V rows of temporal group g cannot attend to V keys of groups < g), `video_to_text` (T rows cannot attend to V), `video_to_last` (L cannot attend to V),
  `text_to_last` (L cannot attend to T). Blocking sets the additive mask cell to the dtype minimum, only inside a window of k=9 decoder layers centred on layer l (Geva et al. attention knockout, as in the base paper),
  clipped to layers 1-28. Centres l in {2,5,8,11,14,17,20,23,26} (9 windows; the base paper's stride-1 sweep is thinned for cost). Hook audit: only intended cells change, each layer hit exactly once (unit-tested).
- Metric: relative change of the probability of the originally predicted answer token (the 5 option-label tokens' full-vocabulary probability at the last position, fp32 readout): %p = (p_knockout - p_base) / p_base x 100.
  Analysis set: videos whose unmodified top option equals the gold option (as the base paper analyses correctly answered samples); baseline-incorrect videos are recorded but not knocked out.
- Model: Qwen2.5-VL-7B-Instruct (revision cc594898...), 4-bit weights, BF16 compute, eager attention, 32 frames, seed 20261023.
- Cohort: NExT-GQA test split, hash-ranked outcome-blind candidates (`evidence_focusing_test_confirm_candidates_20261023_02`, 300 extracted), first N = 80 eligible by the freezer (ranks 1-80); empty exclusion manifest (no exposure
  exclusion; these test videos were used by unrelated earlier experiments, ASSUMED-CLEARED). Videos 81-300 stay reserved for other plans. Expected baseline-correct subset about 50-60 videos.
- Cost: 1 + 36 forwards per baseline-correct video (~4 s each) => about 2-3 h plus ~15 min decoding. Smoke on 3 videos first. Per-video append to a partial JSONL (crash-tolerant, resumable).

## Prespecified replication criteria (descriptive; no hypothesis test is the primary output)
Report mean %p per edge x centre with 95% t intervals (video = unit). A "drop" is a negative mean %p; a drop counts as meaningful if mean %p <= -5% (the base paper's own 5% selection threshold) and the 95% interval excludes 0.
- R1 (cross-frame): the most negative cross-frame mean over centres <= 14 is at least 10 percentage points lower than the mean at centres >= 20.
- R2 (video->text): the most negative video_to_text mean occurs at a centre in 5-20 and is meaningful.
- R3 (text->last): the most negative text_to_last mean occurs at a centre >= 14 and is meaningful.
- R4 (video->last): the most negative video_to_last mean is smaller in magnitude than the most negative video_to_text mean.
Outcome labels: "replicates" if >= 3 of R1-R4 hold, "partially replicates" if 2 hold, otherwise "does not replicate" (the profile is then reported as a Qwen2.5-VL-specific result, which is also informative).
Secondary (descriptive): fraction of videos whose predicted option changes, per edge x centre; edge-wise heterogeneity (median and 10%-trimmed mean).

## Limits
Centres thinned to stride 3; single cohort of ~50-60 analysed videos (wide intervals for small effects); the question and the options are not separated (T contains both plus template tokens); NExT-GQA multiple-choice differs from the
base paper's TVBench tasks; fine-tuned LLaVA-NeXT-Video-FT vs off-the-shelf Qwen2.5-VL; the base paper's appendix details (e.g. per-frame edge specification) were only partly readable, so the cross-frame edge here is our own definition
(all earlier temporal groups). Not explanation faithfulness.
