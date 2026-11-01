# 2026-11-01 PATHSEARCH stage 1: effective-layer search on Qwen2.5-VL-7B (4-bit), TVBench Action Sequence
Setup: sliding 5-layer windows (start step 2, layers 1-27 starts; prespecified after a stride-1 smoke test cost 5 min/video) x 4 edges (cross_frame, video_to_text, video_to_last, text_to_last); selection cohort = batch B, `--skip 200`, 40 videos requested/recorded, 0 excluded, 29 baseline-correct; 1960 forwards. Runner `scripts/run_tvbench_pathway_search.py`, analysis `scripts/analyze_pathway_search.py` (run once; derived `results/tvbench_pathway_search_summary_20261101.json`, mean relative change of predicted-option probability %, baseline-correct, bootstrap 95% CI, video unit, no multiplicity correction). Receipt local-only `data/private_nextgqa/tvb_pathsearch_full_20261101.json`.
## Observations (strongest windows)
- cross_frame: W9-13 -20.4 [-29.2,-12.3], W11-15 -19.5, W13-17 -16.1, W5-9 -13.8, W7-11 -13.0; W19-23 and later ~0.
- video_to_text: W15-19 -21.9 [-33.9,-9.6], W13-17 -10.7, W11-15 -7.6, W9-13 -5.7; W1-9 and W19+ ~0 (CIs include 0 except W23-27 +0.3).
- text_to_last: W23-27 -20.7 [-23.5,-17.9], W15-19 -13.4, W17-21 -11.4, W21-25 -7.9; early windows (1-13) -0.7 to -2.9.
- video_to_last: all windows within about +-0.6 % (no effect).
## Interpretation (not observation)
On Qwen, cross-frame mixing peaks in layers ~9-17, video->text in ~13-19, text->last in late layers (~15-28); video->last is not used. The LLaVA-scaled ranges (cf 5-13, v2t 5-18, last 14-22; corrected 2026-11-01) roughly overlap but are not identical (the Qwen-derived v2t range 15-19 is narrower and the text->last range extends to layer 27). Limits: 40 videos (29 baseline-correct), single task, descriptive/exploratory, 5-layer windows blur boundaries, single-edge-block effects do not equal pathway sufficiency, 4-bit model. 
## Next (stage 2, not run)
Pre-specify Qwen-derived ranges from this table (e.g. layers of windows with effect >= half of the edge's max), then compare keep-only pathway (Qwen-derived vs LLaVA-scaled vs random same-amount) on batch A. Must be fixed before running; this stage's cohort must not be reused for the test.
