# Map the Flow Figs 4-5 / Table D: logit lens on video tokens, Qwen2.5-VL-7B, TVBench Action Sequence (2026-10-29)

Status: PRESPECIFIED before the full run. Protocol read (not executed) from the official analysis/logit_lens_analysis.py: top-1 logit-lens token of every video token at every layer; counts of the task's spatial (21 object nouns) and temporal (14 action verbs) keywords (first sub-word), per layer.

Our transfer: Qwen2.5-VL-7B (28 layers), 100 Action Sequence videos (the first 100 of the hash-ranked cohort used in the other TVBench runs), layer-l residual + final norm + fp32 lm_head (layer 28 already normalized); word matching on the first sub-word of word / " word" / capitalized variants. Only counts are stored.

Prespecified criteria (video = unit, paired bootstrap 10,000, seed 20261029, 95% CI; bands: early 1-4, middle 9-20, late 21-28):
- L1 (paper pattern): temporal-keyword top-1 rate in the middle band minus the early band has CI lower bound > 0.
- L2 (extension, not in the paper's analysis as far as read): blocking cross-frame attention in ALL layers lowers the middle-band temporal rate (CI upper bound < 0) and the drop is larger than for spatial keywords (difference-of-differences CI upper bound < 0).
Reporting only: spatial pattern, per-word totals (words with <100 total hits are flagged, as in the official filter), answers under blocking.
Limits: logit lens on a 4-bit model with the final-layer unembedding is a crude probe; "temporal concepts" are approximated by 14 verbs; descriptive of representations, not explanation faithfulness.
Run: `.venv/bin/python scripts/run_tvbench_logit_lens.py --mode full --n 100 --tvbench-root data/private_tvbench --frames 8 --user-authorized-local-only --output data/private_nextgqa/tvblens_full_20261029c.json`
