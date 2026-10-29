# Map the Flow Figs 4-5 / Table D: logit lens on video tokens, Qwen2.5-VL-7B, TVBench Action Sequence (2026-10-29)

Status: PRESPECIFIED before the full run. Protocol read (not executed) from the official analysis/logit_lens_analysis.py: top-1 logit-lens token of every video token at every layer; counts of the task's spatial (21 object nouns) and temporal (14 action verbs) keywords (first sub-word), per layer.

Our transfer: Qwen2.5-VL-7B (28 layers), 100 Action Sequence videos (the first 100 of the hash-ranked cohort used in the other TVBench runs), layer-l residual + final norm + fp32 lm_head (layer 28 already normalized); word matching on the first sub-word of word / " word" / capitalized variants. Only counts are stored.

Prespecified criteria (video = unit, paired bootstrap 10,000, seed 20261029, 95% CI; bands: early 1-4, middle 9-20, late 21-28):
- L1 (paper pattern): temporal-keyword top-1 rate in the middle band minus the early band has CI lower bound > 0.
- L2 (extension, not in the paper's analysis as far as read): blocking cross-frame attention in ALL layers lowers the middle-band temporal rate (CI upper bound < 0) and the drop is larger than for spatial keywords (difference-of-differences CI upper bound < 0).
Reporting only: spatial pattern, per-word totals (words with <100 total hits are flagged, as in the official filter), answers under blocking.
Limits: logit lens on a 4-bit model with the final-layer unembedding is a crude probe; "temporal concepts" are approximated by 14 verbs; descriptive of representations, not explanation faithfulness.
Run: `.venv/bin/python scripts/run_tvbench_logit_lens.py --mode full --n 100 --tvbench-root data/private_tvbench --frames 8 --user-authorized-local-only --output data/private_nextgqa/tvblens_full_20261029c.json`

## Result (full run `tvblens_full_20261029c`, 100/100 Action Sequence videos, 200 forwards; summary results/tvblens_summary_20261029c.json)
Analysed once, as prespecified. Top-1 keyword rate over video tokens (percent of video tokens; layers 1..28):
- temporal keywords: ~0.00-0.03% in layers 1-23, 0.10 / 0.20 (L25, L26), 1.54 (L27), 3.24 (L28).
- spatial keywords: 0.00% until L23, then 0.35 / 0.59 / 0.63 / 1.99 / 5.01 (L24-L28).
- **L1 not met**: middle-band minus early-band temporal rate -0.007 pt (CI [-0.018, +0.001] pt); there is no middle-layer emergence. The rise is in the last 4 layers and is larger for spatial than for temporal words (late minus middle: temporal +0.64 pt, spatial +1.07 pt).
- **L2 not met**: cross-frame blocking in all layers changes the middle-band temporal rate by +0.0001 pt (CI [-0.002, +0.003] pt); difference-of-differences vs spatial CI [-0.002, +0.003] pt. Both are effectively zero because the middle-band rate is ~0 in both arms (floor), so L2 is uninformative rather than refuting: there was nothing in the middle layers to remove.
- Exploratory (not prespecified): in layers 25-28 temporal 3.24 -> 3.03 and spatial 5.01 -> 4.31 under cross-frame blocking (means of per-layer percentages); not tested.
Interpretation: in Qwen2.5-VL-7B the logit lens decodes video tokens into object/action words only at the very end of the stack, so the paper's mid-layer emergence of temporal concepts is not reproduced with this probe. This can mean a real model difference or that the logit lens is uninformative for intermediate Qwen residuals (the final norm/unembedding is not trained for them; 4-bit). We cannot separate these; a tuned/linear lens would be the follow-up, not a change of this analysis.
Limits: 100 videos of one task; 35 keywords, several are common words ("on", "up", "do", "down"); words with <100 total hits flagged (sandwich 7, drink 7, hold 6, throw 2, take 14, sit 36, do 48, dish 76, food 88); not explanation faithfulness.
