# 2026-10-31 WP2 on TVBench Action Localization (4 options, `start`/`end` annotation), Qwen2.5-VL-7B 4-bit

Same design/analysis as batch A (`2026-10-31_tvbench_window_expansion_plan.md`, priority-queue addendum). Receipt (local-only) `data/private_nextgqa/tvb_window_al_full_20261031.json`; derived `results/tvbench_window_expansion_al_summary_20261031.json`; analysis run once.
Counts: 160 questions in the file; only 135 videos decode/qualify (25 `decode_rejected`; requests for n=160/155 failed with `requested_n_but_cohort_has_135`); 135 recorded, 2 excluded (annotation group budget), 133 analysed, **53 baseline-correct** (chance 25 %; accuracy 40 %); 3612 forwards, 1.89 h. Separate dataset, not pooled with Action Sequence.
## Primary (eff = video->text layers 6-20) 
Evidence-coverage coefficient at equal blocked amount: **+0.6 pp, 95 % CI [-32.9, +34.6]**; t p=0.97, sign-flip p=0.97, Wilcoxon p=0.69; TOST(+-6.7 pp) p=0.36. Rule not met (needs <= -6.7 pp and p<0.05); not equivalent to zero (interval too wide) => **inconclusive**; the point estimate is ~0, unlike the a-priori expectation for this task (options describe when the action occurs).
Blocked amount: eff -57.7 pp per full video [-91.8, -20.3]; all layers -56.0 [-89.9, -20.4]; non-effective layers -6.4 [-14.1, +0.8]. Whole video: -55.4 % (eff), -53.7 % (all), -3.4 % (non-eff).
Group-0 sensitivity: indicator +7.3 pp [-0.7, +15.4] (eff) — no sign of the Action Sequence group-0 drop (-8.5 pp) here.
## Interpretation
Same picture as Action Sequence for amount and layer location (effect only through layers 6-20, grows with the amount blocked) but no evidence-specific effect beyond amount, with wide intervals (53 videos). The group-0 effect does not replicate on this task, so it is task/video-specific. Limits: 25/160 videos excluded for decode/qualification (reason not investigated), low baseline accuracy (40 %), 53 videos, delta_min 6.7 pp taken from Action Sequence (not re-derived), not explanation faithfulness.
