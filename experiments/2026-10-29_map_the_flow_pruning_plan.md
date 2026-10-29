# Map the Flow Table 3 / E: effective-pathway pruning on Qwen2.5-VL-7B and TVBench (2026-10-29)

Status: PRESPECIFIED before the full run (the 3-video smoke run `tvbprune_smoke_20261029a` only checked the code path: 3 records, 15 forwards, blocked fraction 0.45).

## Protocol (read from the official repository analysis/effective_pathway_analysis.py; the code was read, not executed)
Block everything except the effective pathway: cross-frame outside layers 6-15, video->question outside 6-20, question->last outside 16-25 (LLaVA-NeXT-7B, 32 layers, 1-based), video->last in all layers; the paper also blocks video->video after L20, question->question after L25 and last->last. Random control: same number of cells at random. Paper: 58% of edges suppressed in LLaVA-NeXT-7B-Video-FT with marginal accuracy loss on TVBench / TOMATO.

## Our transfer (deviations, all stated)
- Model Qwen2.5-VL-7B (28 layers, 4-bit): ranges rescaled round(i*28/32): cross-frame 6-14, video->question 6-19, question->last 15-23 (receipt: scaled_layers_1based).
- video->video, question->question and last->last (self cells; the knockout audit forbids them) are not blocked, so the suppressed share is ~45%, not 58%.
- Random control: per layer the same number of causal off-diagonal cells as the pathway arm (layer average), uniformly from the region after the first video token; 3 seeds per video.
- Cohort: the same 400 TVBench videos (4 tasks x 100, 8 frames) used in the protocol runs; all videos are scored (accuracy over the cohort). The layer ranges were fixed from the paper, not tuned on these videos (the WP0 peaks on the same cohort were NOT used).

## Prespecified criteria (video = unit, paired bootstrap 10,000 resamples, seed 20261029, 95% percentile CI)
- P1 (retention): accuracy(pathway) - accuracy(baseline) has CI lower bound > -3 percentage points.
- P2 (control): accuracy(pathway) - mean-random accuracy has CI lower bound > 0 and point estimate >= +5 points.
- P3 (reporting only): measured suppressed share; chance accuracy (mean 1/k); per-task results; exact McNemar baseline vs pathway.
A failed P1 means Qwen2.5-VL does not retain accuracy under the paper's (rescaled) pathway; it does not refute the paper for its own models. No post-hoc change of ranges on this cohort.

Run: `.venv/bin/python scripts/run_tvbench_pruning.py --mode full --n 400 --per-task 100 --tvbench-root data/private_tvbench --frames 8 --user-authorized-local-only --output data/private_nextgqa/tvbprune_full_20261029b.json`

## Result (full run `tvbprune_full_20261029b`, 400/400 videos, 2000 forwards, 4-bit Qwen2.5-VL-7B, summary results/tvbprune_summary_20261029b.json)
Analysed once, as prespecified. Suppressed share (all layers, causal cells) 45.1% (paper 58% with video->video etc. also blocked).
| arm | accuracy | chance |
|---|---|---|
| baseline | 57.5% | 37.5% |
| effective pathway only | 53.5% | |
| random, same cell count (mean of 3) | 45.8% | |
- Pathway - baseline: -4.0 pt, 95% bootstrap CI [-8.5, +0.5]; exact McNemar 49 vs 33 discordant, p=0.097. **P1 not met**: non-inferiority within 3 pt is not shown (the interval also does not exclude zero, so neither "retained" nor "lost" is established).
- Pathway - random: +7.7 pt, CI [+2.9, +12.5]. **P2 met**: the rescaled pathway keeps more accuracy than the same number of random cells.
- Random - baseline: -11.7 pt, CI [-16.2, -7.0].
- Per task (n=100 each, EXPLORATORY, not prespecified, no multiplicity correction): Action Sequence 63->64, Scene Transition 78->84, Object Count 46->39, Moving Direction 43->27 (chance 25%). The overall drop is driven by Moving Direction.
Interpretation (observation vs inference): about 45% of attention cells can be removed with a point-estimate drop of 4 pt, clearly better than random removal, so the paper's qualitative claim (effective pathway beats random) transfers; the "marginal drop" claim is not established for Qwen with rescaled ranges at this N (CI reaches -8.5). Whether the Moving Direction drop reflects a different pathway for that task is a hypothesis for a development/test split (roadmap R4), not a finding.
Limits: ranges rescaled, not tuned; video->video, question->question, last->last not blocked; 4 of 5 tasks; no TOMATO; same cohort as other TVBench runs; not explanation faithfulness.
