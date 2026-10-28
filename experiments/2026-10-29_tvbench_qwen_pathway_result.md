# TVBench pathway knockout on Qwen2.5-VL-7B (the base paper's benchmark; 8 frames) — result (2026-10-28/29)

Setup (fixed before the run, same analyzer and criteria as the earlier NExT-GQA transfer; see experiments/2026-10-28_map_the_flow_reproduction_plan.md): scripts/run_tvbench_pathway.py, Qwen2.5-VL-7B-Instruct (4-bit weights, BF16 compute, fp32 lm_head readout, eager attention), TVBench tasks Action Sequence, Scene Transition, Moving Direction, Object Count (Action Antonym excluded: its videos come from NTU RGB+D, which needs the official site; 3 Action Sequence videos failed the uniform-timing decode screen and were replaced by the next ranked), 100 videos per task (hash-ranked, one question per video), 8 frames, 2-4 options with letters A.., window k=9 centered at layers 2,5,...,26, edges cross-frame / video->text / video->last / text->last, only baseline-correct videos knocked out, %p_change of the originally predicted option-letter probability. Run in two segments (paused at 153/400 and resumed with --resume; resume verified; no duplicate videos). Receipt local-only; summary results/tvb0_summary_20261028u.json. Not the paper's models or prompt template (see the reproduction plan): this is the paper's analysis on the paper's benchmark with a different model, not a strict reproduction.

## Verified result
400 videos recorded; baseline accuracy 0.533 (Action Sequence 51/100, Scene Transition 83/100, Moving Direction 38/100, Object Count 41/100); 213 baseline-correct videos knocked out (4,459 forwards). Mean %p_change by layer centre:

| edge | l=2 | 5 | 8 | 11 | 14 | 17 | 20 | 23 | 26 |
|---|---|---|---|---|---|---|---|---|---|
| cross-frame | -4.4 | -12.5 | -20.3 | -23.5 | -19.3 | -10.8 | -1.4 | 0.0 | 0.0 |
| video->text | -3.4 | -7.9 | -10.6 | -25.3 | -35.1 | -36.4 | -18.7 | -2.0 | +0.1 |
| video->last | -0.1 | -0.3 | -0.1 | -0.1 | -0.1 | 0.0 | +0.3 | +0.3 | -0.1 |
| text->last | -2.8 | -12.7 | -2.6 | -10.5 | -12.0 | -93.1 | -90.5 | -35.8 | -38.1 |

The four prespecified criteria (R1 cross-frame early vs late, R2 video->text in the middle, R3 text->last mid-late, R4 video->last smaller than video->text) were all met -> "replicates (4/4)" on TVBench as well as on NExT-GQA.

## Interpretation (inference) and limits
The ordering reported by the paper (cross-frame interaction in early-to-middle layers, video-to-question integration in the middle, question-to-last in middle-to-late layers, a direct video-to-last route that is not used) appears in Qwen2.5-VL-7B on the paper's own benchmark, with the same window size and metric definition. Differences that remain: model and training (the paper's video fine-tunes of LLaVA-NeXT/InternVL and off-the-shelf VideoLLaMA3), 28 vs 32 layers, centers every 3rd layer rather than every layer, the prompt template (the paper's `Best option:(` template was read from the official code but not used here), four of five tasks, 100 videos per task instead of the full tasks (the paper states no counts), and no accuracy-drop table, no token-group routes (non-option question, true option), no layer-wise probability, no pruning. Per-task profiles were not analysed. The strict reproduction with the official code and checkpoint remains blocked on permission.
