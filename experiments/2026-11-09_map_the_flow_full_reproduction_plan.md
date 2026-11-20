# Map the Flow: full-protocol reproduction plan (2026-11-09)

Request (user, 2026-11-09): unify the model with the paper and run a complete reproduction "so that exactly the same conclusions are obtained".
Standing rule: the conclusions cannot be engineered. This plan fixes the protocol and the match criteria BEFORE any new run; the outcome (match / partial / mismatch) is reported as observed, and a mismatch is investigated (data subset, sample size, correct-only filter), not tuned away.

## What is already unified (verified from logs)
- Model: official checkpoint LLaVA-NeXT-7B-Video-FT (bf16, 32 layers) and official code (byminji/map-the-flow, fast copy with mask_num_heads=1, checked identical on 2 questions), isolated venv ~/external/mtf-venv. The Qwen2.5-VL (4-bit) work is a separate cross-model port and is NOT part of this reproduction.
- Done with the official code (see 2026-10-29_official_code_reproduction_plan.md, 2026-10-30_official_code_scaled_run.md, 2026-11-03_llava_layer_window_curve_result.md): cross-frame, vql-to-ql (window 5 and 9, 40 q/task), I1/I2 (layers 1-16 / 17-32), effective-pathway pruning (all 1002 q, 4 tasks), question+options->last and vq->true-option (25 q/task), Logit Lens (Action Sequence).

## Remaining deviations from the paper
| # | Paper | Ours | Effect |
|---|---|---|---|
| 1 | 5 TVBench tasks incl. Action Antonym | 4 tasks; Action Antonym needs NTU RGB+D videos (advisor approval pending; ~/external/ntu_zips holds only TVBench.zip) | blocked until videos exist |
| 2 | (sample size per figure: check paper/appendix; not yet confirmed) | 40 q/task (curves), 25 q/task (C, D), 13-30 correct q/task (I1/I2) | wide intervals, task-dependent mismatch (I1 not met for Action Sequence/Moving Direction) |
| 3 | base ImageLLM comparison (llava-v1.6-vicuna-7b-hf) | failed twice at load_pllava (tokenizer.json not parseable by tokenizers 0.19.1) | "video model > base model" part unverified |
| 4 | Logit Lens on LLaVA-NeXT-13B, Action Sequence, keyword lists from the prompt | official script patched locally (undefined args), 7B, 30 q | temporal share tiny (0.09 -> 0.35 %), direction only |
| 5 | Fig 6/E attention maps | not reproduced | - |
| 6 | Fig 9/10 generation-probability tracing | not reproduced | - |

## Protocol (fixed now)
1. Same code, same checkpoint, same prompt format (conv_mode eval_mvbench, pooling 8-12-12), window k=9 centre, correct-only samples, official seed 42 sampling; sample size is the only free scale and is reported per figure.
2. Match criteria (set before the run, descriptive; no hypothesis test unless a prespecified one is added): (a) early/middle-layer dependence of cross-frame blocking > late layers; (b) video->question and question->last blocking > video->last; (c) true option -> last > non-option question -> last in tasks other than Object Count; (d) effective-pathway accuracy within the paper's -0.3 pt after the same pruning, and above the random control; (e) temporal keyword share rises from early to middle layers.
3. Report effect size, bootstrap CI (video = unit) and which criteria match / do not match. No threshold, metric or subgroup is changed after seeing outcomes.

## Workload estimate (inferred from the w9 queue log, not measured on the full set)
About 1.4 min per baseline-correct video per target (window 9, stride 1, one 16 GB GPU). Full 4-task set is about 500 baseline-correct videos, so about 12 h per target and about 48 h for the four information-flow targets, serial (GPU jobs must not run concurrently). A 100-question-per-task design is about 1/3 to 1/2 of that. VRAM about 15 GB, WSL RAM 15 GiB.

## Blockers and decisions needed
- NTU RGB+D videos for Action Antonym (advisor approval).
- Scope: full 4-task set (about 2 days GPU) vs a larger fixed sample (e.g. 100 q/task).
- Base ImageLLM comparison needs a tokenizers upgrade inside the isolated mtf-venv (not the project .venv).

Status: plan only; nothing from this plan has been run.

## Update 2026-11-09 (later): Action Antonym added, TOMATO queued
- NTU RGB+D videos for Action Antonym were already in data/private_tvbench_extra/video/action_antonym (320/320 present, 1.7 GB, local-only). Linked into ~/external/tvbench_root (empty action_antonym.json backed up as .empty_backup_20261109). Dataset now has 5 tasks / 1322 questions. CAUTION: task_id numbering shifts (sorted by name): AA=0, AS=1, MD=2, OC=3, ST=4 (was AS=0, MD=1, OC=2, ST=3); existing outputs are keyed by the old numbering.
- Smoke (AA, cross-frame, w9, 3 q, output aa_smoke_20261109): ran end to end (1 of 3 baseline-correct; 9.7 min total incl. model load; 1920x1080 avi decoding is slow). Not a result.
- Running: ~/external/queue_llava_aa_w9.sh (AA, cross-frame then vql-to-ql, 40 q, w9) -> outputs/llava_curve_w9_aa_20261109.
- TOMATO (Shangguan et al. 2024): not run before. Paper uses it only for Table 3 (pruning). Official repo has a TOMATO dataset class and effective_pathway_analysis supports it. Data: 1484 q / six tasks on HF (cc-by-sa-4.0 per dataset card); videos are a Google Drive zip, NOT yet downloaded. ~/external/queue_tomato_pruning.sh is queued behind the AA queue but refuses to start until ~/external/tomato_root/{videos,jsons} exist and DATASET_PATH['tomato'] is set. Note jsons/ in the official layout may need conversion from the HF parquet.

## Update 2026-11-10: Action Antonym curves finished; TOMATO data ready
- Action Antonym, LLaVA-NeXT-7B-Video-FT, official code, window 9 (center), 40 q sampled (seed 42), baseline-correct 28/40 videos (70%). Queue finished 2026-11-09 15:28 UTC (cross-frame 12:23-13:03, vql-to-ql 13:03-15:28). Output: ~/external/outputs/llava_curve_w9_aa_20261109 (local only).
- Observed (mean relative change of answer probability, %, layers 1-based; video = unit): cross-frame peaks at L9 (-16) and is ~0 from L17; mean over L6-15 = -11.4, bootstrap 95% CI [-18.7, -4.1] (n=28). video->question: broad -7 to -11 over L8-23, ~0 from L28; mean over L6-20 = -7.8, CI [-12.3, -3.3]. Layers 1 and 32: ~0 (window clipped at the edges).
- Comparison with paper Table E(b) (7B, Action Antonym, 5-layer bins, not the same window): paper cross-frame L6-10 -11.1, L11-15 -6.3, L16-20 -0.2; video->question L11-15 -21.5. Same ordering (early/middle > late, ~0 after ~L20 for cross-frame) but our video->question is flatter and smaller in magnitude and extends later. Different window (k=9 vs. 5-layer bins) so this is a qualitative comparison only; N=28 videos, wide intervals. No test run; descriptive.
- TOMATO: videos.zip (12.1 GB, zip test OK) extracted to ~/external/tomato_root/videos (human 588, object 596, simulated 233 = 1417); HF parquet converted to jsons (1484 q). DATASET_PATH['tomato'] set in the fast copy (backup config_dataset.py.bak_20261109). Local-only; license per dataset card cc-by-sa-4.0 (not independently verified); not redistributed.

## Update 2026-11-10 04:20 JST: Action Antonym pruning (paper Table 3, TVBench column; 5 tasks now complete)
Run: official effective_pathway_analysis.py, task_id 0 (Action Antonym), all 320 q, effective-pathway-7b and random (ratio 0.58), outputs ~/external/outputs/pruning_aa_20261110 (AA effective 18:16-18:48 UTC, random 18:48-19:20 UTC, 2026-11-09). Same settings as the earlier 4-task run (official_20261030_scaled). Analysis: scripts/analyze_official_pruning_scaled.py (question = unit, paired bootstrap 10000, seed 0, exact McNemar); pooled root = symlinks of the 4-task run + AA (~/external/outputs/pooled_5task_pruning_20261110). Local-only derived summaries in session scratchpad (not yet published).
Observed (accuracy %, pathway-baseline in pt with 95% bootstrap CI):
- Action Antonym n=320: baseline 60.62, pathway 60.31, random 49.69; pathway-baseline -0.31 [-5.31, +4.69], McNemar p=1.0 (34 vs 33 discordant); pathway-random +10.62 [+3.12, +18.12].
- Pooled 5 tasks n=1322: baseline 53.33, pathway 51.36, random 43.04; pathway-baseline -1.97 [-4.08, +0.15], McNemar p=0.081 (116 vs 90 discordant); pathway-random +8.32 [+4.77, +11.80]. Paper (Table 3, 7B): 51.5 / 51.2 / 40.1.
- Per task pathway-baseline (pt): Action Sequence -2.75, Moving Direction -1.29 (baseline 25.4% = chance for 4 options, pathway-vs-random uninformative there), Object Count -3.38; all CIs include 0.
Interpretation (observation vs inference): the point estimate for Action Antonym (-0.31 pt) equals the paper's pooled -0.3, but CI width (about +/-5 pt) cannot distinguish it from losses of several points; the pooled 5-task estimate (-1.97) is closer to the paper than the 4-task one (-2.5) yet its CI still excludes the paper's -0.3 only marginally (upper bound +0.15; -0.3 lies inside). Pre-specified criterion (d) 'within -0.3 pt of baseline and above random': above random: met (CI excludes 0); within -0.3 pt: not established or refuted at this N (non-inferiority margin not prespecified; no TOST done). Limits: Action Antonym videos are NTU RGB+D (1920x1080), tasks pooled by question count (paper weighting unknown), single seed for the random arm.

## Update 2026-11-10 21:13 JST: TOMATO pruning finished (paper Table 3, TOMATO column)
Run: official effective_pathway_analysis.py, dataset tomato (HF yale-nlp/TOMATO, 1484 q, 6 tasks, all questions, no sampling), LLaVA-NeXT-7B-Video-FT, effective-pathway-7b and random 0.58; outputs ~/external/outputs/tomato_pruning_20261110 (effective 19:20-22:01 UTC, random 22:01-00:42 UTC; queue done 2026-11-10 00:42 UTC). Edges kept: 43.2% (effective), 42.0% (random control).
Observed (accuracy %, question = unit, paired bootstrap 95% CI, exact McNemar; scripts/analyze_official_pruning_scaled.py): baseline 28.37, pathway 27.09, random 22.44; pathway-baseline -1.28 pt [-3.30, +0.74], McNemar p=0.234 (124 vs 105 discordant); pathway-random +4.65 pt [+1.55, +7.75]. Paper (7B): 30.2 / 29.2 / 23.1 (pathway-baseline -1.0).
Per task (pathway-baseline pt [CI]): Count -1.02 [-5.48,3.42]; Direction -1.24 [-4.96,2.48]; Rotation -3.50 [-8.74,1.75]; ShapeTrend -3.14 [-8.52,2.24]; VelocityFrequency +2.85 [-1.90,7.62]; VisualCues 0.0 [-10,10] (n=70).
Interpretation: direction and size match the paper (small drop, above random; CI of pathway-baseline includes 0 and the paper's -1.0). Our baseline is 1.8 pt below the paper's 30.2 (preprocessing/frame sampling not verified identical). Direction: baseline 21.1% is below the random arm (26.3%) and near chance for 5-6 options, so pathway-vs-random is not interpretable there; the pooled random contrast is dominated by other tasks. No non-inferiority margin prespecified. TOMATO is used locally only (cc-by-sa-4.0 per dataset card, unverified), nothing redistributed.
