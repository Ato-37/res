# 2026-10-31 Audit of the interim-presentation deck (as a research talk) and preparation of improvements

Scope: the Google Slides deck (22 slides after today's edits). Findings are split into (1) slide structure, (2) design, (3) the experiments themselves. Every item has a proposed fix and its status. Nothing here is a faithfulness claim.

## 0. Items previously flagged unverified / to be careful about: status
| Item | Status 2026-10-31 |
|---|---|
| Paper's 5 TVBench tasks | VERIFIED from the official README (task ids 0 Antonym, 3 Action Sequence, 5 Moving Direction, 6 Object Count, 8 Scene Transition); Antonym needs NTU RGB+D (account + request form + supervisor). |
| Official-code runs quantized? | VERIFIED: LLaVA-NeXT-7B-Video-FT in bf16 (not quantized); the Qwen port is 4-bit. |
| Qwen "effects remain in late layers" | VERIFIED in the paper's metric from tvbpp: layers 15-28 -3.5 to -9.9 % (CIs exclude 0), layers 1-14 -17 to -39 %. |
| Why 25/160 Action Localization videos were excluded | VERIFIED (sample of 70): 15 fail the frozen "uniform sampling grid tolerance" check of the frame-time decoder (variable frame-rate / timestamps), i.e. a technical filter, not content; it may favour constant-frame-rate videos. |
| Group-0 hypothesis | PARTLY SUPPORTED: videos whose evidence contains group 0 drop much more (A -14.1 %, B -28.1 % vs -2.7 %, -11.4 %); sham containing group 0 also drops in batch B (-7.7 %). It does NOT explain why batch B > batch A. A vs B is not significantly different (Welch p=0.345). |
| Why batch 2 differs from batch 1 | Compatible with sampling noise (per-video SD 34-38 %p, n=43/64; difference 6.9 %p, SE 7.2). Not explained by evidence length (5.2 vs 5.7 groups) or start (4.4 vs 4.6). |
| "Pre-defined criterion" on slide 11 | Replaced by test p-values and proportions (slide 11). |
| How the paper chose the layer ranges | STILL UNVERIFIED (read from the official code only; paper text not retrievable here: arXiv search returned nothing). |
| Logit Lens "no mid-layer emergence" | RE-EXAMINED with the linear (tuned) lens (n_test=100, n_train=40): fidelity KL(final||lens) in middle layers 8.7 nats lower than the raw lens (CI [-8.8,-8.6]); with the tuned lens the temporal-keyword probability mass rises middle vs early (+0.32 pp [0.26,0.38]) while the raw lens shows -0.04 pp; cross-frame blocking lowers the middle-band temporal mass by 0.35 pp [-0.39,-0.31]. Caveats: tuned-lens KL is still 1.5-2.5 nats in the middle layers; absolute tuned mass (~1.6 % already at layer 1) is a prior, only changes are interpretable; the raw lens' null was partly a probe artefact. Slide 9 still shows the raw lens (to be replaced). |
| OOD of attention masking | UNRESOLVED (see section 3). |

## 1. Structure of the talk
1. Contribution is not stated as a claim with evidence. Slides 10-12 now give a negative/inconclusive answer (evidence interval: not distinguishable from amount) -> the headline should be framed as a question with an answer: "Does the answer depend on the annotated interval beyond the amount of video blocked?" Fix: add a one-slide "research question and answer" before the results, and a conclusion slide that states the answer.
2. Two datasets (TVBench main, NExT-GQA appendix) and three models (paper's LLaVA, public-code LLaVA, Qwen) appear across slides without a map. Fix: one "what was run where" table (task x model x experiment) near slide 6.
3. The related-work position is missing (what the paper did vs this work is on slide 13, but no other work on interval/masking faithfulness, ROAR/deletion/OOD).  Fix: one slide.
4. 22 slides for a 5-10 min talk: main body should be about 12; appendix the rest. Candidates for the appendix: slide 7 (layer table, keep one figure), 8, 14.
5. Slide 9 (Logit Lens) still shows the superseded raw-lens result; the order introduction -> flow -> methods -> reproduction -> addition is good (reproduction first).
6. Limitations (slide 16) and future work (slide 17) are generic; they should list the three concrete limits found today: masking OOD, small N (43-121), exclusion of ~45 % of videos for lack of an equal-count sham.

## 2. Design / readability
- Slides with 4-5 line captions under tables (6, 9, 12, 20) are dense; the layout shifts when text wraps (several overlaps were found and fixed today: 11, 12, 20). Fix: fixed-size text boxes with a measured maximum, one automated render check per edit (done manually so far).
- Numbers are given with different units on neighbouring slides (%, pp, nat in the appendix). Fix: show only % / pp on main slides; nat only in the appendix.
- Terms: "根拠区間", "比較用の区間" are explained on slide 10; "窓", "同じ量で比べて", "先頭の区間" (slide 12) are not defined. Fix: add a small schematic (16 groups, window sliding, group 0 highlighted) on slide 12.
- Colour: red is used for the key rows (good); the table style is consistent across 6, 11, 12, 14.
- Title slide contains the lab logo + name only; the talk title/date are missing.

## 3. The experiments themselves
1. OOD of the intervention (largest threat): masking creates unseen inputs; the amount-effect (larger windows drop more) cannot separate lost information from distribution shift. Countermeasures (not yet run):
   a. Frame removal: delete the evidence frames (and an equal number of control frames) from the input and re-encode a shorter video - in-distribution for the model; compare evidence-removal vs control-removal and vs masking. Expected cost: ~1 forward per arm, Action Sequence 200 videos <= 1.5 h. Needs a runner (`run_tvbench_frame_removal.py`) and a plan.
   b. Replacement/patching: replace the evidence group's keys/values by those of the same group from another video (resample ablation) - keeps magnitudes in-distribution.
   c. Soft attenuation: scale attention logits by a factor (e.g. 0.5, 0.25, 0) to see a dose-response that is not all-or-none.
   d. Amount-matched controls with position control: windows matched on width AND on containing group 0 or not (done post hoc in the regression, should be built into the design).
   e. Report OOD-sensitivity honestly: the paper's own metric is also masking-based.
2. Position confound: the evidence interval often starts at the video start (27/111 start at group 0), and windows containing group 0 drop more (attention-sink-like). Fix: stratify sham and windows by group-0 inclusion (design), or exclude group 0 from the sham comparison.
3. Statistics: the earlier "criterion" (margin <= -0.3 nat) was self-chosen; use tests + effect sizes + proportions on the displayed metric (slide 11 done); correct for the two batches (Holm); an equivalence margin should be justified from data (delta_min = 0.2 SD was used in WP2).
4. Sample size: 64 + 43 baseline-correct videos for WP1-TVB, 121 + 83 for WP2; large per-video SD -> wide CIs. Fix: more videos need more tasks with annotation (only Action Sequence and Action Localization have start/end).
5. Exclusion (~45 %) selects short-evidence questions; Action Localization excludes 25/160 for decoder-tolerance reasons. Fix: report the retained/excluded counts on the slide and show that the conclusions are the same on the unfiltered set where possible.
6. The Qwen port (4-bit, 28 layers, ranges rescaled from the paper) is not the paper's model; the layer ranges are not tuned for Qwen (FINE shows cross-frame effects in layers 7-17, consistent with the rescaled 6-14). Fix: derive Qwen ranges from the single-layer sweep before using them for the effective pathway.
7. Image-model comparison (official run G) failed twice at tokenizer loading and is not done.
8. Logit Lens v2 results are available (section 0); the lens is probe-limited (4-bit, 40 train videos); a larger train set would be the next step.

## Correction 2026-11-01
The Qwen layer ranges used by `run_tvbench_pruning.py` (`scaled_layers`) are cross_frame 5-13, video->text 5-18, text->last 14-22 (1-based; recorded in the receipt). Earlier notes and slide 5 quoted 6-14 / 6-19 / 15-23, which was off by one layer; slide 5 and its speaker notes were corrected on 2026-11-01 (all 24 slides re-rendered and audited; slide 19 updated with the Qwen-derived-range result, slide 10 caption spacing adjusted). See `experiments/2026-11-01_tvbench_pathway_search_stage2_result.md`.

## Overall revision 2026-11-01 (deck now 27 slides)
Added appendix slides 25-27 (frame-removal check; Qwen's own layer ranges, stage 1/2; Logit Lens on the evidence interval), each with speaker notes; slide 18 (limits) now states that frame removal gives the same direction as masking; slide 19 (future work) reports the Qwen-derived-range result. New slides checked after rendering (overflow at 18pt fixed by using 16pt and 4pt paragraph spacing); slides 1-24 were rendered in the earlier pass today and are unchanged apart from slide 18.
Numbers: frame removal from results/tvbench_frame_removal_{A,B}_summary_20261031.json; Qwen ranges from results/tvbench_pathway_search_summary_20261101.json and results/tvbench_pruning_qwenranges_summary_20261101.json; lens evidence from results/tvbench_lens_evidence_summary_20261031.json.

## Revision 2026-11-01 (pooling, p-values)
Slide 12: single column (107 baseline-correct videos, both collection batches pooled post hoc; per-batch values kept in speaker notes), added a plain-language p-value note and an all-video reference row (-38.5% [-43.5, -33.3], computed from the two evidence receipts). Slide 14: Action Sequence batches pooled (204 baseline-correct) via `scripts/analyze_window_pooled.py` -> `results/tvbench_window_expansion_pooled_summary_20261101.json` (amount -29.75 [-51.18, -9.04] p=0.006; evidence coverage -20.92 [-46.2, +3.61] p=0.105; group-0 -10.49 [-15.62, -5.38] p=0.0001; post-hoc pooling, descriptive). "Exploratory" wording replaced with "not conclusive" on the slides. Slide 18 count range updated (53-204). All 27 slides re-rendered after the edits.
