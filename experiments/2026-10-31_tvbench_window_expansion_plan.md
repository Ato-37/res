# 2026-10-31 WP2-TVB: window-width expansion (amount vs location) — prespecified plan

Status: PLAN (written before any WP2 outcome is observed). Runner: `scripts/run_tvbench_window_expansion.py`. Analysis: `scripts/analyze_window_expansion.py` (to be written before the full run).

## Why (what the previous experiments could not separate)
WP1-TVB (200 requested, 111 analysed, 64 baseline-correct): V2T layers 6-20, relative change of the predicted-option probability: whole video -37.05 % [-43.5, -30.43];
annotated interval -5.58 % [-12.11, +0.59]; equal-count sham +0.09 % [-3.0, +3.15]; evidence - sham -5.67 % [-14.16, +2.43] (not distinguishable from 0; margin-based P1 rule not met).
WP1S (NExT-GQA) slid a window of the *same width* over all positions: it varies *location* only. Neither design varies *how much* is blocked, so
"evidence interval is a small share of the video effect" cannot be separated from "blocking less simply does less". WP2 varies the amount and the location jointly.
Also noted (verification of an earlier statement): on NExT-GQA the evidence effect (-9.2 %) is about 22 % of the whole-video effect, about equal to the evidence share of keys (~22 %), i.e. consistent with a pure amount effect.

## Design
- Model/readout/cohort: same as WP1-TVB (Qwen2.5-VL-7B 4-bit bf16, sdpa, 32 frames = G=16 temporal groups, TVBench Action Sequence, 2 options, fp32 readout, cohort ranking `tvb-ev-v1`, N=200 requested, full video).
- Edge: video->text only (video->last was equivalent to zero in WP1 and WP1-TVB). Three layer conditions, per the user's request (2026-10-31: expand the window on the non-effective part and on the whole, not only on the effective pathway):
  - `eff` (PRIMARY): layers 6-20, the effective pathway found in WP1; up to 8 starts per width.
  - `noneff` (secondary, control): the same edge in the layers OUTSIDE 6-20 (1-5, 21-28); up to 4 starts per width. Expected ~0 if the effective pathway is where the dependence sits.
  - `all` (secondary): every layer 1-28; up to 4 starts per width.
  Each condition also has a whole-video arm (w = G).
- k = number of annotated evidence temporal groups. Widths w in {k, 2k, 4k, 8k} capped at G-1 (de-duplicated), plus the whole video (w = G) as positive control.
- For each width w < G: contiguous windows of w groups over start positions 0..G-w; if more than 8 starts, 8 evenly spaced starts plus the evidence-centred start. Per window record overlap o = |window ∩ evidence groups|, %p change (predicted option, relative to baseline) and gold-foil margin change.
- Eligibility: annotation groups valid (1 <= k < G). No equal-count sham is needed, so the ~45 % exclusion of WP1-TVB (no disjoint sham) is expected to shrink; any exclusion is reported.
- Forwards/video ~30 (smoke: 90 forwards for 3 videos, 1.87 s/forward) => ~56 s/video; 200 videos ~ 3 h. Smoke (3 videos, completed, outputs deleted as not for analysis) passed: hooks fire once per layer in every condition.

Post-smoke note (3 videos seen, no inference drawn): windows that include temporal group 0 sometimes produced very large drops (e.g. -93 %) while same-width windows elsewhere did ~0. A position-0 effect is therefore a likely confound; it is added as a pre-declared SENSITIVITY covariate (indicator 'window includes group 0') in a second regression. The primary regression (w/G, o/k) is unchanged.

## Primary analysis (one, video = unit, no interim looks)
Per baseline-correct video, OLS of window %p change on (a) blocked width w/G and (b) evidence coverage o/k over all non-whole-video windows (both in [0,1]).
Primary estimate = mean over videos of coefficient (b): the extra drop when 100 % vs 0 % of the evidence is covered **at equal blocked amount**; evidence-specific if < 0.
Test: one-sided-by-hypothesis reported two-sided (paired t, sign-flip, Wilcoxon), bootstrap CI (10k, seed 0). Practical threshold delta_min = 6.7 pp, = 0.2 x SD (33.7 pp) of the per-video evidence-minus-sham %p contrast in WP1-TVB (standardised smallest effect of interest, derived from data collected before WP2; the noise is large so power is limited, N baseline-correct expected ~ 100).
Decision rule: "evidence-specific" iff mean(b) <= -6.7 pp and two-sided p < 0.05 on all three tests; "no evidence-specific effect beyond amount" iff TOST at +-6.7 pp rejects both sides; otherwise inconclusive. Positive control: whole-video arm mean <= -20 %.
Secondary (descriptive): mean %p vs width for windows covering the evidence vs windows with zero overlap; fraction of the whole-video effect reached at each width; coefficient (a).

## Limitations
Contiguous windows only; TVBench Action Sequence has only 2 options (chance 50 %); 4-bit model; one edge/band; position confound (evidence start position varies 0-12) handled only by using all positions; amount and overlap are correlated by design at small w (reported via the regression). Not explanation faithfulness: dependence on the annotated interval.

## Queue addendum (2026-10-31, user request: do not leave the GPU idle; supplement the low-N items)
Batch A (200 videos) yields only 64 baseline-correct videos in WP1-TVB, so power is limited. TVBench Action Sequence has 336 eligible unique videos; batch B = the 128 not in A (`--skip 200`, disjoint, same ranking `tvb-ev-v1`).
`scripts/queue_tvbench_batchB_20261031.sh` runs serially after batch A of WP2: smoke of the skip path (3 videos) -> WP1-TVB batch B (128) -> WP2-TVB batch B (128).
Rule: batch B is analysed with the SAME prespecified analyses as batch A and reported as a separate replication; no pooling with A and no change to A's primary decision after seeing B (AGENTS.md section 5: no batch-by-batch pooling). Estimated GPU time: WP1-TVB B ~0.5 h, WP2-TVB B ~2 h.

## Priority queue (2026-10-31, user decision): Action Localization first, Logit Lens v2 second
The remaining timed TVBench task, Action Localization (160 questions, 4 options, `start`/`end` annotation; videos downloaded 2026-10-31, sizes match the Hugging Face listing; 160 files extracted to `data/private_tvbench_extra/video/action_localization`), is run with the SAME WP2 design (`--task action_localization`; readout over the 4 option letters; foil = best baseline non-gold option; chance 25 %).
Run after batch A, as a separate dataset/replication (never pooled with Action Sequence). Same prespecified primary analysis and delta_min rule (6.7 pp, derived from WP1-TVB Action Sequence; limitation: not re-derived for this task, the SD may differ).
Caveat specific to this task: the answer options are descriptions of WHEN the action happens (beginning/middle/end/throughout), so the annotated interval is the content the answer is about; a large evidence-specific effect would be expected a priori, which makes it a weaker test of "evidence use" than Action Sequence.
Order: `scripts/queue_tvbench_priority_20261031.sh`: Action Localization (smoke -> full, n=160 with fallbacks) -> Logit Lens v2 (smoke -> full) -> batch B (Action Sequence). Supersedes `queue_tvbench_batchB_20261031.sh` (stopped before it started).
