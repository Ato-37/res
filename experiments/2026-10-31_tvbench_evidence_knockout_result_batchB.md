# 2026-10-31 WP1-TVB batch B (independent replication, Action Sequence videos not in batch A)

Batch B = the 126 qualifying TVBench Action Sequence videos not used in batch A (`--skip 200`; 128 were requested, only 126 qualify). Same runner/analysis as batch A (`2026-10-31_tvbench_evidence_knockout_result.md`), analysed once. Receipt (local-only) `data/private_nextgqa/tvbench_evidence_full_batchB_20261031.json`; derived `results/tvbench_evidence_pathway_knockout_summary_batchB_20261031.json`, `results/tvbench_evidence_relative_change_summary_batchB_20261031.json`. **Not pooled with batch A** (AGENTS.md section 5).
Counts: 126 recorded, 55 excluded (no disjoint equal-count sham / budget), 71 analysed, 43 baseline-correct.
## Prespecified analysis (margin, nat)
Positive control (whole-video V2T) -40.65 %, passed. **P1 (V2T band evidence - sham): -0.714 nat, 95 % CI [-1.309, -0.119]; t p=0.019, sign-flip p=0.018, Wilcoxon p=0.0099; rule (<= -0.3 nat, p<0.05, Holm) MET.** TOST not equivalent (p=0.92). P2 (V2L band): -0.005 [-0.022, +0.013], equivalent to zero (TOST p=8.5e-46). S1 all layers -0.708 [-1.317, -0.099].
Base-paper metric (relative change of answer probability %, 43 baseline-correct): whole video -40.65 [-48.5, -32.6]; evidence -13.38 [-22.9, -4.9]; sham -0.85 [-6.2, +3.6]; evidence - sham -12.53 [-24.0, -1.6].
## Comparison with batch A (separate replications)
| | A (64 baseline-correct) | B (43 baseline-correct) |
|---|---|---|
| P1 margin, nat | -0.307 [-0.717, +0.103], rule NOT met | -0.714 [-1.309, -0.119], rule met |
| evidence - sham, %p | -5.67 [-14.16, +2.43] | -12.53 [-24.03, -1.64] |
| V2L | equivalent to zero | equivalent to zero |
The two batches disagree on P1: A inconclusive, B supportive; the intervals overlap. Neither decision is changed by the other; no pooling was done. A fixed-effect combination would be an exploratory sensitivity analysis only and is not reported as a result.
## Interpretation / limits
Direction is the same in both batches (negative), sizes differ (sampling noise at N=43-64 is large; the per-video contrast SD is ~34 pp). Exclusion removes ~45 % of videos (long evidence windows) in both batches. WP2 batch A showed that the apparent evidence effect is entangled with a group-0 position effect at equal blocked amount; the group-0 analysis for batch B (WP2 B) is queued. Not explanation faithfulness.

## Addendum 2026-10-31 (user feedback): replace the ad-hoc rule by tests and proportions on the displayed metric
The slide row "pre-specified criterion met / not met" (margin <= -0.3 nat and p<0.05) was my own threshold and is not shown to the reader; the slide now reports tests on the displayed metric (relative change of the predicted-option probability, %p; per-video contrast evidence - sham, baseline-correct videos):
- Batch A (n=64): mean -5.67 %p, d_z -0.17; paired t p=0.183, sign-flip p=0.188, Wilcoxon p=0.247; evidence arm lower than sham in 35/64 videos = 54.7 % (Wilson 95 % CI 42.6-66.3 %, sign test p=0.53); evidence arm vs 0 p=0.093.
- Batch B (n=43): mean -12.53 %p, d_z -0.33; paired t p=0.039, sign-flip p=0.037, Wilcoxon p=0.026; 29/43 = 67.4 % (52.5-79.5 %, sign test p=0.032); evidence arm vs 0 p=0.0056.
- Holm over the two batch tests (t): B 0.078, A 0.183 (B is not significant after correction for two tests).
- A vs B difference of the contrast: 6.9 %p, SE 7.2, z=0.95; Welch p=0.345, Mann-Whitney p=0.212 (the batches are not distinguishable).
- Group 0 (from the stored sham/evidence group indices): evidence arm when the evidence contains group 0: A -14.1 % (n=16), B -28.1 % (n=5); when it does not: A -2.7 %, B -11.4 %. Sham arm containing group 0: A 0.0 % (n=23), B -7.7 % (n=19). Group 0 matters in both batches but does not explain why B is larger (B has fewer such videos): the A-B difference is compatible with sampling noise (per-video SD 34-38 %p, n=43-64).
