# 2026-10-31 WP1-TVB result: evidence-interval knockout on TVBench Action Sequence (Qwen2.5-VL-7B 4-bit)

Plan: `2026-10-31_tvbench_evidence_knockout_plan.md` (prespecified). Runner `scripts/run_tvbench_evidence_knockout.py` (full, N=200 requested, 32 frames, sdpa, 4-bit bf16 compute, fp32 readout).
Receipt (local-only): `data/private_nextgqa/tvbench_evidence_full_20261031.json` (1221 forwards, 2650 s). Derived: `results/tvbench_evidence_pathway_knockout_summary_20261031.json`, `results/tvbench_evidence_relative_change_summary_20261031.json`.

## Counts
200 requested, 200 recorded; 89 excluded (80 no equal-count disjoint sham `annotation_sham_inputs_invalid`, 9 `annotation_temporal_group_budget_invalid`); 111 analysed; 64 baseline-correct.
Exclusion is not random (long evidence windows cannot have a disjoint equal-count sham); selection limits generalisation to short-evidence questions.

## Prespecified analysis (margin, nat; one analysis, no interim looks)
- Positive control (whole-video V2T band): mean -37.05 % (%p), passed.
- P1 V2T band evidence - sham: mean -0.307 nat, 95 % CI [-0.717, +0.103]; t p=0.14, sign-flip p=0.143, Wilcoxon p=0.058; TOST(0.3) p=0.514. Rule (mean <= -0.3 nat, all p<0.05, Holm): **NOT met**. Not equivalent either => **inconclusive**.
- P2 V2L band: -0.004 [-0.018, +0.010]; TOST p=6.6e-69 => equivalent to zero (as on NExT-GQA).
- S1 (all layers) -0.293 [-0.720, +0.133] p=0.176; S2 V2L all -0.011.

## Base-paper metric (descriptive; relative change of answer probability %, baseline-correct 64, bootstrap video unit 10k seed 0)
whole video -37.05 [-43.50, -30.43]; evidence -5.58 [-12.11, +0.59]; sham +0.09 [-3.00, +3.15]; evidence - sham -5.67 [-14.16, +2.43]; V2L evidence -0.09 [-0.30, +0.12].

## Interpretation (separate from observations)
On TVBench the evidence-interval effect has the same sign as on NExT-GQA (-5.7 % vs -11.7 %) but is smaller and its interval includes 0: the NExT-GQA finding is not confirmed in this second dataset with this N. The V2L null replicates. This measures dependence on the annotated window, not explanation faithfulness.
Verification of an earlier claim (deck slides 11/12 appeared inconsistent on NExT-GQA): on the 61 videos having both a 100 %-overlap and a 0 %-overlap window in WP1S, the WP1S 100 % window gives -6.03 % and 0 % gives -0.53 %, versus WP1 evidence -7.03 % / sham +2.18 % on the same videos. The deck's -4.5 % (pooled windows, 85 videos) vs -9.2 % (78 videos) difference is largely cohort/aggregation (baseline-correct sets differ: 78 vs 85, 75 in common; 10/75 annotations non-contiguous).
Verification of the public-code model: official-code runs use LLaVA-NeXT-7B-Video-FT in bf16 (not quantized; `official_code_reproduction_plan.md`); the Qwen2.5-VL port is 4-bit.

## Next
WP2-TVB window-width expansion with eff / non-eff / all layer conditions is running (`2026-10-31_tvbench_window_expansion_plan.md`). Effective-pathway keep-and-cut dose-response (how much of the effective pathway can be removed with accuracy equivalent to baseline) is the next planned experiment.
