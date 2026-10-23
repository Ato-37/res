# WP1 result: evidence-specific pathway knockout on Qwen2.5-VL-7B (test split, N=112) — primary hypothesis supported (2026-10-23)

Plan (written before cohort selection and any run): [2026-10-23_evidence_pathway_knockout_plan.md](2026-10-23_evidence_pathway_knockout_plan.md). Base paper: Kim, Kim & Han, *Map the Flow* (ICLR 2026, arXiv:2510.13251);
WP0 replicated its pathway profile on Qwen2.5-VL ([result](2026-10-23_pathway_knockout_replication_result.md)). Label-free; scope = pathway-level dependence of the answer margin on the NExT-GQA annotated interval; not explanation faithfulness.

## Provenance
NExT-GQA test split, hash-ranked outcome-blind candidates 301-500 (disjoint from WP0 and the evidence-focusing cohorts), 112 structurally eligible videos (0 sham-ineligible after the pre-selection screen) -> requested = recorded = analysed N=112 (78 baseline-correct), 0 failures,
1,232 forwards (11 per video + the 3-video smoke), 70 min. Qwen2.5-VL-7B-Instruct (cc594898...), 4-bit/BF16, eager attention, 32 frames, fp32 readout, seed 20261023. Knockout = attention-mask cells set to the dtype minimum (audited: only intended cells change, each layer hit once; unit-tested).
Edge definitions (rows <- blocked key columns): V2T = text rows after the video cannot attend to the selected video keys; V2L = the last position cannot attend to them; selected keys = the annotated evidence groups (3.56 of 16 temporal groups on average) or an equal-count disjoint hash sham.
Layer sets: BAND = layers 6-20 (the base paper's video->question range), ALL = layers 1-28. Summary (no IDs): `results/evidence_pathway_knockout_summary_20261023_60.json`; analysis `scripts/analyze_evidence_pathway_knockout.py`.

## Results (margin change in nat; mean [95% t CI]; video = unit)
| quantity | value |
|---|---|
| positive control: whole-video V2T knockout, BAND, mean %p on baseline-correct videos (n=78) | **-42.6%** (rule: <= -20%) -> **passed**; whole-video V2T changes 50/112 predictions |
| **P1 (V2T, BAND): evidence knockout minus sham knockout** | **-0.681 [-0.983, -0.379]**; t p=1.9e-5, sign-flip p=3e-5, Wilcoxon p=5.8e-5, Holm p=3.9e-5; median -0.251, 10%-trimmed -0.480, 64% of videos negative |
| decomposition (BAND) | evidence knockout: -0.448 (median -0.182, 12/112 prediction flips); equal-count sham knockout: **+0.233** (median +0.106, 7/112 flips); whole video: -1.172 (50/112) |
| %p on baseline-correct videos (BAND, V2T) | evidence -9.2%, sham +2.5%, whole video -42.6% |
| **P2 (V2L, BAND): evidence minus sham** | -0.003 [-0.019, +0.014]; t p=0.75; TOST at +/-0.3 nat p ~ 1e-62 -> **equivalent to zero**; 0/112 prediction flips for both |
| secondary S1 (V2T, ALL layers) | -0.615 [-0.971, -0.258], t p=0.0009, Wilcoxon p=0.0040; evidence -0.171, sham +0.443 |
| secondary S2 (V2L, ALL layers) | +0.011 [-0.005, +0.028], equivalent to zero |
Prespecified rule P1 (t, sign-flip, Wilcoxon all < 0.05, Holm < 0.05, mean <= -0.3 nat): **met** (mean -0.681, about 2.3x the threshold).

## Interpretation and limits
- Answer-relevant information from the **annotated evidence frames reaches the answer through the video -> text-row pathway in the middle layers**, and not through a direct video -> last route: blocking the video keys of the evidence groups on V2T lowers the gold-minus-foil margin much more than
  blocking an equal-count disjoint sham (-0.68 nat), whereas the same blocking on the V2L edge changes nothing (tight equivalence). The evidence groups are about 22% of the video keys yet their blocking carries about 38% of the whole-video V2T margin loss (-0.45 of -1.17), while the sham carries none.
- This extends the base paper from whole-video pathways to **evidence-specific** pathways and to Qwen2.5-VL, with a passed positive control and a prespecified threshold.
- The contrast has two sides: blocking evidence hurts (-0.45) and blocking the sham slightly helps (+0.23), the same sham-side improvement seen in the earlier key-bias and keep-mode runs (removing random visual context can help); part of the contrast is therefore sham-side. Heavy tails (SD 1.6; median -0.25 is about a third of the mean).
- Limits: NExT-GQA annotation intervals are uncalibrated to decoded PTS and may not contain all evidence; the sham differs in position and content; two layer sets only (no layer search); one model, one split, one cohort of 112 videos (test videos were used by earlier unrelated experiments, ASSUMED-CLEARED);
  the margin endpoint is label-free dependence on the annotated interval and not explanation faithfulness (直接評価 `未実施`).

## Next action
Independent confirmation on another fresh cohort (test candidates 501-990 via the Stage 2b cohort, to be run with this design) is the natural replication; meanwhile Stage 2b (evidence-focusing) and SGC are queued.
