# WP0 result: Map the Flow pathway knockout replicated on Qwen2.5-VL-7B / NExT-GQA (2026-10-23)

Plan (written before the run): [2026-10-23_pathway_knockout_replication_plan.md](2026-10-23_pathway_knockout_replication_plan.md). Base paper: Kim, Kim & Han, *Map the Flow*, ICLR 2026 (arXiv:2510.13251).
Label-free; scope = attention-pathway dependence of the predicted answer, not explanation faithfulness.

## Provenance
NExT-GQA test split, hash ranks 1-80 (outcome-blind), requested 80 = recorded 80 videos, 0 failures; 55 baseline-correct videos analysed (as the base paper analyses correctly answered samples); 2,060 forwards (1 + 36 per correct video;
3 more for the 3-video smoke), 180 min on the RTX 5070 Ti; Qwen2.5-VL-7B-Instruct (cc594898...), 4-bit/BF16, eager attention, 32 frames, fp32 readout, k=9 layer windows centred at l in {2,5,...,26}; per-video appends to a partial JSONL (crash tolerant).
Receipt SHA/paths stay local; summary: `results/pathway_knockout_wp0_summary_20261023_40.json`; analysis `scripts/analyze_pathway_knockout.py` (unit-tested; its JSON-output bug found at the end of the run was fixed before the summary).

## Result: all four prespecified criteria met ("replicates", 4/4)
Mean relative change of the predicted option's probability (%p), 95% t interval, n = 55 videos (centre l = middle of a 9-layer window):
| edge | strongest centre | mean %p [95% CI] | median | predicted option changes |
|---|---|---|---|---|
| cross-frame (V group g cannot see earlier groups) | l=5 (layers 1-9) | -14.6% [-20.3, -8.9]; l=14: -8.0%; l>=20: ~0% | -5.0% | 5-11% (l<=14) |
| video -> text rows (V2T) | **l=14 (layers 10-18)** | **-39.7% [-50.5, -28.9]**; l=11: -33.2%; l=17: -24.5%; l>=23: ~0% | -41.0% | 33-36% (l=11,14) |
| video -> last (V2L) | l=5 | -0.4% [-0.6, -0.1]; all centres within +/-0.5% | ~0 | 0% |
| text rows -> last (T2L) | **l=17 (layers 13-21)** | **-92.4% [-95.4, -89.5]**; l=20: -73.1%; l=23: -47.4%; l=26: -56.2%; also l=5: -52.4% | -95.7% | 13% (l=17), 45% (l=26) |
R1 (cross-frame early vs late) met: most negative early mean -14.6% vs late mean ~0.1% (gap 14.5 pp >= 10); R2 (V2T strongest centre in 5-20 and meaningful): l=14; R3 (T2L strongest centre >= 14 and meaningful): l=17;
R4 (V2L smaller than V2T): met. "Meaningful" = mean <= -5% with the 95% interval below 0.

## Interpretation and limits
- Same profile as the base paper's model family: video information reaches the answer through the text rows in the middle layers (V2T, strongest around layers 10-18), the last position reads the text rows in middle-to-late layers (T2L), and there is essentially no direct
  video -> last route. This also explains the earlier key-bias findings: edits at the final row only were inert, edits at all post-video rows worked.
- Differences from the base paper: cross-frame blocking matters much less in Qwen2.5-VL (-6% to -15% mean with medians of -1% to -5%, heavy-tailed) than the large accuracy drops reported for LLaVA-NeXT-Video-FT; Qwen2.5-VL merges two frames per temporal group and was
  not fine-tuned in this study, and the metric differs (%p change of the predicted option vs accuracy). The cross-frame edge here is our definition (all earlier temporal groups).
- T2L %p is dominated by the loss of the option-label format (relative drops of -90% with only 13% prediction changes at l=17): probabilities of all five option tokens fall, so this edge's large %p values must not be read as "answer-specific" evidence; l=5 (-52%) shows an
  early-layer dependence of the last position on the text rows as well.
- Thinned centres (stride 3), one cohort of 55 analysed videos, T = question + options + template tokens, NExT-GQA multiple choice, k=9 windows (narrow windows can be bypassed per the base paper). Test videos were used by earlier unrelated experiments (ASSUMED-CLEARED).
- Faithfulness status: 直接評価 `未実施`; this is an attention-pathway measurement.

## Next action
WP1 (evidence-specific pathway knockout: annotated evidence keys vs equal-count sham on the V2T and V2L edges, band layers 6-20 and all layers) is queued (plan: 2026-10-23_evidence_pathway_knockout_plan.md); the V2T band is the WP0-supported route to test.
