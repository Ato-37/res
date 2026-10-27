# Queue h results (2026-10-26 12:11 -> 2026-10-27 02:23 UTC), sdpa, all analyses per their prespecs

Provenance: master_queue_20261026h.log, post_queue_20261026.log (local); each stage = smoke (n=3) -> full -> analysis. 0 failures. Scope: answer margin / accuracy on NExT-GQA, Qwen2.5-VL-7B, one model; none of this is explanation faithfulness (WP2 is the first step toward it).

| stage | cohort | result | verdict |
|---|---|---|---|
| PERM (5 option-order shifts, mean) | val 300 | 0.740 -> 0.763 (+2.3 pt, +14/-7, McNemar p=0.19); gated (pmax<0.6, 14%) 0.757 (p=0.27) | not supported |
| PERM | test 278 | 0.773 -> 0.781 (+0.7 pt, +6/-4, p=0.75); gated 0.788 (p=0.22) | not supported |
| CEN (random 4-group crops x6) A=base+2 crops | val 300 | 0.740 -> 0.740 (+10/-10, p=1); gated 0.757 (p=0.125) | not supported |
| CEN | test 278 | 0.773 -> 0.784 (+11/-8, p=0.65); all 6 crops 0.791 (p=0.36); gated 0.799 (+2.5 pt, p=0.039, secondary, uncorrected) | not supported |
| WP1D-val (hash overlap dose) | val 300 | slope +0.354 nat [+0.181,+0.527], t p=7e-5; positive control passed | supported |
| WP2 (model-cited interval minus sham) | val 289/300 (11 excluded) | +0.298 nat [+0.161,+0.435], t p=2.6e-5, TOST p=0.49 | inconclusive (just under 0.3) |
| WP1S (sliding window dose) | test 112 | +0.550 nat [+0.179,+0.920], t p=0.004 | supported |
| WP1D (hash dose) | test 112 | +0.556 nat [+0.246,+0.867], t p=5.6e-4 | supported |
| selective prediction (acc@80% coverage, base vs ensemble) | val 508 (exploratory) / test 278 (prespec) | +0.5 pt [-2.0,+2.7] / +5.4 pt [+1.4,+8.6] | val no / test yes; inconsistent |
| progressive knockout V->T (WP0) | 80 (61 baseline-correct) | forward 1..N: -1.5% (4), -7.9% (8), -19.7% (12), -42.9% (16), -43.8% (20+); reverse from 28: ~0 down to 20, -6.7% (16), -39.3% (4); all 4 criteria met | met |

## Exploratory offline check (no GPU; CSL receipts, eager; not prespecified)
Ensembles of the baseline with CSL arms, accuracy vs baseline (exact McNemar). val150 (baseline 0.640): csl4 only +2.7 pt (p=0.34); base+csl4 +2.7 (+4/-0, p=0.125); base+random4 +1.3 (p=0.82); base+csl4+random4 +4.0 (+10/-4, p=0.18). test278 (baseline 0.755): csl4 only +2.9 (p=0.096); base+csl4 +1.1 (p=0.55); base+random4 +3.2 (p=0.078); base+csl4+random4 +2.2 (p=0.21); oracle +3.6 (p=0.087).

## Interpretation (inference)
- No annotation-free correction reduces errors with a confirmed effect: PERM and random-crop ensembles are null on both cohorts; the earlier crop-ensemble test result (+4.3 pt, p=0.029, eager) is not reproduced by the compute-matched random-crop control and is not clearly attributable to the self-guided component (offline arms above are all n.s.). Treat it as a single positive test result, unconfirmed; eager and sdpa baselines differ (0.755 vs 0.773 on the test cohort) so these are not directly comparable.
- Diagnosis side strengthened: three dose-response analyses (WP1D val, WP1S test, WP1D test) show monotone-direction dependence on overlap with the annotated interval; WP2 shows the model's own cited interval matters about as much as the 0.3-nat threshold (inconclusive); progressive knockout answers the "6-20 band is arbitrary" reviewer concern for this model (saturation by ~layer 16-20).
- Selective prediction: confidence-based abstention helps (base-only acc@80% gains +4-6 pt), no novelty; ensemble confidence add-on is inconsistent across cohorts.

## Next
Reviewer-concern items (attention-vs-causal score agreement, correct/wrong split of WP1, failure modes, limitations); decide whether any new correction idea is worth GPU time (open: abstain-or-defer using confidence drop under crop; answer-order + crop combination is not motivated by the null PERM/CEN results).

## Exploratory CPU check: attention-mass (SGC) vs causal (CSL) group scores (2026-10-27; same videos matched by media hash, gold and annotation groups)
Per-video Spearman between the 16 SGC attention-mass scores and the 16 CSL causal scores: val (N=150) mean -0.015 [-0.059, +0.027]; test (N=278) +0.013 [-0.023, +0.049] (paired bootstrap over videos). Top-4 overlap 1.39 / 1.42 groups vs 1.00 expected by chance (the overlap above chance coexists with a near-zero rank correlation, so it should not be over-read). Split by baseline correctness: correct -0.027 (n=96) / -0.007 (n=210); wrong +0.006 (n=54) / +0.075 (n=68). Inference: attention mass is not a proxy for causal importance here, consistent with SGC failing where CSL did not. Answers the reviewer concern "attention vs causal" for this model only.
