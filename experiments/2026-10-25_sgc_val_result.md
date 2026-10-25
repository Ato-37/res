# SGC (self-guided evidence cropping), val primary cohort: result (2026-10-25)

Plan: experiments/2026-10-23_self_guided_cropping_plan.md (fixed before any SGC run). Receipt: sgc_val_full_20261025_84.json (local-only), summary results/sgc_val_summary_20261025_84.json. Cohort: NExT-GQA val, N = 508 requested = 508 recorded, 0 excluded, 3,048 forwards (6/video), seed 20261023, Qwen2.5-VL-7B 4-bit/BF16, fp32 readout, 32 frames. Smoke (n=3) passed; attention capture worked on GPU.
Engineering notes: earlier full-run attempts (tags _80/_81/_82/_83) failed in the resource guards (host RAM held by 508 decoded videos); fixed by `LAZY_FRAMES` (re-decode per video, verified by source hash and PTS) and loading the model before the cohort freeze. No outcome data were produced by failed attempts.

## Prespecified primary results (Holm over 2; delta_min 0.3 nat)
| contrast | mean (nat) | 95% CI | p (t) | Holm | decision |
|---|---|---|---|---|---|
| P1 selfguided - random (matched budget) | +0.364 | [+0.156, +0.573] | 6.4e-4 | 1.3e-3 | supported by the prespecified rule |
| P2 selfguided4 - random4 (annotation-free budget) | +0.112 | [-0.067, +0.292] | 0.22 | 0.22 | equivalent within 0.3 nat (TOST p=0.020) |
Reference: oracle - random +0.669 [+0.455, +0.883] (positive-control logic satisfied: >= 0.3); selfguided - oracle -0.305 [-0.493, -0.116].
Localization (self-guided top-|A| vs annotation): mean overlap 1.047 vs 0.907 by chance (1.15x; prespecified gate 1.5x not met); top-1 hit 0.301 vs 0.203 chance.

## Exploratory follow-ups (not prespecified; post-selection, for interpretation only)
- The "random" comparator of P1 is the hash sham **disjoint from the annotation**, i.e. systematically evidence-poor: random - baseline = -0.114 nat, while the non-disjoint random4 - baseline = +0.265. Against random4, the matched-budget selfguided arm gives -0.014 [-0.216, +0.188]. So P1's support is largely produced by the comparator being anti-evidence, not by self-guided selection beating chance.
- Localization is near chance where the budget is large (|A|=4: overlap 1.11 vs 1.00; |A|=5+: 2.59 vs 2.54) and above chance only for |A|=1-2 (0.16 vs 0.06; 0.49 vs 0.25).
- Accuracy does not improve: correct counts out of 508 (baseline 345): oracle 347, random 308, selfguided 326, selfguided4 336, random4 338. Net flips vs baseline (wrong->correct minus correct->wrong): oracle +2, random -37, selfguided -19, selfguided4 -9, random4 -7.
- Mean margin gain over baseline: selfguided4 +0.377 vs random4 +0.265 (difference +0.112, CI includes 0); on baseline-wrong videos (n=163) selfguided4 +0.609 vs random4 +0.716.

## Interpretation (inference) and status
- Under the prespecified rule P1 is supported, but I judge this not to be evidence of annotation-free evidence localization: the comparator is disjoint from the annotation, the non-disjoint comparison is null, localization is only 1.15x chance, and no arm improves accuracy. The honest summary is "attention-based self-selection does not reliably find the annotated evidence; the margin gains of cropping are mostly a context-reduction effect shared with random 4-group keeps".
- Next prespecified steps: test replication of P1 (running, tag 20261025_85; will be reported with the same caveat, plus the random4 comparison added as a pre-announced secondary); CSL. CSL workload is ~22 forwards/video (~2 min/video, ~17 h for 508), not the 2-3 h in the plan; use a prefix subset or decide with the user.
- Scope: answer margin on NExT-GQA, label-free; not explanation faithfulness (補助診断のみ).

## Update 2026-10-26: SGC test replication (N=278, one prespecified analysis; results/sgc_test_summary_20261025_85.json)
P1 selfguided - random +0.225 nat [-0.043, +0.493], t p=0.0995, Holm p=0.199 -> inconclusive (not replicated). P2 selfguided4 - random4 -0.060 [-0.287, +0.166], TOST p=0.019 -> equivalent within 0.3 nat. Oracle - random +0.480 [+0.204, +0.755] (supported; the Stage 2b result again). Localization: overlap 1.029 vs 0.929 by chance (1.1x); top-1 hit 0.266 vs 0.210. Correct counts of 278: baseline 210, oracle 214, random 199, selfguided 212, selfguided4 213, random4 216.
Conclusion: annotation-free attention-based evidence selection is not supported on either cohort; the val P1 support did not replicate and the val exploratory reading (comparator artefact) stands.
Prespecified crop-ensemble confirmation (experiments/2026-10-26_crop_ensemble_prespec.md) on this receipt: baseline 0.755, ensemble (baseline+S4+R4) 0.799, +0.043, wrong->correct 19 vs correct->wrong 7, exact McNemar p=0.029: supported under that rule (p<0.05 and >= +0.02). Caveats: 3 forwards vs 1, not compute-matched; the attribution to crop diversity is tested in experiments/2026-10-26_crop_ensemble_compute_matched_plan.md.
