# Evidence-focusing (keep-annotation) steering, Stage 1 development result (val, N=123; exploratory)

Plan: [2026-10-23_evidence_focusing_steering_plan.md](2026-10-23_evidence_focusing_steering_plan.md) (hypothesis formed post hoc from the key-bias series, so this is the exploratory stage). Label-free; scope = answer-margin
effect of oracle (annotation-defined) evidence-interval attention cropping, not a deployable method and not explanation faithfulness.

## Provenance
Cohort: NExT-GQA val, outcome-blind hash ranks 401-525 (125 fresh media), 2 media lacked a disjoint equal-count sham -> requested = effective N=123, 369/369 forwards per dose, 0 failures. Keep mode: additive bias on the keys of all temporal
groups EXCEPT the kept ones (annotation groups vs the equal-count disjoint hash sham groups), all post-video query rows, all 28 layers, fp32 readout, 4-bit/BF16 compute, seed 20261021. Doses -ln4 and -ln256, one run each. The -ln256
run was interrupted once by a PC bugcheck (see the crash investigation) and rerun from the same lock/preflight; receipts are the `_20` (-ln4) and `_30` (-ln256) full runs. Summary: `results/evidence_focusing_dev_valE123_summary_20261023_30.json`.

## Results (nat; mean [95% t CI]; video = unit)
| dose | keepAnn - keepRand (primary) | keepAnn - baseline | keepRand - baseline | correct (base / keepAnn / keepRand) |
|---|---|---|---|---|
| -ln4 | +0.221 [+0.036,+0.406]; t p=0.020, Wilcoxon p=0.051; median +0.177 | +0.482 [+0.223,+0.740]; p=0.0003 | +0.261 [+0.033,+0.489]; p=0.025 | 79 / 80 / 81 |
| **-ln256** | **+0.659 [+0.225,+1.094]**; t p=0.0033, Wilcoxon p=0.018; median +0.433; 10%-trimmed +0.442 | +0.564 [+0.121,+1.008]; t p=0.013, Wilcoxon p=0.058 | -0.095 [-0.535,+0.346]; p=0.67 | 79 / 86 / 72 |

## Gate (fixed in the plan) and interpretation
- Gate: proceed to Stage 2 only if some dose has primary mean >= 0.3 nat. **-ln256 passes (0.659); -ln4 does not (0.221).** Stage-2 dose = -ln256 (larger primary mean).
- Observation: keeping only the annotated frames (suppressing the others) raised the margin over the unmodified model and over keeping an equal-count random subset; at -ln256 the random-keep arm is about neutral, so the contrast comes from
  the annotation-keep arm improving (+0.56), unlike the earlier bias/suppression runs where the sham arm drove the contrast. Heavy tails (SD 1.0-2.4; Wilcoxon weaker than t). The dev cohort selected the dose, so these p-values are not confirmatory.
- Limits: the annotation is used at inference (oracle), the intervention is extreme and out of distribution, one model/dataset, dev cohort reused only for dose selection. Faithfulness status: 直接評価 `未実施`.

## Next action
Stage 2 confirmation on the NExT-GQA test split at -ln256 (single run, prespecified rule: t, sign-flip and Wilcoxon all p < 0.05 with mean >= 0.3 nat). The test cohort excludes the 80 videos used by the pathway-knockout replication (WP0).
