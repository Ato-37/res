# Prespecified plan: QA-annotation key bias, confirmatory val cohort N=160 (written 2026-10-22T08:5xZ, before candidate extraction/selection)

## Scope
Label-free design: no human review is used or required (user instruction 2026-10-22). The estimand is dependence of the
five-choice gold-minus-baseline-best-foil margin on key bias placed on temporal groups overlapping the dataset's QA time
annotation, relative to an equal-count disjoint hash sham. It is NOT explanation faithfulness. Model Qwen2.5-VL-7B-Instruct
(revision cc594898...), 4-bit weights, BF16 compute, eager attention, layers 5-16, +ln(2) bias, 32 frames, seed 20261021.

## Fixed before selection
- Readout: fp32 lm_head matmul (`run_nextgqa_annotation_key_bias_fp32_readout.py`), identical for all conditions.
- Cohort: NExT-GQA val, N=160 unique media, outcome-blind SHA-256 hash-ranked order from the existing builder;
  the ranked list is extracted to 200 and the first 40 ranked (the two earlier val cohorts) are excluded, so rows 41-200
  are available and the first 160 eligible by the builder/preflight are used. No row replacement; failures reported.
- Primary contrast per video: annotation-bias minus sham margin (nat). Unit = video. One analysis, no interim looks.
- Practical threshold delta_min = 0.025 nat (chosen from power: N=160 reaches 80% power at delta 0.025 for SD 0.114,
  the conservative SD from the earlier bf16 cohorts; observed fp32 SD was 0.080).
- Inference: one-sided/two-sided as follows. (a) Two-sided one-sample t-test of the mean contrast against 0 with 95% CI
  (primary, alpha 0.05). (b) Equivalence (TOST) with margin +/- delta_min at alpha 0.05, reported as the answer when (a)
  is not significant. Secondary descriptives (not tested, no multiplicity claim): annotation-minus-baseline, sham-minus-
  baseline, accuracy per condition, sign counts, median. Sign-flip permutation test as a robustness check.
- Decision rules (fixed): (1) statistically significant effect: two-sided t-test p < 0.05 on the mean contrast;
  (2) practically meaningful effect: (1) holds AND |mean| >= delta_min; (3) no meaningful effect: TOST rejects both
  one-sided nulls at alpha 0.05 (equivalently the 90% CI lies inside +/- delta_min); (4) otherwise inconclusive.
  Report mean, 95% CI, SD, n, p-values and the TOST outcome regardless of which rule applies.
- Sensitivity: results are not pooled with bf16 cohorts; the exposed cohort-1 fp32 check is reported separately.

## Pre-outcome deviation (recorded 2026-10-22T09:02Z, before any scoring)
The outcome-blind preflight with N=160 failed with `requested_160_eligible_samples_but_found_155` (200 ranked media
extracted; the 40 earlier-cohort media excluded; ineligible rows leave 155). N is changed to **155** (all eligible
ranked media 41-200); everything else is unchanged. Power at SD 0.114 and delta 0.025 is ~78% instead of 80%. No outcome
data existed, so the change is not outcome-dependent. Receipt names use `val155`; the plan file keeps its `val160` name.

## Second pre-outcome deviation (recorded 2026-10-22T09:23Z, before any scoring)
The N=155 preflight failed with `annotation_sham_inputs_invalid`: some candidates have QA annotations covering more than
half of the 16 temporal groups, so no equal-count disjoint sham exists (a structural requirement of the design, not an
outcome). Eligibility criterion added: **annotation groups <= 8 of 16**. `scripts/screen_annotation_key_bias_sham_eligibility.py`
applies it outcome-blind (no model, no scores) to the ranked candidates 41-200, adds ineligible media hashes to a new
exclusion manifest, and the confirmatory N becomes the number of eligible candidates E (<=155). The estimand population is
therefore videos whose QA annotation covers at most half the clip. A chained driver (`scripts/run_annotation_key_bias_confirmatory_pipeline.sh`)
runs screen -> preflight(E) -> smoke -> full -> analysis with no analyst intervention. Power for E < 163 is below 80% at
delta 0.025 for SD 0.114; the TOST/inconclusive rules cover that case.
