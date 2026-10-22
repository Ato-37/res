# Prespecified plan: fresh-cohort replication of the multi-query key-bias design (written 2026-10-23, before screening/preflight/any run on this cohort)

## Why
On the N=149 cohort the multi-query suppression control (-ln256, all layers, all post-video rows) lowered the answer margin more
than the equal-count sham (mean -0.515 nat), after last-row controls had failed. That cohort was also used to design the dose
sequence, so a fresh cohort is needed before treating the finding as established. Label-free; scope = answer-margin dependence on
the QA time annotation, not explanation faithfulness.

## Cohort (fixed now, outcome-blind)
- NExT-GQA val, hash-ranked outcome-blind candidates 1-400 (builder run 2026-10-23, `annotation_key_bias_val_replication2_candidates_20261023_01`).
  Excluded: the 1,034-item exposure union, the 40 earlier-cohort media, the 6 sham-ineligible media from the N=149 screen and all 149
  media used in the N=149 runs (exclusion manifest `known_media_sha_union_val_rep2_20261023_01.json`, 1,229 hashes). Fresh pool = 205 media.
- Eligibility (structural): annotation groups <= 8 of 16 temporal groups (equal-count disjoint sham must exist), applied by
  `scripts/screen_annotation_key_bias_sham_eligibility.py`. N = number of eligible media in the pool (<= 205), no other filtering, no replacement.

### Pre-outcome adjustment (recorded 2026-10-23, before any scoring)
The cohort freezer reported `requested_205_eligible_samples_but_found_198`: 7 of the 205 fresh media fail its own structural eligibility
(strict interval / decode checks), so the pool is 198. The screen is therefore run with pool-n 198 and N = eligible media after the
sham-geometry screen (<= 198). Nothing else changes; no outcome data exist.

## Runs (same settings as the N=149 multi-query runs: 4-bit/BF16 compute, fp32 readout, layers 1-28, post-video rows, 32 frames, seed 20261021)
- A (control replication): bias -ln256.
- B, C (positive steering): bias +ln4 and +ln16.

## Prespecified decisions
- A replicates the control if the annotation-minus-sham mean contrast is negative, two-sided t-test p < 0.05 and |mean| >= 0.025 nat.
- B and C: Holm over the 2 doses (alpha 0.05) for effect; TOST at +/-0.025 nat for equivalence (needs p < 0.025). B and C are read as
  informative about selectivity only if A replicates; otherwise they are reported as uninformative.
- Primary reporting regardless of outcome: mean, SD, 95% CI, p, TOST, sign counts, predicted-choice flips, requested/effective N, failures.
- No re-runs, no dose changes, no pooling with the N=149 cohort in any test (a descriptive side-by-side is allowed).
