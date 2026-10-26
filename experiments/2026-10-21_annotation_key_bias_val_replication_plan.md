# Second SHA-ranked validation cohort: annotation-key bias replication plan

## Status and rationale

Prospective-within-this-workflow exploratory replication plan, written after
the first 20-video run was inspected. It is **not confirmatory** because the
primary endpoint has already been measured once. The purpose is to assess
whether the first cohort's near-zero/mixed annotation-bias-versus-sham margin
contrast is stable on a non-overlapping, outcome-unseen validation subset. No
claim of explanation faithfulness is planned.

## Fixed design

- Use the same NExT-GQA validation split, model revision, 32 decoded frames,
  layers 5–16, requested `+ln(2)` BF16 visual-key attention bias, gold answer,
  baseline-best-nongold fixed foil, and equal-count disjoint hash sham as
  `2026-10-21_annotation_key_bias_val_20_gpu.md`.
- Do not change metric, intervention sites/dose, prompts, frame count, sham
  algorithm, selection rule, failure handling, or analysis threshold based on
  the first cohort's effects.
- Scan the complete local validation archive with the same prior-exposure
  manifests and 1,034-item private raw-media SHA union. Select the **next 20**
  eligible media after the first cohort in the pre-existing deterministic
  ordering by SHA-256 of `[nextgqa-zip-candidate-selection-v1, val,
  media_sha256]`; explicitly exclude the first cohort's 20 media hashes.
  Selection uses neither gold choice scores nor this experiment's outputs.
- Keep the two 20-video cohorts disjoint and report them separately. Do not
  pool or replace rows. If 20 exact rows do not pass preflight and all 60
  forwards do not complete, treat the run as incomplete and do not shrink or
  replace the cohort.
- Primary outcome remains the row-level difference in gold-minus-fixed-foil
  answer margin between annotation-key bias and hash sham. Report the finite
  cohort mean, median, range, signs with a `1e-7` numerical-zero tolerance,
  and 5-choice accuracy by condition. An unadjusted one-sample t interval may
  be shown descriptively; no p-value, population claim, or cross-cohort pooled
  estimate will be used.

## Interpretation before observing cohort two

Positive and negative replication outcomes are both informative. A mean
contrast near zero or mixed directions would strengthen the conclusion that
the measured internal sensitivity is not a stable answer-margin improvement.
A contrast with a consistent direction would motivate an independent
preregistered study, not a faithfulness claim. Neither outcome establishes
claim-specific explanation support: the intervals are NExT-GQA QA-time
annotations, human review and clock calibration remain absent, and rights
status remains unverified. All data and results stay local.

## Planned workload and gates

The archive candidate builder streams the approximately 24 GB local ZIP and
extracts at most the next 20 videos. Outcome-blind preflight must show exactly
20 eligible, distinct rows, all 20 strict interval checks, no decode failures,
and exact disjoint sham key budgets. Then run the usual 1-row / 3-forward GPU
smoke. Only a successful source-, model-, cohort-, and score-bound smoke
permits the fixed 20-row / 60-forward run. Based on cohort one, budget about
8–15 minutes including hashing the approximately 16.7 GB model shards,
preprocessing, model loading and inference; peak VRAM was 11.09 GiB. Check
actual GPU/RAM availability before both GPU phases. Expected spend: one
archive streaming pass, one fresh 16.7 GB shard-hash per scoring phase, three
smoke forwards, and sixty full-run forwards. The result is local-only.

## Frozen first-cohort result, for context only

Annotation-key bias minus sham was +0.012500 nat mean, approximately zero
median, range [−0.125000,+0.250000], with 8 positive / 5 approximately zero /
7 negative rows at the stated tolerance. Accuracy was 15/20 baseline, 15/20
sham, 14/20 annotation bias. The unadjusted descriptive t interval was
[−0.04085,+0.06585]. This result motivates a fixed-design replication; none of
its settings will be tuned.
