# QTA v4 full GPU run `_03_01` (2026-10-18)

## Purpose and setup

Execute the frozen four-condition annotation-union dependence proxy on the
147-video outcome-blind geometry-feasible NExT-GQA test-associated cohort.
For each row, `baseline_full` fixes the predicted A–E answer and highest
probability foil. Those same two choices are scored under baseline-full,
baseline-union-mask, QTA-full, and QTA-union-mask conditions. The primary
finite-cohort contrast is
`(margin_intervention_full - margin_intervention_union_mask) -
(margin_baseline_full - margin_baseline_union_mask)`.

The run used the current repository `.venv`, pinned checkpoint revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, Qwen2.5-VL-7B-Instruct with
4-bit weights, BF16 compute, eager attention, and seed `20261018`. The same
frozen 147 rows and four forward conditions were used for every record. A
preflight, one-row GPU smoke, live resource check, and fresh checkpoint byte
audit passed before the 147-row scorer ran. Rights remain unverified; raw and
derived scored rows stay local.

Command executed with per-command elevated GPU access:

```bash
.venv/bin/python scripts/run_nextgqa_qta_v4_union_scorer.py \
  --mode full \
  --output-name nextgqa_tlvs_qta_v4_score_local_20261018_03_01.json \
  --user-authorized-local-only
```

## Observed result

- Status `v4_completed_test_associated_exploratory_finite_cohort`;
  requested/effective records **147/147**.
- Requested/attempted/completed forwards **588/588/588**; all four branches
  have exact requested/attempted/completed count **147/147/147**.
- Every row has a unique video ID, and ordered video/QID, geometry digest, and
  selected-row binding match the frozen plan. All `delta_union_dependence`
  values are finite.
- Recomputing each Δ from the four stored fixed-answer/foil margins produced
  maximum absolute error **0 nat**. Recomputing the stored summary with the
  runner's formula matched exactly.
- Finite-cohort `delta_union_dependence`: mean **+0.006802729 nat**; sample SD
  **0.137073068 nat**; min **−0.374999682 nat**; max **+0.375000772 nat**;
  positive **74**, negative **68**, zero **5**.
- Baseline-full multiple-choice answer was correct on **82/147** rows; this is
  a cohort-specific auxiliary description, not a general model accuracy claim.
- Runtime **2,539.39 seconds** (42.32 minutes). Recorded peak allocated VRAM
  **11.93 GiB** on RTX 5070 Ti (15.89 GiB total); initial free VRAM **14.66
  GiB**. CUDA runtime **12.8**. Observed monitoring snapshots ranged from
  0–100% utilization and 62–80°C; snapshots are not time-averaged telemetry.
- Primary revalidation confirms the private receipt against the frozen plan,
  census, preflight, smoke, 24 source hashes, 4 data hashes, and a fresh 5/5
  checkpoint-shard byte audit. The independent outcome audit confirmed all
  row/branch counts, recomputed all 147 deltas exactly, and reproduced the
  finite-cohort summary; no discrepancy was found.

## Provenance

- Local private receipt: `data/private_nextgqa/nextgqa_tlvs_qta_v4_score_local_20261018_03_01.json`, mode `0600`, SHA-256 `60f3b5a1847f042406ea727932d3960c8a64be11a2b0ecb8e81808f6289b41cc`.
- Frozen plan SHA-256: `482ffbfc2eb3c8f260a5d70dd26428f29bebd91ff40c7948024a8efddebe8b99`.
- Geometry census SHA-256: `bd4814b231c71b936e043f73e200941b0e623f37c51219b9d68c0cd7d30bdce4`.
- Full CPU preflight SHA-256: `bcbbb2a77070881b5084db144a4650c0a24f0920e0278f8e61d2b6d176315332`.
- GPU smoke SHA-256: `e721edc9e70b6dd3b65b5ff445e2e7a9858182189e2ac6d92a47695f457fb4e8`.

## Interpretation and limitations

The observed finite-cohort mean is close to zero relative to the sample SD,
with effects in both directions. No hypothesis test, multiplicity adjustment,
or population confidence interval is reported; the protocol is explicitly
descriptive for this finite selection-conditioned cohort. This is not evidence
that the intervention has no effect, and the observed signs are not a
confirmatory estimate of a population effect.

The experiment measures model prediction-margin dependence on benchmark
annotation-union masks under this QTA implementation. It does **not** measure
faithfulness to generated explanation claims. The selected QID is
processor-geometry conditioned; benchmark intervals are unreviewed; prior
exposure status remains unknown/contaminated; union masks can create
out-of-distribution inputs; four rows mask all temporal groups. These are
material alternative explanations and limit interpretation.

**Faithfulness progress:** direct claim-linked scoring remains **0/3**; stage-1
claim reviews, stage-2 prediction-link judgments, and 15/15 browser-to-decoder
frame correspondences are still absent. No direct faithfulness effect,
uncertainty interval, or significance test exists. This is the first completed
full-cohort GPU measurement for v4, but it is an auxiliary annotation-mask
prediction-dependence proxy only.

## Next action

The independent audit closed the output-verification task. A separate local
prior-exposure diagnostic found zero exact-byte matches to an explicitly
partial, unauthenticated index, so all 147 remain unknown rather than certified
fresh. Continue the human-review branch after actual role-to-key assignments,
external root pin, signed review-only receipt, and witnessed unopened handoff
are available. The next technical question is whether this small, mixed-sign
finite-cohort annotation-mask contrast justifies any revised experiment; do
not change the current estimand on these outcomes and do not resume the closed
head/layer/alpha search.
