# Annotation-proxy receipt and source-lineage re-audit (2026-10-21)

## Purpose

Recheck the completed 30-video annotation-interval proxy against the current
worktree after mapper/runner files changed. Keep the proxy result distinct from
direct explanation faithfulness and determine whether its execution provenance
can still be independently verified.

## Checks

- The local receipt requests/completes **30/30** records, records **0
  failures**, and reports **180/180** forwards. It remains private.
- Independently recomputed the fixed branch arithmetic from each receipt row:
  **30/30** recorded `delta_f` values equal
  `(intervention_control_ablation - intervention_evidence_ablation) -
  (baseline_control_ablation - baseline_evidence_ablation)` within `1e-12`.
  Recomputed descriptive values are mean **+0.008333 nat**, sample SD
  **0.142837**, range **[−0.250000, +0.250000]**, with **17 positive / 13
  negative / 0 zero** rows and baseline full-input restricted-choice accuracy
  **23/30**. These are finite-cohort annotation-proxy descriptions only.
- Current interval mapper, validation runner, and aggregate analyzer tests
  passed **53/53**. They verify the current code's interval geometry and
  receipt arithmetic contracts; they do not recreate the historical GPU run.
- The strict production analyzer rejected the old receipt with
  `source_manifest_hash_mismatch`. Five receipt-pinned files differ from the
  worktree. Historical/alternate exact copies were recoverable for three code
  files (the temporal selector in Git history, the executed interval mapper
  copy, and the gold-margin pilot in Git history). The exact validation-runner
  source and matching experiment-log bytes were not found in Git history or
  alternate repository files. Therefore this turn could not rerun the strict
  source-bound analysis.

## Interpretation and limits

The receipt's row counts and per-row delta arithmetic reproduce, but that is
weaker than reproducing the historical model execution from its exact source
tree. The source-lineage gap means the earlier proxy result should be cited
with its original private receipt and the current hash mismatch disclosed; it
must not be presented as freshly reproduced under the current checkout. No
confidence interval or significance test is added. This experiment measures
dependence on benchmark annotation intervals, not whether an explanation's
claimed video evidence caused the answer.

Direct faithfulness remains **0/3** formally reviewed claims, **0/15**
verified clock/frame correspondences, **0/3** prediction-link reviews, and
**0/3** direct scores. No direct `DeltaF`, interval, or test exists. No GPU
work was run in this re-audit. Protected receipt contents and videos were not
published; rights status remains unverified.

## Next action

Do not rerun or retune the closed finite cohort. Preserve content-addressed
source snapshots and the exact experiment-log bytes for any future run before
model loading. For the current direct study, prioritize a fresh, bound human
review and the independent 15-anchor clock/frame audit; the proxy receipt
cannot substitute for those gates.
