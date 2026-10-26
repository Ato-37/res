# QTA v4 annotation-union treatment protocol

## Purpose and frozen estimand

This is a test-associated, exploratory prediction-dependence proxy. For each
selected NExT-GQA evaluation video, freeze the model's top A–E answer and its
highest-probability foil from `baseline_full`. Reuse that same answer/foil pair
in four forward conditions: `baseline_full`, `baseline_union_mask`,
`intervention_full`, and `intervention_union_mask`. The mask removes all Qwen
temporal token groups covered by the union of valid gsub intervals. The QTA
intervention applies the existing question-token attention boost in layers
5–16, with the existing fixed `ln(2)` bias and top `max(1, ceil(0.10*T))`
groups selected from baseline-full question attention.

For treatment `t`, `D_t = margin_t(full) - margin_t(union_mask)`. The sole
primary descriptive contrast is `delta_union_dependence = D_intervention -
D_baseline`; positive values mean the intervention increases the fixed
answer/foil margin's sensitivity to the annotation-union mask. This is not the
earlier six-condition `delta_f`, and no branch may recompute the answer or foil.

## Cohort and selection

Preserve all 147 evaluation video IDs from the already frozen v2 split. For
each video, inspect every test.csv QID with structurally valid gsub intervals.
Before model outcomes, require exact agreement between actual PTS and scalar-FPS
temporal group assignment, then select the minimum SHA-256 feasible QID per
video. This selection is conditioned on processor geometry feasibility and is
not a fresh or confirmatory sample. No video is dropped: if any of the 147 has
no feasible QID, the full plan is no-go. Full annotation coverage is recorded
as a mask property and never used to exclude a row.

## Execution gates and expected workload

The intended full run has exactly 147 rows and 588 forwards. First run the
147-row CPU decode/processor geometry preflight and require exact agreement
with the frozen census/plan; then require a successful one-row four-forward
GPU engineering smoke, pinned checkpoint byte audit, CUDA BF16 support, and
the runner's free-VRAM gate before loading the 4-bit BF16-compute model. The
full run has no substitutions and is all-or-none for the denominator. Expected
cost is approximately 1–1.5 hours of accelerator time for 588 forwards, plus
the CPU preflight; this is a planning estimate, not a measured runtime.

Outputs remain mode-0600 local receipts under `data/private_nextgqa/`; stdout
contains aggregate status only. No row-level content is published. Report
mean, sample SD, min/max, positive/negative/zero counts, and baseline-full
correct count as finite-cohort descriptions. Do not report a p-value or
population confidence interval.

## Limits and current status

The official test-associated media and benchmark annotations are used; prior
exposure status remains known-contaminated/unknown, and unmatched records are
not certified fresh. gsub intervals are benchmark annotations, not reviewed
generated-explanation claims. The outcome therefore does not measure generated
claim faithfulness. Union masking may create out-of-distribution inputs and may
remove every temporal group for some rows; report the full-coverage indicator
without excluding those rows. Rights remain unverified; artifacts and data are
local only.

The four-condition arithmetic, union geometry policy, census selection contract,
and scorer denominator/preflight receipt gates pass 36 focused CPU tests. A
full 147-row scorer preflight and a fresh source-bound CPU geometry smoke have
not yet run against this finalized scorer source. No v4 outcome scores or GPU
run have been produced. Next: regenerate the owner-local input hash gate, run a
one-row CPU geometry smoke, then request independent source audit before any
full CPU census. GPU remains unstarted.
