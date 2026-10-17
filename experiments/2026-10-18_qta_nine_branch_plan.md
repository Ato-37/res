# QTA nine-branch plan adapter (2026-10-18)

## Purpose

The QTA GPU smoke runner had an inline branch plan. We extracted it into a
CPU-only helper that binds the same processed input and fixed six-condition
answer-score binding across baseline, QTA treatment, and deterministic sham,
each with full, evidence-key-ablation, and control-key-ablation branches. The
plan records its temporal-group mapping and explicit layer/bias settings,
hashes every branch descriptor, hashes the complete plan, and rejects missing
or semantically inconsistent branches. This is an engineering artifact, not
a real-data gate or faithfulness result.

## Checks and result

- Focused QTA tests passed **22/22**; independent focused audit passed
  **56/56** across the relevant adapter/test set. The independent audit found
  that a coherently rehashed contradictory plan passed the initial validator.
  The validator now reconstructs branch key positions from the stored temporal
  groups and verifies treatment/condition labels, layers, bias, and shared
  input/scorer hashes. Rehashed changes to selected groups and treatment labels
  are covered by regressions.
- Full scripts suite passed **1,565/1,565** in **36.486 s**;
  `compileall`, `py_compile`, and `git diff --check` passed.
- Updated synthetic GPU smoke completed **1/1 record, 9/9 forwards** in
  **49.114 s** on RTX 5070 Ti, CUDA 12.8, Torch 2.11.0+cu128, Transformers
  5.18.0, bf16/4-bit-bf16 eager. Peak allocated/reserved VRAM was
  **10.755/12.439 GiB**. It produced one plan digest and nine branch digests;
  target and sham both reported `bound_logits_adapter_contract_only`.
  `faithfulness_result=false`.
- Protected local artifact is under a mode-0700 directory as a mode-0600 file,
  SHA-256
  `3ddfde961ea7b6134ded4d938c94181441d59e4c15366c2b6971db6f6ac3f1ea`.
  The privacy-reviewed aggregate is
  [`results/synthetic_qta_nine_branch_plan_gpu_20261018_0742.json`](../results/synthetic_qta_nine_branch_plan_gpu_20261018_0742.json),
  SHA-256 `53705b2e2f75f54f90e832119e6334726074a4ed9e70a584275a156675e46ec6`.
  It excludes prompts, QA, row IDs, token IDs, condition logits, local paths,
  and checkpoint weight hashes.
- After the smoke, `nvidia-smi` reported 0% utilization and 14,861 MiB free.

## Interpretation and limits

The helper makes the nine-branch treatment plan explicit and hash-checkable
and removes duplicate branch-composition logic from the smoke runner. Hashes
bind caller-supplied values; they do not authenticate input provenance,
human review, claim mapping, clock calibration, model execution, or output
logits. The synthetic evidence/control assignments have no annotation
meaning. The helper is deliberately non-authorizing and does not compose the
full cohort-wide provenance gate. No inference about `DeltaF`, uncertainty,
significance, or intervention efficacy follows.

Direct faithfulness progress remains **0/3** human claim reviews, **0/3**
prediction-link adjudications, **0/15** verified player-clock/decoder-frame
correspondences, and **0/3** scored records. No real-data `DeltaF`, interval,
or test exists. Rights status remains unverified; protected data remain local.

## Provenance

- Plan helper: `003e6b5d97e68645033ec3f3edb4524d7352ff158f563e1073daaf9e47532788`
- Runner: `2aa3211b98513cb7eb56a78b2d39cbc2d8484fd8e3a33ad44ceac801792b0a5d`
- Focused tests: `ab45e59e68942b3c7121d603b26d8c63df65e9135628c53d7cd7d81a7effeb74`
- Fixed scorer: `84ba18241668fd659ddce0ab25a4e3dfa2cacc656ef43fb494fdc056a5c21744`

## Next action

Continue toward a non-authorizing cohort orchestrator that binds the plan
digests for the complete fixed three-record set and composes them with the
existing review/clock/processor/baseline joins. It must reject any missing
row or upstream receipt before model load and keep treatment forwards blocked
until the separate human, identity, witness, and frame-correspondence gates
are authenticated. Do not resume real-data scoring from the present candidate
inventory.

## Follow-up: row-bound selector budget and fixed-cohort aggregation

Independent adversarial review of the single-row plan found that a coherently
rehashed plan could disagree with its selected groups, and the source inventory
omitted the plan helper. Those defects were fixed before this follow-up: the
plan now stores the temporal partition and fixed selection specification and
reconstructs branch key positions during validation; the helper source hash
is recorded by the runner. A second independent review then found that the
plan did not carry the upstream row-binding digest and did not enforce the
protocol's `max(1, ceil(T/10))` group budget. The plan now binds that digest,
selector and sham IDs, sham seed, layer set, budget, tie-break, and bias; it
rejects oversized target/sham sets. The nine branch descriptors carry the row
binding, typed-input hash, scorer binding, and branch settings.

Added `scripts/qta_three_row_plan_aggregate.py` as a downstream CPU-only,
non-authorizing aggregator. It requires exactly three ordered plans, checks
each against its expected row, processor batch/scorer, temporal-grid,
evidence/control-group, and frozen selection-spec hashes, and requires common
intervention layers/bias/selection spec. Any error returns the fixed requested
denominator of three and zero accepted plans/planned forwards, with no partial
row digests. Success reports three structurally bound plans and 27 planned
forwards, but still reports model loading false, scoring not granted, and data
provenance unverified. This aggregator is not the actual upstream trust/review
preflight or a production launch gate.

- Primary focused QTA + cohort-aggregate tests: **31/31 passed**; independent
  audit confirmed the rehashed selector-budget regression and row-swap block.
  Primary full scripts suite: **1,574/1,574 passed** in **35.986 s**;
  `compileall`, `py_compile`, and `git diff --check` passed.
- Final-version GPU smoke after the row-binding/budget change: **1/1 record,
  9/9 forwards** in **50.046 s** on RTX 5070 Ti; peak allocated/reserved VRAM
  **10.755/12.439 GiB**. It reported nine branch digests and both target/sham
  scorer statuses `bound_logits_adapter_contract_only`;
  `faithfulness_result=false`. Afterward GPU use returned to 1%, 14,863 MiB
  free.
- Independently re-hashed all **22/22** recorded local and installed-code
  source entries against current bytes. The run records pinned model revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5` and five checkpoint shard hashes.
- Protected local result SHA-256:
  `a9b5bd7f1509beb0abcc8721e1d0f6cf6bfd18c4fc85bb8c8bd8cd6e231ae108`.
  Privacy-reviewed aggregate:
  [`results/synthetic_qta_row_bound_budgeted_plan_gpu_20261018_0750.json`](../results/synthetic_qta_row_bound_budgeted_plan_gpu_20261018_0750.json),
  SHA-256 `6b174c9ed13a9391d464f867f300618ffcd1fc32ced6c1cba8fadbf4335bfa00`.

This hardens row linkage and selector-budget checks, and verifies the revised
single-row helper in a synthetic GPU run. It still does not authenticate a
human review or predict faithfulness. Direct progress remains **0/3** claim
reviews, **0/3** prediction-link adjudications, **0/15** frame correspondences,
and **0/3** scored; there is no real-data `DeltaF`, confidence interval, or
test. Rights remain unverified; protected artifacts remain local.

Follow-up provenance hashes:

- Final plan helper: `007591f768e650f249463bcb722f016c39689b798ca176a6d6a512f3e7c3ddbc`
- Three-row aggregator: `50e5d0aa95dc916cd46b3992aae1f715551795a4da4634d9258c4b8ef0e1e400`
- Final runner: `8db0ddcb631cacaeaf6e5327c4179b271a8cc98c0edfe8b4b9cfdc22cdf10d0c`
- Three-row tests: `8bd4477e7f53a82673a51420fe022a1b13ac2306d608a1289db2605e47ed4c45`
- QTA runner tests: `19d79b1aa7a6acd6760bc61bba5b38c409518b964e696842daf47b196e7a2cfb`
- Final sanitized summary: `6b174c9ed13a9391d464f867f300618ffcd1fc32ced6c1cba8fadbf4335bfa00`
