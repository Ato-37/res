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
