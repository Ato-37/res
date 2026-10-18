# Qwen target-appended processor boundary audit (2026-10-19)

## Purpose

Check a prerequisite for the frozen NExT-GQA QA-margin scorer: appending each
single-token answer label (`A`–`E`) to the production prompt must leave the
processor-expanded video/prompt prefix unchanged and place the contextual
answer token at the final sequence position. This is an input-boundary audit,
not a model score or faithfulness experiment.

## Setup

- Data: the existing frozen three-row local NExT-GQA exploratory cohort and its
  lineage/worksheet bindings. Raw questions, answers, identifiers, and frames
  remained local and were not written to the result.
- Model assets: pinned local Qwen2.5-VL-7B-Instruct processor revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; no model weights loaded.
- Runtime: Python 3.12, PyTorch 2.11.0+cu128, Transformers 5.18.0. CPU
  processor path; no CUDA calls and no network downloads.
- Requested/effective: 3/3 QA rows, 16/16 frames per row, 5/5 answer labels
  per row (15 requested checks).
- Final 16-frame result file is local/private, mode `0600`, 2,877 bytes,
  SHA-256 `1a4381d753bf40a831a9ff95442d9226befa2acb6bc22bb2db4f07c9613c1975`.
- The result binds these inputs/code by SHA-256: test CSV
  `0443ede53f521de4cbfb5e4fb02253926b90f81f62041b18bd17207a3511d114`,
  lineage `e1695b6e57b77f3221aa8a9cbe96253510d412053edb5ae89cda01de79b50bba`,
  worksheet `47370c2cf694a0a085f7dedea8f62918d6fbf05c2041b9b57e87476b48f8fc70`,
  audit script `41d7db8a8e1caf96f4a550bf2b4930c8c6be93b68d3d272eca6b9c6c0b60dfda`,
  production runner `92a6173e6efe3606c5baf21c6a580ff589559cd5963057fe07969d174e70b07d`,
  decoder `df0baf0b31afcac9715e35da2027e3e510f63cfc2bdf1aa1cd568e63045e38dd`,
  source adapter `c977d1723c9715ec06abd23a454a958ab49c861cb687abd2ee5dcd7ec42a1e70`,
  and processor files digest
  `fa7014adedb135009ad2abb19ec2abbf31b4d61d32ce8b48be5362c506087438`.

## Implementation and verification

The audit now processes the production prompt once and then processes the same
prompt with each answer label appended. It checks exact token-prefix equality,
the final token ID, attention and modality-token prefixes, and typed digests of
all non-text processor fields (including video tensors and timing metadata).
It releases each candidate batch before making the next one to limit memory.

The initial smoke exposed a timing-check false failure: Qwen emits
`second_per_grid_ts` as float32, so the third row's value `83.88200378417969`
differed from the rational expectation `83.882` by `3.784e-6` seconds. The old
absolute-only `1e-6` threshold rejected this valid float32 rounding. The shared
production validator now uses `math.isclose` with relative tolerance `5e-8`
and absolute tolerance `1e-6`; at the observed 83.882-second interval this
allows about `4.19e-6` seconds (about 0.55 float32 ULP), covering the observed
`3.784e-6` error. Tests accept that emitted value and reject `83.882005` and
`83.9`. The real processor's fps and returned interval were consistent.

- Focused unit suites: **35/35 passed** across the prompt-boundary audit and
  production QA-margin runner, including float32 long-interval acceptance and
  mismatch rejection.
- Python compilation and scoped `git diff --check`: passed.
- Frozen 16-frame real-data audit: **3/3 rows, 15/15 label checks**, exact
  processor-expanded prompt prefixes preserved, contextual label token IDs
  matched standalone token IDs. `target_appended_prefix_stable=true` and
  `contextual_target_tokens_match_standalone=true`.
- Result reports zero model-score calls, zero weight loads, zero downloads,
  and no emitted question/answer text or decoded frame pixels.
- An independent audit verified the final receipt's provenance, permissions,
  counters, hashes, and privacy-safe schema. It independently recomputed the
  timing differences and confirmed that the observed float32 value passes while
  `83.882005` and `83.9` fail.

## Interpretation and limitations

This removes an implementation-level uncertainty in answer-token positioning
for the full unmasked production QA prompt and shows that the prior 3.8e-6
failure was float32 rounding against an overly strict absolute tolerance. It
does not establish model-logit alignment, masked-condition equivalence, claim
truth, explanation-to-prediction linkage, or faithfulness. No GPU experiment
was appropriate for this processor-only contract check. Rights status remains
unverified; all data and its result remain local.

Direct faithfulness progress remains **0/3 claim-reviewed, 0/15
clock/player-to-decoder frame correspondence, 0/3 prediction-link judgments,
and 0/3 directly scored**. No direct effect estimate, interval, or test exists.

## Next action

Use the nickname-first informal review artifact only for exploratory impression
feedback; it cannot supply claim/frame/prediction-link evidence. For direct
faithfulness scoring, the next prerequisite remains frozen, reviewable evidence
and baseline/prediction bindings. Do not resume closed head/layer/alpha or QTA
proxy searches.
