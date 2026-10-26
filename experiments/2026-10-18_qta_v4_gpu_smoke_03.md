# QTA v4 one-row GPU engineering smoke `_03` (2026-10-18)

## Purpose and setup

Verify the frozen four-condition scorer on the first selected real NExT-GQA
video before the 147-row run. The smoke uses the already audited `_03` plan,
147-row CPU geometry preflight, local pinned Qwen2.5-VL-7B-Instruct checkpoint,
eager attention, 4-bit weights, and BF16 compute. It performs one forward for
each of `baseline_full`, `baseline_union_mask`, `intervention_full`, and
`intervention_union_mask`. All output is a mode-0600 local receipt; rights
remain unverified.

Command executed with per-command elevated GPU access:

```bash
.venv/bin/python scripts/run_nextgqa_qta_v4_union_scorer.py \
  --mode smoke \
  --output-name nextgqa_tlvs_qta_v4_gpu_smoke_local_20261018_03.json \
  --user-authorized-local-only
```

## Observed result

- Status: `v4_one_row_four_forward_gpu_smoke_complete`; requested/effective
  rows **1/1**, total forwards **4/4**, with **1/1** completed in each branch.
- Runtime **54.13 s**; RTX 5070 Ti, pinned revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, `4bit_bf16_compute_bf16_eager`.
- Peak allocated VRAM **11.08 GiB**. Before launch, CUDA BF16 was supported
  with **14.66 GiB** free. The runner enforced at least 11 GiB free before
  model load and 2 GiB before each forward.
- The primary agent independently reran the checkpoint byte audit and
  `validate_smoke_receipt`: checkpoint audit, exact source/data maps, plan,
  census, preflight hash, processor manifest, first-row geometry, and all
  branch counts matched. A separate auditor independently verified all the
  same receipt bindings and rehashed all **5/5 checkpoint shards** with zero
  failures. Full-cohort scoring had not run at the time this smoke was logged.
- Smoke `delta_union_dependence` was **−5.96×10⁻⁸ nat**. This is a single-row
  wiring/numerical diagnostic, not an efficacy estimate or direct faithfulness
  evidence.

## Provenance

- Private receipt: `data/private_nextgqa/nextgqa_tlvs_qta_v4_gpu_smoke_local_20261018_03.json`, mode `0600`, SHA-256 `e721edc9e70b6dd3b65b5ff445e2e7a9858182189e2ac6d92a47695f457fb4e8`.
- Bound plan SHA-256: `482ffbfc2eb3c8f260a5d70dd26428f29bebd91ff40c7948024a8efddebe8b99`.
- Bound census SHA-256: `bd4814b231c71b936e043f73e200941b0e623f37c51219b9d68c0cd7d30bdce4`.
- Bound CPU preflight SHA-256: `bcbbb2a77070881b5084db144a4650c0a24f0920e0278f8e61d2b6d176315332`.

## Interpretation and limitations

This verifies one real-data/model execution path and the Qwen eager-attention
hook for four conditions. It does not validate benchmark intervals as reviewed
claim evidence, estimate an intervention effect, establish generalization, or
measure generated-explanation faithfulness. The cohort remains
test-associated and exposure status remains unknown/contaminated; full masking
may be out of distribution. No p-value or population interval is appropriate
for this one-row smoke.

**Faithfulness progress:** direct generated-claim scoring remains **0/3**.
There is no v4 full-cohort score, effect, confidence interval, or test yet.
This run is engineering evidence only.

## Next action

Run the exact frozen full cohort, **147 rows and 588 forwards**, only if the
fresh checkpoint audit, smoke receipt, all four per-branch counters, source and
data hashes, processor manifest, and full preflight still match. Expected GPU
time is **about 1–1.5 hours**; this is a planning estimate from the frozen
protocol, not a measured full-run duration. Record any all-or-none failure
without substituting rows. Keep all detailed outputs local.
