# QTA v4 full CPU geometry preflight `_03` (2026-10-18)

## Purpose and setup

Verify every selected row in the independently audited `_03` geometry census
and frozen 147-video plan against current decoded frames and processor token
geometry before any GPU scoring. This preflight performs no model-weight load,
CUDA call, forward pass, or score computation. It uses the existing `.venv`,
local pinned processor config and local-only protected test media. Rights remain
unverified; all outputs remain local.

Command executed:

```bash
.venv/bin/python scripts/run_nextgqa_qta_v4_union_scorer.py \
  --mode preflight \
  --output-name nextgqa_tlvs_qta_v4_cpu_geometry_preflight_local_20261018_03.json \
  --user-authorized-local-only
```

## Observed result

- Status: `v4_full_cpu_geometry_preflight_complete_non_authorizing`.
- Requested/completed rows: **147/147**; all 147 video IDs unique.
- Ordered video/QID identities, geometry hashes, and row-binding hashes match
  the frozen plan exactly. The scorer's selected-item loader also rechecked
  current media and raw annotation row bindings for all 147 rows.
- The receipt binds exact census and plan bytes, all 24 source hashes, all 4
  dataset hashes, processor manifest and library versions. These were
  independently recomputed by the primary agent and a separate auditor; all
  matched the current files and the plan.
- The mode-0600 receipt contains no scores, logits, model-load authorization,
  CUDA use, or forward counts.

## Provenance

- Private receipt: `data/private_nextgqa/nextgqa_tlvs_qta_v4_cpu_geometry_preflight_local_20261018_03.json`, mode `0600`, SHA-256 `bcbbb2a77070881b5084db144a4650c0a24f0920e0278f8e61d2b6d176315332`.
- Bound plan SHA-256: `482ffbfc2eb3c8f260a5d70dd26428f29bebd91ff40c7948024a8efddebe8b99`.
- Bound census SHA-256: `bd4814b231c71b936e043f73e200941b0e623f37c51219b9d68c0cd7d30bdce4`.
- Source and data hash maps are retained only in the private receipt; the auditor independently confirmed exact-set equality against current code/data.

## Interpretation and limitations

This closes the processor-geometry reproducibility gate for the selected
finite cohort. It does not show model outputs or a treatment effect, validate
benchmark annotations as generated explanation evidence, or establish
faithfulness. The protocol remains a test-associated exploratory
annotation-mask prediction-dependence proxy.

**Faithfulness progress:** direct generated-claim evaluation remains **0/3**;
no real score, effect, uncertainty interval, or test exists for this v4 branch.
The preflight is engineering evidence only.

## Next action

Run a one-video, four-forward GPU engineering smoke using the pinned Qwen
checkpoint and exact `_03` plan/preflight. Require fresh checkpoint-byte
verification, CUDA BF16, at least 11 GiB free before load, at least 2 GiB free
before every forward, and receipt validation against all current hashes and
the full preflight's first row. If any gate fails, preserve the failure receipt
and continue on independent human-free or review-preparation work.
