# Local QTA GPU and review-material smoke (2026-10-18)

This note and both referenced JSON outputs are local-only. The run used the
cached local model; keep its outputs and derived details in the private local
area under the current project instruction. Rights status is unverified.

## Purpose and setup

After correcting the review-player-to-decoder-PTS interval conversion, run the
smallest QTA GPU integration that exercises the current question selector,
internal key bias, evidence/control key ablation, fixed prediction/foil score
extraction, and deterministic sham. This is an engineering smoke, not a
faithfulness efficacy experiment. The one-row synthetic video has 32 frames;
the runner performs one baseline attention-capture forward and eight branch
forwards. It uses the pinned Qwen2.5-VL-7B-Instruct snapshot
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, eager attention, 4-bit weights,
bf16 compute and activations. Seed:
`qta-sham-seed-20261018-v1`. Runtime: RTX 5070 Ti, PyTorch 2.11.0+cu128,
Transformers 5.18.0, bitsandbytes 0.50.2, Python 3.12.14.

## Observed GPU result

The smoke passed: **1/1 synthetic record and 9/9 requested forwards** across
the complete baseline/QTA/sham × full/evidence/control condition set. It
completed in 50.93 s. Peak allocated/reserved VRAM was 10.755/12.439 GiB;
free VRAM was 14.657 GiB before loading. The runner's post-forward reading was
1.874 GiB before model cleanup; after process exit, independent `nvidia-smi`
showed 14,860 MiB free, 1,136 MiB used, 49°C, and 1% utilization.

The runner recorded a synthetic target `ΔF` of −5.96×10⁻⁸ and target-minus-
sham contrast 0. Independently recomputing the fixed-p/foil margins from the
nine condition log probabilities reproduced the `ΔF` to <1e-10. These values
are numerical integration diagnostics on synthetic pixels and have **no
semantic or research-efficacy interpretation**. The artifact itself states
`faithfulness_result=false`, `treatment_frozen=false`, and
`execution_authorization=not_granted`.

The private output is
`data/private_nextgqa/qta_synthetic_integration_20261018_0605/qta_nine_condition.json`,
SHA-256 `669cc1c573d0f6776125fda3e99a6206f312f5aedde201771c7336bab70ba190`.
An independent local verification confirmed the 9/9 count and exact
condition set, recomputed the fixed-p/foil contrast, and matched all 15 local
script-source hashes in the artifact. The result directory/file modes are
0700/0600.

A separate preceding 32-frame internal-ablation smoke passed 2/2 forwards in
22.87 s, with peak allocated/reserved VRAM 10.755/12.439 GiB and 14.657 GiB
free before loading. Its output is
`results/synthetic_qwen_internal_ablation_32f_20261018.json`, SHA-256
`e63cff469ff0c19623e0d67b7fa592be63f6a82bca4cea6f28ba38ee92288850`.
This output is synthetic-only and marked non-authorizing; it is retained
locally.

## Review package inventory

The current v6 offline reviewer page and its three token-named media files
were re-inventoried using the current inventory builder, binding 65 artifacts
with zero missing required slots. An independent hash pass confirmed the
standalone HTML equals the packaged `index.html`, and all three packaged MP4s
match their staged counterparts. The private inventory SHA-256 is
`65efb499355431ddca2f8206827304c1de06f7a9999d6c13b753ec5a8c71e49d`; output
modes are 0600 (file) and 0700 (directory). It explicitly remains
`candidate_only`, with review not started/completed, no joint freeze, no score
readiness, and no execution authorization. It is not a review or release
receipt.

## Faithfulness status and next action

Empirical faithfulness progress is unchanged: video claim review **0/3**,
prediction-link review **0/3**, scored **0/3**, verified player/decoder frame
correspondences **0/15**, and no real-data `ΔF`, interval, or test. The
implementation and smoke path are better exercised, but the local review
release still lacks an independently pinned trust root and witnessed
pre-open handoff; a human reviewer and independent frame-pair reviewer also
remain necessary. Do not read the page/video as an agent substitute, open the
review package, or load the model on real review rows until those gates pass.
Next code task: implement a CPU-only non-authorizing join that recomputes the
first-stage reviewed-interval digest and verifies the review token, video
hash, calibration hash, exact inverse-mapped PTS intervals, group/control
mapping, processor batch digest, and baseline/prediction-link source fields.
The key regression substitutes a different valid batch/mapping while leaving
the review and prediction-link receipts unchanged; the join must reject it
for the full three-row cohort, not silently drop that row.
