# 2026-10-13: exploratory top-8 head smoke

## Purpose and scope

At the user's explicit request, run one bounded exploratory GPU smoke with
`top-k=8` to exercise a head set outside the closed preregistered `top-k=4`
primary. This result is a separate exploratory branch and does not reopen or
alter the 79-video confirmation, its practical threshold, or the primary
head/layer/alpha decision.

## Setup and verified result

- Qwen2.5-VL-7B-Instruct, 4-bit BF16, SDPA, RTX 5070 Ti, all 28 layers.
- Existing 1-video head smoke manifest, 16 frames, seed `20261013`, alpha
  `0.5`, eight selected heads, one matched control group.
- Output: `results/head_exploratory_top8_smoke_20261013_01.json`;
  SHA256 `eb4c71f779f9c6440e9ca132c80c9c6d9e4e23b45afd19e6daf6a46331757e8c`.

Observed: requested/effective `1/1`, failures `0`, exploratory selective
margin `+0.25 nat`, and no confidence interval (`n=1`). The runner records
`research_status=exploratory_smoke` and `exploratory=true`; the primary
threshold flag is false.

A fixed three-sample exploratory repeat was also completed under the same
settings: `3/3`, failures `0`, mean selective margin `+0.0417 nat`, 95% t CI
`[-0.4327,+0.5160]`, and primary-threshold flag false. Its published artifact
is `results/head_exploratory_top8_smoke_20261013_02.json` (SHA256
`6bc1ae9f2b5804a12b1265b2a0aeb61163c7a6dde2b0446800c242625070b764`).

## Interpretation and limitations

This single observation only verifies that the top-8 intervention path runs on
the GPU and produces finite output. It cannot estimate an effect, compare
head sets, establish faithfulness, or justify changing the closed primary
conclusion. The selected heads and one control group were not a preregistered
confirmatory search, and no post-hoc tuning is permitted from this result.

## Next action

Keep the explicit `--exploratory` guard for any future head scan. A larger
exploratory inventory requires a fixed cohort, control policy, and analysis
plan before execution; it must remain separate from the primary estimate.

## Provenance repair run (2026-10-13)

The exploratory one-sample rerun was executed after adding source/result provenance to the publisher manifest. It used the same exploratory top-8 configuration (`Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit, bfloat16, 16 frames, cap-pixels, alpha 0.5, one matched control, seed 20261013) and completed on the RTX 5070 Ti with 1/1 effective sample and 0 failures. The observed selective margin drop was +0.2500 nat for this single sample; no interval or inferential claim is possible. The result is explicitly `research_status=exploratory_smoke` and remains excluded from the preregistered confirmation.

The published manifest binds the run to source commit `ff3be6875f4bb7b8f0acd9b573682447b17789f8` and result SHA-256 `9c64eff1b1461b932dfb864faaf07d32298c167274bf5f5a1b7ab0650c87a727`; normalized command arguments are recorded in the manifest. This repairs the earlier provenance gap for this rerun only; earlier exploratory artifacts remain descriptive and are not retroactively upgraded. Next action: independently audit this binding and implement a separate fail-closed claim-mask protocol validator while NOAA attestation and NExT-GQA rights remain unresolved.

### Independent audit disposition

The independent audit reproduced the result SHA and source-commit binding and confirmed the GPU/device, model revision, settings, and 1/1/0 accounting from the published JSON. A live `git ls-remote` check confirmed that GitHub main points to commit `11ea577950de4ef4bb1fbf3b897f29c89582ec5a`. The run is not SSS: the normalized manifest omits complete argv and some driver/dependency details, and one smoke sample cannot support a faithfulness inference. The audit accepts provenance for this artifact only; the preregistered 79-sample conclusion is unchanged.
