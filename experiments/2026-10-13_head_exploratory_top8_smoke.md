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
