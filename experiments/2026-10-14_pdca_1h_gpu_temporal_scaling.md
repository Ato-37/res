# PDCA 1-hour GPU temporal scaling (2026-10-14)

## Window and purpose

The authorized one-hour window began at 2026-10-14 00:45:42 JST. The purpose
was to prioritize a bounded GPU workload while measuring the safe temporal input
limit for the existing fixed local engineering smoke cohort. This branch is not
claim-selective efficacy and does not reopen the closed 79-sample head search.

## Plan and gate

The RTX 5070 Ti had 13,815 MiB free at the start. The fixed configuration was
Qwen2.5-VL-7B-Instruct, 4-bit BF16, eager attention, seed 20261014,
frame_size=1024, max_pixels=200704, cap enabled, and one sample from the
existing three-sample local smoke manifest. A 64-frame run was attempted first
to test the upper bound; a 40-frame run followed after the failure.

## Verified results

| frames | requested/effective | failures | peak CUDA allocation | result |
|---:|---:|---:|---:|---|
| 64 | 1/0 | CUDA OOM | allocation failed while requesting 7.36 GiB | no output summary |
| 36 | 1/1 | 0 | 13,252,189,184 bytes (12.34 GiB) | mean margin contrast −0.5000 nat; no CI |
| 40 | 1/1 | 0 | 14,789,873,664 bytes (13.77 GiB) | mean margin contrast −0.3750 nat; no CI |

The 40-frame run used CUDA on the RTX 5070 Ti, torch 2.11.0+cu128, and
transformers 5.17.0. The 36-frame privacy-safe derived artifact records raw SHA `bd896460b4a6b65466f00cc88b14b5a7c18672e6c762c55c8f9cadc1efd17623`. The 40-frame privacy-safe derived artifact records raw SHA
`4da022bb910eb3565ccd8762784f8a7dac3e01fdba6b426f8fe9ecc5fda734e7`.
The raw result remains local and is not published.

## Interpretation and limits

Forty frames at this resolution completed but left little headroom on a 16-GiB
GPU. Sixty-four frames exceeded available memory. The −0.3750 nat value is a
single engineering smoke observation using masking and a fixed-choice margin;
it is not direct faithfulness evidence, has no uncertainty interval, and cannot
be compared as an efficacy result with the 79-sample primary estimate.

## Faithfulness progress and next action

Direct faithfulness remains `直接評価済み` only for the prior 79/79 frozen-answer
margin confirmation (−0.0756 nat, 95% t CI [−0.1578,+0.0065], threshold unmet).
This temporal scaling run is `補助診断のみ`; claim-selective real-data work is
`未実施`/`no_go` because NOAA human attestation and NExT-GQA rights remain
unresolved. Next action is to use 32 frames as the high-resolution safe default; 36 frames is a measurable but tighter 12.34-GiB condition, and 40 frames is near the 16-GiB limit,
retain 40 frames only for explicitly bounded diagnostics, and avoid 64-frame
runs unless the memory protocol is changed and smoke-tested.
