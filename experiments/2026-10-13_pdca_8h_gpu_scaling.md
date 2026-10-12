# 2026-10-13: 8-hour PDCA GPU scaling checkpoint

## Window and purpose

This checkpoint belongs to the authorized window started at
`2026-10-13T04:36:47+09:00` and ending at `2026-10-13T12:36:47+09:00`.
It measures whether a larger fixed video input keeps the RTX 5070 Ti occupied
and records peak CUDA allocation. It is an engineering/preprocessing diagnostic,
not a new faithfulness efficacy experiment.

## Setup and verified result

Qwen2.5-VL-7B-Instruct, 4-bit BF16, eager attention, seed `20261013`,
`frame_size=1024`, `max_pixels=200704`, cap enabled, 32 frames, and the same
fixed three-sample local smoke manifest used by earlier diagnostics. The run
used source commit `2452d0d24663591ded1c636209d2dd06e6de9cad`.

| item | observed |
|---|---:|
| requested/effective | 3/3 |
| failures | 0 |
| peak CUDA allocation | 12,034,031,616 bytes (about 11.21 GiB) |
| primary baseline-correct n | 2 |
| primary mean selective margin | +0.4375001 nat |
| 95% pointwise interval | [+0.1250003, +0.7499999] |
| sign-flip p-value | 0.5095 |

The privacy-safe derived artifact is
`results/necessity_choice_cap_frame1024_32frames_smoke_derived_20261013_01.json`.
The raw artifact remains local; its SHA256 is
`11eaf4348ab8a974d273779acfc1646c9e222e252d94a880913e2ff360b0f94a`.

## Interpretation and limits

The larger input increased peak allocation relative to the one-sample
32-frame witness, while completing without failure. The estimate is unstable
(n=2), uses masking and a fixed-choice margin, and is not direct faithfulness
evidence. No multiplicity correction or confirmatory claim applies. The direct
faithfulness status remains the separate 79/79 head-selective confirmation,
which did not meet its +0.25 nat practical threshold.

## Next action

Use the recorded peak-memory field for future workload sizing. Before any
claim-selective real-data run, resolve the NOAA human-attestation and NExT-GQA
rights gates and implement the separate versioned claim-mask protocol contract.
