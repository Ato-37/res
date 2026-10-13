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

## Independent audit and publication verification

The independent audit reproduced the 40-frame derived SHA (`fe1fe489fb9f3cdc414be2230edde77784f80655d005f9d0390fca6dafb84fe1`) and confirmed 1/1/0 accounting, CUDA device, RTX 5070 Ti, 13.77-GiB peak, and the absence of an interval. A live `git ls-remote` check later verified GitHub `main` at `4162b5e96f0579934940dd1c3b88ee16472b5d79`. The raw JSON remains local but is present for local hash verification; only the derived summary is published. The 64-frame attempt produced no summary, so its 7.36-GiB allocation request is a failure diagnostic rather than a measured total peak. The difference between start-of-run free memory and allocator peak remains an unresolved allocator/driver accounting detail; this does not affect the safe-default decision. SSS is not claimed.

## Failure-artifact repair

The all-failed 64-frame path previously exited before writing JSON, which made
an OOM difficult to audit. The runner now writes a privacy-safe result with
`n_effective=0`, failure records, and null summaries instead of raising before
serialization. Existing successful-run behavior is unchanged; CPU syntax and
necessity utility tests pass. This repair is intended for future bounded memory
probes and does not reinterpret the 64-frame failure as a measured effect.

## GPU compute-path separation

A bounded BF16 CUDA benchmark (4096x4096 matrix multiplication, 20 iterations)
ran on the RTX 5070 Ti with torch 2.11.0+cu128 in 0.483 s (41.38 matmuls/s;
peak allocation 142,737,408 bytes). This confirms the CUDA compute path is
available and fast in the elevated `.venv`. The low utilization observed during
64-frame video processing is therefore consistent with video decoding,
preprocessing, and memory pressure being the bottleneck; the benchmark is an
engineering diagnostic only and is not faithfulness evidence.

The failure serializer was hardened after audit: future failure records contain
only `sample_index` and generic codes such as `cuda_oom`, never media paths or
raw exception text. Three focused CPU tests pass.

The repaired runner's 64-frame rerun reproduced the same CUDA OOM and wrote a
local failure JSON with `n_requested=1`, `n_effective=0`, and null summaries.
Because that pre-sanitization file contains a private video path and raw error,
only a derived failure artifact with `failure_codes=["cuda_oom"]` is published;
the raw file remains local.

The allocator counter in an OOM path reported a value above physical VRAM, so
`peak_cuda_memory_bytes` is intentionally set to null in the published 64-frame
derived artifact. The reliable observation is the CUDA OOM and failed allocation
request, not a total peak estimate.

An additional BF16 CUDA benchmark at 8192x8192 for 10 iterations completed in
0.302 s (33.07 matmuls/s; peak allocation 545,390,592 bytes). Compared with the
4096x4096/20-iteration result (41.38 matmuls/s), this confirms the GPU executes
larger dense workloads efficiently; the lower video-job utilization is not a
CUDA availability failure. These synthetic throughput numbers are engineering
only and have no faithfulness interpretation.

Using the successful 32/36/40-frame allocator observations only as a rough local
linear interpolation gives approximately 344,480,256 additional allocated bytes
per frame; it predicts about 14.37 GiB at 42 frames and 15.01 GiB at 44 frames.
These are workload-sizing estimates, not measured results, and are not used to
claim that those frame counts would succeed. The safe operational choice remains
32 frames; 40 frames is near the limit and 64 frames is rejected.

## Synthetic attention backend diagnostic

The preregistered synthetic Q/K/V shape diagnostic completed on the RTX 5070 Ti
in 1.258 s with 9 finite records and 40,897,536-byte peak allocation. For
length pairs 584/585/586, manual BF16 eager showed raw max differences
0.000488–0.000977, while the FP32-accumulation and SDPA paths were exactly zero
for this synthetic tensor. This is a numerical-runtime observation only; it
contains no video, QA, or faithfulness evidence. The privacy-safe synthetic JSON
SHA is `81bbea1e6c9ba913175f5b7d0d374acb5b0acdf6d43b7ad5199b0c8ae241fea3`.

A CPU-only analyzer and two tests now summarize privacy-safe temporal artifacts,
checking frame ordering, status, and requested/effective/failure accounting. The
published summary covers the new 36-, 40-, and 64-frame conditions; the earlier
32-frame result remains in its original dated artifact because its schema
predates the normalized temporal fields.
