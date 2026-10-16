# Real-video Qwen CUDA path smoke (2026-10-17)

## Purpose

Verify the actual local NExT-GQA video decode → Qwen2.5-VL preprocessing → CUDA model-forward path before the separate score-bearing faithfulness branch is ready. This is a pipeline and resource diagnostic only; it does not estimate explanation faithfulness.

## Setup

- Runner: `scripts/run_nextgqa_real_video_gpu_smoke.py` (source SHA-256 recorded in the private run manifest: `cd2c60f96083eecb6e226cc18e9111c7a1820558cf28536d074b6a731d894076`).
- Selection: deterministic hash order over valid annotated local test videos absent from existing sample manifests; inspect candidates until one passes the 1 ms uniform-PTS sampling check.
- Requested/effective workload: 1 video, 4 PTS-derived frames, one Qwen2.5-VL-7B-Instruct forward; 4-bit quantization with BF16 compute, eager attention, pinned local checkpoint revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- No QA text, answer, annotation interval, generation output, or faithfulness score was used. Output was a mode-0600 aggregate local JSON; it omits sample identity/path, media, QA, and generated text. Pixel and PTS digests remain local and are not published.
- Hardware/software: NVIDIA GeForce RTX 5070 Ti (16 GiB), CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 5.18.0, bitsandbytes 0.50.2, qwen-vl-utils 0.0.14.

## Verified result

The runner inspected 3 candidates; 2 failed the uniform timing constraint and the third passed. The selected video decoded to 1,402 frames; all 4 requested frames were sampled and passed the decoder pixel check. The processor produced 536 input tokens and video grid `[2, 28, 36]`. One CUDA forward completed with finite logits.

| Measurement | Observed |
|---|---:|
| Runner-measured segment (candidate decode through forward) | 25.90 s |
| Model forward | 3.09 s |
| Peak PyTorch allocated VRAM | 6.04 GiB |
| Peak PyTorch reserved VRAM | 6.29 GiB |
| CUDA driver free VRAM after run | 8.30 GiB of 15.89 GiB |
| Requested/effective video frames | 4 / 4 |
| Model forwards / nonfinite logits | 1 / 0 |

Reproduction command (choose a fresh output path):

```bash
.venv/bin/python scripts/run_nextqa_real_video_gpu_smoke.py \
  --output results/nextgqa_real_video_gpu_smoke_run02.json \
  --frames 4 --uniform-tolerance-ms 1
```

The completed run's privacy-minimized output remains local. The command above is for reproduction and was not the literal output path used in this run.

The runner's measured interval begins after dataset/candidate discovery and environment imports; the 25.90 s figure should not be read as end-to-end process startup time.

## Follow-up: 32-frame resource smoke

After the four-frame smoke, a second bounded run used the same runner, model revision, quantization, and local-only output policy with 32 requested frames. Of 7 deterministic candidates inspected, 6 failed the fixed timing check and one passed. The selected video produced 32/32 sampled frames, 4,064 input tokens, grid `[16, 28, 36]`, and one forward with finite logits.

| Measurement | Observed |
|---|---:|
| Runner-measured segment (decode through forward) | 34.59 s |
| Model forward | 4.25 s |
| Peak PyTorch allocated VRAM | 10.50 GiB |
| CUDA driver free VRAM after run | 2.22 GiB of 15.89 GiB |
| Requested/effective frames | 32 / 32 |
| QA / interval / faithfulness effect | none / none / none |

The process released its model allocations afterward; a separate post-run check observed 14.85 GiB free. This is one technical example, not a worst-case estimate for the three claim videos. At 32 frames the peak approached the available card capacity, so an eventual nine-forward run must check each exact input's free VRAM and fail closed on an OOM rather than drop a record. No QA, answer, interval, generated text, or claim score was used.

A fresh-path command for reproducing this resource check is shown below; it has **not** been run at that output path:

```bash
.venv/bin/python scripts/run_nextgqa_real_video_gpu_smoke.py \
  --output results/nextgqa_real_video_gpu_smoke_32f_run02.json \
  --frames 32 --uniform-tolerance-ms 1
```

## Interpretation and limits

These runs establish that the pinned 7B model can process one local NExT-GQA video with both 4-frame and 32-frame inputs. They do **not** test answer correctness, explanation generation, grounding, or faithfulness; there is no effect estimate, confidence interval, or inferential test. Both runs are technical smokes on one selected video, not representative samples. The score-bearing claim-mask study remains blocked on independent review of 3 videos, 15 screenshot-to-decoder correspondences, a trusted pre-outcome joint freeze, and a production preflight.

Rights status is unverified. Raw media, local paths, model weights, QA, and per-frame digests remain private and are not included here.

## Next action

Decoder-frame PNGs are now prepared and their source/PTS joins independently verified. The next step is genuine OS/compositor capture, then crop and blinded visual review of all 15 screenshot-to-decoder pairs. Use the exact 32-frame input characteristics of the eventual cohort to check peak VRAM before scoring; keep every condition batch atomic and abort the cohort if any row fails. The GPU smoke itself remains non-authorizing and is not a faithfulness result.
