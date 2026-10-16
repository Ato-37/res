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

## Interpretation and limits

This establishes that one local NExT-GQA video can traverse the pinned 7B preprocessing and CUDA forward path within the observed memory/time envelope. It does **not** test answer correctness, explanation generation, grounding, or faithfulness; there is no effect estimate, confidence interval, or inferential test. The single selected video and four frames are a technical smoke, not a representative sample. The score-bearing claim-mask study remains blocked on independent review of 3 videos, 15 screenshot-to-decoder correspondences, a trusted pre-outcome joint freeze, and a production preflight.

Rights status is unverified. Raw media, local paths, model weights, QA, and per-frame digests remain private and are not included here.

## Next action

Use the current v6 candidate plan to prepare decoder-frame PNGs and independently verify their source/PTS joins. Assemble the blinded 15-pair human review only after genuine video-only screenshots exist. Keep the result non-authorizing until the separate trust and scoring gates pass.
