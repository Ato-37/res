# 2026-10-09 iterative round 2: temporal sampling audit and prior-art gate

## Purpose

Continue the Ubuntu PDCA loop with a synthetic end-to-end preprocessing check and an independent code/prior-art audit before spending GPU time on the proposed video-grounding intervention. This round tests timestamp representation and processor metadata only; it does not test model behavior, grounding quality, or faithfulness.

## Work performed

- Three read-only subagents independently reviewed video faithfulness/activation-intervention literature, NExT-GQA/VidOR rights, and time-mapping/VRAM implementation. Their findings were checked against official conference/model/dataset sources and the local installed implementation.
- Replaced endpoint-rounded frame indices with a constant integer-stride sampler. The extractor now uses decoder-reported timestamps and rejects nonuniform timestamps when they cannot be represented by one FPS value. It no longer fabricates 25 FPS when metadata is missing.
- Added a synthetic MP4 smoke that runs through OpenCV extraction and the cached Qwen2.5-VL processor. Added a future single-synthetic-video attention-forward runner, but it has not been run because model weights are absent.
- Added fail-closed preflight for CUDA/BF16, a 4-bit-only path on <24-GiB GPUs, at least 12 GiB free VRAM, and at most 8 requested frames on 16-GiB GPUs. Added exact video-token/grid count checks, requested/effective frame reporting, disabled KV cache, logits_to_keep=1, finite attention checks, and peak-VRAM reporting. The forward path has not been run against weights.
- Added a model download helper that checks free disk before writing to the external Hugging Face cache. It has not been run.

## Verified software checks

- Unit tests: 9/9 passed (5 audit tests and 4 time-mapping tests).
- Python byte-compilation and CLI help checks passed for the modified scripts.
- Synthetic source: 24 generated MP4 frames at 8 FPS. Requesting 8 samples now yields 8 equal-stride frames, 2.666667 sampled FPS, 2.625 seconds between first/last sampled frames, Qwen temporal patch size 2, 4 grid steps, 0.750 seconds per grid step, and 1,024 video tokens.
- The prior endpoint-rounded sampler used frame offsets [0, 3, 7, 10, 13, 16, 20, 23] for this synthetic clip: gaps were 3/4 frames, while a single average FPS concealed the nonuniform positions. The new uniform-stride sampler uses [1, 4, 7, 10, 13, 16, 19, 22]. This demonstrates and repairs a preprocessing mismatch; it is not a model or research result.

## Prior-art and data-rights gate

- Cai et al.,+§uçâç\Mitigating Hallucination in VideoLLMs via Temporal-Aware Activation Engineering+§uçâç] NeurIPS 2025, studies inference-time activation engineering for video hallucination and reports Qwen2.5-VL results on VidHalluc. This directly rules out claiming novelty for"éÝyø§yÜfirst internal activation intervention in a video VLM+§uçâç] Its target/evaluation is hallucination mitigation, not NExT-GQA temporal-IoU plus explanation/claim faithfulness.
  - Official abstract: https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html
  - Official paper PDF (model results, including Qwen2.5-VL, §§4ºw^~)Þw5 / Tables 6éÝyø§yÓ6): https://proceedings.neurips.cc/paper_files/paper/2025/file/5683583b88da51c79d2eb263f295286e-Paper-Conference.pdf
- Chen et al., ºw^~)ÞtCross-modal Causal Relation Alignment for Video Question Grounding+§uçâç] CVPR 2025, uses explicit causal interventions in a video question-grounding framework. Therefore generic causal intervention for video grounding is also not a novelty claim.
  - Official paper: https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Cross-modal_Causal_Relation_Alignment_for_Video_Question_Grounding_CVPR_2025_paper.html
- Kim et al.,+§uçâç\Map the Flow.éÝyø§yÝ ICLR 2026, analyzes causal information pathways in VideoLLMs. Mechanistic analysis alone is not a novel gap.
  - Official paper: https://proceedings.iclr.cc/paper_files/paper/2026/hash/537b3416a0c8035794a12b3fa596f941-Abstract-Conference.html
- The plausible unresolved question is narrower: whether a prespecified inference-time internal intervention changes both temporal evidence localization and selective dependence of a fixed answer claim on the model-selected interval, under a valid annotation/protocol. This is only a candidate gap; the literature comparison is not exhaustive and does not establish novelty.
- VidOR's official page says its videos are drawn from YFCC100M, but does not grant a uniform license for all clips. The individual source video and the annotation/question rights remain unverified. No dataset video or annotation question was used in this round.
  - VidOR official dataset page: https://xdshang.github.io/docs/vidor.html
  - YFCC100M source: https://multimediacommons.wordpress.com/yfcc100m-core-dataset/
- The official Qwen2.5-VL-7B-Instruct Hub page states Apache-2.0 and approximately 16.6 GB repository size. The local cache is 12 MB, so no weights are available yet.
  - https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main

## Interpretation and limits

The code-path failure found here is real and could have invalidated timestamp/chunk correspondence on arbitrary video lengths. The synthetic test confirms the repair for a constant-frame-rate MP4 with the local OpenCV decoder. It does not prove accurate timestamps for every codec or variable-frame-rate source. Such sources now fail closed when decoder timestamps are nonuniform at sampled frames, but decoder timestamp behavior still requires validation on licensed examples.

The processor check establishes only that the installed Transformers/Qwen preprocessing metadata is internally consistent with the synthetic sampled frames. No model weights were loaded; no GPU forward, output-attention measurement, intervention, dataset result, significance test, CI, or faithfulness estimate exists. Attention overlap remains a grounding proxy only.

The WSL environment currently reports 15 GiB total RAM (about 14 GiB available), despite the user-reported 32 GB machine capacity. It reports an RTX 5070 Ti with 16,303 MiB total and approximately 14.5 GiB free; filesystem free space is approximately 897 GiB. Any model smoke must therefore remain 4-bit, four frames, one synthetic clip, and use the recorded VRAM cap. A model download would transfer approximately 16.6 GB to the external HF cache, not to Git; this round's download helper has a 24-GiB free-space preflight.

## Next action

1. Do not run dataset-video inference until the exact clip license and annotation reuse rights are established.
2. First validate the 4-bit forward/attention code on one generated synthetic clip, if the authorized model download succeeds and the VRAM/RAM preflight passes. Treat it strictly as a runtime smoke and discard its synthetic overlap score.
3. Compare TA-AE, CRA, and ºw^~)ÞtMap the FlovéÝyø§yÝ methods/estimands against the proposed claim-level faithfulness test before freezing a research novelty statement or starting an NExT-GQA pilot.
4. Keep the closed head/layer/alpha search closed; do not choose a new metric or subset after observing outcomes.
