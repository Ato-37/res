# 2026-10-09 iterative round 3: Qwen synthetic forward smoke

## Purpose

Verify that the corrected video-time path reaches an actual Qwen2.5-VL model forward on a generated clip, and measure one short-input GPU memory point before considering any dataset evaluation. This is an execution smoke, not a research efficacy experiment.

## Provenance and setup

- Repository: `main`, commit `441415b06c22ed7d10003c3818270407fbee50eb`; live `git ls-remote origin refs/heads/main` matched this commit at check time.
- Model: `Qwen/Qwen2.5-VL-7B-Instruct`, resolved revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; downloaded to the external Hugging Face cache, never staged in Git.
- The five model shard symlinks resolved to existing blobs with byte sizes 3,900,233,256; 3,864,726,320; 3,864,726,424; 3,864,733,680; and 1,089,994,880. The model-specific cache directory reported only 12 MB because weight blobs live under the shared cache `hub/blobs` directory; the shared blob store reported 23 GiB.
- Synthetic test: `PYTHONPATH=scripts .venv/bin/python scripts/smoke_qwen_attention.py --model Qwen/Qwen2.5-VL-7B-Instruct --quant 4bit`.
- One generated 24-frame, 8 FPS MP4; the runner sampled exactly four frames, one sample total, 224x224 source frames with the configured pixel cap. No dataset video, annotation, question, or answer was loaded.
- Runtime environment: WSL Ubuntu; RTX 5070 Ti, 16,303 MiB total VRAM; 4-bit model; BF16 compute; WSL reported about 15 GiB total RAM, so host-reported 32 GB was not assumed available to the process.

## Verified results

- Model load and one actual video attention forward completed successfully.
- Requested/effective frames: 4/4; effective samples: 1/1.
- Peak allocated GPU memory reported by the runner: 6.446 GiB.
- The synthetic overlap score was intentionally discarded and is not reported as a grounding result.
- Earlier checks: 9/9 unit tests passed; Python compilation and CLI checks passed; synthetic processor metadata was consistent (`grid_steps=4`, `seconds_per_grid=0.750`, `video_tokens=1024`) for a separate eight-frame fixture.

## Interpretation and limitations

This validates that the installed stack can load this pinned Qwen checkpoint in 4-bit mode and complete one short synthetic video forward with all requested attention outputs under the observed memory conditions. It does not validate real-video timestamp fidelity, grounding quality, attention as causal evidence, intervention efficacy, faithfulness, statistical power, or memory use on longer/denser videos. One successful forward does not guarantee repeatability across other GPU states or input lengths.

No dataset inference was performed.

The broad novelty claim "first video-VLM activation intervention" is contradicted by Cai et al. (NeurIPS 2025), which studies temporal-aware activation engineering and includes Qwen2.5-VL. Generic causal video-question grounding intervention is also prior art (Chen et al., CVPR 2025). These works do not, on the current review, settle the narrower possible question of joint temporal localization and claim-level dependence. The literature search is not exhaustive; no novelty claim is established.

## Next action

1. Unit-test the grounding score construction on synthetic fixtures and label it unambiguously as an attention-overlap proxy, not faithfulness.
2. Complete a methods/estimand matrix for relevant activation-engineering, causal grounding, video hallucination, and mechanistic-faithfulness papers before finalizing a research gap.
3. Preregister one intervention estimand, comparator, claim/interval association, exclusions, uncertainty method, and minimum useful effect before any efficacy pilot. Keep the closed head/layer/alpha search closed.

## Reproduction command

This command has run once as stated above; it uses only a generated temporary MP4 and the external cached checkpoint:

```bash
cd /path/to/res
PYTHONPATH=scripts .venv/bin/python scripts/smoke_qwen_attention.py --model Qwen/Qwen2.5-VL-7B-Instruct --quant 4bit
```
