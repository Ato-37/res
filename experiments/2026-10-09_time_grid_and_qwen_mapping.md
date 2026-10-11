# 2026-10-09: NExT-GQA time-grid audit and Qwen preprocessing smoke

## Purpose

Continue the Ubuntu PDCA loop after the initial data audit. Quantify exact video-key coverage and the relation between human gold intervals and the NExT-GQA 32-point time grid; then repair and smoke-test the existing Qwen grounding preprocessing so the temporal spacing supplied to Qwen matches the actual extracted frames.

This is measurement/preprocessing validation only. It does not test an internal intervention, grounding performance, or faithfulness.

## Setup

- Data audit: `scripts/audit_grounding_plan_data.py`, schema v2, reusing local validation/test CSV, gsub, video map, and frame2time inputs. Keys and rows are compared locally but never emitted.
- Processor smoke: `scripts/smoke_qwen_video_time_alignment.py`; cached Qwen2.5-VL processor only, eight synthetic 224×224 frames spaced 5 seconds apart. No model weights or video data were loaded.
- Code change: `scripts/grounding_check.py` now computes effective sampled FPS from selected timestamps, passes Qwen-compatible `video_metadata`, caps pixels per frame, checks the actual temporal grid against the configured temporal patch size, and reports skipped samples separately from evaluated samples.
- Environment observed: RTX 5070 Ti, 16 GB; `.venv` uses PyTorch 2.11.0+cu128 with CUDA and BF16 support and Transformers 5.17.0. System Python 3.14 has no torch. The Hugging Face Qwen2.5-VL-7B-Instruct cache was only 12 MB, so model weights are not present.
- Tests: 5 data-audit unit tests and 3 time-mapping unit tests; Python byte-compilation and `grounding_check.py --help`.

## Verified results

### Data key/time-grid audit

| Split | CSV rows | Rows with all metadata keys | Valid intervals in denominator | Intervals containing a grid timestamp | Coverage |
|---|---:|---:|---:|---:|---:|
| validation | 3,358 | 3,358 | 3,926 | 3,854 | 98.17% |
| test | 5,553 | 5,553 | 6,560 | 6,479 | 98.77% |

All unique video IDs present in each CSV had exact gsub, frame2time, and video-map keys. Validation excludes 3 rows whose intervals exceed the annotated video duration. Test excludes 36 rows containing at least one malformed interval value. The interval-coverage denominator includes only structurally valid, duration-valid annotations with a finite matching frame2time vector; no annotation values or IDs are in the summary.

- Distinct timestamp points per video: validation median 32 (range 21–32); test median 32 (range 21–32).
- Positive adjacent grid spacing: both splits median 1.0 s, maximum 5.5 s.
- Nearest grid-point distance to interval midpoint: validation median 0.25 s, maximum 2.3 s; test median 0.25 s, maximum 2.6 s.
- Nearest grid-point distance to either interval endpoint: validation median 0.30 s, maximum 3.7 s; test median 0.30 s, maximum 4.9 s.
- No frame2time vector was non-monotonic. Adjacent duplicate timestamps remained in 63/570 validation and 123/999 test vectors.

These are deterministic, descriptive counts over the local tables, not estimates from a random sample. “Coverage” means at least one timestamp in the NExT-GQA grid falls inside an annotation interval. It does not show that a generated claim is supported, that a Qwen-sampled frame equals one of the dataset frames, or that a visual token causally depends on that interval.

### Qwen processor smoke and implementation repair

The first synthetic processor attempt passed `qwen_vl_utils`'s `fps=[0.2]` return value directly. Transformers 5.17.0 rejected the list because its typed `fps` parameter expects a scalar. No inference was run in that failed attempt.

The successful route requests `return_video_metadata=True`, passes the metadata object with `do_sample_frames=False`, and sets the synthetic average sampling rate to 0.2 FPS. The processor smoke then verified:

- 8 input frames → temporal patch size 2 → 4 time-grid steps.
- 10.0 seconds per temporal grid step, matching the 5-second spacing of pairs of input frames.
- 1,024 video placeholder tokens, consistent with the processor's spatial/temporal grid.
- Result: passed; CPU preprocessing only. The explicit per-frame pixel cap avoided the previous warning.
- Unit tests: 8/8 passed (5 audit + 3 time-mapping); byte-compilation and CLI usage passed.

## Interpretation

The local NExT-GQA time grid has high interval inclusion under this simple descriptive criterion and complete key coverage. It is therefore plausible to use it for further temporal preprocessing validation, but this is not proof of Qwen-token alignment. The synthetic test verifies that this installed processor receives coherent timestamps and creates the expected grid for synthetic input; it does not validate the model's video understanding or attention behavior on real videos.

The source-code review identified a real issue in the legacy `grounding_check.py`: it discarded the sampling-rate information from `qwen_vl_utils`, while Transformers uses video metadata to determine seconds per temporal grid step. Its old chunk count was inferred indirectly from frame count and could silently map the wrong timestamps. The revised code explicitly checks temporal-patch count and uses the processor metadata API. The reported hit rate remains a grounding proxy, not a faithfulness or causal-dependence measure.

No model inference, intervention, outcome comparison, significance test, confidence interval, or multiplicity correction was run. The Qwen 7B weights are not cached; the installed GPU alone is insufficient to run inference.

## Limitations

- The audit's 32-point grid is NExT-GQA metadata. The legacy script re-samples source videos; actual sampled-frame timestamps must be obtained from the source video and retained in the per-run artifact.
- A scalar average FPS can encode uniform timestamp spacing only. The current uniform endpoint sampler approximates this; exact nonuniform timestamps would require a model/API path that preserves per-frame time positions.
- Temporal patch bounds are coarse: attention is aggregated over all spatial tokens in each temporal grid step.
- Neither high interval coverage nor attention overlap establishes faithfulness; masking/activation interventions can be out of distribution and can affect unrelated evidence.
- No video file was decoded and no Qwen model weights were loaded. Runtime, VRAM, failure rate, and effect size for a real-video run remain unknown.

## Next action

1. Confirm local availability and license of a small NExT-GQA source video without publishing its path or content. If available, run a one-video processor/model smoke only after the Qwen weights are available; explicitly estimate download/storage, runtime, and VRAM first.
2. Before using any grounding-hit outcome, test the revised extractor on a tiny local source video and verify sampled timestamps and metadata together; record requested/effective frame counts and failures.
3. Continue implementing the proposed KFP-inspired internal evidence intervention only after its fixed control, target claim/score, and sample protocol are specified. Do not restart the closed head/layer/alpha search or call this preprocessing audit a faithfulness result.
