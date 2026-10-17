# Synthetic six-condition Qwen visual-key intervention smoke (2026-10-18)

## Purpose

Integrate the fixed-prediction selective-dependence arithmetic, frozen
full/evidence-mask/control-mask Qwen inputs, and the new internal visual-key
attention-bias hook. This is a synthetic implementation/capacity check only.

## Protocol exercised

The runner creates three deterministic 32-frame synthetic videos with dummy
five-choice questions. For each synthetic record it builds full, claim-interval
black-mask, and equal-count temporal-control black-mask batches. It verifies
input IDs match across the three image conditions and freezes all nine CPU
batches and their hashes before loading model weights. For each record, the
baseline full-video forward selects the top 10% video keys by mean final-query
self-attention over layers 5–16; that selector remains fixed across all
branches. The treatment adds `ln(2)` to those key logits at the final prompt
query in layers 5–16. Baseline and treatment full/evidence/control conditions
then use the same fixed A–E prediction and foil chosen from baseline full.
Three records × six conditions require 18 forwards.

## Verified setup and results

- Pinned Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; Transformers 5.18.0;
  PyTorch 2.11.0+cu128; CUDA 12.8; bfloat16 / 4-bit weights with bfloat16
  compute; RTX 5070 Ti.
- A separate CPU-only real-processor smoke built the same three-record,
  nine-condition synthetic bundle: 9/9 effective batches, exact frame-pixel
  reconstruction verified, zero model weights/forwards, and CUDA not
  initialized.
- GPU preflight observed 14.51 GiB free before model load. The completed run
  attempted and completed **18/18 forwards** and scored **3/3 synthetic
  records**. Each video had 4,096 video tokens; the fixed selector chose
  410/4,096 per record.
- Synthetic `ΔF` mean was **−0.1250 nat**, sample SD **0.1250 nat**. No
  confidence interval or hypothesis test was computed. These values have no
  research interpretation because both videos and QA are generated fixtures.
- Peak allocated/reserved VRAM was **10.83/12.44 GiB**. PyTorch reported
  8.67 GiB free at the end of the scoring phase; after process exit,
  `nvidia-smi` showed 14.51 GiB free, 1,140 MiB in use, and 1% GPU utilization.
  Total script time was **72.16 seconds**.
- Independent artifact checks confirmed exact requested/completed counts,
  fixed layer/bias settings, finite aggregate values, and no source question,
  claim, video identifier, or raw logits in the result.
- The focused protocol, freezer, batch-builder, selector-hook, and model-loader
  guard suites passed **52/52**. The full repository suite passed
  **1,412/1,412** in 32.18 seconds; `py_compile` and scoped
  `git diff --check` passed. The first CLI attempt stopped before model load
  because of a missing `argparse` import; that launch defect was fixed before
  the successful 18-forward run.

Aggregate result:
`results/synthetic_qwen_visual_attention_boost_six_condition_32frames_20261018.json`
(SHA-256
`d2cb883cc121989a1fd02239a5d684d473f45707433db5c511087da4b3d418f9`).
Runner SHA-256:
`98eeb482e7636a4c3d544b2a81875c73df0b7daf470ae7e60f08a7c2a7583852`.
Intervention helper SHA-256:
`c267e170725578b57f046ca69f46c1830779f55d82ffda906ea374682926723e`.
Fixed-prediction scorer SHA-256:
`b6d677973c34663cda6cad4da3a21cb29771992c1e6f3e72c10708d71ee5bbd6`;
condition batch builder SHA-256:
`7486c6e959e507add3190b2af91b92ddf3d9dc84035a11cd44b3c0f36cee14e7`;
atomic freezer SHA-256:
`b6e5c60bc51cdf913c91ec6a2d8acfdee2c4f715eadaf851d7d9ce7f1d569c92`;
attention selector SHA-256:
`061fa5dc96f1ac685953f4c0316873a4a0118c2db326668e76741f8e6802f161`.

## Interpretation and limitations

The run confirms that the frozen selector, six-branch fixed-label scorer, and
per-layer attention-mask treatment can execute together on the pinned model
without non-finite logits, branch-input mismatch, incomplete batches, or OOM.
It is meaningful engineering progress toward a direct experiment, but it is
not evidence that the treatment increases faithfulness. The synthetic negative
mean must not be used to tune the bias, selector, or layers. No real
NExT-GQA video or protected annotation was read, and project-data rights remain
unverified.

Direct claim-linked video-faithfulness progress remains **0/3 independently
human-reviewed, 0/3 scored, no real `ΔF` estimate/interval/test, and 0/15
verified browser-to-decoder correspondences**. The human review packet still
awaits review, player-clock/decoder-PTS calibration is incomplete, and no
production command yet binds the real target, review, timing, source, and
protocol receipts to this synthetic runner. No real-data score is authorized.

## Next action

Move this runner's six-branch control flow into a gate-controlled real-data
entrypoint that requires all-row review and timing/source validation before
model loading, preserves immutable batch and selector hashes, and atomically
withholds cohort summaries on any failure. First test every gate-failure path
without loading weights. Only after independent review and clock-to-decoder
calibration pass should the complete predeclared three-record real-video pilot
run. Do not use this synthetic `ΔF` to select a more favorable treatment.

## Complete-input binding follow-up (2026-10-18)

The first binding prototype hashed input token IDs and attention masks only.
Independent audit found this insufficient for visual inputs. The helper now
requires `pixel_values_videos`, `video_grid_thw`, and `second_per_grid_ts` in
each of its six batches, requires integer input IDs and binary integer/bool
attention masks, hashes each complete typed batch with `batch_sha256`, and
requires equality only for the three corresponding baseline/treatment branch
pairs. It retains separate text and full-batch binding digests. CPU tests reject
missing video payload/metadata, floating IDs, nonbinary masks, changed visual
payloads, and mismatched paired branches.

After this repair, the pinned GPU smoke completed **3/3 synthetic records and
18/18 forwards**. The result records three full typed-batch binding digests.
Peak allocated/reserved VRAM was **10.83/12.44 GiB**, and elapsed time was
**93.15 seconds**. The fixture reproduced a synthetic `ΔF` mean of **−0.1250
nat** and sample SD **0.1250 nat**; these remain fixture outputs, not an
efficacy estimate. No real video, annotation, or user data entered this run.

The smoke artifact is
`results/synthetic_qwen_visual_attention_boost_six_condition_fullbind_20261018_0340.json`
(SHA-256
`0a44d6c6c2382b1cc6db656bfa2a83fc0e4f7ae1c604828cafc30473eccc3448`). The
runner SHA-256 was
`66e0ad09a44c87ba2a55c997262858d4118a87734c0107feb83cffb808859e9a`; binding
helper SHA-256 was
`a6c1b68abc93881be623ae71190b426e657d5d2fbe56a51efbfb5de95e00c21d`. The
result is an engineering integration check only; both branches' evidence and
control inputs are external black-masked videos, not internal activation
ablations. Direct faithfulness progress remains 0/3 reviewed, 0/3 scored, and
no real `ΔF` estimate.

An earlier launch was interrupted during CPU construction/hash of the
synthetic processor batches before model load or any forward; its output is
marked `incomplete` and is not counted as a run. The final repository
discovery, after the fail-closed mask guard and its regression test, passed
**1,429/1,429** tests. `py_compile` and `git diff --check` also passed.
