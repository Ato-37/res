# STEAR trigger and visual-selector GPU wiring smoke (2026-10-21)

## Purpose

Verify the modified Qwen STEAR-inspired engineering path from intermediate
layer entropy calculation through the trigger and, when triggered, the
attention-based visual-token selector. This is a synthetic wiring check, not a
STEAR reproduction or a faithfulness experiment.

## Setup

- Model: pinned local Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; 4-bit weights and BF16 compute.
- Runtime: RTX 5070 Ti, PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.18.0.
- Input: 8 generated synthetic frames, batch size 1, one final-query forward
  per run. The second run explicitly overrides the trigger threshold to `0.0`
  only to exercise the selected-token path; the paper-reported/default `0.85`
  was also tested separately.
- CPU contracts: `test_smoke_qwen_stear_trigger_feasibility` and
  `test_smoke_qwen_stear_attention_feasibility` passed **11/11**.
- Private response data, benchmark videos, and human judgments were not read.

## Results

| Condition | Trigger | Selected layer | KES neighborhood | Visual tokens selected | Runtime |
|---|---:|---:|---:|---:|---:|
| Default threshold 0.85 | No | 15 | none | 0/1024 | 26.09 s |
| Synthetic override 0.0 | Yes | 15 | 14–16 | 103/1024 | 27.43 s |

Both runs completed exactly **1/1 forward**. The maximum normalized entropy
was **0.768464**, below the default 0.85 threshold. In both cases, the
intermediate-layer projection matched the model's final output exactly at the
reported precision (`max_abs_error=0.0`). Peak allocated VRAM was **6.21 GiB**
and peak reserved VRAM **6.73 GiB**. The override run selected 103 of 1,024
visual-token positions (the top 10% under the implementation's fixed selector)
from the radius-one neighborhood around the selected layer.

Aggregate receipts are `results/smoke_qwen_stear_kes_synthetic_20261021_01.json`
(forced-trigger path) and `results/smoke_qwen_stear_kes_synthetic_20261021_02.json`
(default-threshold path). Each receipt's source hashes were independently
matched to the current runner and its two helpers.

## Interpretation and limits

This verifies that the current hook path captures all 12 specified intermediate
layers, reproduces final-layer projection, evaluates the entropy threshold,
and can select visual positions after a trigger. The 0.0 override deliberately
forces the engineering branch and is not a study threshold. The selector uses
mean attention and a fixed radius-one neighborhood; the smoke does not
implement evidence reinjection, counterfactual decoding, or a matched sham.
Synthetic input and one forward cannot show that the intervention improves
faithfulness or answer quality. No real-video or faithfulness sample was added.

Direct faithfulness remains **0/3** formally reviewed claims, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. There is no direct `DeltaF`, interval, or statistical test.

## Next action

Keep this as engineering feasibility only. Do not run a real-video efficacy
comparison until the claim/frame review gates and a frozen treatment/sham
definition pass. In particular, do not treat the synthetic threshold override
or paper accuracy gains as evidence for the thesis estimand.
