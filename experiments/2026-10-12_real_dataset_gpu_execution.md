# 2026-10-12: real-dataset video-only GPU execution validation

## Purpose and frozen scope

The user requested a dataset GPU experiment. This protocol tests one actual local NOAA EX2503 dive03 video forward with Qwen2.5-VL-7B-Instruct, independently of the existing manual-QA pilot. It is a label-free execution/memory/processor consistency experiment, not a faithfulness, intervention, grounding, or QA accuracy experiment. One video and four uniformly sampled frames are fixed before outcomes; no outcome-dependent selection or parameter search.

NExT-GQA media exists locally, but the individual YFCC licenses remain unverified. Its old test manifests are not fresh development cohorts. No NExT inference will be performed here. The original suggestion of three NExT development videos is withdrawn after auditing these gates.

## Authorization and data provenance

The current user explicitly authorized dataset GPU work. That authorization covers this separate, reversible, video-only execution check. The local NOAA manual-QA README's human-verified inference attestation remains pending: this run does **not** complete it, certify human-reviewed labels, or enter the manual-QA evaluation protocol. No `samples.json`, gold, audit, or rights metadata is supplied as model input. Human validation of answer labels is not required to test a label-free tensor forward. Independent auditors raised the historical gate; the primary separates these scopes explicitly rather than treating the user request as human attestation. The source media and its independently verified use terms are the basis for the separate execution check.

- Media SHA256: `13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126` (runner verifies before inference).
- Exact original source: [NCEI dive03 300-second segment](https://www.ncei.noaa.gov/metadata/granule/geoportal/rest/metadata/item/EX2503_DIVE03_20250413T192459Z_ROVHD/html), checked live via browser on 2026-10-12 JST. Its download link matches the rights record.
- [NOAA campaign media terms](https://oceanexplorer.noaa.gov/okeanos-beyond-the-blue-media-resources/), checked live: 2025 operation images/video are public domain with source credit.
- Credit: NOAA Ocean Exploration.
- Cruise citation: NOAA Ocean Exploration (2025), EX2503: Beyond the Blue: Papahānaumokuākea ROV and Mapping, NCEI, https://doi.org/10.25921/vgk8-m692 (subset used).
- Model terms: [official Qwen model card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) lists Apache 2.0. Cached revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- No raw media, QA/answer text, local absolute paths, weights, or personal identifiers will be published. Derived numerical summary only.

## Setup and pre-run checks

RTX 5070 Ti; 16,303 MiB nominal VRAM; CUDA visible through per-command elevated execution. Existing `.venv`: torch 2.11.0+cu128, transformers 5.17.0; torch CUDA 12.8. CUDA memory check reported 15,738,077,184 free bytes. WSL RAM about 15 GiB, approximately 11 GiB available. No environment changes or downloads.

4-bit BF16 compute, eager attention, seed 20261012, fixed generic visual prompt, frame maximum dimension 224, processor pixel budget 200704, four requested frames, one full-video input. Video-only: OpenCV RGB frames, no audio. One forward, no generation, `use_cache=False`, `logits_to_keep=1`, all attention layers. Expected runtime a few minutes including model load; previous synthetic four-frame peak was 6.446 GiB, not a guarantee for this input. Require >=12 GiB free VRAM; fail on missing/hash-mismatched video, processor/grid/FPS mismatch, nonfinite tensors, or CUDA error.

CPU tests: 14/14 existing time-mapping/proxy unit tests passed. New runner byte-compiles. Actual dataset GPU results are pending at this preregistration stage.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python scripts/run_dataset_video_gpu_smoke.py --output results/noaa_video_only_gpu_smoke_20261012_01.json
```

Use a new filename on reproduction; existing outputs are refused. This command has not run at preregistration time. The primary will execute it, inspect the artifact, and publish scoped safe results.

## Interpretation and next action

A successful run would establish one finite real-video model forward and internal processor consistency at this sparse input size. It cannot establish visual correctness, semantic annotation fit, temporal evidence localization, intervention benefit, faithfulness, or novelty. Four frames across 300 seconds yield very coarse temporal coverage. No effect estimate, confidence interval, significance test, or multiplicity correction is applicable because there is no treatment/control outcome. The original human QA and NExT per-video rights gates stay open; do not reopen the closed head/layer/alpha searches.
