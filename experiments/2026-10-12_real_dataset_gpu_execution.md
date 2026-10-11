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

## Verified execution and primary audit

The reproduction command above **ran** on the existing `.venv` with individual GPU elevation. Result: `results/noaa_video_only_gpu_smoke_20261012_01.json`.

| Item | Observed |
|---|---:|
| Requested/effective videos | 1/1 |
| Requested/effective frames | 4/4 |
| Failures | 0 |
| Grid (temporal, height, width) | [2, 24, 46] |
| Video / total input tokens | 552 / 584 |
| Attention layers with finite values | 28/28 |
| Finite output logits | true |
| Peak allocated CUDA memory | 6.521810 GiB |
| Forward plus subsequent checks | 1.788918 s |
| Extraction/load/forward through summary construction | 11.549180 s |

Source commit at execution: `20a018f853e0a2ca8e9326e283cbed67e7ef82c9`. Result SHA256: `d3ac15aa1029cb4e339c2723dc8c654c2e8e88d733795249a94abaa9534b0415`. Primary independently rehashed all three referenced source files, checked requested/effective counts, the finite flags, and the absence of absolute local paths or dataset QA fields. No model generation or answer scoring occurred.

Sample timestamps were 0, 99.9999, 199.9998 and 299.9997 seconds. Two chunk spans [0,99.9999] and [199.9998,299.9997] are sparse preprocessing spans, not continuous evidence coverage. Processor `seconds_per_grid_ts=199.999802` matched `2 / sampled_fps` within the runner's fixed tolerance. Neither this identity nor finite tensors establish semantic timestamp validity for arbitrary videos.

Independent pre-run audits caught and resolved: arbitrary input overrides could mislabel provenance (media/hash now fixed together); eight frames had no memory evidence (runner restricted to four); runtime failures lacked an artifact (exception type/counts now preserved without sensitive message text); OpenCV package name mismatched the installed headless distribution (corrected and all version lookups checked before run). Preflight argparse failures still abort without a runtime result artifact. Initial normal-sandbox Git staging failed because `.git` is mounted read-only; scoped per-command elevated Git commits succeeded. No automatic review rejection occurred.

## Evidence-led stop / next action

The label-free real-data execution question is answered for this exact one-video/four-frame condition. Repeating the same forward or expanding to additional clips without a distinct question would not establish faithfulness and is not planned. No efficacy pilot is justified until the annotation/claim, control, estimand and rights gates are resolved. The historical NOAA human-QA attestation is still pending, and NExT clip-level rights are still unverified. This is a completed execution smoke, not an SSS efficacy result or thesis conclusion.

CPU verification (no GPU needed; these 14 tests already passed):

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest scripts/test_grounding_metrics.py scripts/test_video_time_mapping.py
```

For a separately authorized reproducibility check use a **new** result filename. No additional GPU repetition is scientifically required at this checkpoint.

Post-run overwrite guard was checked using the existing output filename: CLI exited 2 before CUDA/model imports and preserved the result; this was a CPU-only safety check, not a second experiment.

Final independent artifact audit passed: the auditor separately recomputed media, all referenced source, and pinned model-config hashes; reconciled counts/token math; and inspected privacy/claim boundaries directly. No critical execution-scope issue remains. The auditor emphasizes that sampled-pair endpoint spans differ from complete 200-second processor cells; this result must not be called a validated timestamp-to-evidence alignment. Independent `ffprobe` inspection found h264 640x360, CFR nominal/average 30000/1001 FPS, duration 300.033067 seconds and 8992 frames, consistent with the sampled endpoints within that duration. These are decoding metadata checks, not semantic evidence validation.
