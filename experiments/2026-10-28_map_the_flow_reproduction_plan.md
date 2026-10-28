# Map the Flow: fidelity audit and full-reproduction plan (2026-10-28, written before any strict-reproduction run)

Sources: the paper text (arXiv:2510.13251v2, saved locally; items below were read by a delegated subagent from the full text and spot-checked by the primary agent: k=9 window, five TVBench tasks, Table 3 numbers, "accuracy drops among correctly predicted samples"), the project page https://map-the-flow.github.io (links code and a model collection), the repository https://github.com/byminji/map-the-flow (README fetched; code not yet read; released 2026-03-03), and the Hugging Face collection byminji/map-the-flow (LLaVA-NeXT-7B-Video-FT, LLaVA-NeXT-13B-Video-FT, Mini-InternVL-4B-Video-FT, VideoChat2-IT-clean). Statements about the repository contents come from its README only.

## 1. What our earlier "WP0 replication" actually was
A **transfer**, not a strict reproduction: our own re-implementation of the four edge classes (cross-frame, video->text, video->last, text->last) on **Qwen2.5-VL-7B** (4-bit weights) on **NExT-GQA** (5 options, 32 frames) with the option-letter token probability at the last position, window k=9 (same as the paper), centers every 3 layers (2..26), baseline-correct samples only (same as the paper), %p_change formula (same). Four prespecified criteria were met. Differences from the paper: dataset (TVBench 5 tasks, 2-4 options, 8 frames), model (the authors' own video fine-tunes of LLaVA-NeXT / Mini-InternVL plus off-the-shelf VideoLLaMA3-7B), metric token (the paper: first sub-word of the complete answer; ours: the option letter), layer sweep density, quantization, prompt template (the paper does not specify the multiple-choice template). So "Qwen2.5-VL shows the same kind of pathway on a different benchmark" is supported; "we reproduced the paper" is not.

## 2. Paper inventory and our status
| # | Paper item | Status |
|---|---|---|
| 1 | Data: five TVBench tasks (Action Antonym, Action Sequence, Scene Transition, Moving Direction, Object Count), 8 frames, correctly answered samples | not done on TVBench (queued: 4 of 5 tasks; Action Antonym needs NTU RGB+D from its own site) |
| 2 | Fig 2 cross-frame knockout, video-FT vs image-only model | transfer only (Qwen, NExT-GQA); strict: not done |
| 3 | Table 2 accuracy drop, cross-frame blocked in the first half of layers (-24.1/-20.2/-18.0/-44.8/-60.8) | not done |
| 4 | Fig 3 video->question, video->last, question->last | transfer only |
| 5 | Figs 4-5, Table D logit lens on video tokens (spatial vs temporal keywords) | not done |
| 6 | Fig 6 / E video->question attention maps with and without cross-frame blocking | not done |
| 7 | Fig 7 question parts (non-option question, true option, false option) -> last | not done |
| 8 | Fig 8 flow into the true option (direct vs via the non-option question) | not done |
| 9 | Fig 9 layer-wise true/false option probability | not done |
| 10 | Table 3 / E effective-pathway pruning (58% of edges suppressed in 7B-Video-FT; TVBench and TOMATO accuracy; random blocking control) | not done |
| 11 | Table A LongVideoBench with the pruning | not done (data size) |
| 12 | Figs U-X long-video flow (VideoLLaMA3, LongVideoBench) | not done |
| 13 | Fig Y spatial (Video-MME) | not done |
| 14 | App B.1 open-ended single token; B.2 VCGBench multi-token anchors | not done |
| 15 | Sec 4 / Fig 10, Z failure cases (wrong-answer probability) | not done |
| 16 | Fig AA / G.4 progressive blocking (leakage check) | partly (NExT-GQA, WP0 cohort, video->text and text->last only) |
| 17 | Table B second-half cross-frame (-0.5/-0.7/-0.8/-1.7/-1.2) | not done |
| 18 | Fig AB window size k in {1,5,9,13} | not done |
| 19 | G.7 speculation about late-layer blocking | no experiment in the paper |
| 20 | Other models (13B, InternVL-4B, VideoLLaMA3) | not done |

## 3. Strict-reproduction route (new, because code and checkpoints are released)
1. **Official code + official checkpoints** (LLaVA-NeXT-7B-Video-FT, base llava-v1.6-vicuna-7b-hf) run through the repository's own analysis scripts on TVBench, compared with the paper's numbers (Table 2 and Table B drops, Table 3 accuracies and the 58% edge count). A match of the paper's own numbers is the test of faithful reproduction.
2. Obstacle: the repository pins torch 2.2.1 / CUDA 11.8 / transformers 4.40.0; this machine's RTX 5070 Ti (Blackwell) needs a recent CUDA build of torch. Plan: an isolated virtualenv outside the research repository (python 3.12, recent torch cu128, transformers 4.40.0, tokenizers 0.19.1); the research `.venv` is untouched. Deviations forced by the port (precision, attention backend, memory: a 7B fp16 model on a 16 GB card may need 8-bit weights) will be recorded and their effect on the paper's numbers reported.
3. Data: TVBench (4 of 5 tasks downloaded, license CC-BY-4.0 with an Apache-2.0 notice for source datasets; Action Antonym videos from NTU RGB+D need the official site and are not available); TOMATO, LongVideoBench, Video-MME, VCGBench: to be checked per dataset (size, access, license) before any download.
4. Order: (a) information-flow script, cross-frame target, Table 2 / Table B check; (b) the other targets (vql-to-ql, question-and-options-to-last, vq-to-true-opt); (c) generation-probability tracing; (d) effective-pathway pruning on TVBench (and TOMATO if obtainable); (e) logit lens; (f) failure cases; (g) the other models and the long-video/spatial/open-ended parts as data allow.
5. In parallel, our own Qwen2.5-VL TVBench run (already queued after the running jobs) gives the same analysis on a different model with a common benchmark.

## 4. What is new in this thesis relative to the paper (to be stated in the talk; claims limited to the models, data and metrics tested)
The paper maps where video information flows (layers, edge classes) with benchmark-level questions that have no evidence annotation. This thesis adds: (N1) evidence-specific pathway dependence: with annotated evidence intervals, knocking out the evidence groups on the video->text pathway hurts more than an equal-count sham (WP1, N=112, -0.68 nat; 7B) with a dose-response in the overlap (three analyses, +0.35..+0.56 nat), the direct video->last route unused; (N2) the model's own cited interval as the evidence: necessity (~0.3 nat), sufficiency, input-level agreement, citation stability (WP2-WP6), including that most citations are start-of-video defaults and that the effect is carried by non-start citations that coincide with the annotation (post hoc, WP7 replication running); (N3) a test of whether pathway knowledge helps annotation-free correction (causal self-localization raises the margin but only sharpens confidence; order-averaging and crop-averaging do not reduce errors); (N4) a cross-model check (3B: direction reproduced, effects weaker). None of these is a faithfulness verdict; they are dependence measurements. A literature scan (experiments/2026-10-28_related_work_scan.md) found no paper with the same combination, within a limited search.

## 5. Limits and honesty notes
The strict reproduction has not been run yet; no number of the paper has been reproduced by us so far. The subagent's inventory is not a substitute for reading the official code; the code will be read before use. If a paper number cannot be matched, the discrepancy and its likely causes (port, precision, unspecified template, sample counts the paper does not state) are reported as such.
