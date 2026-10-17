# Internal visual attention-access ablation feasibility (2026-10-18)

## Purpose

Prototype an internal intervention compatible with the fixed-prediction margin
contrast in `docs/internal_intervention_grounding_faithfulness_plan.md`. Earlier
six-condition GPU work used black-masked video inputs and did not implement this
internal ablation.

## Operation and verification

`scripts/qwen_internal_attention_ablation.py` clones the Qwen eager 4-D
additive attention mask and sets caller-supplied visual key positions to
`-inf` on one fixed causal answer-query row. Other rows and positions remain
unchanged. It rejects malformed/cache-shaped masks, selected keys that were
already hidden or biased, NaN/+infinity, and masks with no zero-valued causal
key remaining. Hook installation freezes positions and supports cleanup. The
installed Transformers 5.18 Qwen eager implementation adds this mask to
query-key logits before softmax; an independent code audit checked that source
behavior and helper.

CPU tests cover exact masking, untouched positions, invalid mask cases,
`finfo.min` rejection, position freezing, and cleanup (**5/5 passed**). The
temporal-grid mapping and runner fixture tests pass (**6/6 focused tests**).
The full repository test suite passed **1,435/1,435** after the runner was
classified as an authorized synthetic engineering loader. The current
self-contained runner fixture has one deterministic-fixture CPU test.

A final synthetic-only GPU smoke is recorded in
`results/synthetic_qwen_internal_attention_ablation_20261018_0445.json`. It
constructed three synthetic clips and forwarded one through the pinned
Qwen2.5-VL-7B checkpoint: baseline and ablated branches completed **2/2**, both
with finite logits. It masked three temporal groups (768 visual keys) in
1-based layers 5, 10, and 16. Runtime was 19.46 s. On the RTX 5070 Ti, peak
allocated/reserved VRAM was 10.75/12.44 GiB out of 15.89 GiB; after execution,
1.94 GiB was free. Environment: torch 2.11.0+cu128, CUDA 12.8, Transformers
5.18.0, bf16 model / 4-bit quantization / bf16 compute, eager attention.

The runner reports synthetic mapping diagnostics: max adjacent PTS gap 1.0 s,
first/last clipped edge-cell duration 0.5/1.5 s, and maximum group support
duration 2.5 s. These are descriptive values only; there is no prespecified
acceptance threshold or coverage guarantee. They do not indicate real-video
timing quality.

Runner source SHA-256:
`20a5a994442c65cce8770d5cbdf97e2ed56e8b5b700871c399886b3da09d33c2`.
Temporal mapper SHA-256:
`98ee69dd153e679cc6a9e564967592a6f808816bd8bd748609c14779f8d6373b`.
The JSON records these and the ablation helper and batch-builder hashes.
Result SHA-256:
`9321bf3723c396edfdc45c1ca11f221fcb6bf72df6b2494ec67c1332bf481c71`.

## Interpretation and limitations

This tests only direct access from the selected answer-query row to selected
visual keys in selected layers. It does not remove information propagated
through other visual/text states and cannot establish full evidence necessity
or explanation faithfulness. The helper does not authenticate a reviewer,
source clock, video, processor batch, or scoring authorization. The caller must
use `use_cache=False`; the square mask shape alone cannot prove cache state.
`-inf` behavior was tested only for the eager backend. All videos/times in the
GPU smoke were synthetic. No real-data score or treatment effect was measured.

Direct faithfulness evidence remains **0/3 independently reviewed, 0/3 scored**,
with no real-data `ΔF` estimate, confidence interval, or test. Human review is
incomplete.

## Next action

Bind PTS, duration, processor grid, token positions, and frame-sampling metadata
to one immutable processor batch. Then establish source/browser-clock
correspondence and independent claim review before any real-data scoring.
Treat a future real-data result as an internal-access contrast under the tested
layers/query/backend only; it will not by itself prove complete evidence
necessity or general faithfulness.
