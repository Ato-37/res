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
`results/synthetic_qwen_internal_attention_ablation_20261018_0605.json`. It
constructed three synthetic clips and forwarded one through the pinned
Qwen2.5-VL-7B checkpoint: baseline and ablated branches completed **2/2**, both
with finite logits. The typed batch hash matched immediately before and after
each forward. Evidence/control groups were `[2,3,4]` and `[5,6,7]`; selected
positions hash, query position, processor fingerprint, and sampling metadata
digest are recorded in the JSON. It masked three temporal groups (768 visual
keys) in 1-based layers 5, 10, and 16. Runtime was 18.32 s. On the RTX 5070 Ti,
peak allocated/reserved VRAM was 10.75/12.44 GiB. Environment: torch
2.11.0+cu128, CUDA 12.8, Transformers 5.18.0, qwen-vl-utils 0.0.14, bf16
model / 4-bit quantization / bf16 compute, eager attention.

The runner reports synthetic mapping diagnostics: max adjacent PTS gap 1.0 s,
first/last clipped edge-cell duration 0.5/1.5 s, and maximum group support
duration 2.5 s. These are descriptive values only; there is no prespecified
acceptance threshold or coverage guarantee. They do not indicate real-video
timing quality.

The JSON records hashes for all repository helpers and the imported Qwen model,
processor, image/video processor, and qwen-vl-utils source modules. Result
SHA-256: `b24c72f921a7a6a2e12acf56232e53d670fd436271f27a77945973d13dc41110`.

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

### Repeat smoke and audit (2026-10-18 18:54 UTC)

After an independent audit found missing before/after forward hash checks, the
runner was repaired to verify the exact typed batch immediately before and
after both the baseline and ablation forward. It now hashes the installed Qwen
processor and image-processor source files in addition to the model, video
processor, and `qwen-vl-utils` files. The rerun completed 2/2 forwards, finite
logits, and all five digest checks. A separate auditor independently matched
all 15 repository/installed-source hashes in the result to the current source.
The final result hash is
`b24c72f921a7a6a2e12acf56232e53d670fd436271f27a77945973d13dc41110`; runner
hash is `7f50633859e07d1ad88e4c2764c659d3309b75c637f509f0ca1fc9fd834abb3a`.
The full repository suite passed **1,438/1,438** in 36.05 s, and focused
compilation/whitespace checks passed.

The local checkpoint weight bytes are not hashed; this run therefore pins the
declared cached model revision, not cryptographic authenticity of every weight
or configuration file in that cache. The typed batch hashes bind rendered
input IDs and pixel tensors, but this synthetic check still says nothing about
real-video timing or score validity.

## Next action

The synthetic mapping and model-forward plumbing is now verified. The selected
thesis estimand still has no frozen non-circular improvement treatment. Next
freeze that treatment and its sham/control before outcomes, and independently
establish source/browser-clock correspondence and claim review before any
real-data scoring. Do not infer efficacy from this GPU smoke.
Treat a future real-data result as an internal-access contrast under the tested
layers/query/backend only; it will not by itself prove complete evidence
necessity or general faithfulness.
