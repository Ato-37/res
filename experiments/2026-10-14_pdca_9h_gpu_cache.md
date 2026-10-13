# PDCA 9-hour window: 32-frame GPU cache-path diagnostic (2026-10-14)

## Window and current checkpoint

The requested nine-hour window began conservatively at 2026-10-14 02:45:12 JST,
the first clock reading after the request, and ends at 11:45:12 JST. At the
02:50:30 JST checkpoint, about 5 minutes 18 seconds had elapsed; the window is
still active. Continue to the next justified research/protocol task at each
branch boundary and record later checkpoints in this entry or linked dated logs.

## Plan and reason

The previous dated log recommended 32 frames as the high-resolution safe default
after a 64-frame OOM and 36/40-frame memory observations. The existing cache
runner only covered four frames, so this cycle extended its fixed profile to 32
frames to check whether prefill positions, fresh KV cache lengths, and cached
decode coordinates remain consistent at longer video context. This is an
engineering compatibility question; it does not reopen head/layer/alpha search
or the closed head-faithfulness confirmation.

The run used the fixed, already-cleared NOAA DIVE03 media hash
`13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126`, pinned
Qwen2.5-VL-7B-Instruct snapshot
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit quantization with BF16
compute, eager attention, and seed 20261012. The 32-frame profile uses a 1024
maximum extraction dimension, 200704 max pixels per frame, and requires at least
13 GiB free VRAM before model loading. Initial check: RTX 5070 Ti, 14,885 MiB
free, no other CUDA process listed, WSL available RAM about 13 GiB. The GPU run
was individually elevated because the normal WSL sandbox hides `/dev/dxg`.

## Implementation and validation

`scripts/run_qwen_video_cache_gpu_smoke.py` now offers explicit fixed 4- and
32-frame profiles. The 32-frame processor profile is fail-closed at grid
`(16,24,44)` and 4,256 total input tokens. The temporal-position guard checks
all 16 native anchors against the installed integer-interval rule and all 16
candidate anchors against the declared float32 fractional rule. CPU tests retain
the four-frame regression, check both processor profiles, and reject a mutated
32-frame candidate anchor. The independent read-only protocol audit found no
privacy or provenance blocker; it did identify the per-anchor assertion gap,
which was repaired before the final run.

Focused validation passed: 10 CPU unit tests, Python byte-compilation, CLI help,
and `git diff --check`. The two-frame-profile processor test uses synthetic
frames; actual media extraction geometry is separately recorded by the run and
bound to the source/media hashes.

## Verified GPU result

Final-code result: `results/qwen_video_cache_gpu_smoke_20261014_32f_02.json`.
The artifact SHA-256 is
`6c6e2b36b81f51fee37376c625291065b75fc8130dbd295c9e7cbbff747efd90`. Its
recorded runner SHA matches the current runner file. The result binds the media,
checkpoint, configs, helper sources, and installed Transformers/Qwen processor
code by hashes; the source Git commit was `8bdaa79586f28ab8a03dd88df363122bbafbc72b`
before this window's uncommitted implementation was committed.

Requested/effective/failure counts were 1/1/0 samples, 3/3 conditions, and 6/6
model calls. The actual extraction produced 32 frames, grid `(16,24,44)`, 4,224
video tokens, and 4,256 total input tokens. Each condition recorded one full
prefill and one single-token cached decode: cache lengths `0 -> 4256` and
`4256 -> 4257`; all position checks passed and all generated scores/raw logits
were finite. Native automatic and explicit-native sequences, scores, and logits
matched exactly. Maximum PyTorch allocated memory was 11.089 GiB. During the
first run, a single live NVML sample observed 95% utilization, 252.49 W, and
14,436 MiB used; this is one point-in-time observation, not sustained utilization
telemetry. The second, final-code run also completed in 22.88 seconds.

The native temporal anchors were `[15,53,91,129,167,205,243,281,319,357,395,433,471,509,547,585]`; the candidate anchors were `[15,53,92,131,169,208,247,285,324,363,402,440,479,518,556,595]`. The candidate changed 3,696 temporal entries while non-temporal coordinates were preserved. These values validate only the implementation's coordinate and cache invariants under this fixed processor/model/runtime setup.

## Faithfulness progress and interpretation

Direct faithfulness progress remains the frozen 79/79 head confirmation from
the prior experiment: estimate `−0.0756 nat`, 95% t interval
`[−0.1578,+0.0065]`, below the prespecified `+0.25 nat` practical threshold;
the head search remains closed. This cycle added zero direct faithfulness
samples and zero efficacy estimates. NOAA human-review attestation remains
incomplete, so this cache-path check does not authorize claim-selective QA or
faithfulness evaluation. NExT-GQA item-specific rights also remain unresolved.

No answer text, token IDs, raw logits/scores, annotations, QA labels, or local
absolute paths are in the result. The summary contains derived aggregate
geometry, cache and runtime measurements only.

## Limits and next action

The run covers one fixed NOAA clip, one pinned checkpoint, one precision and
processor profile, and three cache/position conditions. It establishes neither
candidate temporal semantics nor answer correctness, grounding, human
faithfulness, statistical performance, or generalization. The single sampled
GPU-utilization point is not a utilization profile. SSS is not claimed.

Next, continue useful CPU-only protocol work on the claim-mask validator while
the human-review and dataset-rights gates remain closed. Review structural type
and source-commit validation, add narrow tests if justified, and keep the
validator explicitly non-authorizing. Do not launch another inference unless a
new discriminating question or a cleared upstream gate supports it. At the next
substantive checkpoint, reread the clock and update elapsed time against the
11:45:12 JST deadline.

## Repeated cached-decode horizon extension

### Plan and implementation

The first 32-frame smoke verified only one cached decode call after prefill. A
separate engineering invariant remained: whether the Qwen generation loop keeps
advancing all three rotary rows by one and grows `DynamicCache` by one across a
longer uninterrupted decode. The runner now permits exactly 1 (baseline) or 16
cached decode steps. Sixteen steps mean 17 generated tokens because the prefill
produces the first token; each of the three conditions therefore makes 17 model
calls, 51 total. CPU tests inject both an interior position error and an
interior cache-length error. The independent design/audit review found the
question distinct and the workload bounded.

### Verified GPU result

Result: `results/qwen_video_cache_gpu_smoke_20261014_32f_16decode_01.json`,
SHA-256 `5756469f633bd54d42365fd002b9f16cda6b5215b185a2400eaf84f95e9ebfc8`.
Its runner source hash matches the working file. Requested/effective/failures
were 1/1/0 samples, 32/32 frames, 3/3 conditions, and 51/51 completed calls.
Each condition checked one 4,256-token prefill followed by 16 one-token cached
decodes. Across the cached steps, past-cache lengths advanced 4,256 through
4,271, resulting lengths advanced 4,257 through 4,272, and all rotary rows
advanced from coordinate 54 through 69. All 17 score tensors and all 17 raw
logit tensors per condition were present and finite; output sequence length was
4,273. Native-auto and explicit-native outputs remained bitwise identical.
Maximum PyTorch allocation was 11.089 GiB and total elapsed time was 23.15 s.

A single concurrent NVML observation during this run saw 42% GPU utilization,
123.21 W, 8,639 MiB used, and 7,357 MiB free. This is an instantaneous sample,
not a utilization profile. The current `.venv` used torch 2.11.0+cu128,
Transformers 5.17.0, and CUDA 12.8 on the RTX 5070 Ti.

### Interpretation, limits, and checkpoint

The test supports repeated cache/position compatibility over this one fixed
32-frame input and the pinned software/model setup. It establishes no semantic
meaning for the candidate temporal positions, no correctness or grounding, and
no faithfulness. The direct-evidence count remains unchanged at 79/79 for the
closed head confirmation; no new faithfulness sample was run. The NOAA review
and NExT-GQA rights gates remain closed.

At the 2026-10-14 02:57:30 JST checkpoint, 12 minutes 18 seconds had elapsed in
the nine-hour window and the deadline remains 11:45:12 JST. Continue only with
another justified task; no further same-condition cache repetition is warranted
unless a code or runtime change changes the tested path. The claim-mask
validator CPU hardening is now in progress in
`experiments/2026-10-14_claim_mask_validator_hardening.md`.

## Long-horizon cached decode (128 steps)

### Plan and workload

Sixteen cached decode steps closed the short-horizon cache trace but did not
cover a long generated response. The next bounded engineering question was
whether the cache length and all three rotary rows remain consecutive over 128
cached tokens. The fixed profile still uses one cleared NOAA clip, one pinned
Qwen2.5-VL-7B-Instruct checkpoint, 32 frames, 4-bit weights, BF16 compute, and
the same three native/explicit/candidate prefill conditions. Expected work was
387 model calls (one 4,256-token prefill plus 128 single-token decodes per
condition), with no generated text retained. The preflight saw 14,652 MiB free
GPU memory and 14.2 GB host memory available; it passed the 13 GiB free-VRAM
gate.

The CLI now admits only the fixed horizons 1, 16, or 128. CPU tests check all
128 fake trace entries and corrupt both an interior and final decode position.
The focused suite passed 12 tests; `py_compile` and `git diff --check` passed.

### Verified GPU result

Artifact: `results/qwen_video_cache_gpu_smoke_20261014_32f_128decode_01.json`,
SHA-256 `3b7ec1d9fe21b4fe5ae1f8b7654de54a8fda63e01e6ab6432fcfd59c527783d8`.
Its runner hash `601c1ece8ae746305d34436c92a1c4aa29de96c24c8ef91a1c1f3f0cdda74724`
matches the current runner. Requested/effective/failure counts were 1/1/0
samples, 32/32 frames, 3/3 conditions, and 387/387 model calls. For each
condition, cache length advanced from 4,256 to 4,384 and the decode coordinate
advanced from 54 through 181. All 129 score/logit sets per condition were
finite; native automatic and explicit-native outputs were exactly equal.
Peak PyTorch allocation was 11.089 GiB and the full run took 39.37 seconds.
Per-condition generation took 10.90, 7.88, and 10.51 seconds respectively.

This verifies only cache/position stability for 128 generated tokens on this
single fixed input and runtime. It is not a QA, semantic, grounding, or
faithfulness result. No GPU utilization telemetry was sampled during this run,
so call count and allocated memory must not be described as utilization. Direct
faithfulness evidence remains the prior closed 79/79 result (−0.0756 nat,
95% t CI [−0.1578,+0.0065], below +0.25 nat); the current window adds zero
faithfulness samples.

## Item-level NExT-GQA rights gate audit

An independent read-only audit found no local YFCC license metadata or
item-rights manifest. The 10,000-entry NExT-GQA video-to-VidOR mapping contains
no license field; the local dataset README describes annotations and the video
mapping but makes no item-level rights grant. The repository's MIT license
does not establish rights to the separately downloaded third-party videos.
Primary-source review indicates VidOR videos came from YFCC100M, whose media
use terms are assigned per item by the original uploader, rather than under one
uniform dataset-wide license. See the [official VidOR page](https://xdshang.github.io/docs/vidor.html),
[NExT-GQA repository](https://github.com/doc-doc/NExT-GQA), its
[data README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md),
and the [Flickr/Yahoo YFCC100M license description](https://code.flickr.net/2014/10/15/the-ins-and-outs-of-the-yahoo-flickr-creative-commons-100-million-dataset/).

This is a conservative research gate, not a legal conclusion that every clip
is prohibited. Without an item-to-license join and review of the applicable
conditions for inference, retention, and publication, NExT-GQA video inference
and derived-result publication remain `no_go`. The independent audit neither
opened nor modified untracked user manifests or media. Next useful technical
work is a fixed-input comparison of the pinned Qwen eager and SDPA backends at
the already-used 4,256-token context, if the installed runtime supports both;
the question is whether backend numerical sensitivity changes at long context.
That result would remain an engineering diagnostic, not faithfulness evidence.

At the 2026-10-14 03:02 JST checkpoint, about 17 minutes 10 seconds had elapsed
and the 11:45:12 JST deadline remained active. Continue with the separately
scoped backend diagnostic or another useful audit; do not repeat the 128-step
condition without a code/runtime change.

## Qwen workload utilization measurement

The prior 128-step run proved substantial model work and VRAM allocation but
captured no utilization trace, so it could not directly answer why the GPU
looked idle at times. The runner now samples aggregate `nvidia-smi` utilization,
used/free memory, and power once per second during generation. It never queries
PIDs or process names and stores only per-condition mean, median, maximum,
sample count, error count, and duration. CPU tests validate field parsing and
summary arithmetic (14 cache-runner tests pass). A separate agent audit found
no privacy or cache-correctness blocker; it cautioned that one-second polling
is coarse and may slightly perturb host scheduling.

The 32-frame, 128-decode run was repeated only to capture this missing
measurement, producing
`results/qwen_video_cache_gpu_smoke_20261014_32f_128decode_telemetry_01.json`
(SHA-256 `af564e92cddffd52661ceacd8e23fcc80af34d2cd3060dd209291780b9b5f8f2`).
The runner hash in the artifact matches the source used. It completed 1/1
samples, 3/3 conditions, and 387/387 model calls with no failures in 36.50
seconds. Peak PyTorch allocation remained 11.089 GiB. During generation, the
three conditions produced 10, 8, and 9 telemetry points. Their sampled mean
utilization was 44.2%, 50.4%, and 45.3%, with sampled maxima of 100% for each;
sampled mean board power was 133.5 W, 160.8 W, and 153.2 W. NVIDIA-reported
memory-used maxima were 14,570, 14,559, and 14,555 MiB respectively. These
figures describe sparse samples over roughly 8–9 seconds per condition, not
continuous utilization or a sustained percentile profile.

Observation: the model did use CUDA and reached 100% in sampled moments, but
average sampled utilization stayed around 44–50%, while reported device memory
use approached the card's capacity. A plausible explanation is the sequential
single-prompt autoregressive workload plus host orchestration and a memory-heavy
video context; the current measurement does not isolate the cause. Raising
batch size may improve throughput but has little VRAM headroom and would need a
separate explicit bounded memory test. No artificial matrix load is warranted
for a faithfulness claim. The new telemetry is an engineering diagnostic only
and adds zero faithfulness samples or estimates.

At the 2026-10-14 03:06 JST checkpoint, about 21 minutes 10 seconds had elapsed
in the nine-hour window. Continue with a distinct, useful backend or protocol
check while respecting the closed data gates; do not repeat telemetry without
a code or runtime change.
