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
summary arithmetic. After the first telemetry run, the sampler was hardened to
snapshot safely if shutdown times out, always remove the model trace hook,
reject a second start, and count malformed/failed counter queries. The expanded
cache-runner suite passes 16 tests. A separate agent audit found no privacy or
cache-correctness blocker; it cautioned that one-second polling is coarse and
may slightly perturb host scheduling.

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

## WSL memory and pending estimand audit

Read-only host-runtime inspection after the GPU run showed WSL-visible Linux
memory total 15,921,528 kB (about 15.2 GiB), `MemAvailable` 14,327,624 kB,
`Cached` 11,679,956 kB, swap 56 MiB used of 4 GiB, and memory-pressure averages
near zero. The largest process RSS in the sample was under 20 MiB. The
workspace filesystem reported 893 GiB available. This snapshot does not
support current WSL memory pressure; much of the displayed Linux usage is file
cache that Linux can reclaim. It cannot identify which Windows processes
contribute to a separate Windows Task Manager reading of 30/32 GiB, since that
host-wide view is not exposed by these Linux counters.

The independent CPU protocol audit found a real design-contract mismatch:
`experiments/2026-10-13_claim_selective_protocol_design.md` describes a
teacher-forced log-margin, while the claim-mask validator requires only generic
teacher-forced log-probability; `target_text_sha256` and `answer_text_sha256`
are undefined; and fixture labels say claim/nonclaim *tokens* are masked even
though the design maps a claim to a video interval and masks frames. The
design's baseline-correct primary cohort also differs from the broader
faithfulness plan, which specifies all eligible videos as primary and baseline
correctness only as a secondary analysis. No real-data experiment is currently
gated on this schema, and the reviewer recommended the minimal consistent
meaning of mean teacher-forced log-probability over a preselected claim span
under claim-mapped frame masking versus an equal-size temporal control.

A user preference question is pending before changing this estimand. Until it
is resolved, keep the validator fail-closed and do not build a dependent runner.
This does not block separate runtime, rights, or protocol-audit work.

At the 2026-10-14 03:09:55 JST checkpoint, 24 minutes 43 seconds had elapsed
and the 11:45:12 JST deadline remained. Faithfulness progress remains 79/79
prior direct samples with estimate −0.0756 nat and 95% t CI
[−0.1578,+0.0065], below +0.25 nat; this window adds zero direct samples.

## Independent recomputation of the closed faithfulness confirmation

The primary agent independently reran the concise CPU integrity analyzer on the
published result and run manifest. SHA-256 values were recorded for both inputs
and for the analyzer, and the analyzer reconciled 79 requested/effective
records, zero failures, unique sample keys, the 4-head/5-control intervention,
all per-record target-minus-control contrasts, and the published inferential
summary. It reproduced target mean drop −0.0601 nat, control mean drop +0.0155
nat, selective effect −0.0756 nat (SD 0.3667), 95% t CI
[−0.1578,+0.0065], p=0.0706 versus zero descriptively, and p=1.52e−11 versus
the +0.25 nat practical boundary. The registered success rule remains false.

This analyzer invocation intentionally omitted the private sample manifest, so
it reports unique video identities as unverified; unique sample keys alone do
not establish distinct videos. The run manifest binds a sample-manifest hash,
but this recomputation did not reopen the private mapping. This does not change
the frozen estimate, but is a remaining provenance audit gap and prevents an
SSS claim. No new faithfulness samples were collected. The single primary
contrast has no multiplicity adjustment; secondary subgroup analyses remain
excluded from this confirmation. Next, inspect whether the private manifest can
be safely hashed and checked in-process without exposing identifiers, or retain
the explicit uniqueness limitation if access/provenance is not independently
verifiable.

An independent auditor reproduced the estimates from the per-record contrasts
and confirmed that the input result, manifest, and analyzer hashes match this
checkpoint. It also notes that the +0.25 nat boundary was prespecified as a
design choice (approximately a 1.28 change in relative odds), but was not
empirically calibrated as a universal practical-effect boundary. The protocol
decision remains valid against the declared boundary; its substantive
calibration remains a limitation. The auditor likewise could not verify
video-identity uniqueness without the private sample mapping.

## Planned repeatability check for attention-backend resource results

The first eager-versus-SDPA comparison had one run per backend, so observed
runtime/resource differences could be run-order or clock variation. The new
question is whether the lower SDPA allocation and any runtime difference persist
on the same fixed input in a second run per backend. This remains an engineering
benchmark with one unique clip, not independent sample evidence. The sequence
will be SDPA then eager, opposite the first comparison's eager-then-SDPA order.
Each run uses the existing six-call smoke-validated path followed by 387 calls
(one prefill and 128 cached decodes in each of three conditions), 32 frames,
4-bit weights/BF16 compute, and the same pinned NOAA clip and Qwen revision.
Expected generation time is about 33–40 seconds per run, plus model setup;
the preflight requires at least 13 GiB free VRAM. New output names are unique,
and the runner refuses overwrite. Stop if the current free-VRAM gate fails.

### Verified repeatability observations

Both second-run artifacts used the same runner SHA as each other and the same
media/checkpoint/configuration. The independent GPU audit checked both new
artifacts and the earlier pair. All four completed 1/1 unique source clip,
3/3 conditions, and 387/387 model calls with zero failures; each condition had
129 calls. Hashes for the new SDPA and eager artifacts are respectively
`aeb3b1c1bb1580cb8268999bde1ac83e7cff45609ea3addb8f3f554b11fc072d` and
`9063b35e5786badffffc7cfc82762cabae127f93ab3f28916fe0b6a185f119db`.
The old SDPA artifact took 33.10 s and peaked at 6.728 GiB allocation; the
new one took 30.99 s with the same allocation. The old eager artifact took
36.50 s and peaked at 11.089 GiB; the new one took 41.30 s with the same
allocation. Thus, in both runs SDPA used 4.361 GiB (39.3%) less peak
PyTorch allocation. Runtime advantage varied from 3.40 s (9.3%) in run 1 to
10.31 s (25.0%) in run 2; the two-run means were 32.04 s SDPA and 38.90 s
eager, but this is descriptive only (n=2 per backend, single source clip).
The 1-second telemetry samples again showed brief 100% readings and sampled
condition means between 33.7–36.9% for SDPA and 36.4–44.8% for eager in run 2.

The run-1 eager artifact used a pre-SDPA runner hash, whereas run-2 eager and
SDPA share the same runner hash. The unchanged eager code path and same recorded
inputs make the comparison informative, but the runner-version difference is
another reason not to treat this as a controlled performance claim. No
cross-backend logits or kernel-per-layer introspection were added. The
artifacts contain derived counters/geometry only, have no raw text, IDs, media,
or local absolute paths, and were prepared for scoped publication. Focused
CPU validation passed all 30 tests across the cache runner and fail-closed
claim-mask validator; `git diff --check` passed. None of this is faithfulness
evidence; direct evidence remains the prior 79/79 result, with zero new samples.

Next, do not interpret two runs as a speedup claim. Continue only if a useful
runtime question justifies an additional design (for example, synchronized
per-decode-step timing to distinguish host gaps from accelerator work); keep
the claim-mask protocol validator fail-closed pending resolution of the
estimand/cohort mismatch and the independent human-review and data-rights gates.

## Per-forward CUDA event timing: implementation and smoke

The runtime repeatability check still could not separate time inside
`model.generate()` from model-forward work. A CPU support/audit review agreed
that same-stream CUDA events plus host wall-clock call/gap measurements can
provide a bounded timing breakdown if there is no per-call synchronization and
the result clearly avoids calling event time kernel-active time. The runner
now has an opt-in `--measure-cuda-forward-time` mode. It brackets each outer
multimodal `model.forward()` with start/end events on the current CUDA stream,
measures Python call and between-call wall time, and only reads elapsed events
after the existing single synchronization at the end of generation. The summary
stores one prefill time, decode mean/median/p90/p95/max, host-call summaries,
inter-call-gap summaries, and the exact measurement scope. Raw per-call timing
records, media, text, logits, and IDs are not written. The option is off by
default so prior benchmark paths remain unchanged.

The first GPU smoke used one fixed four-frame NOAA clip, SDPA, 16 cached decode
steps, and 3 conditions (51/51 calls, 0 failures). It took 12.03 seconds total,
with 6.056 GiB peak PyTorch allocation. Its artifact SHA-256 is
`79513f30d8d3cd46cdd2b0e12cb5d6d2fc5b3b7f64295f768db0afe9a080707f`. All three
conditions report 17 outer-model timing events pairs and 16 cached-decode
measurements, finite timing values, and the existing cache/position checks
passed. The first native condition had 1.070 s prefill stream elapsed, mean
45.36 ms across 16 decode events, median host call 39.26 ms, and median
inter-call gap 1.74 ms. The other two conditions had prefill event times near
300–318 ms, decode event means 43.81–46.54 ms, and median inter-call gaps
1.62–1.67 ms. These large prefill differences are not interpretable as a
position effect from this one smoke; warm-up/order/cache behavior are
alternatives. The first-condition p95 for all host calls was 1,069 ms because
it includes prefill; the later-condition all-call p95s were 220 and 201 ms.
Inter-call-gap p95s were 123, 99, and 100 ms. These long tails show pauses in
the observed sequence but do not identify their cause. After independent review,
the summary schema was refined to report prefill host time separately from
decode-only host-call percentiles; this first smoke artifact predates that
reporting refinement.

Focused CPU suite passed 19 tests, including event pairing, stage counts,
summary arithmetic, invalid timings, and existing cache/telemetry checks;
byte-compilation and `git diff --check` passed. The event interval may include
same-stream waits and omits work outside `model.forward`; it is not kernel-busy
time or a complete host/GPU decomposition. Next, run one 32-frame/128-decode
SDPA profile with timing enabled after a new VRAM preflight (387 expected calls,
about 31 seconds based on repeated SDPA runs, minimum 13 GiB free). This tests
the timing path at the established long context.

### Verified long-context timing result

The 32-frame SDPA run completed 1/1 clip, 3/3 conditions, 387/387 model calls,
and zero failures in 34.26 seconds; PyTorch peak allocation was 6.728 GiB.
Artifact SHA-256 is
`6b48b6c70aea25e721e4c85929da02308431e78d5c65942be35acf2c0e5afcb1`. Each of
the 387 outer `model.forward()` calls had two CUDA events, a host-call duration,
and (except the first) an inter-call gap; all values were finite and all
existing cache/position/finiteness checks passed. No synchronization occurred
per forward. CUDA event sums by condition were 6.929, 7.782, and 9.003 seconds;
condition wall durations were 7.304, 7.909, and 9.202 seconds. Median decode
event durations were 25.14, 52.49, and 51.46 ms, with nearest-rank p95 values
66.52, 72.68, and 98.41 ms. Median inter-call gaps were 1.62–1.71 ms and
p95 gaps 1.92–2.00 ms. Sampled mean GPU utilization was 36.0%, 30.75%, and
22.11% (9–10 one-second samples per condition); maxima were 85%, 100%, and
51%. These are three ordered conditions from a single clip and one run, not
independent repetitions or a sustained profile.

The event sums being close to condition wall time does not mean the GPU was
busy for those full intervals: events can include same-stream waits and are not
kernel-active counters. Short median inter-call gaps suggest little Python
between-call delay in the typical step, but do not explain utilization; rare
long tails, in-forward CPU work, stream scheduling, kernel occupancy, and
telemetry sampling remain plausible factors. The candidate condition's
utilization maximum was lower, but this was not a randomized or replicated
comparison and cannot establish that the position edit caused it. The result
adds no faithfulness samples.

### Instrumentation control and schema correction

An independent code/result auditor found that the first smoke's all-host-call
p95 mixed the prefill call with decode calls, making its p95 pause claim
misleading. The source was corrected to report `prefill_host_call_ms` and
decode-only host-call percentiles separately, while keeping the combined
summary. CPU tests cover the added fields and percentile arithmetic, and the
revised four-frame/16-decode smoke passed 51/51 calls with 0 failures in
11.77 seconds (6.056 GiB peak). Its SHA-256 is
`d72100a5a12b5674ee0445f5532361d8b7e5247392fdf2acce737dc0b7294687`.
Across conditions, prefill host calls were 974, 184, and 192 ms; decode-only
host-call medians were 36.07, 29.84, and 42.08 ms, while inter-call-gap p95s
were 102, 82, and 96 ms. The unusually long first prefill host call illustrates
why the split matters; it does not by itself identify whether CPU or stream
waiting caused the time.

One timing-disabled 32-frame/128-decode SDPA run using the same worktree code
completed 387/387 calls, 3/3 conditions, 0 failures in 35.36 seconds at the
same 6.728 GiB peak. Its artifact SHA-256 is
`3044a145098663fd6ab4e266eb48890814c35c4dcb06c00329aa6e6bc58c7eed`. Compared
with the timing-enabled run (34.26 s), this single on/off pair does not show a
measurable slowdown from the optional events; the timed run was 1.10 s faster.
That direction is consistent with run-to-run/condition-order variation already
seen in the repeated backend benchmarks, so instrumentation overhead remains
unresolved. Per-condition times also shifted substantially while total runtime
stayed within about 3.2%. Peak allocation was identical. Both artifacts are
single-clip engineering runs, not an overhead estimate or faithfulness result.

The first 32-frame timing artifact lacks the new decode-only host-call fields.
Next rerun that long-context profile once with the refined committed schema
(387 expected calls, approximately 35 seconds, at least 13 GiB free VRAM),
then independently audit it. Keep both timing-enabled and disabled paths
available, and continue to describe events as same-stream elapsed rather than
GPU busy time. Direct faithfulness evidence remains 79/79 from the prior closed
head confirmation; this window adds zero direct samples.

## Planned SDPA long-context runtime comparison

Before running it, the discriminating question is whether Transformers' SDPA
backend completes the same fixed 32-frame, 128-cached-decode workload with
different runtime/resource use from the measured eager condition. The previous
four-frame attention diagnostic found backend-dependent numerical behavior at
short context, while the present 32-frame eager run recorded about 36.5 seconds,
11.089 GiB PyTorch allocation, and sparse mean utilization around 44–50%.
The upcoming SDPA run is a runtime/cache-path comparison only; this runner does
not retain raw logits for cross-backend numeric comparison, so it cannot answer
whether SDPA changes model scores or semantics.

Predeclared settings: one fixed, already-cleared NOAA clip; 32 frames;
Qwen2.5-VL-7B-Instruct pinned revision; 4-bit weights/BF16 compute; same prompt,
seed, processor profile, three native/explicit/candidate position conditions,
and 128 cached tokens per condition. Expected full work is 387 model calls and
about 40 seconds based on the eager result, with roughly 11 GiB PyTorch peak
allocation and a 13 GiB minimum-free-VRAM preflight gate. First run the fixed
six-call/one-decode smoke. No metric or cohort changes are involved, and even a
successful comparison advances zero faithfulness samples.

### Verified result

The SDPA smoke completed 1/1 sample, 32/32 frames, 3/3 conditions, and 6/6
calls without failures. Its positions, cache invariants, finite scores/logits,
and native-auto/explicit-native parity passed; peak PyTorch allocation was
6.728 GiB. The fixed 128-step SDPA run then completed 1/1 sample, 32/32 frames,
3/3 conditions, and 387/387 model calls without failures. Its artifact is
`results/qwen_video_cache_gpu_smoke_20261014_32f_sdpa_128decode_01.json`,
SHA-256 `2866224cee27610b4bde9d9b4460f4c791408090a29b151ba6df29920228cb16`;
the recorded runner hash matches the source used. It took 33.10 seconds and
peaked at 6.728 GiB of PyTorch allocation. Every condition reached cache length
4,384 and decode coordinate 181; all 387 calls passed and native automatic vs
explicit native remained exact.

Against the single eager telemetry run with the same input/settings (36.50 s,
11.089 GiB allocation), this one SDPA run was 3.40 s (9.3%) faster end-to-end
and used 4.361 GiB (39.3%) less peak PyTorch allocation. Per-condition forward
times were 9.48/5.75/8.21 s for SDPA versus 9.47/7.54/8.91 s for eager.
NVIDIA-reported sampled memory maxima were 9,354/9,338/9,334 MiB for SDPA;
sparse sampled mean utilization was 31.1/44.3/27.0%, with sampled maxima
100/100/99%. These telemetry means use only 6–10 samples per condition. The
first native-auto condition had nearly identical runtime (9.48 vs 9.47 s), and
condition order, clocks, and sparse polling remain alternative explanations for
the other runtime differences. No cross-backend logit comparison was run, so
this is an exploratory backend runtime/memory result, not numerical or
semantic equivalence. The run supports the narrow observation that SDPA
completed this particular workload with materially lower allocated memory;
it does not establish a general speedup or model-score effect.
The run manifest records that `sdpa` was requested through Transformers, but
does not introspect the installed model modules to prove which kernel each
attention layer selected. The result is therefore a requested-backend runtime
comparison, not a per-layer kernel audit.

The GPU is therefore not idle in this workload: observed conditions reached
100% sampled utilization and board power up to 202.81 W. However, the sparse
sampled means are below saturation, and the result does not identify whether
the limiting factor is sequential decode, host gaps, kernel selection, or
another source. Batch scaling would be a separate throughput design, not a
faithfulness experiment; the lower SDPA allocation creates apparent memory
headroom, but any batch increase must first verify exact model/processor inputs
and a safe maximum allocation. Direct faithfulness remains unchanged at 79/79,
−0.0756 nat (95% t CI [−0.1578,+0.0065]); this window adds zero direct samples.
