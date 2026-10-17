# Review interval to Qwen temporal-grid positions (2026-10-18)

## Purpose

Make the mapping from a reviewed source-time interval to complete Qwen visual
key groups testable before scoring. The intended estimand ablates internal
access to groups associated with reviewed evidence and compares it with an
equal-key-budget control fixed without inspecting model outcomes.

## Implementation and verification

`scripts/qwen_review_interval_token_groups.py` validates input token-block
contiguity and the supplied processor-grid dimensions, merge size, temporal
patch size, sampled PTS, source duration, and review interval. It constructs
midpoint (Voronoi) frame-support cells, clipped to `[0, video_duration]`;
each temporal grid group spans the cells for its temporal-patch frames.
Evidence groups are those with positive-duration overlap with the review
interval. The control is a contiguous, disjoint window with the same number of
groups, minimizing duration difference, then center-time distance, then start
index. It reports the largest adjacent PTS gap, first and last clipped edge-cell
durations, and largest temporal-group support duration so sparse sampling is
visible to callers. These are diagnostics only; no gap threshold has been
preregistered. The rule is deterministic and the helper accepts no outcome or
score.

The focused CPU contract tests pass **5/5**, including sparse-PTS coverage and
diagnostic values, single assignment at an exact cell boundary,
duration-over-center control
priority, tie-breaking, and invalid geometry/time/control cases. An independent
audit reviewed the diagnostic formulas and found the implementation correct;
it caught a mistaken test expectation (the final clipped cell is 1.5 seconds),
which was corrected. A separate
`scripts/qwen_processor_batch_review_mapping.py` adapter now clones the full
CPU processor payload before mapping/hashing, enforces a single unpadded
sequence, integer IDs/grid, floating flattened video pixels and temporal
spacing, and checks `pixel_rows == T*H*W` and
`second_per_grid_ts == temporal_patch_size / FPS`. It hashes the complete typed
batch snapshot and a canonical digest of supplied frame indices, PTS, source
duration/count, nominal FPS, processor revision, geometry, and reviewed
interval. It reports maximum PTS cadence deviation from nominal `1/FPS` as a
diagnostic only. Independent audit caught and prompted repairs for empty pixel
width, mutable-batch wording, and one-shot interval binding. Focused mapping,
binding, ablation, and fixture tests pass **14/14**.

The adapter was exercised on an actual pinned-processor CPU batch built from
synthetic frames. Its schema was `(1,4163)` int64 input IDs/mask,
`(16384,1176)` float32 video pixels, grid `[16,32,32]`, and spacing `[2.0]` s.
For synthetic interval `[5,8]`, it selected evidence groups `[2,3,4]`,
disjoint control `[5,6,7]`, and equal budget 768 keys per arm. PTS cadence
deviation was 0.0 s. No real dataset or GPU was used for this CPU adapter check.
The returned digests bind the supplied snapshot and metadata for integrity;
they do not authenticate their origin.

`py_compile` and `git diff --check` pass. The full-suite count is recorded in
the companion internal-ablation experiment log.

## Interpretation and limitations

This is processor-layout geometry and integrity binding, not authenticated
evidence mapping. The
midpoint cells are an operational frame-support convention, not proof that a
Qwen temporal patch semantically represents every time inside the cell. It
assumes review timestamps, decoded PTS, and duration share an authenticated
zero-based seconds clock. Sparse samples can create wide edge cells. The
diagnostics expose sparse support but do not establish an acceptable threshold.
The adapter binds supplied values to a cloned batch payload, but cannot
authenticate grid/frame order, padding declaration, processor implementation
revision, timestamps, or source video; the processor revision and sampling
metadata remain caller-supplied. The control can overlap another salient event and is not a
semantically neutral negative control. The hook only blocks direct answer-query
access to selected keys, not evidence propagated through other keys/layers.

Direct faithfulness evidence remains **0/3 independently reviewed, 0/3 scored,
and no real-data ΔF estimate, interval, or test**. This mapping work does not
measure treatment efficacy.

## Next action

Before real scoring, authenticate the source/frame-sampling and browser/decoder
clock chain, independently review each claim, and prespecify any PTS cadence or
gap/edge coverage acceptance threshold before outcomes. The current digest
adapter is integrity bookkeeping, not that authentication gate. Proceed to
efficacy scoring only after review and provenance gates pass.
