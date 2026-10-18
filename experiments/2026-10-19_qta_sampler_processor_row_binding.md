# QTA sampler-to-processor row consistency (2026-10-19)

## Purpose

Reduce the CPU integration gap between the local 32-frame sampler receipt and
the processor batch row adapter. The protocol and human/release gates are
unchanged. This is a structural consistency check, not an inference gate.

## Implementation

`qta_prescore_processor_row_adapter.processor_binding_row` now requires one
caller-provided sampler row. It checks the source-video SHA-256 against the
processor adapter argument, requires exactly 32 sampled frames, recomputes the
endpoint-inclusive frame-index grid from the sampler's decoded-frame count,
and compares the mapping's sampled indices with that grid. Sampler PTS pairs
are parsed as exact fractions and compared to canonical rational PTS strings
from the processor mapping; floating-point approximations are not used. The
projection still omits processor tensor snapshots and does not return protected
media, claims, or questions.

## Verification

Synthetic unit cases cover a matching row, wrong source hash, wrong frame
count, wrong endpoint grid, exact-rational PTS mismatch, and invalid PTS
denominator. Synthetic adapter-composition cases now pass a 32-frame sampler
row through the existing review/clock/processor join, and downstream mapping
binding and non-authorizing cohort-preflight suites still pass. The combined
focused regression run passed **23/23**. No actual media, protected review
values, GPU, or model was used.

## Interpretation and limits

The check proves only that the caller-supplied sampler row and processor
mapping agree on source hash, sampled index set, and exact PTS values. It does
not authenticate the sampler receipt, establish that either record came from
the source video, prove pixel identity after Qwen preprocessing, validate
reviewer/clock evidence, or authorize scoring. The real source sampler preflight
and the processor mapping still require a trusted production atomic preflight
to bind them with the remaining gates. Claim review, player-to-decoder
correspondence, prediction-link review, and direct scoring remain **0/3,
0/15, 0/3, and 0/3** respectively; no direct effect estimate or uncertainty
result was produced.

## Next action

Have an independent audit review the adapter's exact rational conversion and
failure behavior. Keep real scoring blocked until authenticated human/release
and clock/frame gates plus the production all-row atomic preflight exist.
