# QTA processor-batch manifest receipt v2 integration (2026-10-19)

## Purpose and implemented contract

The prior design audit found the pre-baseline `full_video_processor_batches`
receipt role was opaque, even though phase records already bind one typed base
batch digest per row. This adds a separate pre-baseline v2 path using role
`processor_batch_digest_manifest`; the existing v1 role, signature domain,
body, and post-baseline v3 schema remain compatible.

The v2 verifier validates canonical manifest bytes against explicit caller
expected row pins and an explicit expected protocol SHA-256, matches ordered
record tokens and base batch digests to the signed receipt rows, and matches
each manifest `source_video_sha256` to the corresponding ordered Stage-1
source row. The phase-2 wrapper forwards the same expected pins when it
recursively verifies a v2 phase1 receipt. The receipt and report remain
non-authorizing (`model_load_permitted=false`, `scoring_authorized=false`,
`execution_authorization=not_granted`); the report states that the protocol-pin
origin is unauthenticated.

## Verification

- Primary CPU test run passed **68/68** across the processor manifest,
  pre/post phase receipt, same-source phase-2 composition, and full phase-2
  verifier suites.
- Independent audit passed the phase-freeze suite (**45/45**) and the
  same-source/full-verifier suites (**8/8**), then separately passed the
  v2-to-phase2 wrapper test (**1/1**).
- The wrapper integration test verifies success with matching pins and
  fail-closed behavior when the expected protocol pin is omitted or altered.
  Synthetic receipt tests also cover source-video mismatch, record binding
  mismatch, signature domain/version separation, and recursive phase2→phase1
  verification.
- `git diff --check` passed for the changed receipt and phase-plan files.
  No GPU, protected videos, or real reviewer records were used in this
  integration test.

## Interpretation and remaining limits

This closes a schema and byte-binding gap; it does not establish the external
origin of caller-supplied expected pins, prove that processor/runtime metadata
was derived from the consumed tensors, prove the model used the same tensors
through each forward, authenticate human identity or judgments, or grant model
loading/scoring. The exploratory head scan did not change direct faithfulness
progress: formal claim review **0/3**, clock/frame correspondence **0/15**,
prediction-link review **0/3**, direct scores **0/3**. There remains no direct
`ΔF`, interval, or test.

Next, connect these explicit pins to independently rooted artifacts, then
build the model-load gate around reconstructed processor batches that remain
the exact objects rehashed at each pre-forward boundary. Until the human,
clock, baseline, and prediction-link inputs are independently present and
validated, this candidate receipt is not a basis for real-data scoring. Rights
status remains unverified; local media and derived metrics remain local.
