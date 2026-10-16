# Atomic batch tensor hashing and memory check — 2026-10-17

## Purpose

Reduce avoidable host-RAM pressure in the hash checks used by the fixed
three-record, three-condition scorer. The original freezer encoded tensor
values with `.tolist()` and JSON, which can expand model-sized pixel inputs
into many Python objects. This work changes only the integrity implementation;
it does not authenticate data, approve the draft protocol, load a model, score,
or authorize execution.

## Implementation

`nextgqa_atomic_batch_freeze.batch_sha256` now uses a versioned domain
(`nextgqa_typed_batch_stream_v1`), canonical mapping/scalar encoding, and
contiguous CPU tensor raw bytes preceded by dtype and shape. The byte view is
fed to SHA-256 in 8 MiB chunks. Floating-point finiteness checks scan at most
1,048,576 elements per temporary mask, avoiding one full-size Boolean tensor.
Unsupported sparse/quantized layouts and complex tensors fail closed. Lists
and tuples have distinct hash markers; snapshots preserve tuple type. The
freezer result records the hash schema. Old hashes are not compatible and
must be recomputed for any future pre-run freeze.

An independent code audit found no correctness defect for the fixed WSL/x86
workflow. Hash schema v1 uses the tensor's native byte order; cross-endian
replay is not promised. The production freeze must bind the expected digest
manifest and the freezer/scorer source version before model loading.

## Checks and observed measurements

Focused freezer/scorer tests passed **24/24**, including BF16, scalar and empty shapes,
dtype changes, raw negative-zero distinction, nested mapping order, list/tuple
type, non-contiguous/contiguous logical equality, non-finite values, unsupported
layout, snapshot type preservation, and the existing synthetic nine-forward
software test. The full scripts suite passed **1171/1171** in 16.850 s.
Expected failure diagnostics printed by negative-path tests did not change the
final `OK` status.

A CPU-only synthetic 192 MiB BF16 tensor hash took 0.186 s in the chunked
implementation. Process peak RSS was 714.3 MiB, with a 10.3 MiB increase above
the pre-hash high-water mark after allocating the tensor. An earlier run of
the same-size input with an unchunked whole-tensor `isfinite(...).all()` reached
1,187.7 MiB peak RSS (482.5 MiB increase). This comparison isolates the
finiteness temporary, not the older `.tolist()` implementation; no claim is
made here about measured legacy-versus-new runtime or total experiment RAM.

A separate elevated GPU smoke transferred a synthetic 2,000,000-byte BF16
tensor to the RTX 5070 Ti and confirmed the pre/post-transfer hashes match.
No checkpoint, dataset, video, model forward, or score was involved.

## Interpretation and limits

The hash path no longer creates a Python list of tensor values, and the
finiteness scan's temporary peak is bounded by its chunk size. This removes a
specific avoidable hashing allocation; `freeze_all_batches` still clones the
complete nine-batch CPU snapshot, and a non-contiguous tensor may require one
contiguous copy. The end-to-end nine-batch peak has not been measured on actual
Qwen batches. CUDA validation covers a small synthetic transfer only. The hash
algorithm changed before any production freeze, so code and hash-schema bytes
must be included in the future joint freeze.

## Faithfulness progress and next action

Direct claim-to-video faithfulness remains **0/3 human-reviewed and 0/3
scored**, with no effect estimate, interval, or test. The available RTX and
this hashing smoke do not clear the three-video human review, fifteen
browser-to-decoder frame correspondences, protocol/trust-key freeze, or
production atomic gate. Rights status remains unverified; protected inputs
stay local.

Next, integrate the existing condition-batch builder with verified record
identity and planner assignments only after mapped intervals are bound through
the frozen clock calibration; complete the authenticated all-nine preflight;
then run the three-item nine-forward GPU feasibility experiment once the
human-only evidence gates are present. Do not resume head search while these
direct-evidence and integration tasks remain actionable.
