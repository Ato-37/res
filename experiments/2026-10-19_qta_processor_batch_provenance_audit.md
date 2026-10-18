# QTA processor-batch provenance and GPU eligibility audit — 2026-10-19

## Purpose

Reassess the next action after phase-1 record tokens were bound to the ordered
Stage-1 cohort. Determine whether the remaining processor-batch digest gap is
implementable from an existing frozen artifact schema, whether a production
claim-mapped scorer exists, and whether a direct-faithfulness GPU run is
currently justified.

## Evidence inspected

- `scripts/qta_phase_freeze_receipts.py`: `_record_bindings` validates three
  unique 64-character tokens and digest syntax. Phase-1 creation/verification
  now also binds record-token order to the pinned Stage-1 review tokens.
  `full_video_processor_batches` is included as an exact-byte artifact role,
  but its contents are not parsed and per-row digests are not recomputed from
  it.
- `docs/qta_phase_receipt_verifier_bundle_ja.md`: names the processor artifact
  as an opaque `.bin`; no internal schema is specified.
- `scripts/nextgqa_atomic_batch_freeze.py`: defines `batch_sha256` for typed
  mappings/tensors, including container structure, tensor dtype/shape, and
  tensor bytes, but no persisted safe serialization/parser was found in the
  phase-receipt path.
- `scripts/qta_prescore_mapping_binding.py` and
  `scripts/qta_phase2_plan_manifest.py`: check equality of caller-supplied or
  previously signed processor hashes across records. They do not derive those
  hashes from persisted processor tensor contents.
- `scripts/qta_baseline_pq_candidate.py` is a non-authorizing structural
  candidate builder; inspected claim-mapped QTA paths do not provide a
  production real-video scorer joining reviewed claims, evidence/control
  groups, processor batches, and intervention outputs.
- A local-only search found no quick-review response JSON in the repository.

An independent source audit reached the same conclusions. It did not access
protected review content or run a model.

## Current runtime and run eligibility

Read-only individually elevated check in the existing `.venv` observed an
NVIDIA GeForce RTX 5070 Ti with 15,112 MiB free of 16,303 MiB, Torch 2.11.0
`+cu128`, CUDA 12.8, and `torch.cuda.is_available() == True`. WSL reported 15
GiB total RAM and approximately 13 GiB available. No model or GPU workload was
launched.

GPU availability is therefore not the current blocker. A direct real-video
faithfulness run remains unjustified because no signed claim-review judgments,
15/15 player-clock/decoder-frame correspondences, frozen baseline `p/q`, or
separate Stage-2 prediction-link judgments exist; the claim-mapped production
scorer and a batch provenance contract are also missing. This audit does not
open those gates. Existing question-swap and selected proxy cohorts remain
closed and were not repeated.

## Interpretation and limitations

The processor digest currently provides cross-artifact consistency, not proof
that a digest was computed from the exact typed batch consumed by a model. A
parser cannot be safely added until the persisted representation defines
canonical keys/containers, dtype and shape encoding, tensor bytes, row order,
and a bounded safe decoder. Simply copying hashes into another JSON manifest
would retain the same provenance gap. The least speculative next step is to
freeze that representation alongside the actual scorer call contract, then add
synthetic tamper/order tests before any real baseline forward.

Direct faithfulness remains **0/3** claim reviews, **0/15** clock/frame
correspondences, **0/3** prediction-link judgments, and **0/3** direct scores;
no direct `ΔF`, interval, or test exists. The informal review artifact remains
unanswered and cannot substitute for these judgments. Rights status remains
unverified; all protected material and derived responses stay local.

## Next action

Specify the processor artifact and production scorer interface as one design
unit, including exact tensor serialization or a reproducible in-process digest
procedure, and an enforced pre-forward gate sequence. Keep the interface
non-authorizing until independent reviews are actually complete. After the
schema is fixed, implement synthetic forged-digest/reordering tests and a
fail-closed smoke before considering any real-data GPU baseline run. Do not
repeat closed GPU proxies or count this audit as faithfulness evidence.
