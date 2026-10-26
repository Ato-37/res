# Synthetic event QTA construct CPU processor preflight (2026-10-19)

## Purpose

Validate the revised, prospectively frozen 40-pair synthetic event cohort and
its processor-time evidence/control geometry before writing or running model
scoring code. This is a construct/measurement engineering check, not a
faithfulness, hallucination, or model-performance result.

## Setup

- Protocol: `experiments/2026-10-19_synthetic_event_qta_construct_validation_v1.md`
- Generator: `scripts/qta_synthetic_event_construct_v1.py`
- Preflight: `scripts/run_qta_synthetic_event_cohort_preflight.py`
- Pinned local Qwen2.5-VL-7B processor revision:
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- CPU only; `CUDA_VISIBLE_DEVICES` was emptied before importing torch and
  transformers. No model weights were loaded and no model forward was run.
- Requested/effective cohort: 40 pairs / 80 clips, 32 RGB frames per clip.
  Each clip was built, passed through the local processor, summarized, and
  discarded before moving to the next clip.
- The assignment used 41 distinct uint32 seeds: the first was reserved for an
  engineering smoke outside the cohort; the remaining 40 were assigned to four
  onset strata. The full assignment manifest was hashed but its seed list was
  not written to the receipt.

## Verified result

The preflight completed successfully in 13.996 seconds. All 80 clips produced
the same processor geometry `[16, 26, 42]` with 4,443 sequence tokens and
4,368 merged visual tokens. Every clip mapped its four-frame event to exactly
two temporal groups and selected exactly two disjoint nearest non-event control
groups. All 40 matched pairs had equal rendered question/options and equal
processor geometry.

The cohort balance checks found 10 pairs at each onset `[4, 10, 18, 24]`, 5/5
circle-left/right at each onset, each of the ten orientation × answer-rotation
combinations once per onset, and each target's correct label exactly eight
times at every A–E position. The receipt is owner-only mode `0600`, 3,647 bytes,
and has SHA-256
`55a344b942fc14f299427a4301eb1eb627698bb3cd5def6db3e3fbe1e6b26763` at
`data/private_nextgqa/qta_synthetic_event_construct_preflight_20261019_v1.json`.
The receipt's assignment-manifest hash is
`23da4f9c8e654bb9c54c668e985e51903268eff02225b0f15185a9056c3bfc44`; the
processor fingerprint is
`e688ec7b8133bf3adf8beb5caec451562492b50f3dc4ab88f696104302159930`. The
receipt's protocol, generator, preflight, and QA/video-builder hashes were
independently recomputed from the current files and matched.

The targeted CPU unit suite passed 9/9. An independent auditor separately
recomputed the seed, onset, orientation/rotation, and correct-label balances.

## Interpretation and limits

This verifies the cohort generator and pinned processor geometry only. It does
not verify model behavior, QTA selection, nine-branch scoring, temporal
comprehension, event-key dependence, explanation faithfulness, or natural-video
validity. The controls are nearby non-event temporal groups, not visual
placebos. Rights status remains unverified; all inputs and receipts remain
local.

The simple human-check artifact remains nickname-first and asks only a broad
three-choice impression; its 25/25 software tests do not count as human review
or direct faithfulness evidence. Current formal direct-data counts remain
0/3 claim reviews, 0/15 player-clock/decoder-frame correspondences, 0/3
prediction-link reviews, and 0/3 direct scores; no direct `DeltaF`, interval,
or test exists.

## Next action

Implement the nine-branch runner and its CPU/scoring tests against the exact
protocol. Audit the source and memory guards, then run an 18-forward
engineering smoke (nine per clip in the matched pair) on the reserved external
seed only. It must never enter the
40-pair estimate. Only after independent code/result audit and live VRAM/host
memory checks should the 720-forward synthetic cohort be considered. This
synthetic branch does not substitute for the separate real-video human review
and direct-faithfulness gate.
