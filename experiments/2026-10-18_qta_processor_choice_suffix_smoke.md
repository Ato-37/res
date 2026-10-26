# Pinned Qwen processor A–E suffix smoke (2026-10-18)

## Purpose

An independent code audit noted that the earlier token tests covered the text
tokenizer but did not compare tokenizer-derived A–E IDs with the actual
multimodal processor output. This bounded CPU smoke checks that alignment using
the repository's pinned local Qwen2.5-VL processor and synthetic video only.

## Setup

- Model/processor revision: `cc594898137f460bfe9f0759e9844b3ce807cfb5`
- Model weights: not loaded
- Inputs: synthetic question/options and two synthetic 448×448 RGB frames
- Processor: same chat-template builder and Qwen video utility path as the
  research runner; fixed video input reused across all six processor calls
- Random seed: none; fixture deterministic
- Runtime: Python 3.12.14, Transformers 5.18.0, qwen-vl-utils 0.0.14,
  NumPy 2.5.2, Pillow 12.3.0
- Device: CPU only; no CUDA call, dataset file, real media, protected review
  artifact, or model forward

## Result

Command:

    OMP_NUM_THREADS=2 .venv/bin/python scripts/smoke_qta_pinned_processor_choice_suffix.py

Result: **5/5 A–E continuations passed**. For each label, the multimodal
processor's synthetic `input_ids` exactly matched the base sequence plus the
same single token derived by the pinned tokenizer. The kernel's existing token
resolver returned the same five IDs. The output was `status=passed`,
`weights_loaded=false`, `provenance_authenticated=false`, with local processor
fingerprint `a8aa40930855b070f83b7d559409013d4d0f2f8bcf8c08300c92d826821015dc`.

An independent audit found that the first CLI draft accepted arbitrary model
directories while displaying the fixed revision. The override was removed;
the smoke now uses only the expected local snapshot directory and reports a
processor fingerprint. This is a local consistency fingerprint, not an
independent authenticity claim. The final smoke was rerun after this correction
and passed 5/5.

The first attempt passed NumPy arrays to `qwen-vl-utils`, which requires PIL
images for in-memory frames and failed before processing. Replacing the
fixtures with equivalent PIL images resolved this adapter mismatch; the final
run passed. No data or model changes were made.

## Provenance

| File | SHA-256 |
|---|---|
| `scripts/qta_processor_choice_suffix.py` | `6594479994915fc9bdb2d83c9b03f030f8f0628003d6766aedf517d0ef56d390` |
| `scripts/smoke_qta_pinned_processor_choice_suffix.py` | `6164e05890d19ec230156805f663dfb5ca4de2989bde0963a5d8ea7250c3b7db` |
| `scripts/test_qta_processor_choice_suffix.py` | `ae5bb2eb2f4a8b1c08dbc8a61c7451cd268bc517299494811e5b2e4c8eab810e` |

The local snapshot directory named for the configured revision and its
processor bytes were used; this smoke does not independently establish
upstream checkpoint authenticity or rights status. Keep all execution local.

## Interpretation and limitations

This is direct engineering evidence that tokenizer and multimodal processor
continuation IDs align for the tested pinned processor revision and synthetic
prompt/video path. It does not validate every real example's prompt, a video's
claim support, model predictions, QTA intervention, or faithfulness. It also
does not show that the model's natural-language answer formatting always uses
exactly one bare A–E letter.

No faithfulness metric was measured. The human review remains at 0/3 claim
judgments, 0/15 verified browser-to-decoder frame pairs, 0/3 prediction-link
judgments, and 0/3 direct scores. The review-only trust root, role assignments,
signed release receipt, and witnessed unopened handoff remain unresolved.

## Next action

Retain this as a repeatable preflight utility for the pinned processor. Continue
human-free contract checks that do not reuse the exploratory cohort outcomes;
do not start real claim-linked scoring until the human release and frame-pair
gates pass.
