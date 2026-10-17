# Qwen visual-key attention-bias smoke (2026-10-18)

## Purpose

Test whether a small, explicit internal video-evidence intervention can run in
the pinned Qwen2.5-VL stack. This is an engineering smoke for a candidate
adaptation, not an efficacy test and not a STEAR reproduction.

## Treatment under test

The new helper clones Qwen's 4-D additive causal self-attention mask and adds
`ln(2)` to selected visual-key logits only at the fixed final-prompt query row
in layers 5–16. That doubles the pre-softmax odds assigned to each selected key
relative to unselected keys, before normalization. Selected positions are
chosen once from the untreated full-video forward by averaging final-query
self-attention over layers 5–16 and taking the top 10% video-token positions;
the treatment hook then reuses those frozen positions. This is named a
**Qwen visual-key attention-bias adaptation**. It is distinct from STEAR's
memory reinjection and temporal counterfactual decoding.

## Setup and observed results

- Pinned Qwen2.5-VL-7B-Instruct revision; Transformers 5.18.0; PyTorch
  2.11.0+cu128; CUDA 12.8; bfloat16 with 4-bit weights and bfloat16 compute.
- RTX 5070 Ti, 16,303 MiB total VRAM. Before model load: 13,992 MiB free by
  `nvidia-smi` and 14.657 GiB reported by PyTorch.
- One deterministic 8-frame synthetic video, 1,024 video tokens, four temporal
  groups, and two model forwards (baseline attention capture, then treatment).
- The frozen selector chose 103/1,024 video tokens. Final logits remained
  finite. Treatment versus baseline max absolute logit change was 1.0 and RMS
  change was 0.4473; the top A–E choice did not change on this synthetic item.
- Peak allocated/reserved VRAM was 6.216/6.730 GiB; reported free VRAM after
  the treatment forward was 7.862 GiB. End-to-end script time was 14.85 s.
- Artifact integrity checks independently confirmed status, model/source
  bindings, counts, finite values, selected-token arithmetic, and that no
  question, claim, video identifier, or raw logit array was serialized.
- Focused intervention and loader-classification tests passed 12/12. The full
  repository suite passed 1,412/1,412 in 33.15 s. The first full-suite run
  found that the new loader needed an explicit engineering-script classification;
  the guard list was updated and the rerun passed.

The inspectable aggregate is
`results/synthetic_qwen_visual_attention_boost_8frames_20261018.json`. The
runner and utility are `scripts/smoke_qwen_visual_attention_boost.py` and
`scripts/qwen_visual_attention_boost.py`. Aggregate SHA-256:
`3a704aa7e0786e31442fcb6a7b3d96ea76eb42ca572943ee907297d67bb9be0c`;
runner SHA-256:
`d2aa693a355d786e7b706d21b5d4379f7b28f6d5921eb9c6006c43680b0f894f`;
intervention helper SHA-256:
`c267e170725578b57f046ca69f46c1830779f55d82ffda906ea374682926723e`.

## Interpretation and limits

This confirms that the pinned implementation accepts the attention-mask
intervention at all 12 middle layers and that it changes the finite logits in
one synthetic example. The unchanged top choice is also observed, not a
failure criterion. The selected token positions are chosen from model
self-attention rather than human claim review, and the answer/logit change does
not show that a claim's evidence is more causally necessary. The selected
ratio and `ln(2)` dose are engineering constants, not paper-reproduced or
validated hyperparameters. No real video, human review, six-condition
contrast, faithfulness score, confidence interval, or hypothesis test was run.
Rights status for protected project materials remains unverified; this smoke
used synthetic input only.

Direct project progress therefore remains **0/3 independently reviewed,
0/3 scored, no `ΔF` estimate/interval/test, and 0/15 verified
browser-to-decoder correspondences**. The user-authorized local human-review
package is still awaiting a reviewer. Normal sandbox GPU visibility is blocked;
this exact bounded run succeeded through per-command elevated execution.

## Next action

Integrate the frozen full-pass selector and hook with the existing fixed-label
six-condition contract, first on synthetic three-record batches. Verify that
input hashes, score positions, answer/foil labels, selected positions, and
hook lifetimes are identical across matched branches, and that failures leave
the cohort result incomplete. Keep synthetic `ΔF` descriptive and excluded
from research efficacy. In parallel, the actual reviewer and decoder-clock
calibration gates must still be completed before any real-data faithfulness
score.
