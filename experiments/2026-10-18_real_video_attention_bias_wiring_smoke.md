# Local real-video visual-key attention-bias wiring smoke (2026-10-18)

## Purpose and scope

Check whether the candidate visual-key attention-bias hook runs on real local
NExT-GQA videos under the pinned Qwen2.5-VL checkpoint. This is a non-scoring
engineering/wiring diagnostic. It does not test whether the treatment improves
claim-specific faithfulness.

## Frozen procedure

The runner deterministically selected three test videos absent from the
repository's sample manifests, chose the first eligible question by QID order,
sampled 16 decoder-PTS frames per video within a 50 ms uniform-grid tolerance,
and used the question as the model prompt. It used neither answer text nor
gold-interval values in model input, scoring, or treatment. One baseline pass
captured the final-query attention over layers 5–16; the top 10% of video keys
were frozen and received an additive `ln(2)` attention-logit bias at that same
query row in layers 5–16 for one treatment pass. The six forwards ran
sequentially. The per-video PTS/RGB fingerprints and the complete aggregate
remain in a local-only mode-0600 artifact; no IDs, paths, question/answer text,
frames, raw logits, or per-video hashes were published.

## Verified result

- 3/3 videos passed the decoder timing gate; 0 timing rejects.
- 6/6 requested baseline/treatment forwards completed with finite outputs.
- The mean video-token count was 2,048; the mean selected-key count was 205.3,
  consistent with the fixed top-10% rule.
- Mean RMS logit change was **1.593 logit units**; the largest per-video maximum
  absolute logit change was **11.375**. The top vocabulary token did not change
  for any of the three prompts. These are descriptive intervention-wiring
  measurements only; no answer choice, probability margin, correctness, or
  faithfulness score was computed.
- Runtime was **34.01 s**. Peak allocated/reserved VRAM was **7.24/8.08 GiB**;
  free VRAM after scoring was 8.61 GiB. The local artifact has mode 0600 and
  retains decoder fingerprints for private integrity checks.
- Local checkpoint revision was `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
  PyTorch 2.11.0+cu128, CUDA 12.8, Transformers 5.18.0, BF16/4-bit BF16 compute,
  and RTX 5070 Ti. Runner and intervention-helper hashes match the private
  artifact.

## Interpretation and limitations

This is the first verified execution of the selected-key treatment on three
real local videos; it shows that the Qwen attention hook changes logits without
non-finite outputs or a top-token switch in this tiny cohort. It does **not**
show that the increased attention is directed to reviewed claim evidence, that
the prediction depends more selectively on the explanation's evidence, or
that any answer is correct. The cohort was selected from local annotations and
manifest exclusions, not from an authenticated fresh-cohort ledger. It is not
an independent validation set or an efficacy estimate. Rights remain
unverified; protected media and all per-video measurements remain local.

Direct faithfulness progress is unchanged: **0/3 independently human-reviewed,
0/3 scored, no real `ΔF` estimate/CI/test, and 0/15 verified browser-to-decoder
correspondences**. Human claim review, frame correspondence, the non-circular
treatment/scorer freeze, and the production atomic gate remain open.

## Next action

Keep the three-video measurement as a wiring result only. Before any score call,
complete human claim review and timing correspondence, then freeze the selector,
intervention, sham/control, fixed answer/foil, sample cohort, and analysis rule.
The six-condition synthetic integration is already available; a real score run
must remain blocked until those inputs are authentic and bound to every row.
