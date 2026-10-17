# Question-token selector and internal key intervention integration smoke

## Purpose

The existing synthetic attention-bias runner selected visual keys from the
final answer-query attention. That selector shares the scoring query and can
create mechanical coupling. The new question-token temporal selector avoids
using answer options or score logits, but had not yet been composed with the
attention-bias and internal key-ablation hooks. This smoke checks that exact
engineering path before any real-data protocol is frozen.

## Setup and verified result

- Model: cached Qwen2.5-VL-7B-Instruct revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`, bf16 with 4-bit weights and
  bf16 compute, eager attention. Hardware: RTX 5070 Ti; CUDA 12.8, PyTorch
  2.11.0+cu128, Transformers 5.18.0, bitsandbytes 0.50.2.
- Input: one deterministic **synthetic** 32-frame clip, video-first prompt,
  question-token query span, 16 temporal visual groups of 256 keys each.
  The batch used one unpadded row; the runner now asserts that the mapped
  question positions precede the fixed final score-query position.
- Selector: untreated baseline attention from the question-token rows,
  averaged over layers 5–16. The smoke-only budget was `ceil(10% of groups)`
  (2 of 16); treatment bias was `ln(2)` on those keys at layers 5–16.
  Sham used deterministic seed 20261018 and the same operator, dose, layers,
  and key count. The overlap was recorded as-is: target groups `[0,5]`, sham
  groups `[5,6]`, overlap **1/2**. No overlap rejection or resampling occurred.
- Conditions: baseline full/evidence-key ablation/control-key ablation;
  target full/evidence/control; sham full/evidence/control. The baseline full
  pass is also the selector pass, giving 9 total forwards. Synthetic evidence
  and control were the first two versus last two temporal groups; these labels
  have no semantic meaning. In overlapping intervention/ablation positions,
  ablation explicitly takes precedence.
- Final source-matched GPU run completed **1/1 synthetic records and 9/9
  forwards**. Every output was finite; the typed input hash was unchanged
  around every branch. Elapsed time was **40.84 s**. Free VRAM before loading
  was 14.66 GiB, before forwards 8.61 GiB, and after the final branch
  (before model cleanup) 1.90 GiB. Peak allocated/reserved VRAM was
  **10.75/12.44 GiB**.
- The final artifact records 20 source hashes, independently re-computed and
  matched **20/20**. Result:
  `results/synthetic_question_selector_intervention_20261018_0430.json`,
  SHA-256 `da644f56ee753a1ab67b24cdd1bf973e04fa406aca79de0745d4d64a61d94a3d`.
  The earlier 0422 artifact is superseded because query/padding assertions,
  condition-plan tests, and bitsandbytes version recording were added before
  the final run.
- CPU tests cover nine-branch construction and same/partial/disjoint bias-mask
  overlap. The full repository suite passed **1,448/1,448** in 30.4 s;
  compilation, CLI help, and `git diff --check` passed.

## Interpretation and limitations

This confirms hook compatibility, fixed branch count, finite outputs, and
source/input integrity for one synthetic sequence on the pinned local stack.
The run does not compute answer or claim scores, prediction margins, `F`,
`ΔF`, confidence intervals, or a hypothesis test. Its synthetic target and
sham indices are not evidence locations. `ceil(10%)`, layers 5–16, `ln(2)`,
and sham seed are engineering smoke settings, not an efficacy-justified or
frozen treatment. One example with target/sham overlap 1/2 cannot establish
selector specificity. Eager attention still allocates full attention maps
transiently during the baseline selector pass.

Faithfulness progress remains **direct evaluation not performed**: independent
claim review **0/3**, scored **0/3**, no effect estimate/interval/test;
verified browser-to-decoder correspondence **0/15**. The human review-only
trust/witness release, real interactive capture/pair review, and separate
internal-treatment/scorer freeze remain incomplete. Rights status is
unverified; protected data stays local.

## Next action

Use the now-working local inventory builder to bind the current disclosed
review package. Human review and 15-anchor browser/decoder correspondence still
require the independent people and interactive actions in the runbook. In
parallel, independently freeze the outcome-blind treatment/sham and scorer;
do not use this smoke's selected groups or outputs to tune them. No additional
GPU run is justified until a distinct integration change or approved real-data
gate requires it.
