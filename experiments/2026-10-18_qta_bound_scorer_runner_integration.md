# QTA bound scorer wiring to synthetic 9-forward runner (2026-10-18)

## Purpose

Close the contract gap between the fixed-prediction scorer and the current
synthetic QTA internal-attention runner. Ensure the primary six conditions use
one processor batch, baseline-full answer/foil labels, tokenizer-derived option
IDs, and a score position derived from equal branch bindings.

## Changes

- `derive_six_condition_binding()` now returns the validated per-branch
  declarations in `branch_bindings`, alongside the existing digests.
- The synthetic runner derives these declarations from its CPU `FrozenBatch`
  before model inference. Its baseline-full and target logits for full,
  evidence-key-ablation, and control-key-ablation are copied to CPU and passed
  through `score_bound_six_condition_logits()`.
- Runner branch names are projected from `target_*` to scorer `intervention_*`
  names by a small pure helper. An independent audit found and corrected an
  initial `bindings`/`branch_bindings` return-key mismatch; a regression test
  now covers the projection, missing keys, and exact six-condition coverage.
- The existing sham diagnostic remains secondary and retains its previous
  fixed-label arithmetic path.

## Verification

- Focused binding/scorer/runner tests: **40/40 passed** after the final key
  correction, name projection, and multimodal suffix checks.
- Full `scripts` suite on the final tree: **1,559/1,559 passed** in **36.060
  seconds**. The suite prints expected diagnostics from negative-path tests;
  unittest exited successfully with `OK`.
- Python compilation and `git diff --check` passed after the final edits.
- An independent code audit verified the scorer formula and condition mapping.
  It found the key-name defect before the correction and confirmed the
  correction afterward. The audit also confirmed CPU copies release GPU-logit
  lifetime before scoring, and later checked the processor suffix guard's
  pre-load placement, exact prefix/suffix checks, and CPU-only incremental
  memory use.
- A local CPU-only tokenizer check with the pinned processor and a synthetic
  Qwen chat/video prompt found a stable prompt prefix and one-token A–E
  continuations for all five labels. The runner now resolves IDs from the exact
  rendered prompt plus each label, rather than trusting isolated label
  tokenization, and fails closed on prefix retokenization or a multi-token
  continuation. Its pure-function regressions are included in the 40-test
  focused run.
- The exact multimodal processor path was checked on one synthetic video batch:
  all five `rendered_prompt + label` batches retained the complete original
  `input_ids` prefix and appended exactly the expected one token. This check is
  now enforced in the runner before model loading.
- The updated GPU smoke passed on the final runner: **1/1 synthetic record,
  9/9 forwards**, RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128,
  bf16/4-bit-bf16. Peak allocated/reserved memory was **10.755/12.439 GiB**;
  elapsed time was **51.046 s**. The synthetic diagnostic `ΔF` was
  `+1.19e-7 nat`; `faithfulness_result=false`. This is only a code-path and
  numerical smoke, not an empirical effect.
- Local aggregate output SHA-256: `4a48549b51ed4494ce6be02a4c43a03af856755a2396584d0dfba64034096fb9`.
  Runner source SHA-256: `8b1be5ec0193e27892b50a478aabbe625f7466ab86ab32aa80b49987e95961d2`;
  scorer SHA-256: `84ba18241668fd659ddce0ab25a4e3dfa2cacc656ef43fb494fdc056a5c21744`;
  batch-binding helper SHA-256: `65303e20ffb76ed813e36e0857d53db6624018268c09e78b1e8fc74dd7bc0466`.

## Interpretation and limits

This is engineering integration evidence only. The runner is synthetic and
uses diagnostic evidence/control groups without annotation meaning. The
binding helper verifies consistency of caller-supplied tensors/declarations;
it does not authenticate processor/model provenance or prove that logits came
from the declared inputs. No real videos, reviewer judgments, clock/frame
correspondence, or model efficacy outcomes were used here. The GPU run used
one synthetic fixture only.

Direct faithfulness outcome progress remains **0/3** claim reviews, **0/3**
prediction-link judgments, **0/3** scored records, **0/15** player-clock to
decoder-frame correspondences, and no empirical `ΔF`, confidence interval, or
hypothesis test. The user-selected construct remains prediction dependence on
video evidence indicated by the explanation; the primary candidate is change in
that dependence under internal intervention.

## Next action

Continue work on the review handoff/trust-root and player-clock/frame evidence
path without opening protected review media or enabling model loading. Real-data scoring remains
blocked until independent human claim/prediction-link review, role and receipt
bindings, and all 15 clock-to-decoder correspondences pass the frozen gates.
