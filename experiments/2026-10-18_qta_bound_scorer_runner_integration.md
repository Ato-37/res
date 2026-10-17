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

- Focused binding/scorer/runner tests: **38/38 passed** after the final key
  correction and projection test.
- Full `scripts` suite on the final tree: **1,557/1,557 passed** in **37.105
  seconds**. The suite prints expected diagnostics from negative-path tests;
  unittest exited successfully with `OK`.
- Python compilation and `git diff --check` passed after the final edits.
- An independent code audit verified the scorer formula and condition mapping.
  It found the key-name defect before the correction and confirmed the
  correction afterward. The audit also confirmed CPU copies release GPU-logit
  lifetime before scoring.

## Interpretation and limits

This is engineering integration evidence only. The runner is synthetic and
uses diagnostic evidence/control groups without annotation meaning. The
binding helper verifies consistency of caller-supplied tensors/declarations;
it does not authenticate processor/model provenance or prove that logits came
from the declared inputs. No real videos, reviewer judgments, clock/frame
correspondence, or model efficacy outcomes were used here. No GPU run was
needed for this code-contract change.

Direct faithfulness outcome progress remains **0/3** claim reviews, **0/3**
prediction-link judgments, **0/3** scored records, **0/15** player-clock to
decoder-frame correspondences, and no empirical `ΔF`, confidence interval, or
hypothesis test. The user-selected construct remains prediction dependence on
video evidence indicated by the explanation; the primary candidate is change in
that dependence under internal intervention.

## Next action

Complete the full-suite rerun and record its result. Then continue work on the
review handoff/trust-root and player-clock/frame evidence path without opening
protected review media or enabling model loading. Real-data scoring remains
blocked until independent human claim/prediction-link review, role and receipt
bindings, and all 15 clock-to-decoder correspondences pass the frozen gates.
