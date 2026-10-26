# Synthetic QTA cohort source-reconstruction audit (2026-10-21)

## Purpose

Reproduce the already completed frozen 40-pair synthetic QTA analysis from its
private aggregate receipt, and verify whether the current checkout can still
validate the run's source lineage.

## Setup and provenance

- Historical full receipt: `data/private_nextgqa/qta_synthetic_event_construct_gpu_full_20261019_v1.json`.
- Matching frozen inputs: `qta_synthetic_event_construct_preflight_20261019_v3.json`
  and `qta_synthetic_event_construct_gpu_smoke_20261019_v6.json`.
- The original run source commit is
  `407d7f4c8a4dfa016cdbb0edd2e31b7eac32856b`.
- Direct analysis from the current worktree rejected the receipt with
  `full_cohort_source_hash_mismatch`. Inspection found one changed source file:
  `scripts/run_nextgqa_gold_interval_margin_pilot.py`.
- The original Git commit exists, and its bytes for that one file match the
  receipt's recorded hash. For the other files absent from that commit, the
  current local source bytes match the receipt hashes. A temporary isolated
  source snapshot was assembled from those exact bytes and the three local
  aggregate receipts; no model weights, videos, prompts, or raw answers were
  copied. All **12/12** recorded source hashes matched before reanalysis.
- Reanalysis used the current local analysis-only script against the isolated
  run-source closure and wrote only to a temporary private directory.

## Verified results

The reconstructed-source reanalysis reproduced the existing recheck exactly:

- 40 scene pairs, 80 clips, 720/720 completed model forwards, zero recorded
  failures in the original full receipt.
- Frozen primary estimate: **−0.01718749 nat**; approximate stratified
  Welch–Satterthwaite 95% t CI **[−0.05415152, +0.01977654]**;
  two-sided one-sample t test **p=0.3505521**, df **31.68**.
- The CI crosses zero; this fixed synthetic result does not support a positive
  QTA construct effect. The original complete run took 2,633.197 seconds.
- Current transaction/receipt/live-batch regression command passed **116/116**
  tests. These synthetic-fixture tests establish software binding behavior,
  not human authorization.

## Interpretation and limits

This closes the analysis-reproduction gap for the synthetic construct result,
while confirming that the ordinary current-checkout analyzer correctly blocks
when a pinned source file has changed. It is still a synthetic two-shape event
and fixed gold-versus-baseline-foil margin study; it had no generated
explanations and is not a natural-video faithfulness result. No new GPU forward
was run. Dataset/model rights remain unverified and all receipts/derived
results stay local.

Direct faithfulness remains **0/3** formal claim reviews, **0/15** verified
clock/frame correspondences, **0/3** Stage-2 prediction-link reviews, and
**0/3** direct scores. No direct effect estimate, interval, or test exists.

## Next action

Keep the synthetic cohort closed. Advance the direct study only when authentic
reviewer/release inputs and the independent clock/frame evidence are available;
then run the real-video baseline through the reviewed pre-load transaction
path. Do not resume the closed head search or treat software-fixture tests as
direct faithfulness progress.
