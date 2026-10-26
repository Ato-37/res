# QTA full-video QA-accuracy diagnostic (2026-10-21, local exploratory)

## Purpose and scope

This post-hoc diagnostic asks whether the already-tested question-attention
temporal key bias changes restricted five-choice answer accuracy on the same
27 distinct real-video rows used in nine completed fixed-prediction
faithfulness-proxy cohorts (16–18, 20–24, 26). It adds a distinct outcome:
the model's full-video answer choice under baseline, QTA, and equal-budget
overlap-matched hash-sham routing. This is not a direct explanation
faithfulness endpoint and is not an independent or confirmatory cohort.

## Frozen design

- Rows are exactly the 27 rows and order in the nine owner-only anchor
  receipts; no row replacement or outcome-based exclusion.
- Each row receives three full-video forwards: baseline (which also captures
  question-query attention and derives the pre-fixed top-10% temporal groups),
  the existing positive `ln(2)` QTA bias, and one deterministic equal-budget
  hash-sham whose overlap with the existing evidence/control intervals matches
  the QTA group's overlap counts.
- Model revision, 4-bit weights, BF16 compute, eager attention, layers 5–16,
  processor batches, answer token IDs, and 20261021 seed match the anchored
  runs. Treatment masks are observed after application and checked at all 12
  layers.
- Primary descriptive outputs: per-condition restricted-choice accuracy,
  QTA-minus-baseline and QTA-minus-sham count differences, and paired
  wrong-to-correct/correct-to-wrong flips. No inferential test or population
  confidence interval is planned for this post-hoc reused sample.
- Requested full workload: 27 rows × 3 forwards = 81 scoring forwards after
  one model load. Planning estimate: 20–35 minutes including CPU media
  preparation; peak VRAM expected below the previously observed 12.2 GiB for
  a 32-frame Qwen run, but this estimate must be checked against the live
  runtime. A one-row/3-forward GPU smoke runs first; the same input row is
  retained in the full run only if the smoke passes all provenance, mask,
  resource, and finite-score checks.

## Limits and interpretation rules

An accuracy gain would support only better NExT-GQA five-choice QA on this
reused cohort. It would not show that generated explanation claims are true,
grounded, or faithful. A null or negative result also cannot rule out every
internal intervention. The treatment and metric were selected after many
exploratory faithfulness-proxy analyses, so all values are labeled
post-hoc. Human review remains unperformed; direct claim-to-frame review and
direct semantic-faithfulness scores remain **0/25**. Dataset/model rights
remain unverified. All video, question, answer, and row-level score material
must stay under the local private data root and must not be committed or
published.

## Pre-run implementation and validation

Runner: `scripts/run_qta_full_video_accuracy_diagnostic.py`.
The runner verifies every anchor's private input/source hashes, reconstructs
all frozen processor batches, joins the saved QA answer to exactly one of the
five choices, confirms baseline logits against the prior receipt, and confirms
that fresh QTA key selection reproduces the anchored selector. It writes an
exclusive mode-0600 receipt and emits only aggregate counts to stdout.

At protocol freeze, GPU smoke/full had not yet executed. The following
section records the later run and verification.

## Results

### Verified run

- Smoke receipt `data/private_nextgqa/qta_full_video_accuracy_smoke_20261021_02.json`:
  1/1 row, 3/3 forwards, all QTA/sham hooks and 504 visual keys audited; mode
  0600, SHA-256 `c4c35a6a937bc3bd5fa3400bb1007511cc655eba21b3ff32ea5d0a6dd9ccb27e`.
- The first full attempt (`_01`) stopped at the per-forward resource guard
  before finishing the first row's sham condition. It is a failed partial run,
  excluded from estimates. Cache clearing before resource checks and failure
  receipts were added; the identical frozen cohort then passed smoke and full.
- Full receipt `data/private_nextgqa/qta_full_video_accuracy_full_20261021_02.json`:
  27/27 distinct rows and videos, 81/81 forwards, mode 0600, SHA-256
  `f17764b6503d3a499d6acc95cca27c02fbe823c1a1f7298b272d0aac09767bfe`.
  Runtime 459.83 s; peak allocated VRAM 11.924 GiB on RTX 5070 Ti.
- Restricted-choice accuracy: baseline 15/27 (55.56%), QTA 15/27, sham
  15/27. All 27 prediction indices were identical across conditions; QTA vs
  baseline had zero wrong-to-correct and zero correct-to-wrong flips. Thus
  finite-cohort accuracy difference is exactly 0/27. No CI or population test
  is warranted for this post-hoc reused cohort.
- The maximum absolute per-row gold-margin shift was 0.375 nat for both QTA
  and sham vs baseline, while no choice changed. This maximum is an observed
  diagnostic, not evidence of improved confidence or faithfulness; QTA and
  sham both show that score perturbation can occur without decision change.
- Independent auditor revalidated source/anchor/input hashes, row-to-dataset
  joins, gold labels, every choice vector/margin, unique rows/videos, 81/81
  forward counts, mask hooks, equal key budgets and matched overlap. The local
  aggregate analyzer independently recomputed the same counts and flips.
- Local checks: 24 unit tests passed; both scripts compiled and scoped
  `git diff --check` passed. (The initial attempt to use pytest was unavailable
  in `.venv`; tests were run with stdlib `unittest`.)

### Interpretation and next action

This yields an undergraduate-usable **negative/mechanistic diagnostic**:
global question-attention video-key steering perturbed score margins (up to
0.375 nat) but did not alter any five-choice prediction or improve accuracy
on this 27-row finite cohort. Because overlap-matched sham also perturbed
margins, generic logit movement is not evidence for selective evidence
dependence. It is not direct explanation faithfulness and does not establish
that every internal intervention fails. The registered head search remains
closed. Formal claim-to-frame review and direct semantic-faithfulness scores
remain 0/25; no direct faithfulness effect or interval can be reported.

The next scientifically justified improvement is a distinct, selective
intervention: bias only reviewed claim-support temporal keys and compare it
against a key-budget/overlap-matched sham under a fixed claim prediction, then
measure support-ablation margin dependence and separately score claim grounding.
Existing machine/QA intervals do not meet that prerequisite. Meanwhile, do
not rerun broad global bias on this cohort; treat the measured null as the
result and preserve all detailed receipts locally because they contain
protected row-level material.
