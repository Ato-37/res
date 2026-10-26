# Fresh-val QA-interval claim-mask bias, next fixed seven-row cohort

## Purpose and protocol

This is a sequential exploratory replication of the six-branch machine QA
interval claim-likelihood/mask sensitivity estimand defined in
[`2026-10-21_freshval_qa_interval_claim_mask_bias_cohort10.md`](2026-10-21_freshval_qa_interval_claim_mask_bias_cohort10.md).
It asks whether fixed generated claim likelihood's QA-mask versus equal-count
control-mask contrast changes under QA-window key bias relative to baseline
and an equal-budget disjoint sham. The QA interval is machine-provided benchmark
timing, not reviewed support for the explanation claim. This is not direct
semantic-faithfulness measurement.

## Outcome-blind cohort and execution

The existing 27-candidate val lock was continued strictly after its first 13
already-scanned candidates; the next 14 all passed video timing/decode, four
had no frame-level QA/control pair, ten passed that initial geometry, and seven
passed the frozen temporal-group QA/control/sham geometry. These seven were
fixed before claim generation. The private allowlist is mode 0600, SHA-256
`19e148282a5362b7d92503282bfdcce75ad6049c05e481ce2ed37ea09f99d553`; its
geometry audit is SHA-256
`49de6b959822148f886630b7bda20f7d57e3b82844d24a86b1c72b2a093e8ade`.
No claim or intervention score was viewed during selection.

A one-row generation smoke and full 7-row generation both passed; **7/7**
fixed rows yielded structured claims. The scoring smoke passed 6/6 forwards
on its first fixed eligible row. The full six-condition run completed **42/42
forwards on all 7 rows**, with no scoring attrition or row replacement. Model:
Qwen/Qwen2.5-VL-7B-Instruct, pinned revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit/BF16, layers 5–16,
`+ln(2)` key bias. GPU: RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128; peak
allocated VRAM 7.40 GiB; host available before load 12.83 GiB. The run took
549.43 seconds, including a visibly slow model load (~147 seconds from the
terminal progress display). This is an operational concern; it does not affect
the score interpretation.

The 11 focused scorer tests passed on the same source revision; Python
compilation and `git diff --check` passed. The smoke/full receipts match on
input, source, model revision and five model-shard hashes. The primary and
specificity values below were independently recomputed from all six stored
branch means and the 42 token-score sequences. All branch token counts matched,
QA and sham key budgets were equal on all seven rows, and each row had six
completed forwards. All private artifacts remain mode 0600:

- Full claim packet SHA-256:
  `d8c9ae01ab929df856a8fed8e0f3dc5456324ee2076ba8a3bfffcc10c7663b97`.
- Successful scoring smoke SHA-256:
  `1fec214700ae47fe80da83c7e44e85b1cdf0ccf0a4fc398ddf77a0a7135b1787`.
- Successful full scoring receipt SHA-256:
  `9ff2683a1ee2cd8d76c9d8d2b3f82d22116909bb0c9abcd802ca53a127914900`.
- Scoring runner SHA-256 bound into the receipt:
  `2affe074ba6431355b3673c1e822de7eb58d12ed0915f0ff0b14e51221d26e0d`.

## Results

For each row, `F = meanLP(control-pixel-mask) - meanLP(QA-pixel-mask)`.
Primary is `F_QA-bias − F_baseline`; specificity diagnostic is
`F_QA-bias − F_sham-bias`.

| Row | `F_baseline` | `F_QA` | `F_sham` | Primary | QA−sham |
|---:|---:|---:|---:|---:|---:|
| 0 | −0.01957 | +0.01336 | −0.07300 | +0.03293 | +0.08636 |
| 1 | +0.01800 | +0.00956 | +0.01895 | −0.00844 | −0.00939 |
| 2 | +0.18529 | +0.18466 | +0.15485 | −0.00064 | +0.02981 |
| 3 | −0.05379 | −0.02689 | −0.03713 | +0.02691 | +0.01025 |
| 4 | −0.02454 | −0.01441 | −0.03347 | +0.01014 | +0.01907 |
| 5 | +0.05377 | +0.02004 | +0.00966 | −0.03373 | +0.01038 |
| 6 | +0.06854 | +0.01559 | +0.05884 | −0.05295 | −0.04326 |
| **Mean** | **+0.03281** | **+0.02886** | **+0.01441** | **−0.00368** | **+0.01474** |

Primary median is −0.00064, range [−0.05295,+0.03293], positive in 3/7 rows.
QA−sham median is +0.01038, range [−0.04326,+0.08636], positive in 5/7.
These are finite-cohort descriptions only; n=7 does not justify a population
interval or inferential test here.

## Interpretation, thesis use, and limits

The baseline contrast again averages near zero and is mixed. The QA−sham
specificity diagnostic is positive on average in this cohort, but it is highly
heterogeneous (from −0.043 to +0.086 nat/token) and was one of several
exploratory endpoints. It does not override the earlier six-row negative mean
or validate the QA interval as claim support. Across the fixed three, six and
seven-row exploratory batches, signs are mixed and there is no consistent
primary lift. Do not pool them into a confirmatory estimate: they are sequential
small cohorts, share a broader candidate process, and use unreviewed claims.

For a thesis, the defensible contribution is a real-video reproducibility and
measurement-validity result: generic key bias can change claim-likelihood
mask sensitivity, yet positive proxy direction and sham specificity are not
stable across batches. The study does not demonstrate improved explanation
grounding or faithfulness. Direct reviewed claim-support evaluation remains
0 items and direct intervention `DeltaF` remains unavailable. Review was not
completed and is not imputed. Rights status remains unverified; no protected
inputs or detailed receipts are published.

After the run, read-only resource inspection found a ~16 GiB local checkpoint
snapshot on the WSL ext4 virtual disk (`/dev/sdd`), versus 15 GiB visible RAM
and 4 GiB swap (about 1.65 GiB in use). The current post-run GPU sample was
idle (7% utilization, 573 MiB allocated); this is not a trace of utilization
during the run. The slow (~147 s) load may involve cold or contended checkpoint
I/O or memory pressure, but this single observation does not identify the
cause. The runner executed all 42 forwards on CUDA and logged a 7.40 GiB peak
allocation.

## Next action

Preserve this result as a sequential replication, not a selected subgroup win.
Before another costly run, consolidate the finite-cohort diagnostics and
measure whether the slow repeated checkpoint-load phase is avoidable without
changing the fixed model, source, or estimator. Direct faithfulness still
requires actual claim-to-frame support labels; where review is declined, keep
the conclusion explicitly at proxy reproducibility and limitations.
