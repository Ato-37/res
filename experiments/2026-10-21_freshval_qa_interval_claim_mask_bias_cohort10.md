# Fresh-val QA-interval claim-mask bias pilot, fixed 10-video cohort

## Purpose and frozen question

This exploratory replication tests whether a QA-annotation-window visual-key
bias changes a fixed generated claim's QA-mask versus equal-count temporal
control-mask log-likelihood sensitivity, compared with no bias and an equal-key
budget disjoint sham. It does **not** test human-verified claim support or
direct semantic faithfulness. The estimand and six branches are the same as in
[`2026-10-21_freshval_qa_interval_claim_mask_bias.md`](2026-10-21_freshval_qa_interval_claim_mask_bias.md):
`F = meanLP(control-pixel-mask) - meanLP(QA-pixel-mask)`, primary
`DeltaF = F_QA-bias - F_baseline`, secondary `F_QA-bias - F_sham-bias`.

## Cohort, setup, and attrition

An outcome-blind local-history-screened val lock contained 27 candidates
(lock SHA-256 `778ee2f4e3ed2a5088ca449a29cf61de9d77714fbb4d34b92b2791c0f2260c40`;
mode 0600). The deterministic preflight selected the first 10 eligible
candidate rows after scanning 13. Three candidates did not provide an interval
and matched control. The fixed 10-row filter hash is
`c57642caff7afd38b279d18404aba8f0ab4c81b1a92ca4af8af61992ef3bfc15`;
the exclusion-history manifest listed 713 media hashes (SHA-256
`13e26b492a1b71fdb36f258230cef3783ef2c745addce3788c9c224f3d3c74af`).
The private full generation packet requested 10 and produced 9 structured
claims; one fixed row failed with `claim_missing`. Of those nine, one had
multiple QA intervals and two lacked a disjoint sham group set under the frozen
geometry, leaving **6 scorable rows**. No rows were replaced. The three
geometry exclusions and generation failure remain in the requested denominator.

Scoring used Qwen/Qwen2.5-VL-7B-Instruct, pinned revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, 4-bit weights with BF16 compute,
layers 5–16, `+ln(2)` key bias, and the existing six-branch QA/control mask
protocol. Runtime: 254.54 seconds for 36/36 successful forwards (six branches
per scorable row). GPU: RTX 5070 Ti, CUDA 12.8, PyTorch 2.11.0+cu128; peak
allocated VRAM 7.57 GiB. The runner's hash-bound smoke passed 6/6 forwards.
Focused scorer tests passed 11/11, Python compilation and `git diff --check`
passed. A prior smoke attempt failed before any forward because its assertion
did not account for the two prespecified CPU geometry exclusions; it is retained
as a failed zero-forward receipt, and the assertion was corrected before the
successful hash-bound smoke and full run. One earlier generation smoke also
failed to produce a claim; neither failed smoke is scored evidence.

Private, mode-0600 provenance artifacts:

- Full generation packet SHA-256:
  `af63ac026a9dd63d7ce90fc7bacddc94b756af235e8876a69b02834a777f3a73`.
- Successful scoring smoke receipt SHA-256:
  `1048c6aba874aea3d9f183d32561e3bb2d8d623e50625432beda1ca5df69b078`.
- Successful full scoring receipt SHA-256:
  `2c16d5ac6fbbda353370b1bba3f4690ef9c8b045398fa2dd59d2697f5ae23b94`.
- Scorer source SHA-256 bound into the run:
  `2affe074ba6431355b3673c1e822de7eb58d12ed0915f0ff0b14e51221d26e0d`.

## Verified results

The primary is a descriptive finite-cohort contrast; no p-value or population
confidence interval is justified at n=6.

| Row | `F_baseline` | `F_QA` | `F_sham` | `DeltaF` QA−baseline | QA−sham |
|---:|---:|---:|---:|---:|---:|
| 0 | +0.06192 | +0.04622 | +0.03510 | −0.01570 | +0.01111 |
| 1 | +0.13110 | +0.14378 | +0.13319 | +0.01268 | +0.01060 |
| 2 | −0.12988 | −0.18448 | −0.18372 | −0.05460 | −0.00077 |
| 3 | +0.11998 | +0.13656 | +0.15818 | +0.01658 | −0.02162 |
| 4 | −0.26246 | −0.20486 | −0.18711 | +0.05759 | −0.01775 |
| 5 | −0.04717 | −0.08501 | −0.05077 | −0.03785 | −0.03425 |
| **Mean** | **−0.02108** | **−0.02463** | **−0.01502** | **−0.00355** | **−0.00878** |

Primary median is −0.00151, range [−0.05460,+0.05759], positive in 3/6 rows.
QA−sham median is −0.00926, range [−0.03425,+0.01111], positive in 2/6.
All branch means and contrasts were independently recomputed from the stored
six branch means; receipt counts, model revision, hashes, and 36 completed
forwards were checked. These values describe this selected fixed cohort only.

## Interpretation and limits

The earlier three-row pilot's positive mean (`DeltaF=+0.01817`, QA−sham
`+0.03444`) did not replicate in this fixed six-row scorable cohort. Taken
together, the two tiny exploratory cohorts show no stable positive proxy
effect. The mean here is slightly negative and row effects vary in sign. This
is not evidence that internal steering generally fails; it is evidence against
claiming a reliable benefit for this particular machine-defined QA-window,
black-pixel-mask estimand from the current runs.

QA annotation intervals are not human-verified support spans for generated
claims. Black masking can be out of distribution; generated text is
re-tokenized for scoring; attrition leaves n=6; the cohort is selected by
preflight geometry; and the analysis is exploratory. No direct claim-support
review or prediction-link audit was completed. Project-wide direct faithfulness
status therefore remains **0 reviewed items and 0 direct intervention scores**;
there is no direct `DeltaF`, CI, or significance test for semantic faithfulness.
Rights status remains unverified; all private inputs and receipts stay local.

## Next action

Do not relabel the proxy as faithfulness or hunt a positive subgroup. A useful
next step is to run the same frozen six-branch estimand on the next
outcome-blind, non-overlapping fixed candidate batch, with attrition retained,
to establish whether the near-zero/mixed result persists. In parallel, the
human-review artifact must not be marked complete by waiver. If no reviewer
will complete it, keep the thesis conclusion bounded to internal intervention
diagnostics and failed/mixed proxy replication, with direct claim-support
faithfulness explicitly unresolved.
