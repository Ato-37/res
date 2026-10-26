# QTA overlap-matched hash-sham ablation diagnostic (2026-10-21)

## Purpose and status

The preceding fixed-prediction key ablation had mean `DeltaF=+0.04167` nat
on three sequentially analyzed cohort-12 rows. A post-result audit found that
QTA-selected groups overlapped the control window in two rows. This exploratory
follow-up asks whether an equal-size hash-sham key set, matched to QTA's counts
of evidence/control group overlap, produces the same difference-in-differences.
The sham candidate set is selected after seeing the QTA overlaps, so this is a
post-hoc falsification diagnostic, not a confirmatory test or independent
replication.

## Frozen follow-up

- Anchor: the completed fixed-prediction receipt
  `data/private_nextgqa/qta_fixed_prediction_key_ablation_cohort12_20261021_03.json`
  (SHA-256 `ab79530682de6b78e56b8d6d73651f7a1f5e7e07c6bb0d8079916917a33bfe09`).
- Rebuild the exact same three rows and processor batch; check row, video,
  batch, model revision, precision, and choice-token-binding hashes against the
  anchor before loading model weights.
- Choose two sham temporal groups per row, same cardinality as QTA. Candidate
  sets must match QTA's group-overlap counts with both the evidence and control
  windows, and must differ from the exact QTA set. Rank eligible sets by
  SHA-256 using seed `qta-overlap-matched-sham-v1-20261021` and the opaque row
  token.
- For each row run two additional forwards: fixed prediction/foil with
  `+ln(2)` sham bias plus evidence-window key ablation; and the same bias plus
  control-window key ablation. Use the same Qwen checkpoint, layers 5–16,
  BF16 mask-aware check, and 4-bit/BF16 model configuration as the anchor.
- Report `F_sham`, `DeltaF_sham=F_sham-F_base`, and
  `F_QTA-F_sham`, with no gold-label use, significance test, or population
  inference.

## Checks and execution

Runner: `scripts/run_qta_hash_matched_sham_ablation.py`. Four CPU tests passed
for deterministic selection, overlap matching, unavailable sham sets, and
exclusive private receipt creation. Python compilation and scoped whitespace
checks passed. Actual processor preflight verified all three anchor row/video/
batch bindings and generated deterministic sham groups `[4,7]`, `[8,11]`, and
`[1,3]`; each matches the corresponding QTA overlap counts. Independent code
audit passed before GPU execution.

Final result is stored locally at
`data/private_nextgqa/qta_hash_matched_sham_ablation_cohort12_20261021_02.json`
(mode 0600; SHA-256
`fb25eeb99a4b8651d1e290a47e0f9e5d0da0955a480d5b272c24269f24a95e77`). It
contains 3/3 rows, 6/6 attempted/completed forwards, pinned model revision,
library versions, UTC start time, 14 source hashes, and 7 input hashes. The
primary independently checked all row arithmetic and hashes; an independent
auditor reproduced the same checks. Hardware was RTX 5070 Ti, peak allocated
VRAM 11.17 GiB, and elapsed time 296.63 seconds including model load.

| Row | `F_base` | `F_QTA` | `F_sham` | `F_QTA-F_sham` (nat) |
|---|---:|---:|---:|---:|
| 1 | −0.1250 | approximately 0 | −0.1250 | +0.1250 |
| 2 | −0.1250 | approximately 0 | −0.1250 | +0.1250 |
| 3 | −0.1250 | −0.2500 | −0.1250 | −0.1250 |

Mean `F_QTA-F_sham` was `+0.0416667` nat (median `+0.1250`, range
`[−0.1250,+0.1250]`, 2/3 positive). The QTA-minus-sham row signs and scale
match the earlier `DeltaF` because sham `F` stayed near `F_base` in all three
rows. This does not demonstrate a consistent advantage: the sign differs in
one row, the effect is small, and the same post-hoc cohort anchors both
analyses. No interval or significance claim is appropriate for this
three-row diagnostic.

An initial sham run receipt (`..._01.json`) was superseded. Its 11 declared
source hashes and arithmetic passed audit, but its source map omitted two
transitively used choice-binding helpers (recoverable through the anchor's
hash-bound source map). The final `_02` runner adds these helpers and the
private output helper to its source map, plus environment versions. The
initial receipt remains unchanged and private.

## Interpretation and limits

User waived human-review collection as an execution prerequisite, so the
local experiment proceeded without waiting for review. Human review did not
occur; no interval-to-claim validity is established, and direct
human-validated faithfulness evidence remains 0/3. The output stayed in the
private local directory and was not published. Dataset rights remain
unverified. The bias is global and the selected intervals remain machine
selected. This experiment is neither causal evidence about the generated
explanation nor thesis-level efficacy evidence.

## Next action

Inventory whether a separate, unselected candidate cohort can be frozen
without reuse of these three rows. If available, run this same primary/sham
contrast on that cohort with protocol and thresholds fixed before scoring;
otherwise document the data-availability limit and prepare the review-free
simulation as exploratory only. Do not expand or tune on cohort 12 to obtain a
more favorable estimate.
