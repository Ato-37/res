# Cohort 30 QTA fixed-prediction key ablation (2026-10-21, local exploratory)

## Purpose and decision

Complete the missing QTA and overlap-matched sham internal-intervention
comparison on the already-frozen cohort 30. A prior diagnostic on this cohort
found a positive claim-likelihood-window versus temporal-control QA-margin
contrast but an opposite-signed benchmark-QA-interval contrast. The question
here is whether fixed question-attention visual-key bias selectively increases
answer-margin dependence on the frozen claim-likelihood-selected window,
relative to baseline and an equal-budget overlap-matched sham. This is a
post-result finite-cohort diagnostic: cohort30 scores on other endpoints were
already inspected, and this does not count as an independent or confirmatory
sample.

## Frozen procedure

- Use exactly the three rows in the existing mode-0600 cohort30 lock, pair
  filter, claim packet, selected-window receipt, and selected-window QA-margin
  receipt. Do not replace rows or change the cohort after outcomes.
- Retain the existing QTA runner and fixed protocol: question-query top-10%
  temporal groups, layers 5–16, requested bias `ln(2)` (record its BF16
  effective value), fixed baseline prediction/foil, and evidence/control
  ablations derived from the selected-window frame interval and the existing
  temporal control.
- First run a one-row smoke (five QTA forwards), then the full three rows
  (15 QTA forwards). Run the existing overlap-matched sham implementation
  with its one-row smoke (two forwards), then full (six forwards). The full
  estimate is retained only if each full receipt has all 3 rows and all
  forwards complete. Preserve failed receipts; do not count smoke as extra
  independent observations.
- Primary descriptive quantity: `DeltaF_QTA = F_QTA - F_baseline`, with
  `F = margin(control ablation) - margin(selected-window ablation)` under the
  same fixed prediction and foil. Specificity diagnostic:
  `F_QTA - F_sham`. Report row values, mean, median, range, and signs only; no
  population CI, p-value, or inferential claim for this selected n=3 cohort.
- Planning workload: 21 forwards for the full receipts plus 7 smoke forwards
  (28 total; smoke rows repeat in full), sequential model loads. Expect a few
  minutes of model execution plus load/preparation. Use the existing `.venv`,
  pinned 7B checkpoint, 4-bit weights/BF16 compute, offline mode, and per-load
  and per-forward VRAM/RAM guards. Output only unique mode-0600 receipts under
  `data/private_nextgqa/`.

## Meaning and limits

A positive finite-cohort contrast means this QTA manipulation increased the
fixed QA prediction's margin sensitivity to the **machine-selected** temporal
window relative to the matched ablation control. The window is selected by a
3B claim-likelihood procedure, has no independent human support verification,
and may be semantically wrong. This is therefore an intervention on an
automated grounding proxy, not a direct semantic-faithfulness score. It cannot
establish explanation truth, support adequacy, hallucination rate, or
population improvement. Human review remains unperformed; dataset/model
rights remain unverified; detailed receipts and all protected data remain
local.

No head/layer/alpha search is reopened. If this fixed cohort's intervention
contrast is null, mixed, or sham-equivalent, retain that as a negative
mechanistic result and do not tune on this cohort. The next direct-faithfulness
step remains actual review of the current three-claim artifact, followed by a
pre-frozen support-key versus matched-sham experiment only when valid
claim-support spans are available.

## Pre-run validation

The five private inputs are mode 0600 and bind the same three frozen rows.
Their SHA-256 values are recorded in local execution receipts; no row-level
identifiers, claims, questions, or video paths are printed or published. GPU
smoke/full results and independent arithmetic/provenance audit are appended
after execution.

## Execution result and independent audit

Both fixed-source smoke runs and full runs completed without failed rows:
QTA smoke 5/5, QTA full 15/15, sham smoke 2/2, sham full 6/6 (28 forwards
including repeated smoke rows). The two full runs used the RTX 5070 Ti, pinned
Qwen2.5-VL-7B-Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`,
4-bit weights, BF16 compute, eager attention, and 20261021 seed. QTA full
elapsed 96.60 s; sham full 67.48 s; each peaked at 11.10 GiB allocated VRAM.

| Row | `F_baseline` | `F_QTA` | `DeltaF_QTA` | `F_sham` | `F_QTA - F_sham` |
|---:|---:|---:|---:|---:|---:|
| 1 | −0.000000060 | +0.000000184 | +0.000000245 | +0.000000119 | +0.000000065 |
| 2 | −0.000000066 | +0.125000103 | +0.125000170 | −0.000000096 | +0.125000199 |
| 3 | −0.125000145 | −0.000000061 | +0.125000084 | +0.000000023 | −0.000000085 |
| **mean** | **−0.041667** | **+0.041667** | **+0.083333** | **+0.000000015** | **+0.041667** |

For `DeltaF_QTA`, median is +0.125000084 nat and range is
[+0.000000245,+0.125000170]; only 2/3 rows exceed 1e-6 (the remaining row
is numerical zero). For `F_QTA - F_sham`, median is +0.000000065 nat and range
is [−0.000000085,+0.125000199]; only one row has a non-negligible positive
contrast. These are exact finite-cohort descriptions; no population CI,
hypothesis test, or significance claim is appropriate for this post-result
n=3 sample.

Full QTA receipt:
`data/private_nextgqa/qta_fixed_prediction_key_ablation_cohort30_full_20261021_01.json`,
SHA-256 `b27a916cea516e32022d4dc4f11d8bd7a863c3a44b3e4668c83f7e10d8b5fcde`.
Full sham receipt:
`data/private_nextgqa/qta_hash_matched_sham_cohort30_full_20261021_01.json`,
SHA-256 `069dadf6cd5bd4f68234324263dd1aa72931a9ce37a31b77000a8fa200951f8d`.
Both are owner-only mode 0600. Primary-agent checks recomputed row arithmetic
and matched all recorded input/source hashes. An independent auditor confirmed
the same hashes, all three row/video/claim bindings, prior full-choice-vector
reconstruction, smoke-to-full first-row equality, 21/21 full forward counts,
12/12 layer-hook calls, BF16 bias/ablation assertions, equal QTA/sham key
budgets, matched evidence/control overlap counts, and stored cohort summary.
No GPU rerun was used for this audit.

## Interpretation and next action

The finite-cohort QTA-versus-baseline proxy moves positively on two rows, but
the one-row QTA-versus-sham difference is the only material specificity
contrast. The other two rows are numerical zero, and the average is driven by
one +0.125 nat row. This is a weak, heterogeneous mechanism signal on a
machine-selected window; it does not establish a robust internal-intervention
benefit, claim truth, human-grounded support, or direct faithfulness. It does
not contradict the 27-row full-video accuracy result (0/27 prediction changes),
because this endpoint is a fixed-answer margin sensitivity under ablation.

Do not repeat or retune this cohort. Current direct claim-to-frame review and
direct semantic-faithfulness scores remain 0/3 for this cohort; the separate
formal review counts remain 0/25, 0/15 clock/frame checks, and 0/3
prediction-link reviews. The next direct step is to collect an actual response
for the exact current three-claim review artifact. Only a human-confirmed
support span can enable the prespecified support-key versus matched-sham
faithfulness test; model/QA-selected windows remain proxies.
