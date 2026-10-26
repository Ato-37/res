# QTA cross-cohort sensitivity synthesis (2026-10-22)

## Purpose

Independently recompute the fixed-prediction internal-key-ablation contrasts
already collected on nine frozen NExT-GQA development cohorts. The question is
whether the previously observed small positive directions are stable across
cohorts and against matched sham. This is a descriptive reanalysis; no GPU
forward or new outcome collection was performed.

## Inputs and reproducibility

The analysis reads the completed local QTA and overlap-matched sham receipts
for cohorts 16, 17, 18, 20, 21, 22, 23, 24, and 26. For each pair it verifies
the completed-run contract (3 rows, 15 QTA forwards, 6 sham forwards), exact
row-token pairing, source-video hash pairing, fixed prediction/foil pairing,
and the declared contrast against the saved row-level margins. It also verifies
that all receipts label direct faithfulness as false and human review as not
performed. The 27 video hashes were unique across these nine receipts.

Code: `scripts/analyze_qta_crosscohort_sensitivity.py` (SHA-256
`187366d9e2e19230dc6d7ff04b97b1de8716f4ed7a5b40f0ea2005391cf5c4c2`).
Private derived summary: `data/private_nextgqa/qta_crosscohort_sensitivity_20261022.json`
(mode 0600; SHA-256
`e0aa741885aa7469193d386c593e360aac9a130eb6194cef56014cc27ea03b62`).
Reproduce with:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_analyze_qta_crosscohort_sensitivity -v
PYTHONPATH=scripts .venv/bin/python scripts/analyze_qta_crosscohort_sensitivity.py
```

Four focused unit tests pass, covering explicit near-zero counts, formula
recomputation, and rejection of a mismatched saved effect. Byte-compilation
and `git diff --check` pass.

## Observed results

| Contrast | Mean of 9 cohort means (nat) | Median | Range | Positive / negative / near-zero cohorts* |
|---|---:|---:|---:|---:|
| QTA minus baseline selective-ablation sensitivity | +0.013889 | +0.000000036 | [−0.083333, +0.125000] | 3 / 3 / 3 |
| QTA minus matched-sham sensitivity | +0.023148 | +0.000000016 | [−0.041667, +0.125000] | 4 / 3 / 2 |

Each contrast includes 27 paired video rows from 9 cohorts; the run receipts
record 135 completed QTA forwards and 54 completed sham forwards. Near-zero
uses an explicit absolute tolerance of `1e-7` nat. This computation
recalculated each QTA contrast from the four saved baseline/QTA ablation
margins and each sham contrast from its saved QTA and sham selective-
dependence scores; it did not use only the cohort summary fields.

## Interpretation and limitations

The positive grand means are driven by a small number of cohort directions;
the cohort medians are effectively zero, and both contrasts contain negative
and near-zero cohorts. This is not evidence of a stable QTA benefit or a
reliable advantage over matched sham. The cohorts are post-hoc selected finite
development cohorts, so this log reports no confidence interval, hypothesis
test, multiplicity adjustment, or population inference. The endpoint is a
fixed-prediction answer-margin sensitivity proxy under temporal-key ablation;
it does not verify claim truth, human-reviewed video support, or direct
explanation faithfulness. Human review was not performed; rights status remains
unverified; all data and this derived receipt remain local.

Direct faithfulness progress is unchanged at 0/25 formal support reviews and
0/3 on the latest review artifact; no direct `DeltaF` estimate or uncertainty
interval exists. This analysis supports a thesis result about the instability
of a generic internal steering proxy, not a positive grounding or faithfulness
claim.

## Next action

Do not collect more QTA outcomes on these reused cohorts or reopen its layer,
alpha, or cohort search. For the requested positive grounding/faithfulness
claim, the next discriminating evidence remains an actual human-reviewed
claim-to-frame and prediction link followed by a frozen baseline/support-key/
matched-sham support-ablation comparison. In parallel, keep the current
Japanese review handoff available; its current response count is zero. A future
NOAA QA experiment must remain explicitly exploratory and separate from direct
faithfulness unless those claim-level judgments are collected.
