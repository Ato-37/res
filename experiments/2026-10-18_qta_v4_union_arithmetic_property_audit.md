# QTA v4 union-mask arithmetic property audit (2026-10-18)

## Purpose and setup

Independently check the frozen four-condition union-mask contrast while the
source-bound geometry census is repaired. The v4 estimand freezes the
baseline-full top answer and best foil, computes the full-minus-union-mask
margin drop for baseline and intervention, then subtracts the two drops. This
CPU-only synthetic audit compares that contract with an independent scalar
implementation and checks metamorphic properties. It reads no dataset or
private artifact, loads no model, and does not alter the scorer or census.

Command:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/audit_qta_v4_union_arithmetic_properties.py \
  --draws 10000 --seed 20261018
```

## Verified synthetic result

- **10,000/10,000** seeded synthetic four-condition score sets agreed with
  the independent reference to maximum absolute difference **0 nat**.
- Common shifts across all five options in a condition left its margin
  unchanged; the same perturbation to the frozen answer in both masked
  branches canceled from the difference-in-drops; and alphabetical tie
  handling with a zero-delta case passed.
- In **9,534/10,000** deliberately independent random synthetic draws, the
  intervention-full top two options differed from baseline-full. This is a
  property-test stress-case count, not an estimate of treatment-induced answer
  switching in Qwen or the v4 cohort. It confirms why the protocol must keep
  baseline-full answer/foil labels fixed across all four conditions.
- Focused test suite passed **2/2**; the full property audit passed; Python
  compilation and `git diff --check` passed.

Source SHA-256:

- Audit runner: `b553c0797a01392ff09b26d52f94e4f90bc6dffc47c5676a7ed5e9c1eb770194`
- Tests: `0863a29afec9135ad42a49fb3a3cef084e8eeef3f57cce3849ae66deb21eedce`

## Interpretation and limits

The observed result verifies the finite arithmetic contract and two invariance
properties under the synthetic input construction. It does not establish that
the model's scores are correctly bound to the same prompt, token IDs, or
position; that the mask corresponds to valid annotation evidence; or that
QTA changes prediction dependence. It provides no empirical Δ, uncertainty
interval, test, annotation validity, or explanation-faithfulness evidence.
The v4 primary run remains gated on its full-cohort geometry preflight, source
audit, and one-row four-forward smoke.

## Next action

Use the passing arithmetic contract as a bounded software gate only. Finish
the v4 CPU geometry census and independent source audit under their existing
ownership; do not treat this synthetic property run as a substitute for those
gates or as a reason to start GPU scoring.
