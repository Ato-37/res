# QTA v4 placebo-selection exhaustive implementation audit

## Purpose

Independently check that the frozen deterministic translated-mask selector
enumerates the intended nonwrapping offsets and chooses the prescribed
hash-minimum. This is a synthetic implementation audit, not a dataset result or
a faithfulness experiment.

## Setup

- Date: 2026-10-18 (Asia/Tokyo).
- Inputs: exhaustive synthetic group-token counts in `{1, 2}` for every length
  from 1 through 6, paired with every nonempty mask subset.
- Reference: separately written offset enumeration and explicit compact-JSON
  byte construction followed by SHA-256 ranking. The reference does not call
  the production enumeration or digest helper.
- Checks: candidate set equality, nonwrapping bounds, preserved run shape and
  selected token count, deterministic choice equality, and invariance to input
  mask ordering.
- Resources: CPU only; no GPU, model, real data, or protected experiment output.

## Verified results

Command:

```text
.venv/bin/python scripts/audit_qta_v4_mask_specificity_exhaustive.py
```

The audit completed successfully for 126 count vectors, 5,334 masks, and 4,116
eligible offsets. There were 2,812 masks with no eligible offset. No candidate,
invariant, selection, or ordering mismatch occurred. The existing utility suite
also passed (9 tests); `py_compile` and `git diff --check` passed for the new
audit script.

A separate synthetic wiring test patched only the geometry-preparation boundary,
then passed the generated translated positions into the real attention-mask
ablation helper. It verified exact position equality, equal mask cardinality,
distinct annotation/placebo positions, and no changes outside the ablated query
row. The new exhaustive-audit test and wiring test both pass. Together with the
7 result-auditor, 9 utility, and 6 runner contract tests, 24 focused tests pass.

## Interpretation and limits

The frozen selection implementation matches the independent reference on this
finite synthetic domain, and the scorer's position construction reaches the
attention-mask ablation helper unchanged under the synthetic wiring fixture.
This supports software-level reproducibility for group geometries of length at
most six with one or two tokens per group. It does not establish that real scorer
token groups are correctly mapped for unseen geometries, that any selected
placebo is semantically irrelevant, or that annotation masks are faithful. It
does not inspect protected rows.

## Next action

Finish and independently audit the already-running local mask-specificity
experiment. If its receipt passes, report the paired finite-cohort descriptive
results with provenance and limitations; do not treat this geometry audit as
evidence of efficacy.
