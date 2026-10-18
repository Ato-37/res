# QTA two-phase forward-schedule diagnostic (2026-10-18)

## Purpose

Prepare one bounded human-review-free piece of the production workflow: check
that a future three-record QTA schedule keeps its baseline preparation and
post-baseline branches in the required two phases. The documented design has
one baseline-full forward per record (3 total), followed only after baseline
`p/q` and stage-2 review by eight other branches per record (24 total). This
does not address whether the reviews, inputs, or scores are valid.

## Implementation

Added `scripts/qta_phase_schedule_diagnostic.py` and
`scripts/test_qta_phase_schedule_diagnostic.py`. The CPU-only diagnostic checks
three unique opaque record tokens, exact branch sets (baseline-full only in
phase 1; eight non-baseline conditions in phase 2), identical processor-batch
digests across both phases, and syntactically valid hashes for the phase-1
receipt, complete baseline manifest, and stage-2 review receipt. It rejects
partial row sets and any phase-2 branch spillover. Its output does not echo
tokens or input digests.

The output is explicitly `shape_valid_non_authorizing`: it always reports zero
authorized forwards, denies model load/scoring/phase authorization, and marks
artifact bytes, signatures, human identity, and semantic review as unverified.
This is a schedule-shape regression aid only. It is not either production phase
receipt, does not rerun upstream validators, does not verify any digest against
bytes, and is not integrated with a model runner.

## Checks and results

- Focused synthetic tests: **8/8 passed**.
- `py_compile` and `git diff --check`: passed.
- Full CPU repository suite after the final count-field naming and test changes:
  **1,900/1,900 passed** in 54.546 seconds, exit code 0. GPU visibility was
  explicitly disabled for the run. Expected negative-path fixture messages
  and Transformers deprecation warnings appeared.
- No protected review material, video, model, or GPU was accessed.

An independent implementation audit found no authorization path and confirmed
that incomplete phases return zero total schedule branches. It recommended
making the shape-only meaning of counts explicit; the output now includes
`counts_are_schedule_shape_only=true`, `execution_performed=false`,
`requested_rows=3`, and separately named branch-shape counts. The final test
also asserts the fixed requested-row denominator. The audit confirmed that
digest fields are syntax-checked only and cannot establish artifact linkage.

Source SHA-256:

- `scripts/qta_phase_schedule_diagnostic.py`:
  `405020b337de56794b276683768638e73cf04ccc6aa9a0c23305ff5efaa55b40`
- `scripts/test_qta_phase_schedule_diagnostic.py`:
  `eae9a9ab0ca2efee9e9feb2971680e71f0d98bad01ce543ed2cad84bab1b287c`

## Interpretation and limitations

This establishes only that the synthetic contract checks a 3+24 schedule
shape and rejects tested partial, duplicate, branch-spill, binding-shape, and
processor-batch mismatch cases. It does not produce a real phase receipt,
prove that the named hashes identify approved bytes, authenticate stage-2
review, open a model gate, or measure faithfulness. The direct study remains
blocked on external trust/role evidence, human claim review, 15/15 verified
clock/frame pairs, and the still-missing production score-freeze verifier and
runner. Rights status remains unverified; protected materials stay local.

## Faithfulness progress and next action

- `直接評価済み`: unchanged at 0/3 independently reviewed claims, 0/15
  verified browser/player-clock to decoder-frame pairs, 0/3 prediction-link
  judgments, and 0/3 direct claim-linked scores. There is no direct `ΔF`, CI,
  hypothesis test, or multiplicity-adjusted result.
- `補助診断のみ`: schedule-shape tests; no model inference or faithfulness
  metric.
- The next implementation step is a cryptographically verified pre-baseline
  receipt tied to exact upstream artifacts, followed by an independently
  verified phase-2 receipt. Neither may authorize inference until the complete
  human, clock/frame, model/source, and all-row preflight checks are integrated.

## Reproduction

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_phase_schedule_diagnostic -v
```
