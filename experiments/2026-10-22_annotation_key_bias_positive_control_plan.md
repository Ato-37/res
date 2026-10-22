# Prespecified plan: suppression positive control for the QA-annotation key-bias pipeline (written 2026-10-22T12:30Z, before any control run)

## Why
+ln2 (N=149) was equivalent to zero and ln4/ln8/ln16 runs are in progress. Whether those nulls are informative depends on
whether this pipeline (last-query-position additive attention-score bias on annotation key groups, layers 5-16, fp32
readout, same 149 locked videos) can detect an effect at all. A suppression arm (large NEGATIVE bias = near-removal of
the annotated frames' keys at the final query position) is the matched positive control: if the model uses the annotated
interval to answer, suppressing it should lower the gold-minus-foil margin more than suppressing an equal-count sham.

## Design (fixed now)
- Same locked N=149 cohort/lock/preflight chain, model, layers, readout, sham rule, seed. Only the sign and size change.
- Doses: -ln16 (-2.7726) and -ln256 (-5.5452). One run each, no further doses, no re-runs.
- Required code change (after the running dose pipeline finishes, because it edits a hash-bound file): allow a negative
  effective bias in `map`/audit of `run_nextgqa_annotation_key_bias.py` (currently `effective_bias > 0` is required) and the
  wrapper range check; new preflight needed because source hashes change; unit tests for the sign handling.
- Primary per dose: annotation-minus-sham margin contrast; two-sided t-test, Holm across the 2 doses (alpha 0.05).
- Decision rule: the control PASSES if at least one dose has Holm-significant NEGATIVE mean contrast with |mean| >= 0.025 nat.
  If it passes, the pipeline is sensitive and the positive-bias nulls are informative about the tested doses. If it fails
  (both doses equivalent to zero by TOST at +/-0.025, or inconclusive), the pipeline lacks demonstrated sensitivity and the
  positive-bias nulls must be reported as uninformative about selectivity.
- Descriptive: annotation-minus-baseline and sham-minus-baseline means, predicted-choice flips per dose.
- Scope: answer-margin dependence on the QA time annotation; not explanation faithfulness. No pooling with other runs.
