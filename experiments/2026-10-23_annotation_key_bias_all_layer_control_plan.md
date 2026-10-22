# Prespecified plan: all-layer variant of the annotation-key suppression control (written 2026-10-23, before any run)

## Why
The layers 5-16 suppression control at the final query position failed (no effect even at x1/256), so the +ln2..+ln16 nulls
are uninformative. The leading hypothesis is low leverage of a last-position key edit in mid layers. Smallest discriminating
variant that needs no change to the hash-bound base runner: the same last-query-position bias applied in ALL 28 decoder
layers (wrapper option `--layers all`, setting the base module's layer tuple).

## Design (fixed now)
- Same locked N=149 cohort, lock and poscontrol preflight chain, model, fp32 readout, sham rule, seed, scoring.
- Step 1 (control, one run): bias -ln256 (-5.545), layers 1-28. PASS if the annotation-minus-sham mean contrast is negative,
  two-sided t-test p < 0.05, and |mean| >= 0.025 nat. (Single test, no multiplicity.)
- Step 2 (only if Step 1 passes): positive bias +ln4 and +ln16, layers 1-28, Holm over the 2 doses, TOST at +/-0.025;
  these are then the first informative selectivity tests of the key-bias mechanism. If Step 1 fails, Step 2 is not run and
  the conclusion is that last-position key edits are inert across depth for this design; the next candidate (multi-query
  rows after the video, needs a hash-bound runner refactor) would be a separate plan.
- No dose/layer search beyond these; no re-runs. Not pooled with earlier runs. Scope: answer-margin dependence on the QA time
  annotation; not explanation faithfulness.
