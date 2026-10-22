# Prespecified plan: dose-response of QA-annotation key bias (written 2026-10-22T10:40:05Z by file mtime, before the dose pipeline was launched at 10:40:45Z and before any dose run; an earlier draft header said 10:42Z in error)

## Question
The N=149 confirmatory run at +ln2 found annotation-minus-sham equivalent to zero (within +/-0.025 nat) and almost no
shift versus baseline (-0.003 nat). Is the null due to the mechanism or to a weak dose? Label-free; scope = answer-margin
dependence on the QA time annotation, not explanation faithfulness.

## Design (fixed now)
- Same locked 149-video cohort, lock/preflight chain as the +ln2 run (video = unit; paired across doses). Same model,
  layers 5-16, 32 frames, fp32 readout, seed, sham rule. Only `requested_bias_nats` changes.
- Doses (additive pre-softmax attention-score bias on annotation key groups): ln4 (1.386), ln8 (2.079), ln16 (2.773).
  Executed in this order, one run each, no further doses and no re-runs regardless of outcome.
- Primary per dose: annotation-minus-sham margin contrast. Test: two-sided t-test against 0; Holm correction across the 3
  doses (family alpha 0.05). Equivalence: TOST at +/-0.025 nat for each dose (same delta_min), reported unadjusted as
  descriptive of the null; the equivalence verdict for a dose requires TOST p < 0.05/3.
- Manipulation check (descriptive): annotation-minus-baseline and sham-minus-baseline means/SD and the count of rows whose
  predicted choice changes versus baseline, per dose.
- Dose trend (secondary, one test): paired OLS/Spearman of per-video contrast on log2 dose is NOT used for inference; only
  the three means and CIs are reported.
- Reading: if no dose shows a Holm-significant effect, the null holds up to ln16 for this mechanism; if a dose shows a
  Holm-significant effect with |mean| >= 0.025, report it as the dose at which selectivity appears, as exploratory because
  the cohort is reused and previously analysed at +ln2. Possible OOD attention at high dose is a limitation, not corrected.
- Not pooled with the +ln2 result in any hypothesis test.
