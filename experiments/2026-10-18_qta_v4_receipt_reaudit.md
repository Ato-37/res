# QTA v4 local receipt arithmetic re-audit (2026-10-18)

## Purpose

Independently check the arithmetic and currently available source provenance
for the protected local v4 exploratory receipt. This does not rerun inference
or authorize scoring/publication.

## Verified observations

- The private receipt bytes match the experiment log's SHA-256
  `60f3b5a1847f042406ea727932d3960c8a64be11a2b0ecb8e81808f6289b41cc`; the
  file is mode `0600` and remains local.
- Receipt status is completed exploratory finite-cohort; requested/effective
  rows are 147/147 and requested/completed forward calls are 588/588.
- Independently recomputing each contrast from the stored fixed-answer/foil
  margins gave exact equality for all 147 rows. Recomputed mean was
  `+0.006802729000668057` nat, sample SD `0.1370730683903988` nat, range
  `[-0.37499968174961396, +0.3750007720082067]` nat, with 74 positive, 68
  negative, and 5 zero rows. These match the stored summary to floating-point
  representation.
- All four pinned data files match their receipt hashes. Of 24 source files,
  22 match the current checkout. The current selector source differs, but its
  receipt-pinned bytes are present in Git history at
  `93824b3451cf68f986460785b6cf828eb43ce712`. The receipt-pinned bytes for
  `scripts/run_nextgqa_validation_qta_annotation_proxy.py` were not found in
  available committed history; its current bytes differ. Therefore this
  re-audit cannot independently revalidate the complete frozen execution
  source set or the run-time plan verification from the present checkout.

## Interpretation and limitations

The result arithmetic and private-receipt hash are independently reproduced.
The inability to recover one pinned source blob means the end-to-end source
provenance is not SSS-verifiable from the currently available Git objects.
This narrows the provenance claim; it does not show that the recorded result
is arithmetically wrong. The result remains a finite, test-associated
annotation-mask prediction-margin dependence proxy, not generated-claim
faithfulness. No inference, population uncertainty interval, or significance
test is added here. Rights remain unverified and all receipt content stays
local.

## Next action

Preserve the private receipt. Do not rerun the completed GPU cohort solely to
repair missing historical source provenance. For future runs, ensure all
execution sources are committed or content-addressed before launch and bind
the exact source manifest into the private receipt. Continue the separate
capture-to-selection component work; it is not yet connected to an authorized
phase-2 signed receipt.
