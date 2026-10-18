# Research-loader inventory regression check (2026-10-18)

## Purpose

After adding the local QTA CPU processor smoke, run the complete scripts test
suite to check its interaction with repository-wide guards that classify
pretrained processor/model loaders.

## Observed results

The first full run completed **1,865 tests with 2 failures**. Both failures were
from `test_closed_research_branches`: newly tracked QTA v3/v4 local-data
scripts and the new synthetic processor smoke were not present in the static
loader classification sets. No execution or research-result test failed.

Added the relevant scripts to the local-data-pilot and processor-only
classification sets. The focused static guard then passed **8/8** tests.
Re-running the full suite completed **1,865/1,865 tests successfully** in
**54.450 s**. Expected negative-path CLI output and Transformers deprecation
warnings appeared; the command exited 0. No GPU or protected media was used by
the test suite.

## Interpretation and limits

This is repository-wide software-regression evidence. It neither re-runs any
QTA experiment nor authenticates the release roster, establishes human review,
or measures faithfulness. The local processor suffix run and its processor
fingerprint remain local-only; they are not included in a public result
artifact. Rights status remains unverified.

## Next action

Continue with independent outcome-blind design and contract checks. Real
claim-linked baseline/treatment scoring remains blocked until the trusted root,
role assignments, review-only receipt, witnessed unopened handoff, actual
human judgments, and frame-pair evidence are complete.
