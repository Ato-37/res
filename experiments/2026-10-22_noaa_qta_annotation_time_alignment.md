# NOAA QTA annotated-time alignment audit (2026-10-22)

## Purpose

Check whether the temporal group selected by question-attention QTA on the
completed NOAA QA pilot overlaps any annotator-listed evidence timestamp more
often than the equal-budget disjoint hash sham. This is a post-hoc grounding
proxy analysis of an existing GPU receipt, not a claim-support or faithfulness
test.

## Data and method

The analysis joined all six sample rows to the corresponding local annotation
audit by media SHA-256 and hashed question ID. For each listed evidence point,
it found the nearest of the eight sampled frame timestamps and mapped that frame
to its adjacent-frame temporal group (two neighboring frames per group, four
groups per row). A hit means the selected single group equals at least one
group containing a mapped evidence point. The comparison is against an expected
hit count for a uniformly random single group, computed per row as the fraction
of the four groups containing any mapped point. No threshold or subgroup was
selected from outcomes, but this is post-hoc because the QTA/sham receipt already
existed.

The current QTA receipt covers 6 rows across 3 videos, with 4 temporal groups
per row and one QTA and one disjoint sham group. The three Dive03 rows derive
from two video-only annotator passes plus a blind visual auditor; the other
three rows derive from independent AI blind-answer reviews. These annotations
were not independently re-reviewed for this analysis and are question-evidence
points, not reviewer-verified support spans for generated explanation claims.

Reproduce with:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  scripts/test_analyze_noaa_qta_annotation_time_alignment.py -v
.venv/bin/python scripts/analyze_noaa_qta_annotation_time_alignment.py \
  --output datasets/noaa_qa_pilot_20261011/private/noaa_qta_annotation_time_alignment_20261022_01.json
```

The analysis code is `scripts/analyze_noaa_qta_annotation_time_alignment.py`.
An independent code audit found that the first version did not enforce receipt
manifest hashes and did not strictly enforce unique six-row joins. The analysis
now checks exact manifest path/hash equality, six unique row tokens and
video/question keys across exactly three videos, and safe local input files.
Focused tests cover these invariants as well as nearest-frame grouping,
deterministic tie handling, and invalid geometry. The combined analysis and QTA
runner tests pass 11/11 and both scripts byte-compile. A second independent
audit found two remaining edge cases (symlink receipt-path checking and
duplicate group indices); both are now rejected and covered by tests.
Reproduction command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  scripts/test_analyze_noaa_qta_annotation_time_alignment.py \
  scripts/test_run_noaa_qta_distribution_pilot.py -v
.venv/bin/python scripts/analyze_noaa_qta_annotation_time_alignment.py \
  --output datasets/noaa_qa_pilot_20261011/private/noaa_qta_annotation_time_alignment_20261022_03.json
```

The corrected analysis script SHA-256 is
`f252b7f8ae298411da71bf894877481752252b02d9e796d8a506b67eec0fee52`.
The regenerated local summary is mode 0600 at
`datasets/noaa_qa_pilot_20261011/private/noaa_qta_annotation_time_alignment_20261022_03.json`
(SHA-256 `7490f4e1630cc7b6ce316f74a1da75ac5a4ca095984c4587e56650bda2c2ba55`).
The initial `_01` and `_02` outputs predate the complete audit fixes and are
superseded.
The summary contains only aggregate counts and source hashes, not row IDs,
questions, choices, or timestamps. It is not published.

## Observed results

- QTA selected a group containing at least one listed evidence point in **2/6**
  rows.
- The equal-budget disjoint sham did so in **3/6** rows.
- Given the observed number of distinct evidence groups per row, a uniform
  random single group would be expected to hit **1.75/6** rows.
- Paired outcomes were QTA-only 1, sham-only 2, both 1, neither 2.
- By annotation method, the 3 two-annotator rows had QTA 0/3 and sham 2/3 hits;
  the 3 independent-AI-review rows had QTA 2/3 and sham 1/3 hits. These strata
  are too small for inferential comparison.

## Interpretation and limitations

This finite sample gives no indication that the question-attention-selected
group aligned with listed QA evidence points better than the chosen sham. QTA
was near its coarse chance expectation (2 vs 1.75 hits), while sham had 3 hits.
With only six rows, three video clusters, coarse eight-frame sampling, and
annotation methods that differ by video, the difference is descriptive; no
confidence interval, test, or population claim is warranted. Nearest sampled
frame mapping can miss evidence between samples. The sham controls group budget
and disjointness only, not position or visual content. Annotation points relate
to QA questions; they do not validate any explanation claim's semantic support.
Gold JSON, model weights, and media were not opened or used by the analysis;
question/answer correctness fields were not used. No new GPU forward was run.
Human review of the current explanation claims remains unperformed, rights
status remains unverified, and the result remains local-only.

Direct faithfulness remains at 0/25 formal support reviews and 0/3 for the
current claim/frame/prediction-link/direct-score handoff. This analysis adds a
weak, exploratory grounding-validity check for question-attention selection; it
does not change direct faithfulness progress.

## Next action

Do not tune QTA or claim an alignment benefit from these six observations. A
separate exploratory QA prediction-sensitivity experiment may compare evidence-
mapped temporal groups with position-matched equal-budget non-evidence groups,
but it would remain question-time sensitivity rather than explanation
faithfulness. A test of explanation evidence dependence still requires
claim-specific support spans reviewed independently. Keep this result as a
measurement-validity limitation alongside the answer-distribution result.
