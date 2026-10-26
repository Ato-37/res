# NExT-GQA native QA evidence-interval coverage audit (2026-10-21)

## Question and method

The user asked whether NExT-GQA includes temporal evidence intervals. This
aggregate-only audit joins the local `val.csv`/`test.csv` QA identifiers to
`gsub_val.json`/`gsub_test.json` and validates every listed interval against
the source video's declared duration. It prints no video IDs, questions,
answers, interval values, or claims. Dataset README defines `gsub` locations
as time-span annotations corresponding to QA rows, with multiple spans
permitting alternatives.

The runnable audit is `scripts/audit_nextgqa_native_interval_coverage.py`.
Its unit tests cover exact joins, invalid time boundaries, duplicate QA rows,
boolean values incorrectly accepted as numeric timestamps, and aggregate-only
packet-frame overlap. This is a dataset-structure audit, not a model or
semantic review.

## Verified aggregate results

| Split | CSV QA rows / unique videos | Exact QA-to-interval join | Alternative spans | Valid positive intervals | Boundary anomalies |
|---|---:|---:|---:|---:|---:|
| val | 3,358 / 567 | 3,358/3,358; no missing or extra annotated QA IDs | 3,931 | 3,928 | 3 spans end after declared video duration |
| test | 5,553 / 990 | 5,553/5,553; no missing or extra annotated QA IDs | 6,600 | 6,564 | 36 spans have negative start times |

Multiple acceptable evidence spans occur for 339 validation QAs and 591 test
QAs; the rest have one span. Each span is an interval pair. The exact input
SHA-256 values are recorded by the command output and are listed here:

- val CSV: `ac5a1eb1d6dee9cd84ab7f9c94ab345b2404646f24b79ac182cd2aa9e2d424c1`
- val gsub: `4feb23e53102cb2d4779e857ff8f9fd9b552fc38435621406b2ed9b0bf76204b`
- test CSV: `0443ede53f521de4cbfb5e4fb02253926b90f81f62041b18bd17207a3511d114`
- test gsub: `0c16c31699238a0f2b8ace45aeed7f193149ce2f011d6ed4f1ef7170ef5d41e9`

The exact three-item claim-review artifact was also checked against its
source-bound packet timestamps. All 3/3 rows have at least one timestamp inside
the QA interval union; in total, 15/48 timestamps intersect four reference
intervals. This uses inclusive boundaries (`start <= PTS <= end`). Independent
review confirmed the artifact fingerprint, packet/video/row-token bindings,
generator/lock/pair-filter/decoder source hashes, packet hashes of
`test.csv`/`gsub_test.json`/video mapping/frame-time inputs, and frame-time
provenance. The HTML does not pre-highlight QA intervals because they are not
claim-specific and could prime a reviewer.

Four focused CPU tests passed (4/4), the current dataset audit completed, and
`py_compile` passed. No GPU inference ran because these interval counts do not
establish alignment between QA evidence and a generated explanation.

Reproduce the aggregate dataset and current-artifact overlap audit with:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/audit_nextgqa_native_interval_coverage.py \
  --include-current-review-packet-overlap
PYTHONPATH=scripts .venv/bin/python -m unittest -v \
  test_audit_nextgqa_native_interval_coverage \
  test_build_test_split_claim_support_frame_review
```

## Research implication and limitation

The earlier blanket impression that NExT-GQA lacked evidence segments is
incorrect: its val/test QA sets are fully covered by native question-level
time-span annotations. The limitation is semantic granularity. A QA interval
marks evidence for a question/answer; it does not prove that each generated
claim describes the event in that interval. Some annotations also violate
duration bounds and must be excluded or handled under a pre-fixed rule before
time-based scoring; do not silently clip them.

Thus these annotations can support an answer-grounding diagnostic or provide
candidate windows for claim review, but they do not remove the current
claim-to-frame review gap. The exact three-item handoff remains 0/3 reviewed,
0/15 clock/frame correspondences and 0/3 prediction links. No direct
claim-faithfulness effect or uncertainty interval exists. Rights status
remains unverified; files/results remain local.

## Next action

Use these QA spans only as pre-existing question-level evidence candidates.
For the current claim-specific study, preserve the existing local review
handoff and require the reviewer to determine whether the claim is
answer-relevant and supported by the sampled evidence. If a later machine-only
exploratory analysis uses the raw intervals, predeclare how invalid
duration-boundary intervals and multi-span alternatives are treated and label
the result as QA grounding, not explanation faithfulness. Avoid another GPU run
on the same interval proxy without a new discriminating estimand.
