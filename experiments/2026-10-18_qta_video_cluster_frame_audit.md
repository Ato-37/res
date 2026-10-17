# NExT-GQA video-cluster candidate-frame audit (2026-10-18)

## Purpose

The QTA sample-size sensitivity analysis counts independent source-video
clusters, not question rows. The local NExT-GQA test split can show how many
video IDs and QA records exist and whether every test pair has interval
metadata. It cannot establish independent source clusters or claim-review
eligibility. This metadata-only census tests candidate-frame availability
without opening video bytes or emitting dataset identifiers or question/answer
text.

## Method and verified results

Added `scripts/audit_nextgqa_video_cluster_frame.py`. It streams
`test.csv`, retains only the `(video_id, qid)` pairs and row counts, and parses
the test gsub and video-map metadata. Other CSV columns are parsed transiently
by the CSV reader but are not used, retained, or emitted. It reads no media
bytes. The exact command was:

```bash
PYTHONPATH=scripts ./.venv/bin/python scripts/audit_nextgqa_video_cluster_frame.py \
  --test-csv NExT-GQA/datasets/nextgqa/test.csv \
  --gsub-json NExT-GQA/datasets/nextgqa/gsub_test.json \
  --video-map-json NExT-GQA/datasets/nextgqa/map_vid_vidorID.json
```

Observed aggregate counts:

| Quantity | Count |
| --- | ---: |
| Test QA rows | 5,553 |
| Unique `(video_id, qid)` pairs | 5,553 |
| Duplicate QA pairs | 0 |
| Unique test video IDs | 990 |
| Test IDs with interval metadata | 990/990 |
| Test IDs with video-map entries | 990/990 |
| Distinct mapped paths among test IDs | 990 |
| All gsub IDs / those outside test | 999 / 9 |
| Intervals attached to test QA pairs | 6,600 |

QA rows per video-ID ranged from 1 to 12 (nearest-rank median 6, p25 4, p75
7, p90 9). Exact histogram is retained in command output and reproducible
through the script; no ID, path, QA text, or media is emitted. The weighted
histogram sums to 5,553 rows across 990 IDs. The independent auditor
independently reran and recomputed each count and quantile, confirmed the
focused synthetic suite passed **3/3**, and checked that no raw identity or QA
text is emitted. The complete `scripts` suite then passed **1,613/1,613** in
**38.694 seconds** (including this audit's three tests); expected negative-path
fixture output and Transformers deprecation warnings appeared. `git diff
--check` passed. The tests verify duplicate-pair accounting, required-column
rejection, and absence of sentinel identifiers/question/answer strings from
output.

## Interpretation and limitations

The test split contains enough *video IDs* to make the illustrative 23–63
independent-unit scenarios from the preceding power sensitivity numerically
possible, but that does not establish 23–63 independent source clusters,
eligible claims, successful human reviews, available authorization, or a
random sampling frame. Distinct map paths do not prove byte-distinct media or
independent event/source identities. Prior-exposure history is partial, and
all three current feasibility candidates remain quarantined from any later
confirmatory cohort. If multiple QA rows from one video are considered, the
analysis must predefine a video-level selection/aggregation rule; they cannot
be counted as separate independent videos. The 6,600 interval count is only a
metadata count, not a validation of annotation correctness or claim-specific
evidence.

Rights status remains unverified; all inputs were read locally and no
protected content was published.

## Faithfulness progress

- `直接評価済み`: remains **0/3** reviewed claims, **0/3** prediction-link
  judgments, **0/15** verified browser-player/decoder correspondences, and
  **0/3** real QTA scores; no empirical `DeltaF`, uncertainty interval, or test.
- `補助診断のみ`: one metadata-only candidate-frame audit of 990 test video
  IDs, full test-pair interval-key coverage, and **3/3** synthetic tests.
  This is not evidence of explanation faithfulness or treatment effect.
- Source clustering, prior-exposure completeness, claim eligibility, and human
  review completion remain unresolved.

## Next action

Resolve prior-exposure coverage and source/event clustering outcome-blindly
before choosing any confirmatory sample frame or variance model. The immediate
feasibility pilot still cannot enter baseline scoring until the independent
human claim review and all 15 browser-player/decoder-frame correspondence
decisions are complete under the frozen protocol and trusted release chain.

## Provenance

- `scripts/audit_nextgqa_video_cluster_frame.py` SHA-256:
  `4988581e6cb9bd4108c6dc76befd1ec22036a307ee8c87d88ce7d00bda8cff90`
- `scripts/test_audit_nextgqa_video_cluster_frame.py` SHA-256:
  `5f640963a7f1b04a15ada63ee1add5a16ff0aca057e066610da201fee9c21b1b`
- Data source files are the local vendored NExT-GQA test CSV, gsub annotation,
  and video-ID mapping. Their content and source-media bytes are not copied to
  this log.
