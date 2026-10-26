# NExT-GQA test-split annotation-key-bias replication (exploratory)

## Question and status

This is a prospective, outcome-unseen **split-level replication** of the
existing NExT-GQA validation annotation-key-bias-versus-equal-size-hash-sham
answer-margin endpoint. The same three-condition runner, model, frame count,
key intervention, fixed foil rule, sham algorithm, and aggregate analysis are
used on a new test-split cohort. This is deliberately post-result with respect
to the validation cohorts and is exploratory; it is neither a new endpoint nor
a confirmatory population study. The two validation cohorts and test cohort
will be reported separately, never pooled.

The estimand per row remains
`(gold - fixed baseline-best-foil margin under annotation-key bias) - (same
margin under equal-size disjoint hash-sham bias)`. The intervals are NExT-GQA
QA-time annotations. They are not claim-specific explanation support, their
clock correspondence has not been independently calibrated, and human review
is not performed. The outcome is an internal answer-margin sensitivity proxy,
not explanation faithfulness or grounding accuracy.

## Frozen cohort and execution rules

- Before any test-split scores are observed, freeze exactly **20** eligible
  distinct test videos using the runner's deterministic outcome-blind cohort
  ordering and 32 decoded frames. The only exclusions are a private union of
  the latest local raw-media exposure union, the latest claim-exposure union,
  both completed validation annotation-key cohorts, and any locally recorded
  split-bound test cohort locks. Source history remains heuristic and
  incomplete; exclusions do not prove freshness outside those sources.
- If fewer than 20 rows satisfy the runner's fixed decode/timing/interval and
  disjoint-sham geometry criteria, stop this cohort before model loading. Do
  not reduce N, replace rows after viewing outcomes, or substitute another
  split. Preserve the candidate/preflight attrition counts.
- Run the exact existing runner in `preflight`, then one-row/3-forward GPU
  `smoke`, then the exact 20-row/60-forward `full` only after all lock,
  source, model, package, preflight and smoke bindings pass. Use pinned
  Qwen2.5-VL-7B-Instruct, 4-bit weights/BF16 compute, 32 frames, layers 5-16,
  requested `+ln(2)` attention bias, fixed baseline-best foil, and the existing
  disjoint hash sham. No metric, dose, layer, frame count, cohort, or exclusion
  changes after scoring begins.
- Report requested/effective rows, failures, forwards, GPU/VRAM, per-cohort
  mean/median/range/signs and 5-choice accuracy. A finite-cohort t interval
  may be shown descriptively, labeled as such; no hypothesis test, pooled
  estimate, or population claim is allowed. Preserve every receipt owner-only
  and keep protected data local; rights remain unverified.

## Decision rationale and interpretation boundary

The first validation cohort's annotation-bias-minus-sham mean was +0.0125 nat
and the second's was +0.0250 nat; both had a near-zero median and mixed row
directions, with no accuracy gain. A fixed test split can reveal whether the
same small answer-margin pattern recurs out of split. It cannot validate claim
truth, semantic support, clock calibration, or explanation faithfulness.
Negative and near-zero results are valid. This branch does not reopen the
closed head/layer/alpha search or any generated-claim proxy.

## Outcome-blind feasibility result and branch decision

The private exclusion manifest was built from six byte-verified local inputs:
the 1,034-hash known-media union, the 117-hash latest claim-exposure union,
and four split-bound validation/test locks (80 lock assignments, including
byte-identical duplicate lock records). The manifest contains 1,074 unique
media hashes, mode `0600`, SHA-256
`d9c795439a261c2b76aad76084884be269a4027dc2518437ab4f481ec90b529c`.
Historical coverage remains heuristic and incomplete.

The fixed-N=20 test-split preflight failed closed before creating a lock or
loading model weights: `requested_20_eligible_samples_but_found_0`. An
independent CPU-only candidate census found 85 test video groups in the local
archive, all 85 had local video files, and **all 85 matched the exclusion
union**; zero unexcluded local candidates remain. This is not a zero-outcome
experiment and no test-split model score was collected. Do not drop exclusions,
reduce N, or treat exposed videos as fresh to obtain a run. Preserve this
branch as unavailable under the recorded local history.

GPU execution: **0 forwards**. Rights remain unverified. All private material
remains local. The test-split replication is not feasible with the current
local video inventory and defensible exclusion set.

## Next action

Consider only a separately labeled, fixed-design sequential continuation on
still-unexposed validation rows if an outcome-blind preflight establishes a
complete new cohort; keep it exploratory and report separately. Otherwise
continue non-inference work toward the reviewed claim-support measurement.
