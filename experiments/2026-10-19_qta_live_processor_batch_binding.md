# QTA live processor-batch binding diagnostic — 2026-10-19

## Purpose

Connect the source/review ordered join to the actual in-process processor
`FrozenBatch` tensors used by the baseline scorer. The check must fail closed,
validate all three rows atomically, and remain non-authorizing. This is a
provenance/readiness check, not a faithfulness experiment.

## Implementation and setup

- Added `scripts/qta_prebaseline_live_batch_binding.py`. It accepts exactly
  three ordered expected digest rows, live `FrozenBatch` rows, and a source
  crosswalk. It checks opaque review-token/source order, duplicate identities,
  CPU-only supported finite tensor contents, and recomputes each live batch
  digest with the scorer's `batch_sha256` implementation.
- Integrated this leaf into
  `scripts/qta_actual_prebaseline_diagnostic.py`. The diagnostic constructs
  the token/video crosswalk only after its source and review byte validators
  pass. It reports aggregate gate status only; it does not expose tokens or
  hashes, load a model, call the scorer, or grant authorization.
- Expected processor digests remain caller-supplied and unauthenticated. A
  matching digest proves only consistency with that supplied pin, not that the
  batch arose from the claimed source frames, sampler, reviewed interval, or
  pinned processor.

## Checks and observed results

Command, run on CPU with synthetic fixtures only:

```bash
cd scripts && ../.venv/bin/python -m unittest -v \
  test_qta_prebaseline_live_batch_binding \
  test_qta_actual_prebaseline_diagnostic \
  test_qta_prescore_processor_row_adapter \
  test_nextgqa_baseline_only_scoring
```

Observed: **39/39 tests passed**. New integration tests verify 3/3 ordered
success; atomic rejection of reordered rows, changed tensor bytes, duplicate
source/token, non-finite tensor content, and upstream review mismatch; and no
token/digest leakage in the public diagnostic. They also verify that a passing
processor gate leaves model loading, scoring, and faithfulness flags false.
Existing processor-row adapter and baseline scorer tests passed unchanged.

## Interpretation and limitations

This closes a narrow diagnostic gap: the check now recomputes hashes from the
live in-process CPU batches and verifies their ordered labels against the
source/review join. It does not authenticate the expected digests, guarantee
the checked object is the object later scored, or prove source-frame-to-sampler-
to-processor provenance. The baseline scorer independently rehashes its inputs,
but a production caller must preserve and pass the same fixed digest/object
through a validated pre-load transaction. No model was loaded, no GPU work was
run, and no protected video was read in these tests.

## Faithfulness progress and next action

Direct faithfulness remains **0/3** reviewed claims, **0/15** player-clock /
decoder correspondences, **0/3** prediction-link judgments, and **0/3** direct
scores. There is no direct `ΔF`, interval, or statistical test. The informal
nickname-based check remains an exploratory impression tool and does not fill
these counts. The signed trust/release and human semantic-review gates are
still absent, so a real-video score run is not justified. Rights status remains
unverified and all protected material must remain local.

Next, design a typed, independently pinned processor-batch receipt and a
single-use immutable handoff that makes preflight and scoring consume the same
three batch snapshots. Keep it non-authorizing until trust, release, review,
clock/frame, target, and baseline `p/q` gates independently pass. Do not repeat
closed proxy runs or launch GPU work to exercise this CPU-only validation seam.
