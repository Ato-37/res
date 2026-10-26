# PDCA checkpoint: internal-score shift vs grounding outcome (2026-10-21)

## Purpose

Turn the existing real-video internal-steering results into a thesis-ready,
properly scoped result, revalidate its local receipt, and check whether a new
nonduplicate GPU endpoint is available. This checkpoint added no GPU forwards.

## Independently verified existing result

The current-source receipt
`data/private_nextgqa/interval_generation_cohort30_logits_full_20261021_01.json`
was read locally and passed the current runner's `validate_receipt` against the
current frozen packet SHA-256. The receipt SHA-256 is
`807c77719878165ac95832a6e1fa44d260b55084d6360876452f12dc7c28fb41`; file
permissions are `0600`. Its three records and 9/9 condition generations passed
schema, denominator, tIoU arithmetic, and compact-logit checks. Recomputing
from the receipt gave:

| Video ordinal | tIoU: baseline / selected bias / hash sham | Mean selected−baseline token log-probability | Mean selected−sham token log-probability |
|---:|---:|---:|---:|
| 0 | 0.18631 / 0.18631 / 0.18631 | +0.00583 nat | −0.00115 nat |
| 1 | 0 / 0 / 0 | +0.01054 nat | −0.00003 nat |
| 2 | 0.06269 / 0.06269 / 0.06269 | +0.02475 nat | +0.02375 nat |

All three conditions emitted the same constrained interval code on each video.
The aggregate mean tIoU is `0.08300` under every condition, and all paired
tIoU differences are exactly zero. The token-level score changes are a
post-hoc mechanism diagnostic nested within three videos; they are not an
independent sample of 18 and have no inferential test or interval. The
intervention changed its audited attention-mask values, but this receipt alone
does not prove a useful change in explanation evidence use.

## Research interpretation

This is a concrete thesis result about a dissociation: in this fixed pilot,
internal temporal-key steering moved conditional token confidence while
leaving the generated QA interval and benchmark tIoU unchanged. The endpoint
uses NExT-GQA question-answer intervals, not independently validated support
spans for the natural-language explanation. Therefore it is an auxiliary
grounding/behavior result and **not direct explanation faithfulness** or an
efficacy claim. It supports a thesis about the limits of interpreting internal
score movement as evidence-grounding improvement, alongside the mixed
attention-bias, QA-accuracy, and matched-sham results documented in
`docs/thesis_topic_faithfulness_ja.md`.

## Direct-faithfulness state and review handoff

- Direct generated-claim scores: **0/3** in the current claim cohort; the wider
  formal inventory remains **0/25** claim-support decisions.
- Verified player-clock/decoder-frame pairs: **0/15**.
- Stage-2 prediction-link decisions: **0/3**.
- No direct `DeltaF`, uncertainty interval, or test exists.
- The current support-frame handoff is
  `data/private_nextgqa/claim_support_frame_review_test_20261021_06.html`
  (owner-only `0600`). No response for this artifact was found in Downloads;
  its two JSON files are older, differently bound exploratory artifacts and
  do not count as these reviews.

## Runtime and next action

The individual read-only GPU check found RTX 5070 Ti at 0% utilization,
15,471 MiB free of 16,303 MiB; `.venv` PyTorch 2.11.0+cu128, CUDA 12.8,
CUDA available, BF16 supported; WSL `MemAvailable` 14,564,072 KiB. Hardware
is available. However, the similar QA-interval, temporal-order, frame-mean,
and fixed-prediction proxy branches have already been run and closed or show
unstable results. Repeating them would not resolve semantic claim support.
The formal real-video scoring route still requires authentic reviewer/release
inputs, the Stage-1 claim and 15 frame-time checks, frozen source/prompt/
processor bindings, and a production pre-load transaction. Historical QTA
capture-based scoring also included a runner that ignored false authorization
flags; its results remain non-authorizing and the current scoring entrypoint
fails closed. No missing authorization is inferred from spare GPU capacity.

Continue offering the current local review artifact and inspect an actual
response against its exact fingerprint when one exists. In parallel, advance
only a new pre-outcome design that distinguishes claim-support dependence from
generic confidence change and matched-sham drift. Do not reopen closed proxy
cohorts or call the present result a faithfulness improvement. Dataset/model
rights remain unverified; private video, raw responses, and detailed receipts
remain local.

## Direct-scoring readiness re-audit

An independent source audit and primary rerun confirmed that no production
atomic pre-load transaction currently connects the human/trust/timing checks
to model loading and the three-row baseline scorer. The components remain
non-authorizing by design: `qta_prebaseline_transaction_preflight.py` returns
`structural_bindings_consistent_unvalidated`, `model_load_permitted=false`,
and zero authorized forwards; `qta_direct_faithfulness_preload_gate.py` also
returns denial; `nextgqa_baseline_only_scoring.py` requires a caller-owned
model and is explicitly diagnostic-only. The atomic batch freezer is a
structural primitive with caller-provided callbacks, not a production
authority boundary. A mutable batch could also change after preflight and
before a future model call; a production path must bind/copy the exact final
inputs and check them at the scoring boundary.

Primary rerun:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_qta_prebaseline_transaction_preflight \
  test_qta_prebaseline_live_batch_binding \
  test_qta_direct_faithfulness_preload_gate \
  test_nextgqa_atomic_batch_freeze -v
```

Result: **103/103 passed** in 42.94 seconds. The tests exercise synthetic
fixtures and fail-closed structure (including reorder, byte/tensor mutation,
missing clock/review inputs, and denial status). They do not establish a real
reviewer's identity, semantic correctness, out-of-band root origin, or actual
claim faithfulness. No source code or GPU workload changed in this audit.

The concrete engineering gap is a production runner that internally revalidates
the complete signed review/timing/protocol/code evidence before model load,
then runs only the exact three baseline forwards and freezes the resulting
prediction manifest. It must stay denied without authentic external trust-root
and reviewer inputs. The current local intake still does not contain those
inputs, so this test pass improves readiness evidence but does not unblock
scoring.
