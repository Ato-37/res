# Stage-1 pins to baseline v3 and prediction-link composition

## Purpose

Close the fixture-splicing gap between the synthetic Stage-1 source-pin adapter,
the baseline P/Q candidate, and the prediction-link structural validator. The
composition must derive the question, choices, claim, source video digest,
reviewed-interval digest, support decision, and token order from one invocation
of the existing Stage-1 adapter and its matching QA inputs.

## Setup

- Added `scripts/test_qta_stage1_baseline_prediction_link_composition.py`.
- The test instantiates `QtaStage1SourcePinAdapterTests`, calls its actual
  `invoke()` on its all-synthetic Stage-1 and QA fixture, and independently calls
  the same `bind_qta_actual_qa_source()` on those exact QA input bytes.
- The baseline candidate rows use the resulting ordered QA questions/choices,
  claim-manifest text, and Stage-1 pin fields. No alternate baseline or Stage-2
  source-row fixture is joined into this test.
- Baseline log probabilities are synthetic placeholders
  `[-1.5, -1.6, -1.7, -1.8, -1.9]` for every row. A local deterministic fake
  tokenizer and synthetic processor/input IDs are used; no model, video, or GPU
  is loaded.
- The prediction-link receipt is generated from the exact baseline manifest
  rows using `displayed_fields_sha256()`. Its judgments and rationale are
  self-reported synthetic fixture values, and the receipt remains
  `prediction_link_review_recorded_unvalidated` with execution authorization
  denied.

## Verified results

- The new composition module passes **2/2 tests**: the three ordered records
  reach `validate_prediction_link_receipt()` with structural status
  `prediction_link_review_structurally_valid_unvalidated`, and stale
  `reviewed_intervals_sha256` or `source_video_sha256` pins are each rejected
  with `baseline_video_review_binding_mismatch`.
- Focused regression command covering the composition, Stage-1 source pins,
  baseline candidate, and prediction-link validator passes **47/47 tests**.
- Successful composition reports 3/3 rows and 3/3 synthetic support
  judgments. Identity and semantic judgment remain unauthenticated; model load
  is false, score forwards are zero, and execution authorization is
  `not_granted`.
- A first smoke attempt used token ID 99 for the fake video token, colliding
  with the character `c` in the prompt word `Video`. The fixture was corrected
  to use a dedicated synthetic token ID (32000), after which the exact
  single-placeholder expansion check passed.

## Interpretation and limitations

This establishes a CPU-only structural byte/field join from one synthetic
Stage-1 adapter output and its corresponding QA inputs through baseline v3 and
the prediction-link validator. It removes the particular risk of combining
unrelated hand-authored source pins and baseline rows in this tested path.

It does not demonstrate a completed human review, authenticate reviewer
identity or semantic judgment, validate player-clock/frame correspondence,
authenticate model/processor outputs, authorize scoring, or test model
faithfulness. The placeholder scores are not model outputs, and the synthetic
support judgments are not evidence that P is semantically supported by a real
video. This is not a real-data experiment or a faithfulness efficacy result.

Direct faithfulness status remains unchanged: **0/3** formal claim reviews,
**0/15** player-clock/decoder-frame correspondences, **0/3** Stage-2
prediction-link reviews, **0/3** direct scores, and no measured real-data
faithfulness effect or uncertainty interval.

## Next action

Use this single-source composition as the baseline/prediction-review synthetic
handoff. Separately extend the same-source join through the Stage-2 plan
candidate only when its phase-1 preflight batch bindings can be generated from
these same three tokens and baseline rows. Keep all synthetic receipt and
score values explicitly non-authorizing; real-data scoring remains blocked on
the independent review and provenance gates.
