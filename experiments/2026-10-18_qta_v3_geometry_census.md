# QTA v3 outcome-blind geometry feasibility census

## Purpose

The v2 `single_overlap_anchor` CPU preflight could not form an evidence/control
pair for every frozen evaluation video. Before any model outputs, v3 exhaustively
checked all interval-valid test QIDs in the same 147-video evaluation partition
and selected no QID by outcome. The v3 plan was allowed only if every video had
at least one candidate with exactly one evidence anchor and a disjoint control
outside all annotation-covered groups.

## Setup and provenance

- CPU-only operation in the existing `.venv`; no model weights loaded, CUDA call,
  or forward pass. Local Qwen processor/tokenizer geometry was used after decoding
  media for the fixed 147-video cohort.
- One-video smoke passed before the full census: 1/1 video, 5/5 valid QIDs
  examined, 5 geometry-eligible.
- Full census source hash at execution: `7139e29030f65ec205f530c44d479a5ce848b079600a6d1b7a6426ade247336c`.
  Exact executed bytes are retained in the local-only source snapshot
  `scripts/run_nextgqa_qta_v3_geometry_census_executed_20261018_02.py`.
- Private census: `data/private_nextgqa/nextgqa_tlvs_qta_v3_geometry_census_local_20261018_02.json`
  (mode 0600). No v3 plan was written. The output remains local-only; rights
  status is unverified.

## Verified results

- Completed videos: 147/147.
- Interval-valid QID candidates attempted/completed: 702/702.
- Geometry-eligible QIDs: 685.
- Videos with at least one eligible QID: 144/147; three videos had none.
- Candidate-level ineligibility codes: 13 `annotation_anchor_control_unavailable`
  and 4 `processor_implied_grid_changes_group_assignment`.
- All-or-none selection therefore failed and no v3 selected plan was produced.
  The full cohort was not reduced, and no model score was collected.
- The private census file is mode 0600. At run completion, all captured data
  hashes and the executed script hash matched their exact bytes. The remaining
  source hashes were verified against current bytes; the executed script bytes
  are independently preserved by the snapshot noted above.
- CPU test suite after the maintenance-only output collision/stdout repair:
  33/33 focused tests passed; Python compilation and `git diff --check` passed.

## Interpretation and limits

This is an outcome-blind processor-geometry feasibility result for a QID
selection-conditioned exploratory proxy. The 144 feasible videos do not form
an authorized or analyzed subset; dropping the other three would change the
frozen denominator and was not done. The result is not a model result,
annotation-semantic validation, direct faithfulness evidence, clean/fresh
evaluation, or confirmatory evidence. The v3 scoring branch is no-go under the
current one-anchor estimand. Known prior-exposure indices are incomplete, so
unmatched rows are not certified fresh.

## Next action

Keep v2/v3 plans immutable and do not score this cohort. Independently audit the
private census aggregate and provenance. Only after that audit, decide whether
to freeze a distinct alternative mask estimand that can be evaluated for all
147 videos without changing membership; do not select that estimand using model
outcomes.
