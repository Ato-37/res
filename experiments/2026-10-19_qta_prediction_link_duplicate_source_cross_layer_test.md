# Prediction-link duplicate-source cross-layer regression (2026-10-19)

## Purpose

Ensure the Stage-2 prediction-link validator rejects a canonical, correctly
rehashable v3 baseline when two otherwise distinct review rows refer to the same
source-video bytes. This exercises the public reviewer-validator error path,
not only the lower-level baseline parser.

## Change and verification

- Added a synthetic regression case to
  `scripts/test_nextgqa_prediction_link_review.py`.
- The case copies the three-row v3 fixture, duplicates one row's
  `score_binding.source_video_sha256`, recomputes that score-binding digest,
  serializes and hashes the complete canonical manifest, and updates the outer
  receipt's expected manifest digest. It asserts rejection as
  `baseline_manifest_score_binding_invalid`.
- `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_nextgqa_prediction_link_review -v`: **12/12 passed**.
- Added a second regression test at the formal-workspace controller seam. It
  duplicates source bytes in one score binding, recomputes the binding and
  manifest hashes plus the externally expected context digest, and confirms
  Stage 2 fails closed before receipt verification or any authorization.
  `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_nextgqa_formal_review_workspace -v`: **8/8 passed**.
- `git diff --check -- scripts/test_nextgqa_prediction_link_review.py`: passed.
- The fixture is synthetic. No protected video, model, GPU, or human response
  was used.

## Interpretation and limits

This closes a cross-layer test-coverage gap for duplicate video identity after
all outer hashes have been recomputed. It verifies error mapping and regression
behavior, not the provenance or semantics of real review material. The earlier
strict baseline parser already rejects duplicate source-video hashes; this test
proves that the public prediction-link validator does not accidentally accept
that malformed cohort. The workspace-level test also verifies the current
integrated controller with a self-consistent external context pin; both tests
use fabricated inputs and do not establish browser interaction coverage.

## Faithfulness progress and next action

Direct faithfulness remains **0/3** formally reviewed claims, **0/15** verified
player-clock/decoder-frame correspondences, **0/3** Stage-2 prediction-link
judgments, and **0/3** direct scores. No direct effect estimate, confidence
interval, or hypothesis test exists. The exploratory nickname check has no
response JSON in the local private directory. The next direct-evidence step
still requires valid human-review material and upstream receipts; meanwhile,
run the focused baseline/phase-freeze/prediction-link integration suites and
continue the independent CPU-side gate audit. No real-video GPU score run was
launched because its preregistered input gates remain unmet.
