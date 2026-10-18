# QTA stage-1 source inventory closure (2026-10-18)

## Purpose

Ensure the local pre-review candidate byte inventory includes the implementation whose output is consumed by QTA stage-1 source pinning, plus the adapter's direct local dependency. This is source provenance preparation only; it neither creates a review freeze nor allows scoring.

## Finding and repair

An independent audit found that the pre-review inventory's `CODE_FILES` omitted `qta_stage1_bundle_audit.py` and `qta_stage1_source_pin_adapter.py`. The adapter's local dependency `qta_prescore_mapping_binding.py` was also missing. Since static import closure starts from registered code roles, it could not detect these omissions, even though the source-pin adapter reruns the stage-1 audit. The inventory could therefore report zero missing slots while leaving this current stage-1 executable closure unbound.

Added these three source roles to `CODE_FILES` and to the inventory regression's required role set. The test also explicitly requires the resulting import-closure check to remain complete. The existing import closure then verifies the adapters' transitive local imports. Test modules were not registered as production code: they import synthetic fixtures and are verified by test execution instead. Baseline candidate/token-binding modules belong to the later score-stage freeze, not this review-only inventory; they must be bound before any such scoring phase.

## Verification and observed inventory

- Independent source audit confirmed the three omissions and reviewed the transitive imports without opening human-review material.
- Inventory-builder synthetic suite: **20/20 passed**; `py_compile` and `git diff --check` passed.
- Rebuilt an actual local candidate inventory against the current package and staged-media paths. It contains **80 artifact roles, 0 missing slots**, including the current hashes of all three added stage-1 sources. A separate local check independently matched their role hashes and file sizes.
- Candidate directory/file modes are **0700/0600**. The inventory remains `candidate_only`, `joint_freeze=false`, `review_completed=false`, `score_ready=false`, `execution_authorization=not_granted`, and `rights_status=unverified_local_only`. It has no freeze-owner signature, trusted root, release witness, or human review.
- The earlier 77-role candidate is stale for this source closure and must not be used for a future freeze. The new candidate path is `data/private_nextgqa/pre_review_inventory_candidate_20261018_1208_stage1closure_01/`; its bytes remain local.

## Interpretation and limits

This closes a mechanical source inventory omission for stage-1 audit and source pinning. It does not prove that a reviewer watched or correctly interpreted a video, validate browser/player-to-decoder correspondence, bind future baseline-scoring code, or produce a faithfulness outcome. Rights remain unverified. The baseline candidate and text-token-binding code are intentionally outside this review-only candidate; those sources and their dependencies must enter the later score-stage freeze before any baseline or treatment scoring.

## Faithfulness progress

- `直接評価済み`: no change; 0/3 human claim reviews, 0/3 prediction-link judgments, 0/15 verified browser/player-to-decoder correspondences, and 0/3 claim-linked scores.
- `補助診断のみ`: inventory source closure and synthetic tests; no model forward, faithfulness metric, estimate, interval, test, or multiplicity adjustment.
- Human roles and the independent trust-root fingerprint remain outstanding. No protected content was rendered or reviewed in this work.

## Next action

Use the updated inventory builder at the witnessed review-freeze event, with current source/media hashes and actual roster/key material. Do not treat this candidate snapshot as the freeze. Then complete the human review and clock/frame pairing; bind the baseline candidate and tokenizer/processor adapter in the separate score-stage inventory before any score run.

- Full scripts suite after the inventory-closure change: **1,727/1,727 passed** in **51.211 s**; expected negative-path fixture diagnostics and Transformers deprecation warnings were emitted, exit code 0.

Source hashes:

- `scripts/build_nextgqa_pre_review_byte_inventory.py`: `2e5b66207f0424dfc1fb58bd6da9d75bf094292dab628c6ba6eef0369143faed`
- `scripts/test_build_nextgqa_pre_review_byte_inventory.py`: `37d7ae9519446e3ec41f00f5f94d0bd9d8d6907544e85c5e291c6f94effc22fe`
- `data/private_nextgqa/pre_review_inventory_candidate_20261018_1208_stage1closure_01/pre_review_candidate_byte_inventory.json`: SHA-256 recorded locally; do not publish this protected-derived inventory.
