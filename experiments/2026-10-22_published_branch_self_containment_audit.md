# Published-branch self-containment audit (2026-10-22, `/pdca 1h` window)

## Purpose
Check whether `origin/internal-ablation-wiring-20261018` reproduces the locally tested state, using a clean
git worktree (no untracked files, no private data). Reproducibility check, not a scientific result.

## Verified findings
- Clean checkout of the remote tip (`9c9745b`): `unittest discover -s scripts` ran 2353 tests, 4 failures + 36 errors.
  Cause: 187 untracked local scripts and 23 locally modified tracked scripts were never published, while published
  tests import them (8 missing modules incl. `run_nextgqa_qta_v4_union_scorer`, `qta_v4_mask_specificity_utils`,
  `fixed_annotation_union_dependence`, `nextgqa_capture_operator_attestation`). The remote history also lacks 4
  scripts present in local `da704d3` (e.g. `run_qta_intervened_interval_generation.py`).
- Local working tree baseline: 3166 tests, 2 failures + 1 error before this audit:
  (a) `test_closed_research_branches` (2 tests): 43 scripts calling `from_pretrained` were not classified
  (one of them is this session's `run_minerva_vista_residual_smoke.py`); (b) `test_run_test_split_internal_bias_
  frame_mean_interaction`: `run_source_hash_mismatch:scripts/generate_nextgqa_claim_review_pilot.py`, i.e. that script was
  edited after the local receipts it is bound to were produced. (b) is pre-existing, not repaired here.
- Secret/path scan of the 187 untracked scripts (2.8 MB): no absolute home/Windows paths, keys or tokens
  (one hit is JS URL parsing).

## Actions
- Classified the 43 loaders in `scripts/test_closed_research_branches.py` (guard now 8/8 pass): 29 scripts whose CLI
  requires or enforces `--user-authorized-local-only` -> USER_AUTHORIZED_LOCAL_DATA_PILOTS; 3 synthetic-data model
  smokes + `smoke_nextgqa_baseline_only_qwen_gpu.py` -> ALLOWED_ENGINEERING_LOADERS; 4 processor-only scripts ->
  PROCESSOR_ONLY_LOADERS; `run_claim_token_key_order_interaction.py` (docstring: GPU smoke/full scoring fail-closed)
  -> GATE_CONTROLLED_LOADERS.
- **ASSUMED-CLEARED (pending user confirmation):** five scripts have no local-only CLI flag and were classified
  USER_AUTHORIZED_LOCAL_DATA_PILOTS on the strength of their docstrings only (local-only, post-selection):
  `run_claim_frame_index_mask_pilot.py`, `run_head_bootstrap_exploration.py`, `run_head_gold_answer_query_exploration.py`,
  `run_head_gold_answer_query_exploration_executed_20261019_v1.py`, `run_head_query_position_exploration.py`.
  The four head-attention scripts sit near the closed head-search branch (AGENTS.md: do not resume head/layer/alpha
  search); they are measurement-only, but the user should confirm the classification or revert it.
- Published the local `scripts/` directory state plus two experiment logs read by tests to the remote branch
  (code only; no docs/data/results). Clean-worktree rerun: 3166 tests, 0 failures, 23 errors, all from absent
  private/untracked inputs (`data/private_nextgqa/*`, `datasets/runtime_source_audit/private/*`); no import errors.

## Limits / next action
- The 23 errors are environment dependencies on non-publishable private data, not code defects; they can only be
  verified on this machine. The local branch is still diverged from the remote and its 190 files remain untracked
  locally (identical to what was published); the user should decide how to reconcile local history.
- Repair or regenerate the receipts bound to `generate_nextgqa_claim_review_pilot.py` (hash drift).
- Faithfulness status: `未実施`/`no_go` unchanged; this audit adds no faithfulness evidence.

## Follow-up: source-hash drift on the test-split internal-bias receipts
- Verified: the local receipts (`test_split_internal_bias_frame_mean_interaction_row{0,1}_20261021_01`) record source
  digests for three scripts that were edited after the runs. Two (`generate_nextgqa_claim_review_pilot.py`,
  `run_saved_claim_frame_permutation_attention_smoke.py`) match committed blobs in git history; their exact executed
  versions are now kept in `scripts/frozen_sources/<stem>__sha256_<first8>.py`.
- `analyze_test_split_internal_bias_frame_mean_interaction.py` still enforces the recorded digest: it accepts the live
  file or an exact frozen copy, nothing else (`_source_matches`; 4 unit tests in `test_frozen_source_provenance.py`
  cover accept, drift, frozen-accept, tampered-frozen-reject). The hash check was not relaxed.
- **Unresolved:** the executed version of `qta_multiquery_video_key_intervention.py` is in no git revision (it was
  edited while untracked). Its recorded digest cannot be reproduced, so
  `test_completed_local_receipts_reconcile` still fails with `run_source_hash_mismatch`. This is a real provenance gap
  for those two test-split receipts, not a test bug; do not cite them as source-verified until re-run or a copy is found.
