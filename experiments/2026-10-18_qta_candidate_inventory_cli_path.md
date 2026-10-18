# Accept the official pre-review inventory path in QTA QA source binding (2026-10-18)

## Purpose

Allow the non-model QTA QA-source binder to consume the candidate inventory emitted by the current official pre-review builder, so the actual three-row dataset-source binding can be checked without human review or GPU scoring.

## Finding and repair

The inventory builder writes `pre_review_candidate_byte_inventory.json` under a caller-selected private directory. The current candidate uses the canonical `pre_review_inventory_candidate_<suffix>` directory name, but `run_qta_actual_qa_source_binding_smoke.py` accepted only `inventory-<suffix>`. The CLI therefore rejected the new official candidate path before reading it.

Expanded the anchored allowlist to accept both the current builder's `pre_review_inventory_candidate_*` form and legacy `inventory-*`, while retaining the fixed leaf filename, no-follow directory traversal, owner/private-mode checks, file-size limit, exact expected SHA-256 pin, and stable-read checks. Updated the CLI help. Added synthetic tests for acceptance of the canonical form with 3/3 source rows, zero forwards/scores, and all denial flags; existing legacy form still passes. Rejection cases cover empty suffix, traversal through path components, absolute paths, and unrelated paths.

An independent reviewer confirmed the pattern is anchored and does not permit separators in suffixes. The existing `O_NOFOLLOW`, directory ownership/mode, leaf-file ownership/mode/type/size, and read-stability checks still apply. The CLI continues to label the caller-supplied inventory pin unverified, and does not authenticate that an inventory was produced by the official builder.

## Verification so far

- Focused CLI suite: **11/11 passed**.
- `py_compile` and `git diff --check`: passed.
- No GPU, video, HTML, reviewer judgment, or score was used in this code/test subcycle.
- The actual source-binding CLI was run against the fixed local three-row QA cohort after the CLI and scope clarification were committed at local `HEAD` `25ddf9a43c101ef91187f611bd7f2218be74da97`. A freshly rebuilt candidate inventory had 80 roles and 0 required omissions; the caller-pinned inventory digest matched. The aggregate-only receipt reports `source_binding_passed_unvalidated`, 3/3 validated rows, exact protocol/config blob matches, and 0 forwards/0 score calls. `execution_authorized`, `model_load_permitted`, and `scoring_authorized` are all false; inventory and commit pins remain caller-supplied and unverified. The receipt retained no private bundle. The inventory and receipt remain local.
- The CLI run printed only its allow-listed aggregate receipt; no row-level QA/claim text, video, or review material was displayed. No human review or semantic validation was performed.
- The initial fetch in the normal sandbox could not write `.git/FETCH_HEAD`; an individually elevated fetch then succeeded. It showed the remote branch at the prior commit and local `HEAD` one commit ahead. This confirms fetch state only; the newly appended log update still needs its own scoped commit and push verification.

## Limits and faithfulness progress

This change only permits the CLI to read a candidate inventory at the official builder path. It does not authenticate the inventory, trust root, person, review, source annotations, or semantic validity; a successful report remains `source_binding_passed_unvalidated` with model loading, forwards, scoring, and execution authorization denied. It creates no direct faithfulness evidence, estimate, interval, test, or multiplicity-adjusted result. Direct status remains 0/3 claim reviews, 0/3 prediction-link judgments, 0/15 verified timing correspondences, and 0/3 scores. Rights remain unverified and data remain local.

## Next action

Independently audit the aggregate receipt and record the result. Then continue with another useful local-only experiment-preparation task that does not require human review, while preserving the requirement for human claim/prediction-link review before direct faithfulness scoring. Retry scoped Git publication only through an authorized writable Git path; do not claim remote verification from cached refs.

Full scripts unit suite after the path-allowlist change passed **1,728/1,728** in **51.124 s**. Expected negative-path fixture diagnostics and Transformers deprecation warnings were emitted; exit code 0.

Final hashes before the real data-binding step:

- `scripts/run_qta_actual_qa_source_binding_smoke.py`: `262e62c61c018869146332481d987f2e70addd6f9a6af04524f560bab070cf41`
- `scripts/test_run_qta_actual_qa_source_binding_smoke.py`: `9cf0d033fae96a55e044caf5a342546f92fafd93d363c62c8cafe8d2cd5c0466`
