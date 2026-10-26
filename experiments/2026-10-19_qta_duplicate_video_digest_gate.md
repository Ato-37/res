# QTA duplicate-video digest gate

## Purpose

Close a validator gap in the fixed three-row QTA cohort: distinct review tokens
must not silently refer to the same pinned source-video byte digest. This is an
input-integrity check, not a faithfulness result or authorization to run scoring.

## Change and verification

The baseline candidate builder rejects duplicate `source_video_sha256` values
with `baseline_duplicate_source_video_sha256`. The phase-1 source-pin validator
and baseline-manifest validator reject the same condition with their respective
phase-specific errors. Tests cover each entry point, including a baseline row
whose nested score-binding digest is recomputed after mutation so rejection is
not caused merely by a stale internal hash. Synthetic fixtures now use distinct
video digests.

Verified on 2026-10-19:

- Focused five-suite test run: 71/71 passed.
- Broader QTA regression run: 211/211 passed.
- `py_compile` for the modified Python modules/tests: passed.
- `git diff --check` for those files: passed.
- An independent read-only audit confirmed the three duplicate-digest checks
  and that the tests exercise the intended paths.

## Interpretation and limits

This closes only duplicate *byte digest* reuse within the fixed cohort. It does
not establish unique semantic video IDs, catch re-encoded copies, authenticate
the origin of media bytes, or maintain a ledger of videos used in earlier
cohorts. These validators remain non-authorizing. Review evidence, independent
prediction-link adjudication, full production orchestration, and the upstream
trust root are still missing. Direct NExT-GQA faithfulness remains unmeasured;
this change contributes no direct score, effect estimate, or GPU result.

## Next action

Design a fail-closed prior-use ledger keyed by a custodian-controlled stable
video identifier, with the byte digest retained as a separate integrity field.
Audit its provenance and replay behavior before considering any cohort reusable.
