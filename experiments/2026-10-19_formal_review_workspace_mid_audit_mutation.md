# Formal review workspace mid-audit source mutation guard (2026-10-19)

## Purpose

Exercise the existing Stage-1 workspace guard against a bundle tree changing
while its audit is in progress. Keep the regression entirely within synthetic
fixtures and verify that the controller clears both stages and remains
non-authorizing.

## Setup

Added one case to `scripts/test_nextgqa_formal_review_workspace.py`. It first
verifies the synthetic Stage-1 and Stage-2 fixtures, then wraps the Stage-1
audit so it appends one byte to the synthetic review source after the audit
returns but before `verify_stage1()` performs its post-audit tree signature
check. No protected bundle, video, human review, model, or GPU is used.

## Verified result

Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_nextgqa_formal_review_workspace -v
```

Result: **7/7 tests passed**. The new case observes
`source_tree_changed_during_audit`, clears previously successful Stage-1 and
Stage-2 reports, and retains `model_load_permitted=false`, `score_ready=false`,
`execution_authorization=not_granted`, and `authorized_forwards=0`.

## Interpretation and limits

This is synthetic controller regression coverage for a persistent mutation
between bundle loading/audit and the final tree-signature check. It does not
prove resistance to an attacker who changes and restores bytes between
signature checks, validate real bundle contents, exercise a real browser, or
establish reviewer identity, semantic judgments, clock/frame correspondence,
score readiness, or direct faithfulness. Formal human review and score gates
remain unchanged.

## Next action

Keep this test as a non-authorizing regression. Before a human uses a real
bundle, the remaining real trust/release, review, clock/frame, and browser-path
requirements must still be satisfied independently.
