# Review envelope to phase-1 composition test repair

Date: 2026-10-21 (JST)

## Purpose

Repair a stale synthetic browser-export harness that failed before the phase-1 composition checks. The review page now emits one combined envelope for successive reviewers; the old helper expected separate review and index downloads. No production renderer or human-review artifact was changed.

## Implementation and verification

Only `scripts/test_audit_qta_stage1_bundle_cli.py` changed. The Node VM harness now extracts and executes the current rendered page's actual reviewer readiness, export, next-reviewer, and envelope-download logic. It checks that an incomplete review leaves export disabled and inert, that a completed review enables export, and that two synthetic reviewer sessions produce one envelope. It decodes both sessions' payload and index members and checks exact member count/roles, byte sizes, SHA-256 values, index-to-payload binding, reviewer IDs, completion status, and `execution_authorization=not_granted`. The first session's exact decoded bytes then feed the existing Stage-1-to-phase-1 source-pin composition test.

- Independent full focused suite before the final assertion-only hardening: `PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_stage1_phase1_composition -v` — 56/56 passed.
- After final assertion hardening, the specifically changed integration test passed 1/1 with the same module import path.
- After publication, the full owning helper module passed: `PYTHONPATH=scripts .venv/bin/python -m unittest test_audit_qta_stage1_bundle_cli -v` — 12/12 passed.
- `git diff --check -- scripts/test_audit_qta_stage1_bundle_cli.py` passed.

## Scope and limitations

This executes the rendered JavaScript handlers in Node's VM with synthetic controls and capture functions. It is not a real-browser E2E and it creates no human evidence, trusted identity, semantic validation, execution authorization, model forward, or faithfulness score. The current direct human-review counts and model baseline remain unchanged; the runner remains non-authorizing until the separate evidence gates and production pre-load transaction are complete.

## Next action

Keep this harness as a software integration check. Continue the human/release evidence handoff and independently review the production load boundary. Do not treat this synthetic pass as authorization to run real-video inference.
