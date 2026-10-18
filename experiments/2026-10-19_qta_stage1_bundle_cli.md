# Stage-1 bundle CLI (2026-10-19)

## Purpose

Reduce manual path wiring for the formal Stage-1 review audit by accepting one private bundle directory. This is separate from the informal nickname→question check. The CLI must call the existing audit unchanged and must not authorize review opening, model loading, or scoring.

## Implementation

- Added `scripts/audit_qta_stage1_bundle_cli.py`, a read-only wrapper around `audit_qta_stage1_bundle()`.
- The caller supplies one private root directory and four out-of-band values (lineage SHA-256, custody SHA-256, root DER SHA-256, and ceremony ID). The manifest must bind all 12 raw source roles, the media directory, and the exact artifact role set from the inventory.
- The loader rejects malformed/duplicate-key JSON, path traversal, symlinks, wrong owner/mode, missing/extra artifact roles, oversize files, and invalid external pin formats. It prints the original content-free audit report or a stable failure code.
- Added a Japanese local bundle runbook. No protected bundle or derived video data was created by this implementation.

## Verification

- `PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_stage1_bundle_audit test_audit_qta_stage1_bundle_cli`: **24/24 passed** (14 existing aggregate tests plus ten CLI tests). The synthetic valid fixture matched the existing API report exactly and remained non-authorizing; tampered paths, omitted roles, unsafe root/nested-directory permissions, symlinks, malformed external pins, and an injected permissive result failed closed. Mocked aggregate-size limits covering files and a conservative 3×128 MB media reservation were rejected before the audit call.
- Python bytecode compilation and `git diff --check` passed.
- No real review bundle, GPU inference, or human response was used.

## Interpretation and limits

This removes Python/API argument assembly, not the real work of selecting and auditing the review files. The manifest still maps every inventory role and carries the same protected artifacts. The 1 GB limit is a serialized input-byte bound, not a limit on total Python RSS. To keep the input-byte bound valid if media change between preflight and downstream read, the CLI conservatively reserves the maximum size of all three videos (384 MB), so it may reject a bundle whose actual files would fit under 1 GB. The successful test uses synthetic cryptographic fixtures and establishes CLI parity only. It does not establish real reviewer assignment, root trust, a witnessed handoff, semantic claim review, clock/frame correspondence, calibration, or score readiness. Direct faithfulness scoring remains **0/3**. Rights status remains unverified; the formal bundle and all derived media must stay local.

## Next action

Use the CLI only when a real private Stage-1 bundle and independently verified external pins exist. Continue unrelated CPU-side source/provenance review while that operational dependency is absent. Do not convert the informal nickname responses into formal review evidence.
