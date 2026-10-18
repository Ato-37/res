# QTA stage-1 bundle audit hardening (2026-10-18)

## Purpose

Harden the local, non-authorizing stage-1 structural audit against oversized direct API inputs and concurrent changes to the private staged-media directory. This supports reproducible review handoff only; it does not satisfy human review or authorize model execution.

## Findings and changes

A synthetic race test patched the release-order verifier to add an extra valid-looking `.mp4` file after the stage-1 media snapshot. Before the fix, `audit_qta_stage1_bundle()` returned `stage1_structurally_valid_non_authorizing` and `media_snapshot_stability=unchanged`. This was a false pass for the directory's exact-three-media invariant.

An independent read-only design audit found that the shared 128 MiB input limit applied to file paths and staged media, but not to raw byte arguments supplied directly to the stage-1 API. `_json()` could decode and parse oversized bytes before any size rejection.

The audit now checks all 12 direct raw-byte inputs (five JSON inputs, receipt, two rosters, inventory, two attestations, root PEM) for `bytes` type and a nonempty size no greater than `MAX_PRIVATE_ARTIFACT_BYTES` before parsing, signature validation, or media access. It returns a closed `blocked` report on failure. Media-directory snapshot identity now includes `mtime_ns` and `ctime_ns`; final verification also lists the still-open directory descriptor and requires the exact expected basenames. These checks detect additions/removals during the audit as well as changes to the three media files.

## Verification

- Synthetic test first reproduced the extra-file false pass before the fix.
- Stage-1 bundle audit suite: **14/14 passed** after the fix.
- Stage-1 source-pin adapter suite: **21/21 passed** (includes the stage-1 audit fixture tests).
- `py_compile` and `git diff --check`: passed.
- Input rejection tests patch the parser and media opener to fail if called, demonstrating fail-fast behavior without allocating an oversized payload.
- No private review HTML, video, QA contents, or human judgments were opened. No GPU or model run occurred.

## Interpretation and limits

This is engineering validation of a pre-review structural audit. It does not establish reviewer identity, semantic claim validity, frame/time correspondence, prediction linkage, or explanation faithfulness. All outputs remain local; rights status is unverified. The stage-1 report remains explicitly non-authorizing, with clock/frame and calibration audits separate.

## Faithfulness progress

- `直接評価済み`: no change; reviewed claims 0/3, prediction-link judgments 0/3, browser/player-to-decoder correspondences 0/15, claim-linked QTA scores 0/3.
- `補助診断のみ`: synthetic audit-contract tests, no model or outcome measurement.
- No effect estimate, confidence interval, hypothesis test, or multiplicity correction was produced.
- Human role assignment and independently trusted root fingerprint remain outstanding; rights status remains unverified.

## Next action

Continue independent audits and model-free diagnostics while the reviewer/trust information is pending. Once roles and trust root are supplied, rebuild the candidate inventory from current sources, repeat the witnessed release procedure, complete the human/video review and frame clock correspondences, then run only the prespecified baseline phase.

An independent protocol auditor rechecked both changes separately. Its bounded
stage-1/source-pin synthetic set passed **35/35** and found no false pass or
false reject in the changed cases. It confirmed that the audit remains
non-authorizing and does not imply identity, semantic review, clock/frame, or
calibration validity. The audit API still accepts a caller-supplied media root;
fixed-root containment is outside this helper's contract and would need an
explicit path-policy requirement before implementation.

The complete repository scripts suite passed **1,727/1,727** in **51.127 s**.
The run printed expected negative-path fixture diagnostics and Transformers
video-processor deprecation warnings; exit code was 0.

Final source hashes:

- `scripts/qta_stage1_bundle_audit.py`: `a63a0b3bcc1510cc74a8217a0fd421edf9fe0e07a9ea66ef51bfa9e936c3681c`
- `scripts/test_qta_stage1_bundle_audit.py`: `f4d757879ca752ee04aaa847cea313bb24ac4fa22e8824b82abbc1fe7c5ca38e`
