# Current claim-review intake alignment (2026-10-21, local checkpoint)

## Purpose

The Japanese launcher opened the current GPU-generated-claim quick-review artifact, while the JSON inspector/private importer defaulted to an older v14 artifact. Align the default and the Japanese runbook so current schema-v6 responses are checked against their actual wording/version and old responses require explicit artifact selection.

## Changes

- Changed `DEFAULT_ARTIFACT` in `scripts/inspect_nextgqa_quick_review_response.py` to the current generated-claim review target; retained the old v14 path as an explicit legacy constant.
- Added a regression test that invokes the inspector without `--artifact` on a fingerprint-bound v6 fixture and verifies redacted output.
- Updated the Japanese runbook to open the current generated-claim artifact and explain explicit artifact selection for old responses.
- Hardened artifact intake to open the HTML once with `O_NOFOLLOW`, derive item IDs and the embedded fingerprint from the same bounded byte snapshot, and use that same snapshot before private import.

## Verification

- `PYTHONPATH=scripts .venv/bin/python -m unittest test_inspect_nextgqa_quick_review_response -v`: 20/20 passed.
- `PYTHONPATH=scripts .venv/bin/python -m unittest test_import_private_quick_review_response -v`: 14/14 passed.
- After the snapshot hardening, both focused suites passed **36/36**, including symlink rejection and a one-read snapshot regression test.
- `git diff --check`: passed.
- Tests use synthetic response fixtures. No real reviewer response was imported or analyzed in this change; no model or GPU inference ran.

## Interpretation and limitations

The importer now defaults to the current claim artifact and continues to reject a mismatched schema-v6 fingerprint. The artifact IDs and fingerprint are validated against one no-follow snapshot, closing the concurrent-read/symlink edge. An unbound legacy response still requires explicit legacy opt-in and the correct old artifact; it does not become a verified review. This improves workflow alignment but contributes no direct faithfulness evidence and does not satisfy reviewer identity, release-root, claim-review, or frame-timing gates.

## Next action

After a human submits a current v6 response, import it locally, verify fingerprint and completeness, and report only privacy-safe aggregate judgments. Continue independent fail-closed checks of the review-to-preload path; do not run the real-data model forward until upstream human/release/timing evidence and production gate are complete.
