# Source-ledger count metadata validation (2026-10-19)

## Purpose

The source-attributed prior-identity auditor already checked the ledger HMAC,
schemas, token lists, and identity unions. An independent review identified
that authenticated source rows' count metadata was only weakly validated.
HMAC integrity alone does not establish internal consistency if a buggy or
incorrectly generated ledger is signed with the expected local key.

## Change

The auditor now requires ordered source ordinals, validates that parsed-source
row counts are nonnegative integers bounded by `rows_total`, checks that
identity-specific row counts do not exceed `rows_with_any_identity`, and
checks zero/nonzero consistency and maximum cardinality between identity row
counts and their token lists. It verifies source-family/status compatibility.
Failed parses require null row counts and no identity tokens; opaque legacy
records may retain a known `rows_total` while identity-specific counts remain
null. Unresolved-reason metadata must be a map of bounded nonempty strings to
positive integer counts. These checks do not add freshness certification,
source authenticity, or execution authorization.

## Verification

- Related source-ledger and split-identity suites: **30/30 passed**.
- New signed synthetic cases reject row-count overflow, count/token
  disagreement, family/status mismatch, and source-ordinal mismatch. Both
  parseable and unparseable opaque-legacy records are accepted according to
  their intended nullability contract.
- Python compilation and scoped `git diff --check` passed.
- The verification fixtures are synthetic. No model inference, GPU, or
  publication of protected-ledger results is part of this change.

## Interpretation, limitations, and next action

The stronger checks close a metadata-consistency gap in the aggregate-only
exposure auditor. Independent audit found and helped repair one compatibility
bug: parseable opaque-legacy records keep a total row count while their
identity-specific counts remain unknown. Added regression fixtures cover this
distinction. The checks do not prove that a source roster is historically
complete, authenticate the HMAC key's provenance, identify prior semantic
exposure, or certify an unlisted item as fresh. Next, apply the same explicit
validation discipline to any new source adapter before using its outputs for
candidate planning; preserve the closed confirmation cohort and do not use
these checks as model-score authorization.
