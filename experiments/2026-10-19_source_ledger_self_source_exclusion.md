# Exact source self-exclusion for prior-identity audits (2026-10-19)

## Purpose

A source-attributed exposure audit can compare a split to a ledger that also
contains the exact source used to define that split's current candidate set.
Those matches are real byte/identity intersections, but counting them as prior
exposure confounds a source with itself. Add an optional exact-source
exclusion while preserving the full-ledger checks.

## Change

`audit_nextgqa_source_ledger_split.py` accepts repeatable `--exclude-source`
files. It derives the existing source-file HMAC fingerprint from each exact
byte string and excludes only its unique matching source record from overlap
aggregation. Unlisted or duplicate exclusions fail closed. The full ledger
HMAC, schema, token-list, and aggregate source-union checks still run; the
report marks its aggregation scope and compared/excluded record counts. No
identifiers, source paths, HMAC values, or row-level content enter the report.

## Verification

- Focused source-ledger audit tests: **8/8 passed**, including exact-byte
  exclusion, duplicate/unlisted rejection, CLI file loading, and aggregate-only
  output.
- Related prior-exposure ledger and aggregate-index suites: **23/23 passed**.
- `py_compile` and scoped `git diff --check`: passed.
- The unit fixtures are synthetic. No model/GPU execution or inference is
  part of this change.

## Limits and next action

Excluding an exact source prevents a self-match from being mislabeled as
independent history; it does not authenticate who created the ledger, prove
that the roster is complete, establish media freshness, or validate every
source-row count field. Nonmatches remain unknown. The audit remains
non-authorizing and contributes no direct faithfulness score or effect
estimate. Keep data-specific aggregate results in local-only records. Use the
option only when the exact source bytes are known to be represented by one
ledger fingerprint; do not infer an exclusion from a filename alone.
