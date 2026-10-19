# Quick-review v14 artifact binding and legacy isolation (2026-10-19)

## Purpose

The v13 export used stable item IDs only. Because those IDs intentionally survive
wording changes, the inspector could accept a schema-v5 response while pointed
at a later artifact with revised questions or rating guidance. This iteration
binds new responses to the exact local HTML artifact and makes historical
unbound imports an explicit opt-in.

## Implementation

- The local builder now emits v14 and schema-v6 response JSON. It computes a
  SHA-256 over the completed HTML with one fingerprint placeholder, writes the
  digest into a meta element, and exports the same value in the response.
- The inspector reconstructs the placeholder form and rejects malformed or
  byte-modified bound artifacts. Schema v6 responses must have an exact
  fingerprint match. Schemas v2-v5 can be inspected only with explicit opt-in; they are reported as
  `legacy_unbound` and never described as version-verified.
- The private importer rejects legacy-unbound responses by default. Historical inspection and
  archival require `--allow-unbound-legacy`; schema-v6 responses remain
  fail-closed on missing, invalid, or mismatched fingerprints.
- Removed a redundant second `__ITEMS__` replacement in the builder. A regression
  fixture confirms that ordinary item text containing that marker is preserved.
- Updated the Japanese review launcher and runbook. The local HTML and all real
  response JSON remain private and are not committed.

## Verified results

- Local v14 artifact exists with mode `0600`, 3 item IDs, and a successful
  self-fingerprint check. The artifact is 30,656,433 bytes; its fingerprint is
  intentionally omitted from this public-safe log because it identifies the
  private instrument version.
- Builder, inspector, and importer CPU suites: **39/39 passed**. Coverage
  includes exact schema-v6 match, HTML tamper rejection, response mismatch
  rejection before private storage, legacy-import opt-in, both CLI default-refusal paths, and sentinel text
  preservation.
- Synthetic Windows Chrome E2E: **1/1 passed** in 2.15 seconds after the final
  builder revision. It exercised three sequential synthetic reviewers, one
  combined schema-v6 download, exact artifact-fingerprint equality, answer
  order, and abstention persistence. The ordinary sandbox invocation could not
  clean up the Windows Chrome profile; the same bounded test passed when run
  with per-command Windows interop access. It used only synthetic questions and
  a generated tiny video.
- The user-provided historical exploratory JSON was inspected locally against
  v14: 3 respondents and 9 judgments validated; it was explicitly reported as
  `legacy_unbound`. Aggregate counts remained clarity 4/4/1, answer fit 5/3/1,
  and evidence sufficiency 2/5/2 (ordered as positive / negative / cannot tell).
  No nickname or free text was displayed, and the raw JSON was not modified or
  copied by this inspection.
- `py_compile` and scoped `git diff --check` passed.

## Interpretation and limitations

The fingerprint prevents accidental cross-version matching and detects
post-build byte changes when the response retains its original digest. It is not
a signature: it does not establish reviewer identity, prove that a person
watched the HTML, or prevent a deliberate editor from changing both HTML and
JSON. The legacy response has no source fingerprint and therefore cannot be
retrospectively bound to v10-v13; its judgments are descriptive exploratory
feedback only.

Formal faithfulness evidence remains **0/3** claim reviews, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3** direct
scores. This UI/provenance repair does not change those counts or provide a
faithfulness effect estimate. Rights status of the local media remains
unverified; no media, response JSON, or fingerprint value is published.

## Next action

Use the v14 local artifact for any fresh informal review. Inspect its schema-v6
export without `--allow-unbound-legacy`; only schema-v6 files whose digest
matches the exact v14 HTML are eligible for normal private import. Do not treat
this exploratory flow as formal annotation or as authorization to start direct
GPU scoring.
