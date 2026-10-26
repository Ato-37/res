# Local NExT-GQA prior-exposure split audit (2026-10-19)

## Purpose

Check whether the explicit local historical-source ledger overlaps the
canonical NExT-GQA train/validation/test video-ID sets. This is a provenance
screen for candidate planning, not a freshness certification, scoring result,
or authorization. All inputs and outputs remain local because the source
ledger and rights status are unverified.

## Setup and provenance

- Used the protected, owner-only `prior_exposure_source_ledger_run03.json`
  (46 source records; 31 declared `nextgqa` namespace rows) and its local
  HMAC key. The source-ledger auditor verified the ledger integrity HMAC; it
  explicitly leaves HMAC-key/source binding and historical completeness
  unauthenticated.
- Audited canonical `video_id` sets in the local official `train.csv`,
  `val.csv`, and `test.csv`. No media, QID, answer text, model, or GPU was
  accessed.
- An initial attempt used `prior_exposure_source_ledger_run02.json`, which
  lacks the required integrity-HMAC field and correctly failed closed before
  counting. No result from that attempt was used.
- Reproduced the test split count separately with an inline standard-library
  CSV/HMAC calculation using the documented `nextgqa-exposure-v1` token
  encoding, then compared per-class and union counts to the auditor output.

## Observed results

| Split | Unique video IDs | Matches among explicit historical sources |
|---|---:|---:|
| Train | 3,870 | 0 |
| Validation | 567 | 30 |
| Test | 990 | 212 |

The test-split match is attributable across source classes as follows (source
overlap counts can double-count IDs appearing in multiple records):
`sample-json`: 7 source records / 269 source-ID incidences;
`lineage-json`: 2 / 6; `result-json`: 2 / 2; `lock-json`: 0 / 0.
The independent standard-library recomputation returned 212 unique matched
test IDs and exactly the same per-class counts. The old aggregate-only
`prior_exposure_expanded_known_history_20261018_01.json` returned zero test
matches; its frozen older source set has fewer known identities and must not
be treated as equivalent to the later 46-source ledger.

The exact fixed three-row candidate lineage was then checked without printing
IDs. All 3 video-ID and 3 video/QID matches in the full 46-source ledger came
from the candidate's own `lineage-json` source file, which is explicitly one
of the roster inputs. Excluding that self-source, **0/3** candidate video IDs
and **0/3** video/QID pairs match the other 30 canonical-source records. This
avoids misclassifying a source file's self-inclusion as prior exposure, but
the unmatched rows remain unknown because historical coverage is incomplete
and unauthenticated.

Because candidate dataset identifiers do not agree across all historical
artifacts, a separate exact-media-digest join was necessary. It found all
**3/3** candidate media SHA-256 values in **four** other `lineage-json` source
records: one earlier sentence claim-generation file, one claim-mapping file,
and two diagnostic-target files. Within the earlier claim-generation file,
the same **3/3** media+question pairs are present, but the current
claim-text hashes differ for all three rows. This establishes prior
appearance of the same media/questions in the recorded claim-generation
pipeline, not prior human review or a prior faithfulness score. The fixed set
must be described as an exploratory reused-item cohort, never as a fresh
confirmatory holdout.

## Interpretation and limits

The known-source overlap is nontrivial for the test split: 212/990 video IDs
match the explicit 46-source ledger, including the current candidate lineage
as a self-source. After excluding that one self-source, the current three
video IDs and video/QID HMACs are absent from the other 30 source records, but
all three exact media bytes are present in earlier lineage artifacts and all
three media/question pairs occur in the earlier claim-generation file. The
current claims differ from those earlier claim hashes. Treat the three items
as previously exposed to the candidate-generation process and unsuitable as
a fresh confirmation cohort; the ledger does not show that they were
previously scored or formally reviewed. The 778 unmatched test IDs remain
unknown, not certified fresh. The audit does not detect unlisted history,
incomplete rosters, or source-authenticity failures. These counts do not alter
the direct-faithfulness status: claim review 0/3, clock/frame correspondence
0/15, prediction-link review 0/3, and direct score 0/3; no direct `DeltaF` or
uncertainty estimate exists.

## Verification and next action

The source-attributed auditor completed for all three splits with
`freshness_conclusion=not_made`, `faithfulness_scored=false`, and
`execution_authorization=not_granted`. The test result was independently
recomputed from the CSV and HMAC token definition; exact media, question, and
claim-hash joins were recomputed from private lineage records without
emitting identifiers. A reusable `--exclude-source` option was then added to
the source-attributed auditor. It recomputes the same source-file HMAC
fingerprint from exact caller-supplied bytes, rejects duplicate or unlisted
exclusions, checks the full ledger's HMAC, schema, token lists, and source
unions, then excludes only the matched source from overlap counts. This helper
does not validate every row-count metadata invariant. The test split count falls from 212 to
209 when its own candidate-lineage source is excluded (990 split IDs;
30/31 canonical sources compared). The focused suite passes **7/7** tests;
the focused suite now passes **8/8**, including the CLI exact-file exclusion
and aggregate-only output. `py_compile` and scoped `git diff --check` pass.
Exact-source CLI execution returned the expected 209/990 aggregate and no
row-level data.

## Source-ledger metadata validation follow-up (2026-10-19)

The audit helper was subsequently hardened to validate source ordinals,
available row-count bounds and cross-field consistency, identity-token
cardinality, unresolved-reason counts, and source-family/status compatibility.
An independent synthetic family audit caught and helped repair a valid
parseable `opaque-legacy` case: its total row count is known while identity
counts remain null. The repaired code accepts that form and also accepts the
unparseable opaque form with unknown total rows. The combined relevant suite
passes **30/30**; the test-split matching counts above are unchanged when
recomputed. No row-level output was emitted. The ledger format does not retain
`rows_without_any_identity`, so that omitted count cannot be validated from a
ledger alone. This update remains local with the protected audit record.

GPU/model use: none. Next, retain the fixed three items only as an exploratory
reused-item pilot. Any confirmatory candidate set must be sampled and frozen
under a new outcome-blind rule before model outputs or review judgments are
examined; exclude known media digests where available, but do not call
nonmatches fresh while source-history completeness remains unauthenticated.
