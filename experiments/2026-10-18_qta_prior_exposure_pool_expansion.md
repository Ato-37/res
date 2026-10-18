# Expanded historical-exposure reconciliation for the local candidate pool (2026-10-18)

## Purpose

Check whether a broader, explicitly enumerated set of local historical run locks
and model-result records adds exact-byte prior-exposure matches to the 490-video
candidate pool. This is outcome-blind data-integrity work for cohort selection;
it is not a faithfulness experiment. Inputs, keyed identities, and derived
indices remain private and local.

## Method

Hardened the explicit-input HMAC manifest builder to use bounded, no-follow,
stable file-descriptor reads; reject symlink components and non-regular files;
avoid FIFO blocking; and cap source count, per-source bytes, and total retained
source bytes. Added an aggregate-only pool reconciler that opens the pool root
without following symlinks, snapshots the directory tree, streams MP4 bytes
through stable descriptors, compares exact media-byte HMACs against one or
more caller-supplied local exposure indices, and verifies that the tree did
not change. It emits no paths, identifiers, digests, or per-file matches.
Expected pool-count mismatch exits nonzero. The reconciler caps media files at
10,000, each file at 512 MiB, total media bytes at 8 GiB, and combined index
bytes at 256 MiB.

The expanded private index used 19 explicit source files: six run locks and
thirteen prior model-result records. All 19 parsed. Across those sources,
136 rows were represented; 42 rows carried usable media identities, yielding
31 distinct media identities. Ninety-four rows had no usable identity in the
builder's supported fields. The index declares historical completeness
unauthenticated. The existing partial local exposure index was unioned with
this expanded index using the same local HMAC key.

## Observed results

- Candidate pool: **490/490** MP4 files; **1,990,584,365 bytes**; all 490
  media files had distinct byte hashes.
- Combined prior indices: **139** distinct media-identity HMACs.
- Exact-byte matches: **21/490**; unmatched to the supplied indices:
  **469/490**.
- The prior partial index alone had **11** exact-byte matches in the same pool;
  adding the explicitly enumerated local locks/results added **10** matches.
- An independent second implementation rehashed all 490 files and recomputed
  the keyed media tokens; it reproduced **490 unique / 21 matches / 469
  unmatched**.
- Focused synthetic security tests passed **27/27**. The complete `scripts/`
  suite passed **1,738/1,738** in **50.862 s** after the final implementation
  changes. Expected negative-path fixture messages and Transformers deprecation
  warnings appeared; exit code was 0.
- An independent auditor reviewed the final source/tests without reading any
  private inputs or running the real pool. It verified the no-follow root and
  child traversal, snapshot-to-open file identity binding, stable read checks,
  pre-read per-file/aggregate byte caps, aggregate-only output, and nonzero CLI
  exit on expected-count mismatch. Its synthetic run passed **27/27**.

## Interpretation and limits

This adds ten exact-byte exclusions against the enumerated run/result sources.
It does not show that the remaining 469 videos are unexposed: historical
coverage is incomplete and unauthenticated, and 94 rows in the newly processed
records had no usable identity. Exact byte matching misses re-encoded copies,
shared events, overlapping source footage, and related scenes. The lock
records' opaque question hashes were not interpreted as video/question pairs.
No source/event independence or claim-review eligibility was assessed. No
video was decoded or displayed, no model was loaded, no GPU forwards were run,
and no faithfulness score was computed. Rights remain unverified; inputs and
derived indices remain local.

## Faithfulness progress

Direct generated-claim evaluation is unchanged: **0/3** independent claim
reviews, **0/15** verified player-clock/decoder-frame correspondences, **0/3**
prediction-link judgments, **0/3** claim-linked scores, and no empirical
`DeltaF`, confidence interval, hypothesis test, or multiplicity-adjusted
result. This is an auxiliary exposure-coverage diagnostic only.

## Next action

Continue the human review release preparation while obtaining the actual role
assignments and trusted-root fingerprint. Independently prepare source/event
clustering evidence for the 469 unmatched candidates; do not call them clean,
do not use them for confirmatory sampling until exposure/source coverage is
adequate, and do not start direct scoring before the human and phase gates.

## Provenance

- All local exposure indices and exact input locators are owner-only and are
  intentionally not committed.
- No GPU/model/environment dependency applies to the CPU-only reconciliation.
- `scripts/build_nextgqa_prior_exposure_manifest.py` SHA-256:
  `cd48e91ef16fbf2451978501b93237d3417991e7ce6d9fe4a05856769d4efd0e`
- `scripts/test_build_nextgqa_prior_exposure_manifest.py` SHA-256:
  `52c06220827d9ea5fd44bd4d6e7ff709491bd060e97abef3d8d4103e04d78e2d`
- `scripts/reconcile_nextgqa_prior_exposure_pool.py` SHA-256:
  `4a97d683defd303f766a8c68731517a86ead35fa3a5d965c7702aaf74658ddc3`
- `scripts/test_reconcile_nextgqa_prior_exposure_pool.py` SHA-256:
  `3eb19c343ab77dd509d5d450809cf10c664b6cf85a414672e3537ca932229789`
