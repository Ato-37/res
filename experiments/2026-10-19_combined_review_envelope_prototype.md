# Combined formal-review envelope prototype

## Purpose

Create a CPU-only handoff layer that can place multiple formal reviewer exports
into one JSON file while keeping each reviewer session separate and preserving
the existing review, integrity-index, and optional clock artifacts byte for
byte. This is a packaging prototype, not a review UI, semantic audit, identity
check, or scoring authorization.

## Implementation

Added `scripts/nextgqa_combined_review_envelope.py` with `build_envelope`,
`parse_envelope`, and `split_envelope` APIs. One to 16 sessions are permitted;
each reviewer retains a distinct sequence and private identifier. The envelope has exact root,
session, and member key sets; duplicate-key/non-finite JSON rejection;
canonical Base64; per-member SHA-256 and size checks; exact ordered session
sequences; unique reviewer private IDs; required review/index roles and an
all-or-none clock/index pair; and validation through the existing
`stage_nextgqa_review_exports._validate_exports` function. Inner JSON bytes are
not decoded and reserialized when split. The splitter accepts only a new
direct child of a current-user-owned mode-0700 private root, writes nested
session directories mode 0700 and files mode 0600, refuses overwrite, and
removes partial outputs on failure.

The envelope is strictly non-authorizing (`execution_authorization` is fixed
to `not_granted`). It records declared reviewer IDs and byte hashes, but does
not authenticate identity, independence, actual viewing, or semantic review.
The 2 MiB per-member, 16-session, and 64 MiB envelope limits are implementation
caps, not study design limits.

## Compatibility finding

Source inspection found that the current claim renderer emits clock export
`schema_version: 3`, while the staging validator initially accepted only
version 2. The shared staging validator was updated during this task to accept
versions 2 and 3 while retaining the original bytes. The envelope delegates to
that validator and its synthetic regression test verifies that clock-v3 bytes
survive packaging and splitting unchanged. No payload is normalized.

## Verification

Synthetic-only tests in `scripts/test_nextgqa_combined_review_envelope.py`
cover byte-exact round trips, reviewer order and separation, optional clock
pair handling, duplicate reviewer IDs, duplicate JSON keys, non-finite JSON,
exact schemas, out-of-order sequences, canonical Base64, hashes, sizes,
inner non-authorizing declarations, private directory/file modes, refusal to
overwrite, output confinement, and cleanup after a simulated write failure.
No protected review input was read.

Commands run:

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_nextgqa_combined_review_envelope.py' -v
.venv/bin/python -m unittest discover -s scripts -p 'test_stage_nextgqa_review_exports.py' -v
.venv/bin/python -m py_compile scripts/nextgqa_combined_review_envelope.py scripts/test_nextgqa_combined_review_envelope.py
if rg -n '[[:blank:]]+$' scripts/nextgqa_combined_review_envelope.py scripts/test_nextgqa_combined_review_envelope.py experiments/2026-10-19_combined_review_envelope_prototype.md; then exit 1; else echo 'whitespace check passed'; fi
```

The envelope suite passes 22/22 tests; renderer unit tests pass 10/10 after
the sequential-export integration; staging tests pass 11/11; and the quick
review builder/importer/inspector suites pass 28/28. A combined rerun of these
six suites passed **71/71**. Python compilation, fresh rendered-JavaScript
`node --check`, and scoped `git diff --check` passed. A separate synthetic
Windows Chrome E2E passed **1/1** for two formal reviewers, state reset, an
intermediate snapshot, final and retry downloads, and Python validation of the
exact envelope bytes. The E2E used only generated colored MP4 files and
synthetic judgments.

## Limitations and next action

The envelope module is wired into the formal claim renderer's in-memory
sequential-review flow: each completion freezes its review/index and optional
clock/index as one session, next-reviewer resets row and attestation state, and
one download exports the ordered envelope. A retry button repeats that same
envelope. An optional `--embed-videos-from` mode is available for an owner-only
token directory.

The synthetic browser E2E verifies the sequential interaction and download
bytes in Chrome, but does not authenticate reviewer identity, independence,
actual viewing, or semantic judgment. Sequential reviewers on one device are
not made independent by the page. No protected media or response was read,
and no protected result should be published.

The follow-up audit fixes enforce the 16-session maximum in the browser, align
reviewer-ID validation on a 256-Unicode-scalar limit and exact NFKC plus
Unicode full-casefold mapping, and size-check the actual pretty-printed
download bytes. Starting a later reviewer exposes a download snapshot of
completed sessions so earlier entries can be saved while the next session is
incomplete.
