# Candidate clock-anchor decoder PNG export

`scripts/export_clock_anchor_decoder_pngs.py` exports the decoder-side images
for the exact 15 anchors in a hash-pinned candidate plan. It reads a private
plan and matching candidate provenance manifest, verifies both caller-supplied
hashes, validates the candidate-only statuses and `execution_authorization`,
and derives the 3×5 token/anchor mapping from the plan. It also checks that the
provenance's plan hash, token set, source-media hashes, frame counts, PTS
sequence hashes, time bases, and pixel specifications agree with the plan and
the staged media.

It also requires provenance to match the active PyAV version, the exact FFmpeg
library-version mapping reported by PyAV, and SHA-256 hashes of the current
plan builder, decoder, custody auditor, and private-output policy source files.
Any runtime or pinned-source change fails closed. A stale provenance manifest
must not be reused; regenerate provenance and revalidate the candidate plan
before exporting.

For each media file the helper opens one pinned file handle, verifies its SHA,
decodes once in display order, and incrementally checks the full PTS-sequence
hash while retaining only the five selected RGB frames. Each planned relative
PTS must identify exactly one decoded frame. It writes RGB PNGs and a private
crosswalk containing token/anchor, staged-media SHA, decoder frame index,
absolute and relative PTS, RGB hash, and PNG hash. Output goes into a new mode
`0700` directory with files at `0600`; errors remove the partial output.

No image/video content or identities are printed. This produces candidate
decoder-frame artifacts only. It does not capture a browser, establish
screenshot-to-decoder correspondence, authenticate the candidate plan or its
expected hash supplier, establish joint freeze, calibrate a clock, or
authorize scoring. Keep the files local and private.

Example command (substitute the independently verified hashes and current
artifact paths; output path must not already exist):

```sh
.venv/bin/python scripts/export_clock_anchor_decoder_pngs.py \
  --plan /path/to/private/clock_anchor_plan.json \
  --expected-plan-sha256 PLAN_SHA256_VALUE \
  --provenance /path/to/private/clock_anchor_plan_provenance.json \
  --expected-provenance-sha256 PROVENANCE_SHA256_VALUE \
  --media-dir /path/to/private/review_media \
  --output-dir /path/to/private/decoder_anchor_pngs_new
```

Synthetic-only tests use locally generated FFV1 clips:

```sh
.venv/bin/python scripts/test_export_clock_anchor_decoder_pngs.py
```
