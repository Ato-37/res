# Candidate screenshot cropper

`scripts/crop_nextgqa_clock_screenshots.py` makes local, private RGB PNG crops
from the full-window screenshots created by
`capture_nextgqa_clock_screenshots.ps1`. It accepts only an explicit crop-map
JSON with exactly the 15 token/anchor assignments from the actual hash-pinned
candidate plan. The map binds the exact plan, provenance, and capture-receipt
SHA-256 values and supplies one integer `x,y,width,height` box per assignment.
The tool verifies the plan/provenance candidate states, exact receipt bytes,
receipt plan/provenance bindings, exact screenshot coverage and screenshot
hashes, and that each crop box is within its screenshot bounds. It rejects
symlinks and non-private files/directories, writes RGB PNGs and a private
crosswalk to a new mode-`0700` directory (files mode `0600`), and removes
partial output on failure. Standard output contains only a generic status and
row count.

Provenance must also match the active PyAV version and FFmpeg library-version
mapping, the exact builder/decoder/custody-auditor/private-output source hashes,
the worksheet/lineage/custody input hash fields, a UTC creation timestamp and
the exact active Python version, and the three token-to-staged-media hash
records in the candidate plan. Stale or internally misaligned provenance
fails closed.

Example (all paths and hashes must be independently checked; output must not
already exist):

```sh
.venv/bin/python scripts/crop_nextgqa_clock_screenshots.py \
  --plan /private/clock_anchor_plan.json \
  --expected-plan-sha256 PLAN_SHA256 \
  --provenance /private/clock_anchor_plan_provenance.json \
  --expected-provenance-sha256 PROVENANCE_SHA256 \
  --capture-receipt /private/clock_screenshots_run/capture_receipt.json \
  --expected-capture-receipt-sha256 RECEIPT_SHA256 \
  --screenshot-dir /private/clock_screenshots_run \
  --crop-map /private/crop_map.json \
  --output-dir /private/screenshot_video_crops_new
```

The crop map schema is deliberately small:

```json
{
  "schema_version": 1,
  "purpose": "private_candidate_screenshot_video_crop_map",
  "plan_sha256": "...",
  "plan_provenance_sha256": "...",
  "capture_receipt_sha256": "...",
  "records": [
    {"review_token": "24-character token", "anchor_id": "fit_begin",
     "bbox_xywh": [100, 120, 640, 360]}
  ]
}
```

The crosswalk records the receipt screenshot byte hash, crop box, crop PNG
hash and dimensions, and the exact plan/provenance/receipt/map hashes. A
custodian can use the resulting `{token}_{anchor_id}.png` files as screenshot
crops when preparing the existing blinded-pair review manifest.

The crop box is caller-supplied. Hash and bounds checks prove only which bytes
were cropped and that the rectangle fits inside the screenshot. They do not
establish that the rectangle contains only video, excludes browser chrome or
other content, or corresponds to the intended video viewport. The cropper
cannot authenticate the screenshot receipt, capture process, or plan supplier;
crop purity still requires human review. Its output and crosswalk are
`candidate_only`, remain `execution_authorization=not_granted`, and establish
no screenshot authenticity, clock calibration, scoring, or research claim.

Synthetic-only tests:

```sh
.venv/bin/python scripts/test_crop_nextgqa_clock_screenshots.py
```
