# Blinded screenshot-crop / decoder-frame review

`scripts/build_blinded_frame_pair_review.py` builds a local-only, offline review
page for exactly 15 already prepared PNG pairs. It accepts only a strict
custodian manifest with an opaque token, anchor ID, two simple filenames, and
the expected SHA-256 for each file. Inputs and their parent directories must be
owned by the current user with files at mode `0600` and directories at `0700`.
Images are size-limited, decoded, converted to RGB, and re-encoded as PNG to
remove source metadata.

Each build randomizes pair IDs, row order, and A/B sides. The reviewer page and
package manifest live in one private directory; a separate private directory
contains the custodian identity/side map. The reviewer export contains
randomized pair IDs, crop-purity attestations, match/ambiguous/mismatch
decisions, a reviewer pseudonym, independence self-attestation, UTC review
time, per-package nonce, and package-manifest hash. `audit-export` rejects missing or
duplicate rows and treats any impure, ambiguous, or mismatched pair as blocked.
The package and export explicitly remain `not_authorizing` with
`execution_authorization=not_granted`; there are no signatures, reviewer-key
trust claims, or scoring permissions.

Example local invocation (use fresh, private output paths):

```sh
.venv/bin/python scripts/build_blinded_frame_pair_review.py build \
  --manifest /private/input/pairs.json \
  --plan /private/input/anchor_plan.json \
  --expected-plan-sha256 <64-lowercase-hex> \
  --screenshot-dir /private/input/crops \
  --decoder-dir /private/input/frames \
  --viewer-dir /private/output/reviewer_bundle \
  --identity-dir /private/output/custodian_map
.venv/bin/python scripts/build_blinded_frame_pair_review.py audit-export \
  --export /private/reviewer/blinded_frame_pair_review.json \
  --package-manifest /private/output/reviewer_bundle/package_manifest.json \
  --reviewer-html /private/output/reviewer_bundle/index.html
```

The reviewer must save the downloaded export into a mode-`0700` private
directory and set the JSON file to mode `0600` before running `audit-export`;
the auditor rejects exports that do not meet those permissions.

The custodian manifest must bind the SHA-256 of the actual candidate anchor
plan and contain exactly three distinct review tokens, each paired once with all five
fixed anchors: `fit_begin`, `heldout_quarter`, `fit_midpoint`,
`heldout_three_quarter`, and `fit_end`. The CLI requires this expected plan
hash:

```sh
--expected-plan-sha256 <64-lowercase-hex>
```

The command receives the private candidate plan via `--plan` and its expected
SHA-256 via `--expected-plan-sha256`. It checks the plan bytes, requires the
candidate-only status and `execution_authorization=not_granted`, derives the
3×5 token/anchor assignment set from that JSON, and requires an exact match
with the custodian manifest. This binds rows to that hash-pinned candidate
plan; it does not authenticate who supplied the expected hash or establish
joint protocol freeze/authorization.

The reviewer page asks for a pseudonym and independent-review self-attestation.
Its export includes a UTC timestamp, package nonce, package-manifest hash,
reviewer role, and the literal `self_reported_unauthenticated` provenance
status; signature status is `none`. The nonce associates the export with its
package but does not authenticate the reviewer or prevent replay. The package
manifest stores a normalized reviewer-HTML template hash and per-side PNG
hashes. `audit-export --reviewer-html /path/to/index.html` verifies that
template hash and each embedded image payload before considering review rows.
The normalization replaces only the self-referential package-hash and nonce
fields. These checks establish internal consistency, not an external trust
anchor.

The reviewer can see image content and dimensions, and may infer which side is
the screenshot crop versus decoder frame from visual characteristics. RGB
re-encoding may change color-profile appearance, and the page scales images to
fit the viewport. The decisions are human visual correspondence judgments,
not exact pixel-equality measurements.

The input manifest schema is:

```json
{
  "schema_version": 1,
  "purpose": "private_blinded_frame_pair_review_input",
  "plan_sha256": "64 lowercase hex characters",
  "records": [
    {
      "review_token": "24-character opaque token",
      "anchor_id": "predeclared anchor ID",
      "screenshot_crop_file": "pair-01-crop.png",
      "screenshot_crop_sha256": "64 lowercase hex characters",
      "decoder_frame_file": "pair-01-frame.png",
      "decoder_frame_sha256": "64 lowercase hex characters"
    }
  ]
}
```

The script consumes only those image files and this manifest. It does not read
videos, questions, claims, answers, timestamps, fit/held-out labels, or scores.
The reviewer still sees frame content. This tool only presents already
custodian-cropped images; it does not establish that a crop came from an
authentic OS/compositor screenshot, that the crop is the intended video
viewport, that a decoder frame corresponds to a browser display event, or that
the images have scientific validity. A human crop-purity attestation and a
`match` decision are recorded claims, not authenticated evidence. This review
does not calibrate clocks, validate scoring inputs, or authorize model scoring.

Synthetic-only tests:

```sh
.venv/bin/python scripts/test_build_blinded_frame_pair_review.py
```
