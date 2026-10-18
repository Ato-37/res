# Candidate-pool visual-overlap screen protocol (2026-10-18)

## Purpose and status

Outcome-blind exploratory screen for near-duplicate or visually overlapping
clips among the 490 exact-byte-distinct candidate videos. It addresses one
limited threat to treating videos as independent units. It does not establish
source/event independence, validate claims, or measure faithfulness. No model
logits or reviewer data enter the procedure.

## Frozen screen before inspecting outcomes

- Candidate set: all MP4s in the already fixed local 490-video pool. The runner
  requires exactly 490 before processing; no treatment outcomes or human-review
  labels are used.
- Per-video sampling: eight zero-based frame indices at equally spaced interior
  fractions `(i + 1) / 9` of decoded frame count `N`, computed as Python
  `round((i + 1) * N / 9)` (ties-to-even). The decoded count must support eight
  unique indices; otherwise the video is counted as a decode/sampling failure.
- Representation: resize each decoded frame to 32×32 grayscale; compute a
  64-bit perceptual hash from the low 8×8 two-dimensional DCT coefficients,
  excluding the DC coefficient from the median threshold. Frames with grayscale
  standard deviation below 5/255 are excluded from matching as low-texture
  frames.
- Pair screen: a video pair is a visual-overlap candidate if at least two
  distinct sampled frames from each video have a cross-video Hamming-distance
  match of at most 6 bits.
- Primary aggregate: number of screened video pairs meeting that rule, plus
  the count histogram of maximum distinct matching-frame support. No pair IDs,
  paths, images, or per-video hashes are emitted. A private local inventory
  manifest pins all 490 relative paths and content SHA-256 values; only its
  aggregate fingerprint is included in the local receipt, never published.
- Compute: decode locally; transfer only the small grayscale frame tensors and
  hash comparisons to CUDA. No neural model is loaded. Use all sampled frames;
  no outcome-dependent exclusions or threshold changes.

## Interpretation limits

Perceptual hash is a coarse visual fingerprint. A positive pair is only a
candidate for later source/event investigation; common backgrounds, static
scenes, or repeated objects can cause similarity. A negative pair does not
prove distinct source events, and eight frames can miss sparse overlap,
re-encoding, crops, or temporal shifts. Decoder failures preserve the requested
denominator and are reported. Any later threshold exploration is exploratory
and cannot replace independent human/source metadata validation.

Direct claim-faithfulness scoring remains blocked by the separate human review,
timing/correspondence, trust, and phase gates. Rights remain unverified; all
video reads and outputs remain local.

## Verified run (2026-10-18)

The candidate set was pinned before the full screen by a private mode-0600
manifest binding all 490 relative entries and their content SHA-256 values.
The runner checked the manifest's pre-run fingerprint, exact 490-path set, and
all 490 file contents before decoding. The fingerprint and per-video hashes
remain local; only aggregate counts are recorded here. The aggregate receipt
is also mode 0600 and remains under the local private-data root.

- Full-pool status: **completed**; requested **490**, decoded **490**, failures
  **0**. The frozen set contained 490 byte-distinct MP4s totaling
  **1,990,584,365 bytes**.
- Sampling: **8/8** interior frames per video, **3,920** sampled frames total.
  Three frames were excluded by the fixed low-texture rule; all **490/490**
  videos still had at least two informative sampled frames.
- Pair screen: **119,805/119,805** pairs checked; **0** met the frozen rule
  (at least two distinct matching frames per video, pHash Hamming distance
  <= 6). The support histogram totals 119,805 at support zero; all other
  support bins are zero.
- Decoded source frame-count sum: **552,154** across the 490 videos.
- Runtime resources: RTX 5070 Ti, Torch 2.11.0+cu128, CUDA 12.8, OpenCV 5.0.0,
  Python 3.12.14; peak CUDA allocated memory **58,859,520 bytes**. No neural
  model was loaded, no model forwards occurred, and no faithfulness score was
  computed.
- The preregistered four-video smoke passed 4/4 with 0 decode failures and
  reported `bounded_smoke_completed`; it was not represented as a full run.
  Its zero of six candidate pairs did not change any threshold or rule.
- Focused CPU tests passed **10/10**. The full `scripts/` suite passed
  **1,748/1,748** in **52.033 seconds**. `py_compile` and `git diff --check`
  passed for the changed files.
- The independent code auditor verified protocol/code agreement, the manifest
  path/content checks, private output behavior, and synthetic tests without
  inspecting the videos. A separate aggregate-receipt audit confirmed the
  arithmetic, histogram total, scope, and non-faithfulness flags without
  inspecting media, paths, or the membership manifest.

## Interpretation and next action

Observed: under this frozen eight-frame pHash screen, no pair among the finite
490-video set crossed the prespecified candidate threshold. This is an
exploratory visual-overlap diagnostic, not an estimate of general NExT-GQA
source independence. It cannot detect sparse overlap outside sampled frames,
temporal shifts, crops, re-encoding, related scenes, or shared events; pHash can
also miss semantic similarity. The zero flags therefore do not certify that
the 469 videos unmatched to supplied historical exposure indices are fresh or
independent. No pair identities were emitted, so any future source/event
investigation needs a separate authorized private workflow with controlled
pair access and independent review.

Direct faithfulness progress remains **未実施**: **0/3** independently reviewed
claims, **0/15** verified browser-clock/decoder correspondences, **0/3**
prediction-link judgments, **0/3** claim-linked scores, and no empirical ΔF,
confidence interval, hypothesis test, or multiplicity-adjusted result. The
screen used no treatment, model prediction, or faithfulness metric; its result
does not advance the direct efficacy denominator.

Next, continue the human-review readiness work by obtaining the distinct role
assignments and independently pinned trust-root fingerprint requested earlier.
For the human-free branch, do not retune this screen after its zero-flag result;
use independently sourced media lineage or a separately prespecified
validation procedure to assess source/event overlap. Direct scoring remains
blocked until the human attestations, timing/correspondence, trust, and phase
gates pass. Rights remain unverified and all media and receipts stay local.

## Reproducibility provenance

- Source: local frozen 490-video candidate pool; exact input membership/content
  receipt and raw aggregate run receipt remain private and are not committed.
- Protocol: this file's frozen sampler/threshold section; Python ties-to-even
  indexing. No random seed or outcome-dependent selection was used.
- `scripts/audit_candidate_pool_visual_overlap.py` SHA-256:
  `1def797f39fc8344d0dfb5450a4570b2918ab8078d5df954da4af97cbd15ccb1`
- `scripts/freeze_candidate_pool_manifest.py` SHA-256:
  `25d8ea16ebfa2e7730e6e28ea9828b5690a8f970b189e3f07126f023ab7c616e`
- `scripts/test_audit_candidate_pool_visual_overlap.py` SHA-256:
  `eb30352b49cbf145a0af90c1d137a88f08c5b3baf7366f5d95ca5417647a353b`
- `scripts/test_freeze_candidate_pool_manifest.py` SHA-256:
  `13ae8d315b366780bfbd08917e7c8dc53363f44952a21b22b1b220648ee29c91`
