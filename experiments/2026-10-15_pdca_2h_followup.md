# PDCA two-hour follow-up — 2026-10-15

Requested work window: **2 hours**. The user-message timestamp was unavailable,
so accounting started conservatively at **2026-10-15 21:34:23 JST**
(**12:34:23 UTC**) after reading the PDCA skill and repository instructions.
Deadline: **23:34:23 JST / 14:34:23 UTC**. Record checkpoints below; do not report
this window complete until the deadline is reached.

## Plan

Advance the blocked NExT-GQA claim-faithfulness path without repeating the
unstable gold-interval QA-margin pilot or the failed frame-index prompt-format
search. The last source log reports 0 claim-reviewed samples and 0 claim-specific
effect estimates. The immediate dependency is a blinded review of the frozen
model claims against their source videos, without showing the QA answer,
choices, or gold time-span seed. Keep all data/artifacts local under the user's
2026-10-15 local-only authorization; rights status is still unverified. Do not
publish or push.

Before the review packet was prepared, inspected the current protocol and
experiment source, the local dataset README, and current environment. The
official [NExT-GQA dataset README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)
describes `gsub` intervals as QA-corresponding time spans and `frame2time` as a
frame-ID-to-seconds map. Neither that representation nor the earlier 85-video
PTS census independently establishes that a QA interval supports a later
model-generated claim. The existing protocol therefore requires an independent
claim-to-interval review and keeps the QA span hidden until the mapping is
frozen.

## Do and check

An independent protocol audit confirmed that the claim-score branch remains
`no_go`: human claim mapping is incomplete; annotation/player-to-decoder timing
has not been calibrated for the existing gold-span mask; the schema-v13
primary estimand has no justified SESOI; and its preflight does not grant
execution authorization. These gates are separate from the rights gate waived
for this user's local-only operations. The audit also confirmed the previous
head-faithfulness search is closed and the 16/32-frame answer-margin pilot had
unstable signs, so neither branch was repeated.

The prior v3 generation packet is an unreviewed, mixed-reference artifact. A
new CPU-only builder, `scripts/build_nextgqa_claim_mapping_packet.py`, now emits
an allowlisted mapper-facing draft with only the frozen question/claim and the
private source-video reference/hash plus opaque IDs. It excludes QA choices,
gold answer/span, parsed answer letter, model-selected frame indices, and raw
generation text. It binds the exact source-packet byte hash, model revision,
available source/runner hashes, the four dataset-input hashes, and per-claim
text hash; the CLI also rehashes each referenced local media file before writing.
The worksheet explicitly says `draft_not_reviewed`, keeps all reviewer decisions
blank, marks review procedure and occurrence policy `unfrozen`, and says
`execution_authorization=not_granted`. It refuses output inside the repository,
requires the local-only flag, writes mode 0600 exclusively, and will not
overwrite an existing file.

The first application attempt correctly failed closed because the legacy v3
packet lacks the aggregate `fully_structured_claim_records` field. No output
was written by that attempt. After inspecting only schema/count metadata, the
builder was changed to derive availability from the allowlisted per-record
claim field instead of trusting the legacy `status` or expecting the missing
aggregate. A second application succeeded on the three recorded rows, and
verified all three local media-file hashes against the packet before output.

Local draft worksheet:

- Path: `/tmp/nextgqa_blinded_mapping_worksheet_20261015_2145.json`
- SHA-256: `a22834d48e7d22c5d9bb660b8a38b9994469a8c108a0398db4bc57ce661e1d2d`
- Permissions: `0600`; requested/effective claim rows: **3/3**
- Status: `draft_not_reviewed`; human-reviewed: **0/3**; score authorization:
  `not_granted`

The independent implementation audit verified the exact output-key allowlist,
count constraints, repository-path rejection, no-overwrite/mode-0600 behavior,
and actual media-hash checking in the builder. Provenance remains input-bound,
not independently authenticated: model/runner metadata are values in the
source packet, not a signed run attestation. This worksheet clears no semantic,
timing, rights, SESOI, or execution gate.

The new synthetic test file passes **8/8**. Python compilation and
`git diff --check` pass. The full repository suite passes **858/858**. The
current elevated, one-command preflight observed an RTX 5070 Ti, CUDA-enabled
`.venv` (`torch 2.11.0+cu128`, BF16 supported), and 14.66 GiB free of 15.89 GiB
VRAM. No new model inference was run: the dependent interval selector and score
protocol are not ready, and repeating the prior exploratory contrast would not
discriminate a new hypothesis.

## Faithfulness progress at this checkpoint

- **Direct evaluation:** `未実施` — **0 requested / 0 reviewed / 0 scored / 0
  effect estimates** for claim-specific faithfulness on Qwen2.5-VL-7B-Instruct
  and NExT-GQA. No intervention/faithfulness metric was run in this window.
- **Engineering and numerical diagnostics only:** a blinded review worksheet
  is prepared for 3/3 prior model-generated claim candidates; 3/3 referenced
  local media hashes match. The earlier 85-video interval/PTS census remains a
  geometry/timing diagnostic only, not semantic support or faithfulness.
- **Unresolved gates:** a versioned claim-review procedure and occurrence policy
  are not frozen; independent human review has not started; gold-span/decoder
  timing is uncalibrated; the practical SESOI and inference target remain open;
  the scoring preflight still reports no execution authorization. Local-only
  operation is authorized by the user, but rights remain unverified and nothing
  has been shared online.
- **Next action:** freeze the outcome-blind review procedure and occurrence
  policy before a reviewer opens the items; complete full-video review of these
  exact three records, with unsupported/ambiguous claims retained as failures.
  Only after that review may the claim-linked mask/control be derived. Resolve
  clock mapping and the estimand/SESOI separately before any scoring. Do not
  reopen head/layer/alpha exploration, repeat prompt tuning on these three, or
  treat this three-item feasibility cohort as confirmation.

## Act

No results or media were published. The new builder/tests and this log entry are
the scoped local changes for this cycle; the user data directory and existing
private manifests remain untouched and unstaged. Before any future GPU scoring,
run the project fail-closed preflight after the review/protocol gates are
resolved, then smoke on the frozen mapped cohort. At the next checkpoint,
record elapsed time and continue with another independent, justified task while
the requested window remains.

## 21:55 JST checkpoint — frozen review procedure and local reviewer page

Elapsed: **20m44s / 2h**, deadline remains **23:34:23 JST**. The occurrence
policy and display procedure are now fixed in
`docs/nextgqa_claim_review_policy_v1.json` (exact-byte SHA-256
`0fe92fff658aa25280b0e32c6db979d8903c9ad4d1ddfc592bb38601d459bfb7`). The
builder pins this hash and refuses to mark a procedure frozen unless the packet
is exactly 3 requested / 3 effective rows. The procedure requires an independent
human to verify each row against the separate frozen cohort/source-lineage
record, check the selected media hash, review the complete video with audio and
captions hidden, map every relevant visual occurrence, and preserve failures in
the denominator. It explicitly leaves decoder PTS calibration, control
selection, rights, SESOI, and score authorization unresolved.

Added `scripts/render_nextgqa_claim_review_html.py` and tests as an offline
review aid. It has no network-connected assets, a restrictive CSP, local file
SHA-256 checking before playback, a 64 MB media-size ceiling, forced mute,
custom playback/seek controls, per-occurrence interval and rationale fields,
and bounds checks for duration, positivity, overlap, and millisecond precision.
The renderer enforces a closed worksheet schema, recomputes claim-text hashes,
rejects unexpected fields, and excludes local media paths from both page and
export. Explicit export creates an unvalidated review handoff and a separate
companion index binding the exact review-payload hash to the frozen worksheet
and policy hashes; both remain non-authorizing. Browser downloads may have
permissive modes and must be moved to a private directory and chmod 600 before
any use.

The first implementation mistakenly retained local paths in the exported JSON;
that version was discarded and not used for review. A second independent audit
found no remaining source-level privacy blocker for the frozen worksheet schema.
The current full repository test suite passes **863/863** (8.277 s), including
synthetic checks that unknown answer/location sentinels are rejected and claim
hashes are recomputed. Python compilation and `git diff --check` pass. Generated
private page: `/tmp/nextgqa_claim_review_20261015_2210.html`, mode 0600, SHA-256
`0990cb0c952f281ffa4d525725b5c372ce71a36d99d892dd99c6010f6df67e09`. Static
inspection finds no external assets, `connect-src 'none'`, no embedded local
source paths, and no gold-answer/span or raw-generation fields. No browser is
installed, so playback, CSP behavior, and download flow remain untested; this
page is not evidence that anyone reviewed the videos.

### Faithfulness progress at 21:55 JST

- **Direct evaluation:** `未実施` — **0 requested / 0 human-reviewed / 0 scored /
  0 claim-specific effects** for Qwen2.5-VL-7B-Instruct on the 3-item NExT-GQA
  feasibility set. No intervention or faithfulness metric was run in this
  window, so no estimate, CI, hypothesis test, or multiplicity procedure applies.
- **Supporting diagnostics only:** 3/3 local source-file hashes match the
  private packet declaration; the blinded worksheet is 3/3 rows and 0/3
  reviewed; the static page is mode 0600. These establish file/engineering
  properties, not semantic claim support or faithfulness.
- **Open gates:** human review has not started; digest-to-source lineage still
  requires the reviewer’s separate manifest check; annotation-player time to
  decoder PTS is uncalibrated; control construction is separate; SESOI/inference
  target and item-rights evidence remain unresolved; scoring preflight remains
  fail-closed. Nothing was published.
- **Next action:** validate the exported review handoff/index structure locally
  using synthetic fixtures, then conduct actual independent human review of all
  three rows when available. Do not substitute AI review, metadata, or another
  GPU pass for human claim mapping.

## 22:08 JST checkpoint — actual source lineage and pinned review handoff

Elapsed **33m50s / 2h**; deadline **23:34:23 JST** (86m10s remain). The local
lineage audit matched all **3/3** claim rows against `test.csv`,
`map_vid_vidorID.json`, the question text hash, relative NExTVideo key, and
source-media bytes. Its private manifest is
`/tmp/nextgqa_claim_source_lineage_20261015_2220.json`, mode 0600, SHA-256
`11d99a6314e92313672dfd870e84f3162195bbcd1eb1e038301177943759da18`. The three
source videos are present under the local NExTVideo root and are 6.7–9.4 MB,
below the review page's 64 MB limit. This verifies file/metadata consistency;
it does not verify claim semantics or a person's review.

The first independent source audit found that a caller could alter the unsigned
lineage manifest and recompute its digest before rendering. Rendering and
handoff audit now require the independently logged manifest SHA as a separate
CLI argument; the lineage parser also rejects duplicate JSON keys, oversized
input, and paths that traverse above the video root. Synthetic mutation tests
cover digest substitution and path traversal. The remaining trust boundary is
explicit: this is an independently recorded checksum, not a digital signature
or proof of who performed the source audit.

The current review page is
`/tmp/nextgqa_claim_review_20261015_2235.html`, mode 0600, 17,994 bytes, SHA-256
`e10edb3a71ac01f2ca26c92c47b6de245a37add9b47f86f43d9f41c7045b73ee`. It pins
the exact lineage manifest above and displays the QID/video mapping for manual
comparison. Static HTML parsing, inline-JavaScript syntax check, no absolute
repository-path scan, Python compilation, and `git diff --check` pass. The full
CPU suite passes **874/874** in 9.903 seconds. No browser is installed here, so
actual playback, secure-context WebCrypto behavior, and browser download
permissions remain untested; no reviewer export exists.

### Faithfulness progress at 22:08 JST

- **Direct claim-specific faithfulness:** still **0/3 human-reviewed, 0 scored,
  0 effect estimates**. No new faithfulness intervention, model scoring, GPU
  inference, CI, or hypothesis test occurred.
- **Real-data readiness evidence:** the local source and metadata lineage is
  cross-checked for 3/3 records; the blinded worksheet remains 0/3 reviewed.
  This is provenance/readiness evidence only, not semantic claim support or a
  faithfulness result.
- **Reason no GPU job was run:** the valid next score would depend on claim
  mapping and a defensible temporal intervention. A GPU pass before those gates
  close would repeat non-discriminating exploratory inference, and could not
  establish the target faithfulness claim. GPU/CUDA availability alone does
  not make such a run scientifically useful.
- **Open constraints:** a human has not reviewed these items; this runtime has
  no browser for the review UI; the expected lineage checksum is a logged
  checksum rather than a signature; player-to-decoder timing and the frozen
  SESOI/inference target remain unresolved. Scoring authorization remains
  `not_granted`; nothing was published.
- **Next action:** complete a second independent audit of the pinned handoff,
  then spend the remaining authorized window on outcome-blind verification of
  the existing timing/calibration implementation and its stated limits. Do not
  inspect gold spans before claim mapping or represent metadata checks as
  faithfulness evidence.

## 22:13 JST checkpoint — PTS-origin boundary and real-file cadence

Elapsed **39m09s / 2h**; deadline **23:34:23 JST** (80m51s remain). A separate
CPU-only metadata pass over the same three local source videos found H.264
streams at 25 fps or 30000/1001 fps, with decoded frame counts 1615 or 2697.
Thus the frame cadence is roughly 33–40 ms; the review page now says its
0.001-second entry granularity is storage precision, not 1 ms visual evidence
resolution. The updated private page is
`/tmp/nextgqa_claim_review_20261015_2214.html`, mode 0600, 18,191 bytes, SHA-256
`336d543f813dfa9c0b3149166cea04bde149dda9cd8eeeb15acc375b68931fd9`.

Added one synthetic lossless Matroska fixture test through the production
two-pass PyAV decoder. Requested PTS `(100,102,104,106,108)` at exact time base
`1/10` are preserved as sampled PTS `[10000,10400,10800]` with decoded time
base `1/1000`; the selected frame indices are `[0,2,4]` and relative times are
`[0,2/5,4/5]` seconds. Pixel hashes verify the selected RGB buffers. The test
also confirms the adapter labels its origin as the first decoded presentation
timestamp, keeps `annotation_clock_calibrated=false`, and leaves execution
authorization `not_granted`. An independent source review reran all 11 decoder
tests and found no issue. This closes a synthetic PTS-origin plumbing case; it
does not calibrate the source player's clock or validate semantic evidence.
Focused timing suites pass **28/28**; full CPU suite passes **875/875**.

### Faithfulness progress at 22:13 JST

- **Direct evidence remains 0/3 human-reviewed, 0/3 scored, 0 effect estimates.**
  No Qwen model forward or GPU inference was run in this checkpoint.
- **Real-data checks:** 3/3 source-media/hash and CSV/video-key/question
  metadata bindings verified; 3 videos' codec cadence inspected. These are
  media provenance/timing-readiness checks, not faithfulness scores.
- **Independent audit:** source-manifest SHA is required at both the Python API
  and CLI boundaries; unsigned-manifest substitution, duplicate keys, path
  traversal (including backslashes), schema mismatch, and handoff tampering are
  rejected by tests. The independent auditor found the CLI/API checks closed;
  the checksum still depends on keeping the independently recorded digest
  separate from the input file.
- **Remaining gates:** no browser is installed in this runtime, human review is
  incomplete, media rights are unverified, player-to-decoder anchors and
  held-out clock calibration are absent, and SESOI/inference target remain
  unresolved. No result was published. The user was offered the local review
  page to complete the human stage; continue independent CPU/protocol audit
  while awaiting a response.

## 22:16 JST checkpoint — decoder pass on the three source videos

Elapsed **42m07s / 2h**; deadline **23:34:23 JST** (77m53s remain). A
metadata-only, full-frame CPU decode through the production PyAV first-pass
routine succeeded on the three local source files. Their frame counts were
1615, 2697, and 2697; all timestamps were strictly increasing, all streams were
H.264, and every within-video frame delta was constant: 1/25 s for the first
and 1001/30000 s for the other two. Relative decoded durations were 1614/25 s
and 337337/3750 s. The decoder saw RGB source dimensions corresponding to
480x360, 640x360, and 640x480. No frames were shown or saved; no QA, gold span,
claim text, or model score was read by this diagnostic.

The three Python CLIs now also have synthetic integration tests for private
input modes, mode-0600 exclusive outputs, overwrite rejection, and the
non-authorizing audit status. Focused reviewer/handoff/lineage CLI checks pass.
The new PTS-origin fixture's exact PTS and time base are asserted, not just a
nonzero-origin inequality. The independent source auditor reproduced the
actual-video counts and rational timestamp facts without opening annotations
or rendering video content. It also verified all three media SHA-256 values
against the frozen lineage manifest and found zero decode failures. The first
PTS is 1000 for the 1/25000-time-base file and 0 for the two 1/30000 files.
This concrete offset shows why normalizing each stream to its first decoded PTS
does not itself map the media clock to the HTML/source annotation player's zero
point.

### Faithfulness progress at 22:16 JST

- **Direct metric progress:** still **0/3 human-reviewed, 0/3 scored, 0 effect
  estimates**; no GPU inference or faithfulness metric ran in this window.
- **Actual data:** local question/video/source hash binding 3/3; all three
  source files decode and have monotonic constant-rate PTS through the pinned
  PyAV pass. This narrows a media/decoder feasibility risk only; it does not
  align the source-player clock, establish claim support, produce a mask, or
  estimate faithfulness.
- **Protocol decision:** no sampling tolerance or timestamp-origin conversion
  was selected from these observed videos. The annotation/player-to-decoder
  map still requires pre-outcome fit and held-out visual anchors, independent
  validation, and a frozen acceptance rule. Therefore these results do not
  unlock GPU scoring.
- **Open work:** source browser playback remains unavailable here, and
  independent human review is still incomplete. Continue with remaining CLI and
  protocol checks until the recorded 23:34:23 JST deadline unless the user
  explicitly changes the scope.

## 22:18 JST checkpoint — dataset duration metadata versus decoded extent

Elapsed **44m25s / 2h**; deadline **23:34:23 JST** (75m35s remain). For the same
three lineage-verified videos, I read only each matched `gsub_test` duration
field and compared it with `last_relative_PTS + one exact decoded-frame
interval`. Dataset durations were 65, 90, and 90 seconds; the decoded extent
estimates were 64.6 and 89.9899 seconds, leaving differences 0.4 and 0.0101
seconds. All differences fit whole-second rounding, but this agreement cannot
identify subsecond origin, player seek behavior, edit boundaries, drift, or
annotation timestamp semantics. It does not calibrate the clock and does not
expose QA spans or answers.

The independent audit confirmed the media hashes still match the lineage
manifest for all three files and independently reproduced frame counts,
rational time bases, constant frame intervals, durations, and monotonicity with
zero decode failures. The full CPU suite now passes **879/879** after CLI
privacy/overwrite tests and exact nonzero-PTS assertions. `compileall` and
`git diff --check` pass. These remain provenance and decoder diagnostics, not
faithfulness observations.

### Faithfulness progress at 22:18 JST

- **Direct evaluation:** **0/3 reviewed, 0/3 scored, 0 effect estimates**; no
  direct faithfulness metric or model forward was run this window.
- **Real-data progress:** the candidate set's question/video/hash binding,
  full-frame decode, and coarse duration consistency are verified. A 0.4 s
  first-file offset between rounded dataset duration and decoded extent is
  compatible with rounding; it cannot choose an exact offset correction.
- **Next discriminating step:** an independent human must map the frozen claims
  to visible video intervals while blind to QA seeds and scores. Separately,
  freeze fit/held-out visual anchors and an acceptance rule to calibrate the
  source-player clock. Until these are done, no GPU scoring run can test the
  intended claim-mask effect.

## 22:28 JST checkpoint — duration-only parser and independent audit

Elapsed **53m50s / 2h**; deadline **23:34:23 JST** (**66m13s remain**). The
private clock-extent CLI now extracts only the three candidates' dataset
`duration` values; it skips nested annotation payloads without constructing
their Python objects. Its scanner rejects duplicate keys, malformed container
separators, non-finite JSON constants, trailing data, and missing duration
values. The lineage manifest is bounded to 1 MB before reading, requires schema
version 1, checks its independently supplied SHA-256, and reports missing-file
errors through the CLI's controlled error path. Synthetic tests cover these
conditions and the annotation-payload exclusion.

An independent reviewer inspected source and tests, confirmed that nested
`location` payloads are skipped, and raised an important interpretation issue:
the terminal extent adds the preceding PTS interval to the last PTS; it does
not read an independently measured terminal-frame duration. The output field
and interpretation limit now state this explicitly. The reviewer found no
record-level IDs, paths, questions, spans, or claim text in the aggregate
output; stable hashes remain for provenance. The reviewer did not read protected
source data or run the real-media CLI.

I independently reran the corrected CLI against the locally authorized three
videos after the parser changes: **3/3 requested/effective, 0 decode failures**;
frame counts remain 1615/2697/2697, and the reported whole-second duration
differences remain 0.4/0.0101/0.0101 seconds. Output is mode 0600, 2439 bytes,
SHA-256 `71801038fddd1c765c5ffbac75819783d7897e798608812f35bc27d7d31decd5`
at `/tmp/nextgqa_candidate_clock_extent_20261015_2234.json`. This is a
local-only derived artifact, not a faithfulness result. The duration reader
scans the 195,500-byte dataset JSON text but does not parse span structures.

### Faithfulness progress at 22:28 JST

- Direct claim-specific evidence: **0/3 human-reviewed, 0/3 scored, 0 effect
  estimates**; no GPU model forward or faithfulness metric has run.
- Real-data feasibility: lineage and decoder checks remain 3/3. The duration
  differences are compatible with whole-second rounding under the tested
  preceding-PTS extrapolation; they do not calibrate the annotation clock or
  establish claim support.
- Next action: finish the full regression suite and verify whether an available
  local review route can expose the already-prepared blinded worksheet to a
  human reviewer. Continue into independent design/protocol audits if that
  route remains unavailable; GPU scoring still cannot answer the target claim
  before temporal anchors and claim mapping are fixed.

## 22:30 JST checkpoint — NExT-GQA real-video CUDA path smoke

Elapsed **55m54s / 2h**; deadline **23:34:23 JST** (**1h04m06s remain**). The
local NExT-GQA smoke runner executed on the RTX 5070 Ti with the existing
offline pinned Qwen2.5-VL-7B-Instruct checkpoint, 4-bit loading and BF16
compute. It deterministically inspected three eligible local test clips for
the configured 1 ms uniform-grid constraint; two were rejected on timing and
one supplied four sampled frames. Qwen preprocessing returned temporal grid
`[2,28,36]` and 536 input tokens. The one CUDA forward took 3.1535 seconds,
produced finite logits, and peaked at 6.0421 GiB allocated VRAM; total run time
was 25.7241 seconds with 14.6572 GiB free at preflight. Requested/effective
videos and frames were 1/1 and 4/4; no runtime failure occurred.

The private mode-0600 aggregate is
`/tmp/nextgqa_real_video_gpu_smoke_20261015_2238.json` (1,631 bytes,
SHA-256 `23b94de072152fd084b21986a4e956080ee6608b98c388802dfba6e8e69be73c`).
The runner hash is `a75e5242ed7e15db52dfef57cee32a51f5c07e55c7c0f648a6a85a07e6286254`.
The artifact contains no media identity/path, QA text, generated text, or raw
media. However, deterministic candidate selection does read the QA's location
interval and duration to validate annotation eligibility; those values are
neither supplied to the model nor used to select the sampled frames. No QA
question/answer text or generated text is used or saved, and no answer,
grounding, or faithfulness score is computed. The sampled PTS/pixel hashes are
stable linkage fingerprints, so the artifact remains local-only. The
independent audit confirmed the run facts and noted that the result does not
carry an explicit checkpoint hash; the runner pins a local model revision.
This is the first verified NExT-GQA real-video CUDA path result in this window,
but remains an execution smoke only.

### Faithfulness progress at 22:30 JST

- Direct claim-specific faithfulness: **0/3 human-reviewed, 0/3 scored, 0
  effect estimates**. The GPU run did not evaluate the target estimand.
- Real-data execution: **1/1 video, 4/4 frames, 1 finite-logit CUDA forward**;
  two of three timing candidates were rejected under the fixed 1 ms uniform
  sampling constraint. The per-item rejection rate here describes only this
  technical selection pass, not a dataset failure rate or research outcome.
- Next: close the independent artifact audit, then assess whether the current
  generated claims and review interface can be independently validated while
  preserving the blind. Use the remaining window for a distinct protocol or
  data-validity check; do not repeat identical GPU forwards to inflate usage.

## 22:33 JST checkpoint — frame2time versus decoded PTS (same frozen cohort)

Elapsed **58m35s / 2h**; deadline **23:34:23 JST** (**1h00m48s remain**). I
extended the private source-clock diagnostic to read only `frame2time_test`
vectors for the three lineage-frozen video IDs and compare each listed time to
the nearest decoded relative PTS. It did not access `gsub.location`, QA text,
answers, claims, or video pixels as review evidence. Each video had 32 monotonic
frame2time points. Median nearest-PTS residuals were 0.0100 s, 0.00885 s, and
0.00885 s; maxima were 0.0200 s, 0.0162 s, and 0.0162 s. These maxima are at
or below half a frame interval for the respective CFR streams, consistent with
nearest-frame quantization. The comparison uses PTS relative to the first
decoded frame; one file's absolute first PTS is 0.04 s, so these residuals do
not validate the container's absolute timeline.

Private output: `/tmp/nextgqa_candidate_clock_extent_frame2time_20261015_2233.json`,
mode 0600, 3,063 bytes, SHA-256
`56261af2b8997a95f292fda5d952f46171887080fd3c73d0019a5f70478e11b6`.
This is a frame-grid-to-decoder timestamp consistency check, not frame identity
validation, source-player calibration, human claim support, or an interval
mask efficacy result. This first output was superseded by the fully audited
rerun below, which also records nearest-PTS collisions and points outside the
decoded PTS range. The output explicitly keeps `annotation_clock_calibrated`
false. Its terminal duration uses the preceding PTS interval as an
extrapolation.

### Faithfulness progress at 22:33 JST

- Direct claim-specific faithfulness remains **0/3 human-reviewed, 0/3 scored,
  0 effects**. The only CUDA run remains one 4-frame generic-prompt forward;
  no QA or faithfulness outcome was measured.
- Real-data support now includes 3/3 media lineage, 3/3 full decodes, and 32
  frame2time-to-nearest-PTS points per video with residuals no larger than one
  half-frame interval in this cohort. That supports a narrow geometric
  consistency statement only; human visual anchors and source-player identity
  are still missing.
- Continue with independent audit and provenance completeness for the GPU
  artifact, then use remaining time on a prespecified data-validity check.

## 22:36 JST checkpoint — audited geometry counts and regression suite

Elapsed **1h01m21s / 2h**; deadline **23:34:23 JST** (**58m39s remain**). After
independent review, the frame2time audit now reports duplicate timestamp count,
nearest-PTS index collision count, and points outside the decoded PTS range
separately. The six focused tests cover exact rational median/max residuals,
synthetic VFR timestamps, duplicate and nearest-index collisions, range overflow,
and rejection of empty, decreasing, boolean, string, negative, NaN, and infinite
points. The independent auditor rechecked the revised helper/tests and found the
earlier ambiguity resolved.

Final private result from this version:
`/tmp/nextgqa_candidate_clock_extent_frame2time_20261015_2238.json`, mode 0600,
3,211 bytes, SHA-256
`65cfdc108e9e2a5d3e7d1796df953c49184f61a54fd56b7398f359b69385689f`;
runner SHA-256 `ca435f4b80eb7e61160070c000fe2bf1732cbf407a394061cbec62cb75182c26`.
The three videos each contain 32 monotonic mapping points; all points lie
within the first/last decoded PTS range and none collide onto the same nearest
decoded PTS index. Median/max nearest residuals are unchanged from the prior
checkpoint. This is a descriptive result for three selected clips, with no
population uncertainty interval and no claim of exact frame identity.

The repository CPU suite passes **885/885** after these changes; compileall and
`git diff --check` pass. Expected failure-path messages and transformer
deprecation notices appeared in the test stream, but the suite exited zero.
Direct faithfulness evidence remains **0/3 reviewed, 0/3 scored, 0 effects**.

## 22:37 JST checkpoint — 16-frame real-video CUDA smoke

Elapsed **1h02m18s / 2h**; deadline **23:34:23 JST** (**57m42s remain**). A
second, distinct NExT-GQA video passed the pinned-Qwen CUDA smoke with 16
sampled frames and the CLI's fixed 50 ms uniform-grid tolerance. This clip's
2,525 decoded source frames yielded Qwen temporal grid `[8,24,44]` and 2,144
input tokens. The single CUDA forward took 3.1482 s, returned finite logits,
and measured 7.2068 GiB peak allocated memory after model loading; reported
elapsed time was 17.8212 s, with 14.6572 GiB free at preflight. The runner's
elapsed timer starts after imports, CUDA preflight, and candidate enumeration,
then includes timing-candidate decoding, processor/model setup, and forward;
it is not total CLI wall time. Peak allocated tracking resets after model load,
so it includes resident model allocations and the forward high-water mark but
excludes transient model-load peak and is not reserved/total VRAM. There were no
timing rejections (1 inspected, 1 selected). Compared with the earlier
4-frame run, this checks a different actual clip, a fourfold requested frame
count, and a larger processor input. These two small smoke runs are not a
controlled model-quality or efficacy comparison, and memory values exclude
any transient peak during model loading because peak tracking was reset after
loading.

The private mode-0600 artifact is
`/tmp/nextgqa_real_video_gpu_smoke16_20261015_2240.json` (1,632 bytes,
SHA-256 `7d6bf4586e80e6baf67efc77059bfaed8b94ef8dd1a13a0f1a90ba0e3746b0f0`);
it is bound to the same runner SHA as the four-frame result. The fixed 50 ms
tolerance controls only this smoke's candidate eligibility and must not be
reused as the gold-span, annotation-clock, or faithfulness tolerance. QA
questions, generated text, answers, and scores were not used or stored; as in
the first smoke, location/duration metadata contributes to candidate
eligibility but not model input or temporal selection. Independent artifact
audit is pending.

### Faithfulness progress at 22:37 JST

- Direct claim-specific evidence: **0/3 reviewed, 0/3 scored, 0 effects**.
- Actual GPU path: **two distinct videos**, one 4-frame and one 16-frame run;
  both had one finite-logit forward. Allocated-memory observations were
  6.042/7.207 GiB under their respective inputs. This supports only those two
  runtime configurations, not efficacy or a general peak-memory bound.
- Continue with the independent audit and protocol/data checks until the
  23:34:23 JST deadline. Do not infer that 50 ms is an acceptable temporal
  calibration limit from this smoke.

## 22:39 JST checkpoint — independent related-work audit

Elapsed **1h05m18s / 2h**; deadline **23:34:23 JST** (**54m42s remain**).
An independent primary-source review found several close precedents, so broad
priority claims such as “first video explanation faithfulness” or “first
temporal intervention” are not supportable. NExT-GQA already has human temporal
labels and joint QA/grounding work ([CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Xiao_Can_I_Trust_Your_Answer_Visually_Grounded_Video_Question_Answering_CVPR_2024_paper.html));
causal grounding alignment has been studied on NExT-GQA ([CVPR 2025 CRA](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Cross-modal_Causal_Relation_Alignment_for_Video_Question_Grounding_CVPR_2025_paper.html));
and temporal claim deference has been tested with event-order and sampling
interventions ([Caved or Convinced](https://arxiv.org/abs/2608.03160)). Other
adjacent work tests explanation-derived visual edits ([EDCT-Bench](https://arxiv.org/abs/2609.17953)),
or video-model activation interventions for hallucination reduction ([TA-AE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html)).
These works have different outcomes and intervention targets; this short scan
is positioning evidence, not a systematic review.

The narrow candidate distinction is a post-hoc behavioral dependence measure:
for a human-verified mapping between a model-generated answer claim and a
reported temporal interval, compare the fixed claim's gold-versus-best-foil
log-margin drop under that interval mask against a rank-matched temporal control
mask. This particular combination's priority is unresolved. Even a future
positive estimate would not demonstrate hidden-reasoning faithfulness, claim
truth, or causal mechanism; temporal mask OOD effects and semantic differences
between target/control intervals remain alternative explanations. Direct
claim-specific faithfulness evidence is still **0/3 reviewed, 0/3 scored, 0
effects**.

The reviewer worksheet remains **0/3 reviewed**. Thus no claim-mask scoring
should run yet. Continue independent protocol and implementation validation;
do not substitute model-generated labels or geometric timestamp checks for the
required human mapping.

The protocol validator's focused regression suite passed **52/52** on this
checkpoint, including fail-closed checks for incomplete human review, unresolved
SESOI/inference design, uncalibrated execution, and the distinction between
rights recording and execution authorization. This verifies structural guard
behavior only; it does not clear those gates. The final frame2time and two GPU
artifact hashes were re-read and match their recorded values. The claim review
worksheet remains mode 0600 and 0/3 complete.

## 22:41 JST checkpoint — independent timestamp recomputation

Elapsed **1h07m07s / 2h**; deadline **23:34:23 JST** (**52m53s remain**).
I independently re-decoded the three lineage-hash-verified local videos using
PyAV and recomputed, from exact decoded PTS and the three `frame2time_test`
vectors, nearest-PTS residuals without calling the project audit helper. The
sorted decoded frame counts were 1,615/2,697/2,697, with 32 mapping points per
video; medians were 0.01000/0.00885/0.00885 s, maxima 0.0200/0.0162/0.0162 s,
and out-of-range counts were 0/0/0. These independently match the aggregate
artifact's values. This reproduction still cannot establish which source frame
each annotation denotes, player/decoder alignment, annotation-clock origin or
drift, or visual claim support. It is not a temporal calibration pass and does
not change the **0/3** human-review or **0** faithfulness-effect counts.

Additional claim-ledger, private-review-index, clock-extent, and cross-gate
preflight regressions passed **51/51**. These tests cover hash binding,
row reconciliation, omission of private values from summaries, invalid-span
rejection, and the invariant that even a structurally complete preflight stays
blocked without execution authorization. They are synthetic/structural tests,
not human review or empirical faithfulness evidence. A scoped local commit
`f716ca7` records this checkpoint; no remote fetch or push was performed.
