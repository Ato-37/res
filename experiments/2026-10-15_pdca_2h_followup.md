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
