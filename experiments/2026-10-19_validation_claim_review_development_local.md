# Validation-split claim review development run (local-only, 2026-10-19)

## Purpose

Exercise real-video Qwen claim generation and prepare a low-friction human
review artifact on an outcome-blind, validation-split development cohort. This
is review-workflow feasibility work, not confirmatory evaluation or a direct
faithfulness score. Keep this log, the packet, embedded-video page, candidates,
and model generations local; do not publish them. Dataset/model rights remain
unverified.

## Provenance and cohort construction

- The local official validation split contains 567 distinct video IDs and 3,358
  QA rows. Its mapped videos were read from the local 24,253,160,380-byte
  NExTVideo ZIP (SHA-256 recorded in the private pool summary); no data was
  uploaded.
- Before choosing a cohort, the aggregate-only source-ledger audit verified
  ledger integrity over 46 records (31 NExT-GQA namespace records) and found 30
  validation video-ID matches. Historical completeness and key-to-source
  binding remain unauthenticated. A separate ZIP audit previously found 33
  distinct media-byte matches across the supplied partial/expanded indexes.
  Nonmatches are unknown, never certified fresh.
- The bounded pool builder checked all eligible ZIP media against three
  explicitly supplied HMAC indexes before selection. The local HMAC-key digest
  was calculated from the supplied key for this exploratory run because no
  independent out-of-band pin was located; this does not authenticate the
  index/key/source binding. Overlap across these sources is supporting evidence
  of shared tokenization, not a freshness certificate.
- Builder counts: 3,358/3,358 rows mapped; 208 QA rows excluded by known video
  IDs; 26 rows excluded by known media bytes; 3 interval-invalid rows; 3,121
  eligible pairs across 534 unique media. The outcome-blind SHA-256 ranking
  selected ten unique media for decode screening and extracted 45,551,562
  bytes. Ten candidates totaled 43 eligible pairs.
- CPU decode screened four candidates before the fixed rule selected three
  that passed the interval/evidence-control geometry. These three exact pairs
  were frozen before model load; an independent repeat of the deterministic
  CPU freeze reproduced the same cohort hash. The selected rows are consumed
  development items and must be excluded from any later confirmatory cohort.

## GPU run and observed result

- Runtime: existing repository `.venv`, torch 2.11.0+cu128, CUDA/BF16 available;
  NVIDIA GeForce RTX 5070 Ti, 16-frame video inputs, pinned offline
  Qwen2.5-VL-7B-Instruct checkpoint, 4-bit weights and BF16 compute.
- The frozen three-row cohort was run as a one-row GPU smoke followed by the
  remaining two rows. Both runs completed: 3/3 videos and 3/3 claim records were
  structurally valid; each record contains 16 sampled frames. Model-run elapsed
  times summed to 23.8 seconds; maximum recorded allocated VRAM was 7.45 GiB.
- The two private outputs were checked for owner-only mode 0600, then combined
  into one mode-0600 packet. The three media digests and question IDs exactly
  matched the pre-model CPU-frozen cohort. Combined packet SHA-256:
  `8144b73df9ff79c87bede731e5236330648db2dd35ddd4c24c3b51f4d5299d63`.
- Built an offline Japanese quick-review page with three embedded videos,
  nickname entry, one-button progression through the questions, combined JSON
  download, and an optional toggle for original English question/claim text.
  It has a self-fingerprint, CSP `connect-src 'none'`, no file picker, and mode
  0600. Current HTML SHA-256:
  `36e73aa45a3123444a943a6301d74854141ef62bd6839e4e7e4870a554bf1dac`.
  The page runs the existing informal clarity/answer-fit/evidence-sufficiency
  check; it does not implement the formal signed review protocol.

## Checks, limitations, and faithfulness status

- Synthetic CPU tests passed: ZIP candidate pool 11/11, split-aware claim
  runner and score-runner compatibility 29/29, private response import 14/14,
  and quick-review artifact builder 9/9 (63/63 combined). The elevated
  synthetic browser E2E passed 1/1 after the English-source toggle change.
  Python compilation and scoped whitespace checks passed.
- The synthetic Windows Chrome E2E initially failed its final exact-profile
  cleanup assertion in the ordinary sandbox. A follow-up read-only Windows
  query found zero matching temporary profiles and zero Chrome processes. The
  same synthetic E2E then passed 1/1 under per-command elevation; it exercised
  three reviewers and combined JSON download, without opening the protected
  review HTML. Browser playback of the actual protected artifact remains
  unverified on the target user's browser.
- A synthetic response matching the exact new HTML's item IDs and self-bound
  fingerprint passed the response inspector's in-memory validation (3/3
  item mappings, fingerprint verified, nickname removed from aggregate); this
  was repeated on the English-toggle v2 artifact. No
  synthetic or human response file was saved.
- A primary-side artifact audit parsed the HTML data without printing its
  contents, matched all three ordered question/claim pairs back to the private
  GPU packet, verified each of the three embedded video payloads byte-for-byte
  against the browser-compatible local source, and checked the self-fingerprint
  and mode 0600.
- No person has reviewed this newly generated cohort yet. Claim generation is
  not evidence that a claim is true or sufficient. The review page is a
  low-stakes exploratory aid only.
- **Direct faithfulness progress remains unchanged:** formal claim reviews
  0/3, clock/frame correspondences 0/15, prediction-link reviews 0/3, direct
  scores 0/3, and no observed `DeltaF`, interval, or test. Nothing in this run
  supports a claim about model faithfulness or intervention efficacy.

## Next action

Use the private quick-review HTML for exploratory item-quality feedback and
retain its downloaded response locally. Independently audit returned response
integrity before using it to improve wording or review flow. For the main
faithfulness question, continue only after the formal claim/frame/prediction
link and custody gates are independently satisfied; do not treat this
validation development cohort as confirmatory.

## Follow-up: source-language disclosure reset and v3 artifact (local-only)

- Fixed the small navigation issue: opening the optional English source details
  on one item no longer leaves them open on the next item. The state is reset in
  `showQuestion()`; the previous artifact remains unchanged.
- Added a browser E2E assertion that opens the disclosure, advances to the next
  synthetic item, and verifies it is closed. The full three-reviewer combined
  response-flow test passed 1/1 in Windows Chrome using only synthetic videos.
  It also verified the combined schema-v6 JSON flow.
- Added a private, pair-hash-keyed translation sidecar for these three items so
  the Japanese text can be regenerated against the exact English question and
  claim pair without storing those source strings in the translation key. This
  records which translation belongs to which source pair; it is not an
  independent validation of translation accuracy. The page retains an optional
  English source view.
- Rebuilt as a distinct v3 output (no overwrite):
  `data/private_nextgqa/quick_review_val_generated_claims_local_20261019_03.html`.
  SHA-256 is
  `dd5c9e9f8a402ca2ddef6a2fd6f4e6790f62e79c21527144e463487a59bbcdc2`;
  size 12,086,460 bytes; mode 0600. Translation sidecar SHA-256 is
  `f1c1a98177c14a739d25e8a1310bd44b22f2431f473a044d804161d02c9598ec`,
  also mode 0600.
- Primary and independent audits verified 3/3 ordered question/claim mappings
  against the private packet/package, 3/3 item IDs, and 3/3 embedded browser-
  compatible video byte matches. Fingerprint, private file mode, offline CSP,
  and synthetic response-to-artifact binding all passed. v3 differs from v2
  only by the disclosure reset and the correspondingly recomputed fingerprint.
- Relevant CPU suites passed 71/71; Chrome synthetic browser E2E passed 1/1.
  No protected artifact was opened in Chrome. Actual-user-browser playback and
  Japanese translation correctness remain unverified.
- Direct faithfulness status is unchanged: no human review of the new cohort,
  no clock/frame or prediction-link attestations, and no direct score or
  intervention effect. This work improves review readiness only. Rights status
  remains unverified and all media/review artifacts remain local-only.

## Follow-up: translation clarity and actual-browser video smoke (v4, local-only)

- An independent Japanese-language check found no major meaning reversal. It
  suggested preserving the item-2 source ambiguity without hardening its causal
  wording and making the item-3 helping adult's role explicit. I applied those
  two wording refinements; item 1 remains unchanged. This is translation
  checking, not claim verification.
- Rebuilt a separate v4 (v3 remains intact):
  `data/private_nextgqa/quick_review_val_generated_claims_local_20261019_04.html`.
  SHA-256 `bf872db7f18e6d0c7a2c87a9647843a639eda70194e301ae2f7a49e41dd9e731`,
  size 12,086,514 bytes, mode 0600. Its translation sidecar is
  `data/private_nextgqa/quick_review_val_generated_claims_translations_local_20261019_04.json`,
  SHA-256 `9e04a07eb3bc84693dd570e7115e4439d2fa68f18548d3f8a801148154a2be74`,
  mode 0600. Sidecar keys bind exact source question/claim pairs but disclose
  no raw source text; they do not certify translation accuracy.
- Independent language review passed 3/3 translations for basic meaning and
  clarity, with no quoted source in its report. It did not review truth,
  evidence sufficiency, or the video.
- A primary v4 audit reconfirmed packet/package question-claim order, item IDs,
  3/3 browser-compatible embedded video byte matches, self-fingerprint, CSP,
  mode 0600, and an in-memory synthetic schema-v6 response binding 3/3.
- Extended the local Chrome E2E to support an explicitly selected private
  artifact and to require video metadata plus a rendered video frame. Both the
  synthetic artifact and v4 private artifact passed in Windows Chrome. The v4
  run also completed the three-reviewer synthetic-answer flow and combined JSON
  download; the E2E's temporary profile was cleaned. No human response was
  created. The test blocks HTTP(S) requests and uses the local WSL file URL.
  This verifies a local Chrome build, not every user's browser configuration.
- Relevant CPU suites passed 71/71; Python compilation and scoped diff checks
  passed. Full decoding of the three embedded H.264 clips passed 3/3.
- **Direct faithfulness remains unchanged:** human reviews 0/3, clock/frame
  correspondences 0/15, prediction-link reviews 0/3, direct scores 0/3, and no
  `DeltaF`, confidence interval, or intervention test. The GPU claim-generation
  feasibility remains the only new model run; review readiness and translation
  quality are not faithfulness efficacy evidence. Rights remain unverified;
  packet, videos, review HTML, and responses remain local-only.

## Follow-up: retrospective source-identity overlap check (local-only)

- The exact frozen candidate-pair digests were joined back to the validation
  CSV without outputting IDs or question text. All 3/3 selected QID/video pairs
  mapped uniquely by the frozen pair digest (the QID alone is not unique in
  this CSV and is insufficient as a join key).
- The selected three were checked against the union of the supplied partial,
  expanded-known-history, and supplemental validation-pilot HMAC indexes:
  known video-ID matches 0/3, video/QID pair matches 0/3, and media-SHA matches
  0/3. This supports only “unmatched to these supplied records.” The history is
  incomplete and the HMAC key/source binding is unauthenticated, so freshness
  remains uncertified. The rows stay development-only and must not enter a
  confirmatory cohort.
- This retrospective check changes no inference or generation cohort. It is an
  outcome-blind provenance reconciliation only; no video was decoded and no
  model/GPU was used.
- To make the exact join reproducible, added
  `scripts/audit_nextgqa_frozen_candidate_exposure.py` and six synthetic tests.
  The tests include a QID reused by two different video IDs (resolved by the
  frozen pair digest), known video/pair/media matches, wrong-key zero matches,
  missing frozen pairs, packet mismatch, duplicate QIDs, and output privacy.
  The local command reproduced the 3/3 mapping and 0/3 counts for each supplied
  history identity class; it reports key/history authentication false,
  freshness `not_made`, GPU unused, and authorization not granted. Synthetic
  tests passed 6/6; compile and diff checks passed. The tool is not a freshness
  certifier and does not unlock scoring.

### Audit correction after independent code review

- The first script version overstated the packet-to-media association: its media
  digest is supplied by each packet row, but no independently frozen
  video-ID-to-media digest crosswalk is present. The independent audit showed
  that swapping two packet media digests could preserve the aggregate overlap
  count. I revised the output to call these
  `known_packet_claimed_media_sha256_matches`, explicitly mark the pair/media
  crosswalk unverified, and prohibit interpreting this as cohort-verified media
  overlap. Historical index and HMAC provenance remain unauthenticated too.
- Exact pair digests, when present in packet rows, can disambiguate selected
  repeated QIDs, but are still packet assertions and are not authenticated.
  Legacy packets use QID only and fail closed when selected QIDs are ambiguous.
  Duplicate JSON object keys at nested levels and duplicate split CSV pairs now
  fail closed. Synthetic tests include swapped media digests, selected repeated
  QIDs with and without explicit pair digests, duplicate JSON keys and split
  pairs, wrong key, and aggregate privacy.
- The revised synthetic suite passed 11/11. Re-running the local aggregate
  audit still maps 3/3 frozen pairs; video-ID and video/QID pair matches are
  0/3, and the **packet-claimed** media digest overlap is 0/3. This does not
  establish that packet hashes are correctly associated with selected videos,
  that historical indexes are complete, or that candidates are fresh. No
  video/model/GPU was used for this audit. No direct faithfulness evidence or
  human-review gate changed.
- Closed the auditor's remaining integration gap by adding the same canonical
  `candidate_pair_sha256(video_id, qid)` to future private generator records.
  Added a focused generator test that the digest changes when the video changes
  even if QID stays fixed. The combined exposure/source-ledger/pool/generator
  suites passed 46/46; compilation and scoped whitespace checks passed. The
  already-generated private packet predates this code and therefore still uses
  the legacy QID join; its selected QIDs happen to be unique, and media remains
  packet-claimed only. No regeneration or GPU run was needed for this metadata
  binding change.
- Inspected the user-provided exploratory JSON with the repository's
  privacy-preserving response inspector. It is schema v5 with 3 respondent
  labels and 9 categorical item responses; nickname and free-text values were
  not emitted. It validates against the older default artifact's item IDs only
  when legacy-unbound mode is explicitly enabled. It does **not** fingerprint-
  bind to the newer v4 artifact, and it did not validate against available v1–v4
  generated pages. Thus this is exploratory feedback for the older page, not a
  formally authenticated or artifact-bound review.
- Item-level counts (3 responses per item): item 1 clarity unclear 2/3, answer
  fit “does not fit” 1/3 (1 fits, 1 cannot tell), evidence insufficient 2/3;
  item 2 clarity unclear 2/3, answer “does not fit” 2/3, evidence split evenly
  among insufficient/sufficient/cannot tell; item 3 clarity clear 3/3 and
  answer fits 3/3, while evidence is insufficient 2/3. These are counts from
  three self-labeled exploratory respondents, not a representative sample,
  formal claim truth labels, or a faithfulness estimate. The pattern supports
  attention to question clarity, answer fit, and evidence sufficiency in the
  review workflow. Because these response IDs map only to the old unbound
  default artifact and do not map to generated artifacts v1–v4, do not attach
  these item-level findings to the current three GPU candidates or revise those
  candidates from these counts. No direct score, `DeltaF`, interval, hypothesis
  test, or intervention result follows from this feedback.
