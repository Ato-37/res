# QTA human-gate state audit (2026-10-18)

## Purpose

Recheck whether the local three-video QTA review handoff has advanced to actual
browser/player capture or whether a real-data faithfulness run is now allowed.
This audit used filesystem metadata and repository protocol/source only; it did
not open protected images, video, question/claim values, or private JSON
contents.

## Verified observations

- The protected root is owner-only (`0700`). The metadata census found 553
  staged MP4 files and 15 image files. All 15 images are in decoder-export
  directories and owner-only (`0600`); there are no image files in a
  screenshot/capture/crop/evidence-class directory.
- The relevant current-v6 candidate anchor plan, provenance, staged videos, and
  decoder PNG export are already present as candidate preparation. This does
  not constitute player capture or reviewer observation.
- The current human-review runbook requires a frozen review-only receipt,
  independently pinned trusted root/role roster, witnessed unopened handoff,
  and a fresh release-order audit with `package_opening_permitted=true` before
  opening the package. The runbook states that those real artifacts do not
  currently exist. It also states that production pre-baseline and
  post-baseline score-freeze gates/runners are not implemented.
- Independent state audit and primary metadata-only recount agree that
  browser/player screenshots are absent. Stage-1 claim review remains 0/3,
  player-clock/decoder correspondence 0/15, stage-2 prediction-link review
  0/3, and claim-linked scores 0/3.
- Correctly invoked focused decoder export, crop, clock-evidence, evidence
  reader, and calibration-input suites passed **84/84** tests. An earlier
  invocation without `PYTHONPATH=scripts` failed at test-module import; it did
  not execute tests and was immediately rerun correctly.
- A read-only elevated GPU query at 2026-10-18 00:36 UTC saw an RTX 5070 Ti with
  14,851 MiB free of 16,303 MiB and 0% compute utilization. No GPU forward was
  run in this audit because the score-bearing branch remains protocol-gated;
  repeating an unscored wiring smoke would not answer the selected faithfulness
  question.

## Interpretation and limits

This verifies a decoder-side artifact handoff only; no new human or model
evidence was obtained. Metadata cannot prove that a reviewer never viewed a
file, and this audit does not authenticate the inventory or human identity.
Rights status remains unverified; files and derived sensitive outputs remain
local.

## Faithfulness checkpoint

- **Direct evaluation:** `未実施` — 0/3 independently reviewed claims, 0/15
  verified player-to-decoder correspondences, 0/3 prediction-link judgments,
  and 0/3 QTA scores. Tested model/intervention/faithfulness metric: none in
  this audit. There is no empirical `ΔF`, interval estimate, inferential test,
  failure count, or multiplicity adjustment to report.
- **Auxiliary diagnostics only:** 84/84 synthetic/software evidence-pipeline
  tests; private-file metadata census; GPU availability check. These are not
  efficacy evidence.
- **Blocking gates:** no authenticated release root/roster and witnessed
  review-only release; no real stage-1 review; no browser captures/correspondence
  or clock calibration; production phase receipts and score runners remain
  incomplete; stage-2 review must follow baseline `p/q` and precede treatment.
- **Next action:** continue independent source/protocol validation that does
  not create synthetic activity or bypass human evidence. Real baseline and
  intervention forwards become justified only after the corresponding review,
  capture, trust, and phase gates pass.
