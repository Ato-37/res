# Exposure-screened real-video cohort 10 freeze (2026-10-21)

## Purpose

Create an outcome-blind, non-duplicative local NExT-GQA test cohort for a
possible future claim-review handoff. This is a candidate-selection/readiness
step only. It does not authorize model scoring and does not provide a
faithfulness result.

## Selection and provenance

An independent read-only audit reconstructed both current test candidate
pools using their pair-hash allowlists, source media exclusions, the latest
claim-exposure union, and the current local generation/scoring artifact scan.
The exposure union manifest contains 117 media hashes from 240 source files
(manifest SHA-256
`40d5579f1dfb31cef96f0ac471569dace59b7447da93b7eb195ff5b3829fe410`). The
current local artifact scan contains 160 unique media hashes and includes all
117 manifest hashes. These are known local records, not a complete or
authenticated historical exposure ledger.

Pool #03 currently has 20 materialized video candidates and 111 eligible
question-pair hashes. Nine candidate videos overlap the 160 known local
generation/scoring hashes; the other 11 were all decoded and passed the
16-frame uniform-timing preflight. Pool #04 has only two remaining candidates
after the same scan and cannot supply a three-video batch. The current #03
summary and pair-filter hashes differ from those recorded in old locks 08/09;
lock 09 also lacks a hash-bound exposure manifest. Those older locks were not
reused.

The current freezer selected three unique rows from pool #03 after applying
the pool's 93-media exclusion list and the 160-media local exposure set. It
scanned six eligible candidate videos to obtain three; recorded overlap with
the current local scored-claim set was zero. All source assignments, raw media
hashes, questions and paths remain in private mode-0600 files:

- Lock: `data/private_nextgqa/claim_cohort_lock_20261021_10.json`, SHA-256
  `5519e5095e22b7efbc56d1acbd4b3965fce911931e5ddd1c293bff5ff51c1bed`.
- Selected pair filter: `data/private_nextgqa/claim_cohort_pair_filter_20261021_10.json`,
  SHA-256 `447cead66d295a3840e1840400c1800d999a4a5ae3fdab9862b27eae286303a8`.
- Frozen cohort digest: `f3e195ba54f908692e11b21249d71600f6e75992164d549f1705eab2f8e15f1d`.
- Candidate summary SHA-256: `c84936f98dc3894b0b03556146e99756b1ae762747840c7078bb1846fff3b2f1`.
- Candidate-pair allowlist SHA-256: `335716b9ade96f82e1959dffeb9c7732743e15fc22d4bf3bddc8aec8f3beaa38`.

## Checks and interpretation

`PYTHONPATH=scripts .venv/bin/python -m unittest scripts.test_freeze_temporal_order_claim_cohort -q`
passed 7/7 tests. The live freeze returned exactly 3 unique rows, zero known
local exposure overlap, and the expected bound pool, pair-filter, exclusion,
and manifest digests. No outcomes were inspected to select the cohort. No GPU
forward, claim generation, human review, or faithfulness score was performed.

Before claim generation, the lock was independently reconstructed using the
same 93 pool exclusions and 160 local exposure hashes. The resulting private
generation exclusion file contains 175 unique hashes (source overlap explains
why this is below 93+160), mode `0600`, SHA-256
`407c60b656d1a0cb4faea3d484c97163ec125114fa07ef71126aac388e7f39bf`. A fresh
CPU decode of the exact three allowed pairs reproduced the ordered cohort
digest and both ordered media/QID assignments; 3/3 timing checks passed and
decode failures were 0. The claim-generation pilot unit tests passed 10/10.

This is a newly reproducible, locally exposure-screened exploratory cohort,
not a population-fresh or confirmatory sample: the broader historical exposure
ledger and model pretraining exposures are incomplete. Direct faithfulness
remains unmeasured: human claim-support review 0/3, clock/frame correspondence
0/15, prediction-link review 0/3, direct intervention score 0/3. The current
claim-token intervention runner also fails closed because its capture protocol
does not grant scoring authorization. QA-level spans cannot substitute for
claim-specific semantic support.

## Next action

Retain this lock as candidate-review readiness only. Do not score it through
the closed claim-token capture path or reuse it to tune a proxy. A future
machine run needs an independently justified, authorized endpoint; direct
faithfulness additionally needs real claim-support and prediction-link
judgments. Existing review artifact `claim_support_frame_review_test_20261021_06.html`
is still unanswered, and no human labels are inferred. Keep this result local;
dataset/model rights remain unverified.
