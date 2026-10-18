# QTA canonical dataset video-ID uniqueness guard (2026-10-19)

## Purpose

The fixed three-row QTA source adapter already rejected duplicate review tokens,
dataset video/question pairs, and (in a later phase) repeated source-media
digests. An independent gate audit found that three distinct question rows
could still name the same canonical NExT-GQA `dataset_video_id`, including
when each row claimed different media bytes. The protocol requires distinct
videos within the cohort.

## Change and verification

`build_actual_qa_source` now tracks canonical `dataset_video_id` values and
rejects a repeated ID with `dataset_video_id_duplicate` before resolving the
selected CSV pair. The synthetic negative test keeps three distinct claimed
video byte hashes while repeating only one dataset video ID.

- Focused four-suite source/phase-composition run: **87/87 passed**.
- Independent source-adapter rerun and diff audit: **9/9 passed**; the valid
  three-ID fixture still passes and the duplicate-ID case rejects as intended.
- Python byte-compilation and scoped `git diff --check`: passed.
- No video pixels, model, or GPU were used.

## Interpretation and limitations

This establishes exact-string uniqueness of canonical dataset IDs within this
one three-row candidate. It does not exclude IDs used in earlier cohorts,
normalize alternate spellings, authenticate a dataset namespace, or prove the
completeness of any historical-use inventory. The local prior-exposure indexes
remain partial, so an unmatched ID cannot be called fresh. This change is a
structural input-integrity check only; it does not establish human review,
clock/frame correspondence, score authorization, or faithfulness.

## Faithfulness status and next action

Direct NExT-GQA claim review, clock/frame correspondence, prediction-link
judgments, and claim-linked scores remain unverified at **0/3, 0/15, 0/3, and
0/3**, respectively; no direct effect estimate, confidence interval, or test
exists. Next, any prior-use exclusion for QTA must be bound to a custodian-owned
canonical-ID ledger with a defensible coverage declaration. Until such complete
history is independently established, the cohort must remain freshness-unknown
and no confirmatory/direct-score claim may rely on this uniqueness check.
