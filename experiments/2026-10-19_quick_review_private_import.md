# Private import for the exploratory quick-review response (2026-10-19)

## Purpose

Make the low-friction nickname-first review's downloaded JSON safer to handle
locally before anyone inspects its summary. The response contains a nickname,
so leaving it in a normal browser download directory was an avoidable privacy
gap in the previous handoff instructions.

## Change and verification

- Added `scripts/import_private_quick_review_response.py`. It reads one bounded
  regular file without following a symlink, validates the full response against
  the exact selected v4 artifact's opaque item IDs, then writes a unique copy
  into `data/private_nextgqa/quick_review_responses/` using an exclusive
  temporary file and atomic no-overwrite hard-link publication.
- It requires the private root and response directory to be owned by the
  current user and mode `0700`; the response is created and verified as mode
  `0600`. It does not overwrite or delete the source download. The CLI prints
  only the validated broad response counts, per-item choice, and private copy
  path; it omits nickname and free text.
- Updated the Japanese runbook with one import-and-inspect command and clear
  instructions to remove the download copy manually only after verifying the
  private copy.
- Synthetic importer, response-inspector, and artifact-builder tests:
  **25/25 passed**. Cases cover private permissions, source and destination
  symlink rejection, malformed response rejection before copy, exact artifact
  item mapping, source retention, output redaction, cleanup after a simulated
  write failure, and preservation of an existing file on a forced final-name
  collision.
- No real response was present in the local user tree during the filename-only
  search. No response JSON contents, video contents, or protected review answers
  were read. No human response was collected.

## Interpretation and limitations

The new helper reduces accidental disclosure risk and validates that all three
answers belong to the current quick-check artifact before keeping a private
copy. It does not protect the original browser download; the operator must
delete that copy manually if no longer needed. The judgments remain informal
impressions, not claim truth, hallucination labels, formal review, or direct
faithfulness measurements. The helper does not create signatures or satisfy
the independent role, root-pin, witnessed-release, frame-correspondence, or
prediction-link gates.

## Faithfulness progress and next action

Direct faithfulness remains **0/3** formally reviewed claims, **0/15** verified
player-clock/decoder-frame correspondences, **0/3** Stage-2 prediction-link
judgments, and **0/3** direct scores. No direct effect estimate, confidence
interval, or hypothesis test exists. Engineering work improves the local
response handoff only. Next, if a reviewer returns the quick-check JSON, import
and inspect it as exploratory feedback; separately, continue the formal
review-workspace integration and wait for authentic reviewer/root/witness
materials before any direct-score GPU forward. Rights status remains
unverified; protected material stays local.
