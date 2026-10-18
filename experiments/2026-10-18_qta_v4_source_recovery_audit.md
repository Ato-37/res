# QTA v4 scorer source-recovery audit (2026-10-18)

## Purpose

Follow up the documented provenance gap for the local v4 mask-specificity
aggregate. Check whether the exact scorer and validation-helper source bytes
recorded in the private receipt remain recoverable from Git objects, so the
result is not incorrectly described as having or lacking reproducible source.

## Verified checks

Read only top-level provenance fields from the mode-restricted private receipt
`data/private_nextgqa/nextgqa_tlvs_qta_v4_mask_specificity_score_local_20261018_01.json`.
It records scorer SHA-256
`93eab9dd35822250d2b499e9c175e7eab9c83a1e2c8fd3ba253f52f487052dde` and
validation-helper SHA-256
`cfbffce9224a4eb2917c41b3b38dc78efa701cebae4ee08a863a4bfbe454b93d`.

The reachable Git history contains a scorer at commit `7b52fd3`, but its file
SHA-256 is `4db93fd2beecd1a9747eedbd84b433a304095255d0132f91bafd6d45352bbdcb`;
the validation helper at that commit hashes to
`0039e66f063a08c81a44eb16097e45d60b7a5adee53c278781ee9f925d485063`.
Neither matches the private receipt's execution-time hashes. A read-only scan
of all 55 unreachable Git blob objects also found no exact match for either
source.

## Interpretation and limits

Git contains related source, but not the exact execution-time files required
for fresh source-level reproduction. This confirms the prior provenance gap;
it does not invalidate the already independently reproduced row counts,
contrast arithmetic, or aggregate summaries. Those remain exploratory
finite-cohort measurements, not direct claim-faithfulness evidence. No model
was loaded and no protected video or annotation content was read. Rights status
remains unverified; the receipt and all row-level data stay local.

## Next action

Do not elevate or rerun the post-result mask-specificity branch. Preserve the
original receipt and source-hash ledger. Continue human-free work on a
pre-outcome frozen, source-bound intervention pipeline, and continue the human
review handoff when actual independent role assignments and a trusted root
pin are available.
