# Japanese review runbook inventory binding (2026-10-19)

## Purpose

The formal English handoff runbook directs operators to its Japanese operational
copy, but the candidate byte inventory did not bind that Japanese file. Add the
Japanese runbook as a separately hashed public/source role so the actual
instructions handed to reviewers cannot silently drift after the candidate is
prepared.

## Implementation and checks

- Added `review_runbook_ja` for
  `docs/nextgqa_v6_human_review_runbook_ja.md` to
  `PUBLIC_OR_SOURCE_FILES` in `scripts/build_nextgqa_pre_review_byte_inventory.py`.
- Extended the inventory test to require the role and verify it is emitted.
- The focused inventory-builder suite passed **21/21 tests**. `py_compile` and
  scoped `git diff --check` passed.
- The integrity-receipt and release-order-auditor suites passed **33/33
  tests** after the new role was added. The receipt verifier derives its
  required artifact set from the builder's registered roles, so the Japanese
  runbook is now required in signed inventory verification too.
- Built a fresh local candidate at
  `data/private_nextgqa/inventory-jp-runbook-20261019-1525/`.
  It contains **95 roles**, has **0 missing slots**, and includes the Japanese
  runbook role. The inventory and directory are mode `0600` and `0700`.
  Its status remains `candidate_only`; `joint_freeze=false`, `score_ready=false`,
  and `execution_authorization=not_granted`.

## Interpretation and limits

This closes a byte-binding omission in the procedural handoff. It does not
authenticate the candidate inventory, reviewers, trust root, or chronology and
does not grant permission to open the package, load a model, or score results.
The independent review-release prerequisites remain absent. Direct
faithfulness progress is unchanged: **0/3** formal Stage-1 reviews, **0/15**
verified player-clock/decoder-frame correspondences, **0/3** Stage-2
prediction-link reviews, and **0/3** direct scores. No direct effect estimate,
confidence interval, or test exists. Rights status remains unverified; all
protected data and outputs remain local.

## Next action

Obtain and independently pin the real trust root, finalize distinct reviewer
and custodian roles, produce the signed freeze/release/witness materials before
any first opening, and complete the required Stage-1 and clock/frame evidence.
Until those inputs pass, keep the direct GPU baseline at zero forwards. Continue
independent audits and other non-authorizing implementation checks without
repeating closed proxy cohorts.

## Japanese intake navigation follow-up (2026-10-19)

The Japanese operational runbook linked its assignment instructions to the
English template even though a Japanese template exists. The link now points to
`docs/nextgqa_human_review_role_assignment_template_ja.md`. All three local
Markdown links in the Japanese runbook resolve. Because the runbook bytes are
inventory-bound, a fresh candidate superseded the prior one at
`data/private_nextgqa/inventory-jp-intake-link-20261019-1530/`. It again has
95 roles, zero missing slots, and an exact hash match for the current Japanese
runbook; mode is `0700` for the directory and `0600` for the inventory. It
remains candidate-only and non-authorizing. Its SHA-256 is
`690db88252d8eb60cc06ddc49e89541cae8f1acd54246cfb19b6ea463568a32f`.
An independent audit rehashed every registered source and code role with zero
stale roles, checked the Japanese link target, and confirmed the candidate
status. This navigation fix does not add any reviewer, trust-root, clock/frame,
or faithfulness evidence.
