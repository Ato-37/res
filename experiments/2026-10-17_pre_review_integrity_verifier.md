# 2026-10-17: non-authorizing pre-review integrity verifier

## Purpose

Shorten the path to a real, local human review by checking signatures and exact
candidate bytes without treating those checks as permission to expose the
package or run scoring. This is workflow tooling, not a faithfulness experiment.

## Implementation and checks

Added `scripts/nextgqa_pre_review_integrity_receipt.py`, which accepts an
independently pinned Ed25519 root key, a root-signed three-key roster, the
candidate byte inventory, and explicit role-to-file inputs. It checks the
freeze-owner signature and exact artifact sizes and hashes. The builder writes
only a new mode-0600 receipt inside an owner-owned mode-0700 directory; it does
not overwrite. CLI output contains aggregate status and counts only. The tool
does not generate keys, identities, or a real receipt.

A successful verification is `signature_valid_order_unverified`. Local time is
marked non-authoritative; opening, model loading, scoring, and execution
authorization remain disabled. It cannot satisfy the separate review-only
release gate, because there is no trusted timestamp or witnessed release-order
evidence format in the current protocol. The private candidate inventory still
has 50 present artifact roles and zero missing slots, but no independent trust
root/roster or release receipt exists and review has not begun.

The helper's roster binds only freeze owner, claim reviewer, and frame-pair
reviewer. It does not independently bind the custodian. A later release
procedure must record whether the freeze owner also serves as custodian or
include a separately trusted custodian identity/key; the helper does not
verify that assignment.

The synthetic verifier suite passed **17/17**; the existing synthetic
pre-review inventory suite passed **8/8**. Compilation and scoped
`git diff --check` passed. An independent read-only auditor found no
authorization or path/mode bypass; a file-descriptor cleanup issue found in
that review was fixed and covered by a regression test. No private package,
real key, or real receipt was opened or created. No model or GPU run occurred.

## Faithfulness status and limits

Direct generated-claim faithfulness remains **0/3 human-reviewed and 0/3
scored**, with no direct effect estimate, confidence interval, or test. This
integrity helper does not change that evidence. The head-confirmation branch
remains closed: top-4 at n=79 was −0.0756 nat (95% t CI [−0.1578,+0.0065]),
below its fixed +0.25 nat threshold. The existing top-8 n=3 exploratory smoke
used the same sample hash and is not an independent cohort. Do not reopen head
search with these already observed records.

## Next action

First establish the trusted root/role roster and independent review roles.
Then separately establish witnessed or trusted release order before opening
the offline review package. Complete all three claim reviews and the 15
browser-to-decoder frame-pair decisions. Only after those gates and a separate
score freeze pass may the all-three-item GPU score run. Rights status remains
unverified; all protected inputs and derived artifacts stay local.
