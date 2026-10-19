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

## Role-key creation instructions and safety checks (2026-10-19)

The Japanese assignment sheet now gives each actual role holder a WSL/Linux
command to create their own Ed25519 key pair, retain the private key in a
mode-0700 per-user directory, and pass only the mode-0600 public key to the
trusted root authority. Key IDs are restricted to the CLI's accepted
characters and length. The base key directory must already have mode `0700`
or be created that way by the restrictive umask; detected symlink paths are
refused, and permissions on existing directories are not changed. The new
per-key directory is created exclusively so an existing key ID cannot be
overwritten; `set -euo pipefail` and an exit trap stop on signing-key errors
and remove partial output. The root key is expressly outside this procedure
and must be held by the independent root authority.

The embedded shell block passed `bash -n`. A temporary-only synthetic run
verified successful Ed25519 generation and 0700/0600 permissions, produced a
64-hex-character fingerprint, refused a second run without changing the
existing key, refused a path-traversal key ID, cleaned a simulated OpenSSL
failure, and rejected both a symlink key directory and a preexisting
non-private directory without changing their targets or permissions. No real
role keys were generated.

The independent auditor rechecked the final instructions and 96-role
candidate, including exact hashes for both Japanese documents and zero stale
registered code/source roles. It confirmed the candidate remains
non-authorizing. The inventory builder, signed-receipt verifier, and
release-order auditor suites passed **54/54 tests** together after registering
the Japanese assignment sheet. The command assumes the user's home directory
is not under hostile same-user concurrent modification; it does not defend
against a process already controlling the same account.

The Japanese assignment sheet itself is now an inventory-bound source role.
The fresh candidate at
`data/private_nextgqa/inventory-jp-keyinstructions-r2-20261019-1533/`
contains **96 roles**, has **0 missing slots**, matches the current bytes of
both Japanese handoff documents, and remains `candidate_only` with all review,
freeze, scoring, and execution flags denied. Its inventory SHA-256 is
`140c708f23d8ba542c63997be8d074a0b38327619cb33f0bb5798e6f688fd007`.

These commands prepare real reviewers to create keys; they do not establish
which person controls a key. The root pin, actual role-to-person assignments,
signed rosters, release witness, 3/3 Stage-1 review, 15/15 clock/frame
correspondences, and 3/3 Stage-2 review remain absent. Direct faithfulness stays
at **0/3** scored, with no effect estimate, interval, or test. GPU scoring
remains gated; the new work is handoff preparation only.
