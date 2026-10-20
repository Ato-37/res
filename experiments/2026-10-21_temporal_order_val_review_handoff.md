# Current temporal-order validation cohort review handoff (2026-10-21)

## Purpose

The existing low-friction review artifact supported sequential reviewers and a
combined JSON export, but was bound to a different three-claim packet than the
temporal-order GPU experiment. Create a distinct local artifact that binds the
exact three validation questions, generated claims, and source videos used by
that run. The user's waiver allows exploratory experiments to proceed without
waiting for reviews; it does not create a human response or formal review.

## Build and binding checks

Built the artifact from the temporal-order smoke and remaining generation
packets using the existing Japanese quick-review renderer. Each source video
digest matched its packet before conversion. The three artifact question/claim
pairs matched the packet rows, and each embedded browser-compatible video
matched the corresponding private packaged video. The current artifact is
`data/private_nextgqa/quick_review_temporal_order_val_20261021_01.html`,
SHA-256 `a3133e2a24043f3b23947bc1c8ce9d0f6f2b3241a21500c1752a0acd92d51b89`,
mode `0600`; its package directory is `0700` and its files are `0600`. The
artifact embeds protected data and remains untracked/local.

The Japanese launcher and runbook now point to this artifact. The private JSON
response inspector's default artifact was updated to the same filename; older
responses must select their original artifact explicitly and cannot silently
join to these items.

## Verification

- Artifact-to-packet audit: 3/3 question/claim pairs and 3/3 packaged videos
  matched; the fingerprint is present.
- Focused builder and response-inspector tests: 33/33 passed during artifact
  creation; after redirecting the current launcher/default inspector to this
  exact artifact, builder/inspector/importer tests passed 48/48.
- Exact-artifact Windows Chrome E2E: 1/1 passed using synthetic answers. It
  exercised video decode, per-person nickname entry, sequential respondent
  reset, and one combined fingerprint-bound JSON export. No human response was
  created.
- Python compilation and scoped `git diff --check` passed after the launcher,
  runbook, and inspector-default update.

## Meaning and limits

This is a simple impression check for question clarity, answer fit, and whether
the explanation presents enough visible-video evidence. It does not verify
identity, independence, viewing completion, claim truth, or direct
faithfulness. Actual responses remain 0/3; direct faithfulness remains 0/3.
The exploratory temporal-order interaction remains the separate finite-cohort
result reported in
`2026-10-21_temporal_order_claim_dependence_val_replication.md`. Dataset/model
rights are unverified and all review inputs/outputs stay local.

## How to open locally

From the repository root, run:

```bash
xdg-open ./data/private_nextgqa/quick_review_temporal_order_val_20261021_01.html
```

On WSL with Windows desktop integration, the Japanese runbook contains the
corresponding `powershell.exe Start-Process` command. After each person finishes
three items, choose **「次の人へ」**, enter that person's nickname, and begin
again. At the end, choose **「全員分を保存」** once to download the combined
local JSON. The JSON contains nicknames; use pseudonyms and do not share it.

## Next action

Keep exploratory scoring independent of whether anyone completes this informal
artifact. If a response arrives, inspect it only against this exact artifact
fingerprint and keep it private. Do not count its ratings as formal Stage-1 or
Stage-2 evidence without the corresponding protocol and provenance.
