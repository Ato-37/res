# Claim-mask protocol validator hardening (2026-10-14)

## Purpose and gate

The NOAA claim-selective branch remains `no_go`: the human-review attestation
is incomplete, and NExT-GQA item-level rights remain unresolved. This CPU-only
cycle hardens the future protocol's structural validator without authorizing
inference or changing the scientific estimand.

## Changes

`validate_claim_mask_protocol.py` now rejects boolean `schema_version` values
(`True == 1` in Python), requires `runtime.source_commit` to be a full lowercase
40-hex Git commit, requires model and tokenizer revisions to use the same
immutable 40-hex identifier form, rejects split-overlap counts above the
declared video count, and requires distinct selected/control condition labels.
Malformed JSON-shaped protocol stages return a failure summary instead of
raising from set membership. JSON loading closes its file descriptor; duplicate
keys and nonfinite numbers are exercised through generic CLI failure paths. A
valid structural result still declares `execution_authorization: not_granted`.
These checks do not authenticate attestation records, referenced artifacts,
dataset rights, semantic spans, or runtime behavior.
The optional digest accepted by the imported `validate()` function is also
validated before it can be echoed; malformed digests fail generically and are
omitted from output.

## Verification

The updated focused validator suite passed 13 tests, including the valid
structural-only fixture, incomplete review, claim-conditioned sampling,
boolean schema version, malformed commit IDs, impossible overlap count,
mutable model/tokenizer revisions, malformed list/dict/null stage values,
identical intervention labels, attempted execution authorization, duplicate
JSON-key and nonfinite-number rejection, and a valid CLI result that still
cannot authorize execution. It also rejects malformed caller-supplied input
digests without echoing them. An independent final audit found no blocking
structural defect after these changes. The separate faithfulness protocol
validator suite passed 17 tests. The current 32-frame cache runner suite passed 11 tests.
Python byte-compilation and `git diff --check` passed. No GPU or private review
data was used for this validator cycle.

## Interpretation and limits

These are structural fail-closed checks only. A valid result is not evidence
that a reviewer signed the referenced record, that hashes resolve to suitable
artifacts, that the selected claim is semantically grounded, or that the
intervention measures faithfulness. The validator does not unblock the gated
claim-selective run. This schema now requires immutable 40-hex model/tokenizer
IDs; if a future approved model source cannot provide one, its revision format
must be explicitly designed before validation rather than silently accepted as
a mutable label. The overlap-count upper bound assumes the count represents
distinct overlapping videos, which should be retained in future schema docs.

## Faithfulness progress and next action

Direct evidence is unchanged: the closed 79/79 head confirmation remains at
`−0.0756 nat` with 95% t CI `[−0.1578,+0.0065]`, below the `+0.25 nat`
practical threshold. The 32-frame GPU run is an engineering cache/position
diagnostic only. This validator cycle adds zero efficacy samples and zero
faithfulness estimates. Keep claim-selective inference `no_go` until the NOAA
attestation and chosen dataset rights are resolved. Next, independently audit
the validator edits and inspect the proposed claim-mask estimand/control
wording for an ambiguity that can be resolved before any review gate opens.
