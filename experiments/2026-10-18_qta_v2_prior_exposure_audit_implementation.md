# Frozen v2 cohort prior-exposure audit implementation (2026-10-18)

## Purpose

Prepare an outcome-blind, local-only audit of the exact 147-video evaluation
cohort in the frozen QTA v2 plan against the existing partial prior-exposure
HMAC index. This is a cohort-integrity diagnostic. It does not change cohort
membership, exclude or replace matches, certify unmatched media as fresh, or
measure model faithfulness.

## Implementation and verified checks

Added `scripts/audit_nextgqa_v2_cohort_prior_exposure.py`. The runner binds the
exact private v2 plan and preflight bytes, checks the four current NExT-GQA
dataset hashes against both artifacts, preserves the 147-row denominator, and
checks the plan/preflight row bindings. It pins both exact files through the
mode-0600 `_17` local input-freeze receipt and rechecks all four dataset hashes.
It accepts only media-SHA256 HMAC
matches because the index's dataset-video namespace is caller-declared and
not crosswalk-authenticated. It does not emit or persist IDs, QIDs, paths,
questions, media SHA-256 values, or HMAC tokens. The private JSON output uses
owner-only stable reads, rejects symlinks and changed files, and is created
with `O_EXCL` and mode `0600` under the owner-only private directory. The
receipt binds the plan, preflight, input-freeze, dataset, auditor-source,
prior-index, and source-roster hashes; it marks the index as known partial and
states that unmatched does not mean fresh. The local freeze is an owner-
controlled integrity anchor, not an external signature or trust root.

Synthetic checks passed **9/9**. They cover aggregate match/unmatched counts,
redaction, plan/preflight/dataset hash agreement, rejection of a self-consistent
joint plan/preflight substitution against the frozen input gate, media hash
mismatch, symlink and path-boundary rejection, and private exclusive output.
The first real-input attempt failed closed before receipt creation because this
preflight intentionally exposes only each row-binding hash, not a separate
`media_sha256` field. The auditor was corrected to compare ordered row-binding
hashes and to require exact plan/preflight hashes from the local `_17` input
freeze. An independent code audit confirmed the substitution fix. `py_compile`
and `git diff --check` passed. Tests used no protected files.

Source SHA-256 values:

- Runner: `356605cc3cad70aca49af2aa223b42cf5b4de8af8d39e5f0932dab6b4008fcc6`
- Tests: `f24f9918936b84073dc6fdce705b2521f8f2ff0f951490370c9988c9f616ee29`
- Input-freeze `_17`: `c514ed7d9dc87bf36390bae8a691c923849e07895ccfc4edec6c118994c1b2cb`

## Planned execution and limits

The corrected local-only command was executed:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/audit_nextgqa_v2_cohort_prior_exposure.py \
  --repo  \
  --output-name nextgqa_v2_prior_exposure_aggregate_local_20261018_02.json
```

The expected bounded workload is **147 media files** hashed once by the audit,
plus the four current dataset files and small private plan/preflight/index/key
inputs. The frozen candidate-pool audit recorded 1,990,584,365 bytes across
490 files (about 4.06 MB per file); applying that pool average to 147 files
gives a rough planning estimate of **about 0.60 GB**, not a measurement of the
selected v2 subset. The runner enforces a 512 MiB per-file cap and an 8 GiB
cohort cap. It performs no video decoding, model load, CUDA operation, or
forward pass. The first attempted execution failed before media hashing or
receipt creation; the corrected run then completed.

## Verified local result

The 147-row frozen cohort completed with **147/147** distinct media files
hashed (**584,424,771 bytes**) and **0** exact-byte HMAC matches to the supplied
partial prior-exposure index; consequently **147** rows were unmatched to that
index. The primary agent independently rehashed the same 147 files and
recomputed the HMAC comparison, reproducing all counts exactly. The private
aggregate receipt is mode `0600`, SHA-256
`73db890e689c5f39a45b876e40cb7a7692278eda01b923e2fc2128dd0b3bee4b`.
Its input-freeze and auditor-source hashes match the current files. It contains
no IDs, QIDs, paths, QA text, media hashes, or HMAC tokens.

This does **not** show the videos are fresh: the index is known partial and
unauthenticated, and the match is only exact bytes under one key and one
canonicalization. Re-encoded duplicates, shared events, related footage, and
unindexed prior exposure remain unknown. The audit changed no cohort membership
and scored no faithfulness or model outputs. Rights status remains unverified,
with all material and outputs local.

An exact-byte match supports exclusion knowledge only against this supplied
index and the same HMAC key/canonicalization. The index is explicitly partial
and unauthenticated; unmatched media remain unknown, and the audit cannot
detect re-encoded copies, shared source events, or related footage. It makes
no change to the frozen cohort and does not establish source/event
independence, rights, annotation validity, explanation faithfulness, a ΔF
estimate, uncertainty interval, or significance result. Direct claim-linked
status remains 0/3 reviews, 0/15 verified player-clock/decoder correspondences,
0/3 prediction-link judgments, and 0/3 scores.

## Next action

Continue preparing the independent human-review handoff and other human-free
protocol validation. Direct claim-linked progress remains 0/3 reviewed claims,
0/15 verified player-clock/decoder correspondences, 0/3 prediction-link
judgments, and 0/3 scores. Do not call index-unmatched videos fresh or use
these counts as efficacy evidence. Keep the receipt local because
prior-exposure aggregates may reveal information about protected media.

After strict `source_count` type validation was added, the same frozen audit
was rerun without changing selection or interpretation. Receipt `_03` again
records 147/147 unique media files and 584,424,771 bytes, with 0 exact matches
and 147 unmatched to the supplied partial index. The primary agent rehashed
the artifact and independently reproduced the counts. The new private receipt
is mode `0600`, SHA-256
`efe80787cb44e09d1e8738584cd2e34bc6018542bcebad1ff9f3f72ea5c71e2f`. Unmatched
still does not mean fresh.
