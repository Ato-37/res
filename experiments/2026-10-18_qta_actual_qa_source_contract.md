# QTA actual QA source binding contract (2026-10-18)

## Purpose

Add a CPU-only, non-authorizing contract that joins exactly three ordered QTA
claim rows and opaque review tokens to the selected NExT-GQA CSV rows through
the existing source adapter.
This closes a source-material wiring gap before any later baseline run; it is not
human review, prediction-link adjudication, a score freeze, or faithfulness evidence.

## Implementation and checks

`scripts/qta_actual_qa_source_contract.py` accepts exact protocol and config bytes
and caller-supplied expected SHA-256 pins. It checks the pin values and validates
claim text digests and claim IDs against lineage, unique review tokens and
ordered claim/selection/lineage/target joins, including the target's claim-text
digest, then delegates question/choice/answer binding and exact CSV byte
validation to `nextgqa_actual_qa_source_adapter.build_actual_qa_source`. It also
compares the recomputed qid, source-row, question, ordered-choice digests, and
gold index against those already frozen in the supplied target candidate; it
fails closed instead of silently repairing a stale or tampered candidate.
The fixed denominator is three. Any failure returns zero validated rows and no
partial private bundle. Success returns QA-source JSON only through a clearly
marked in-memory private bundle; the public summary contains statuses and
counts, not private source hashes, row identities, review tokens, question,
claim/answer text, choices, or gold indices. The private bundle retains the
source hashes and ordered review-token-to-source crosswalk for later local joins.

The public contract explicitly leaves the prediction link unfrozen: p/q can only
be established by a later, separately authorized baseline full-video output. The
primary score must use that frozen baseline p/q; the dataset gold answer index is
secondary. This binder always reports model-load/execution/scoring authorization
false and all forward/score-call counts zero, on both success and failure.

Synthetic-only tests:

```text
PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_actual_qa_source_contract -v
```

Result: **9 tests passed**. They cover successful ordered binding and summary
privacy, caller-pinned protocol/config mismatch, claim and selection tampering,
target claim-text digest mismatch, invalid source-video digest, target reordering,
missing/duplicate/changed review tokens, five frozen-target source-field tampering
cases (including choice order and gold index), changed CSV choice order even when
the CSV digest pin is updated, duplicate JSON keys, and CSV hash mismatch. The
focused unittest suite and `git diff --check` passed. The complete scripts suite
passed **1,630/1,630** before the final extra CSV-choice-order test; rerun it
before integration.

## Evidence and limits

No protected dataset source, private video, model, GPU, or real QTA input was read.
No actual source-binding result, baseline p/q, internal intervention score, ΔF,
confidence interval, or human-review count is produced. The hashes in tests are
synthetic fixture hashes, not authoritative QTA pins. A caller must provide the
protocol/config bytes and their independently frozen expected digests. This
module verifies byte pins only; it does not interpret the protocol/config schema,
verify human identity or review quality, validate a release-order receipt, verify
clock calibration/custody, or grant any upstream/downstream authority. Private
bundle bytes must remain local and must not be placed in a public summary or
published artifact.

## Next action

Integrate this binder into the CPU-only all-three-row prebaseline readiness
report. That report must independently verify upstream release-order, review,
custody, clock/calibration, sampler, processor-batch, and synthetic-preflight
artifacts, then remain explicitly non-authorizing. Only after those upstream gates
and human review are complete can a separate approved baseline path freeze p/q.
