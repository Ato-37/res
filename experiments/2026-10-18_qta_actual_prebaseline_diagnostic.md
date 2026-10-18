# QTA prebaseline input diagnostic (2026-10-18)

## Purpose

Provide a fixed-three-row, CPU-only diagnostic for the byte-oriented parts of
QTA input preparation. This is neither an execution gate nor a readiness claim.
It reruns the QTA actual-QA source binder and the QTA review-row adapter from
exact raw bytes and caller-supplied expected digests, then checks their ordered
token/video-hash join. It accepts no precomputed success reports.

## Implementation and synthetic verification

`scripts/qta_actual_prebaseline_diagnostic.py` emits one status per component:
`passed`, `blocked`, or `not_supplied`. Every component retains requested rows
at three; successful byte checks count three, and a failure counts zero. The
summary omits deterministic digests of raw source/review bundles as well as row
IDs, claim/question/answer text, choices, interval details, and gold indices.
All component provenance is labelled
caller-pinned/unverified or not-supplied/unverified. Even a complete structural
byte join remains `diagnostic_incomplete`; all model, scoring, phase, forward,
and score-call authorization fields remain false or zero. Structural review
validation never changes the separate human identity/semantic proof gate from
`not_supplied`.

Focused synthetic test command:

```text
PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_actual_prebaseline_diagnostic -v
```

Result: **7 tests passed**. They exercise a consistent three-row synthetic join,
missing-input reporting, a wrong protocol/config pin, a wrong review-lineage
pin, explicit omission of all raw/pin digests from the report, and rejection of
oversized source and review inputs before binding. An independent audit
identified the linkable-digest concern; the report now emits no source or
ordered-join digest. The pre-review inventory builder registers both this
diagnostic and its direct review-adapter dependency. A fresh local candidate
using the repaired page/package contains **76 artifacts, 0 missing slots** and
remains `candidate_only`. A second falsification test builds independently
valid source and review bundles that refer to different claims over the same
videos; both leaf validators pass, but the cross-bundle claim ID/text-hash join
rejects all rows. No protected source text, private video, model, GPU, or
external trust material was accessed by the synthetic tests. `git diff
--check` was run for the new files.

## Observed result and limits

Synthetic source binding, review-byte structure, and their ordered join each
passed 3/3 in the positive fixture. This proves only that these synthetic bytes
agree under the existing adapters. The diagnostic deliberately leaves these
gates `not_supplied`: stage-1 release/custody, clock-frame evidence, clock
calibration, processor-batch binding, prescore mapping, baseline full-video p/q
freeze, stage-2 prediction-link semantic review, and human identity/semantic
proof. Therefore there is no baseline prediction, internal intervention score,
ΔF, uncertainty estimate, or direct faithfulness evidence.

The expected QTA protocol/config digests and expected review-lineage digest are
caller-pinned inputs, not trust-root-authenticated values. The current adapters
do not provide an authoritative score-freeze contract. The report must not be
used to authorize any model or scoring work, and it does not establish actual
human review from structurally valid review declarations or signatures.

## Next action

Once a separately controlled owner supplies and freezes the missing upstream
trust pins and actual review evidence, add narrowly scoped byte-level checks for
clock/frame evidence and calibration. Keep baseline p/q freeze and stage-2
prediction-link review as separate gates; do not infer either from this report.

## Provenance

- Diagnostic SHA-256:
  `7eb8516c1d3e01ddcd8b3b0461de6b73df8e57341e6da1dd28226f071793b739`
- Focused-test SHA-256:
  `b47b48590dc1fe3fd63056842a40a37947c4e14820139b9a29e5b6c95b8bbbeb`
- Inventory builder SHA-256:
  `3f55237cab35199411f88bf4af93feb44da83dcf0a75004b61a2a4c867f34b91`
- Inventory test SHA-256:
  `18de3b341174d8fda59cddfb1d239beff1b4fb723fdcaf145efa3d996b381533`
