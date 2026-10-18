# QTA source-capsule CPU validation

## Purpose

Prevent future local QTA runs from losing the exact source bytes needed to
reconstruct their recorded source hashes. A prior mask-specificity run's
aggregate arithmetic was independently reproducible, but the original scoring
runner was absent from each available source archive. This is a standalone
reproducibility utility and adversarial synthetic test, not a model experiment.

## Implementation and verification

Added `scripts/qta_source_capsule.py` and
`scripts/test_qta_source_capsule.py`. The API records an ordered allowlist of
relative source paths and per-file SHA-256/size in a stored ZIP capsule. It
requires owner-only private output permissions, rejects output inside a
detected Git worktree, and publishes a fully written/fsynced temporary file
atomically with no-replace semantics. Recovery validates owner/mode/link count,
the exact member set/order, archive limits, member metadata, path syntax,
manifest canonicalization, and content hashes; it returns bytes in memory and
never extracts paths. The optional previously retained archive hash supports
integrity comparison but is not an authentication mechanism by itself.

The CPU suite used only temporary synthetic source files. It covers exact
round-trip, changed source after capture, missing/extra/duplicate members,
traversal and path normalization, symlink/hardlink/special-file rejection,
tampering, permissions, overwrite/race refusal, partial-write cleanup, forged
oversize metadata, and early rejection of an excessive ZIP member count before
`infolist()` materialization.

Environment: Python 3.12.14. Command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_source_capsule -v
```

Observed: **20/20 tests passed**; `git diff --check` passed. An independent
review found three issues during iteration (final-name partial visibility,
source hardlinks, and ZIP central-directory allocation before the actual member
count was verified). All were fixed and the independent re-audit passed. Current
source hashes are:

- `scripts/qta_source_capsule.py`:
  `dbc03c5e26f3513d9008bf8b2c305dd0d245ffcfd20519db21d55380763629eb`
- `scripts/test_qta_source_capsule.py`:
  `5cb121c88a514226df09f663a21426c0bec7b119bc5647e26e8ca96eb4fc2448`

## Interpretation and limits

This establishes synthetic implementation behavior for this utility only. It
does not reconstruct the missing historical runner, prove the validity of any
scientific result, or show that a future runner imported or executed capsule
bytes. A prospective gate is now integrated into
`scripts/run_nextgqa_validation_qta_annotation_proxy.py`: its CLI requires an
explicit private capsule directory, the source capsule is captured and verified
before dataset/checkpoint access, and the run receipt binds only the capsule
basename, archive digest, and verified source hash map. A separate test file
checks capture/verification, source mismatch, capsule collision, CLI
requirement, and early receipt-name collision/symlink handling. The focused
capsule utility, integration, and runner suites passed **47/47 tests** after
the collision guard was added; `py_compile` passed. No model or real data was
loaded. Runner SHA-256 is
`3ebd42e2fc1125f81838996f02a7547e82dbba90ca8d523a800aa9d30cc903c9`; capsule
integration test SHA-256 is
`4d651509fa1baf39adc063a108f24fe3cbd9441cb61db3a9d943824bc7c80c8e`.

Same-UID attackers remain outside the protection boundary; source allowlists
and an externally preserved archive digest are required for meaningful
provenance. The runner gate proves only that the declared source bytes were
captured and verified before this process proceeds; it does not prove the model
executed capsule bytes from an immutable imported tree. No protected data,
video, model, or GPU was used. Rights status remains unverified and all
protected outputs remain local-only.

## Next action

Next, assess a disjoint, pre-outcome cohort and direct claim-linked measurement
only after the human review gates provide valid annotations. If an annotation-
proxy run is proposed before then, use a fresh predeclared cohort and preserve
its exploratory label; the previously used 30-row validation pilot and 490
test-video pool are closed to additional outcome-driven searches. Keep prior
results exploratory and do not claim recovered provenance retroactively.
