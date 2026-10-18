# Synthetic atomic QTA input adapter (2026-10-18)

## Purpose

Add a small CPU-only join over the existing exact-byte QTA source and review
leaf validators. Synthetic fixtures are used in tests, but caller input
provenance is not authenticated. The adapter is non-authorizing and unable to
run a model or score. It does not complete the
production preflight or resolve any human, identity, clock/frame, trust-root,
or phase-release gate.

## Implementation

Added scripts/qta_synthetic_atomic_preload_adapter.py. It accepts the exact
byte/pin maps already required by diagnose_qta_prebaseline, reruns that
diagnostic, then reruns the two leaf APIs to reconstruct the fixed ordered
three-row source/review join in memory. A successful result includes per-row
hashes and an ordered cohort digest; it emits no raw text, IDs, paths, or
partial hashes on failure. All result paths hard-code
model_load_permitted=false, score_ready=false,
execution_authorization=not_granted, forward_count=0, and score_calls=0.

The function cannot prove that caller bytes are synthetic or authenticate
human review. Its success status means only that these exact caller-supplied
bytes passed the existing structural validators and ordered join. The output
classification deliberately says only that the fixture was caller supplied
and not independently authenticated.

## Verification

Synthetic-only tests use the existing synthetic source and review fixtures:

    PYTHONPATH=scripts .venv/bin/python -m unittest test_qta_synthetic_atomic_preload_adapter -v

Result: **6/6 passed**. Positive coverage verifies all three ordered row
bindings, aggregate hash, non-disclosure of fixture text/IDs, and unconditional
denial flags. Negative coverage verifies missing source/review inputs, a
reordered review row set, and a tampered source configuration hash all reject
the entire cohort with zero validated rows and no partial hashes. Stateful
mapping regressions verify caller inputs are snapshotted once before digesting
and every validator/row join consumes that same snapshot; arbitrary exception
text is reduced to a fixed generic failure code. No protected
package, receipt, video, checkpoint, model, CUDA, or forward was accessed.

## Limits and next action

This is a human-free structural adapter, not a production preload gate. It
does not consume authenticated stage-1 custody/release evidence, raw clock
capture/index/anchor-plan bytes, verified browser-to-decoder frame
correspondences, actual processor batches, a frozen baseline p/q run, observed
QTA selector attention, nine branch execution receipts, or prediction-link
review. The existing synthetic 9-forward smoke remains capacity/wiring evidence
only. Keep this adapter disconnected from any model-loading path. The next
safe step is a separate synthetic-only composition test for the clock and
processor row adapters once their exact byte-level trust inputs and role/release
contracts are specified; no real scoring is justified by this artifact.
