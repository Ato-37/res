# QTA direct-faithfulness pre-load structure diagnostic — 2026-10-19

## Purpose

Reduce the risk of treating the fragmented source/review adapters as one
complete scoring gate. Add a CPU-only diagnostic that re-runs the existing
actual-candidate bridge from raw inputs and makes all still-missing formal
review/scoring gates visible in one sanitized result.

## Implementation and checks

- Added `scripts/qta_direct_faithfulness_preload_gate.py`. It accepts raw bridge
  inputs, invokes `build_qta_actual_candidate_input_material`, accepts no
  caller-created validator success report, and returns only a fixed aggregate
  schema. Exceptions and failures leave all row counts at zero.
- Structural success is labeled
  `structural_inputs_composed_unvalidated`; the three rows are counted only as
  `structurally_composed_rows=3`. `validated_rows` remains zero because no
  direct-faithfulness score rows have been validated.
- Trusted human review, semantic review, clock/frame correspondence (0/15),
  baseline/prediction-link, and processor-batch provenance remain blocked on
  every path. The module has no model, CUDA, forward, scoring, or authorization
  path.
- Synthetic focused command:

  ```bash
  cd scripts && ../.venv/bin/python -m unittest -v \
    test_qta_direct_faithfulness_preload_gate \
    test_qta_actual_candidate_input_bridge
  ```

  Observed: **27/27 tests passed**. New tests check aggregate labels, no
  authorization, unchanged raw inputs passed to the bridge, and fail-closed
  behavior for tampered/missing inputs. `py_compile` and scoped
  `git diff --check` also passed. Fixtures are synthetic; the wrapper test
  stubs the bridge's leaf validators.
- An independent source/test audit agreed that the implementation stays
  non-authorizing and identified no critical issue after the row-count label
  was tightened. That audit also confirmed this is not a production pre-load
  gate.

## Interpretation and limitations

This is a small orchestration/readiness diagnostic, not an inference runner.
The upstream bridge composes source, review-row, and clock-calibration
candidate structure, but its successful return does not prove human identity,
semantic correctness, or correspondence to decoded frames. Root keys and
expected fingerprints still arrive from caller inputs; their out-of-band trust
origin is not authenticated here. The wrapper does not compose the Stage-1
release/custody verifier, the complete 15-anchor clock/frame evidence audit,
baseline `p/q`, prediction-link receipt, or the signed live processor-batch
binding into an identity-preserving score transaction. It cannot guard a
future model loader because it has no loader/scorer path.

Therefore a structural pass must not be interpreted as permission or readiness
for a direct GPU score. No protected review data, human judgment, model, CUDA,
or GPU was used in this cycle. Rights remain unverified and data remain local.

## Faithfulness progress and next action

Direct evidence remains **0/3 formally reviewed claims**, **0/15** verified
player-clock/decoder-frame correspondences, **0/3** prediction-link judgments,
and **0/3** direct scores; no direct `ΔF`, interval, or test exists. The
preregistered head confirmation remains closed at 79/79, −0.0756 nat, 95% t CI
`[−0.1578, +0.0065]`, below its +0.25 nat practical threshold. Existing real-
video QTA mask and generated-text-likelihood runs are annotation-dependence
proxies, not direct explanation-faithfulness or hallucination evidence; their
fixed cohorts remain closed.

Next, complete an independently rooted production pre-load transaction only
when the exact role/release inputs are available: re-run Stage-1 release and
claim review, all 15 clock/frame/calibration records, baseline `p/q`,
prediction-link review, and signed live-batch binding from exact bytes; retain
the same batch objects through a one-shot guarded scorer. Until that sequence
is implemented and its real inputs pass, do not run a direct-faithfulness GPU
score. Continue independent non-repeating implementation/audit work in the
meantime.
