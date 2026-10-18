# Stage-1 to full Phase-2 verifier composition (synthetic CPU)

## Purpose

Exercise `verify_qta_phase2_plan_candidate` using a single invocation of the
real Stage-1 source-pin adapter fixture and its paired synthetic QA inputs,
carried through baseline-v3, a Phase-2 plan candidate, signed phase receipts,
and capture-sidecar/call-receipt validation. This closes a serialization and
binding integration gap; it is not an inference or faithfulness experiment.

## Setup

- New test: `scripts/test_qta_stage1_full_phase2_verifier_same_source_composition.py`.
- The Stage-1 adapter and QA fixture are inherited from
  `test_qta_stage1_baseline_prediction_link_composition` through the existing
  same-source Phase-2 plan composition test.
- The signed CPU runtime descriptor is fixed before building baseline-v3. Its
  exact byte hash is written into baseline `run_digests.runtime_sha256`; the
  Stage-1 and Phase-2 receipts are newly signed over the resulting exact
  artifacts using the existing temporary synthetic keys and rosters.
- `processor_batch_sha256` is inherited as a synthetic per-token placeholder
  and reused consistently across baseline/preflight/plan/capture joins; it is
  not a digest of an actual typed processor batch.
- The plan and capture bindings derive from the same baseline rows and
  preflight input. Attention arrays are fabricated, not produced by a model.
  Synthetic binary-fraction group masses are used so float32 query/head
  averaging reproduces the declared provenance bytes exactly.
- The call receipt contains `forward_calls=1` as a schema fixture value only;
  no model forward occurred. No GPU, model, protected data, or external service
  was used. No files outside the two new deliverables were edited by this task.

## Verified results

Command:

```bash
cd scripts && ../.venv/bin/python -m unittest -v test_qta_stage1_full_phase2_verifier_same_source_composition
```

Observed: 6 tests passed in about 7 seconds. The run includes the two inherited
same-source plan-candidate checks, one full signed-verifier positive case, and
one full-verifier sidecar-byte tamper rejection. The positive case validated 3
plan rows, 3 capture sidecars, and 3 call receipts; both receipt signatures and
recursive Phase-1 verification passed. The report remained
non-authorizing (`authorized_forwards=0`), with
`forward_source_authenticated=false`,
`attention_capture_authenticated=false`, and
`plan_semantics_authenticated=false`.

## Interpretation and limits

The test establishes that one synthetic Stage-1/QA fixture can be joined by
exact bytes through baseline-v3, plan construction, signed receipt envelopes,
runtime binding, attention sidecar arithmetic, and call-receipt indexing. A
modified sidecar payload is rejected. This does not establish that the
synthetic scores are valid, that a model executed, that an attention hook
captured a real forward, that reviewers or signers are independent, that the
data are authorized, or that any explanation is faithful. The temporary test
keys/rosters are scaffolding, not an external trust ceremony. The verifier
only byte-binds the `stage2_prediction_link_review` role as a Phase-2 artifact;
it does not invoke a reviewer/attestation validator, so this test establishes
no Stage-2 review validity. The test also does not run
`verify_qta_phase2_plan_candidate` on real-data/model artifacts.

## Next action

Keep this composition as a CPU integrity regression. Before any real capture
is treated as evidence, execute the actual runtime path and replace fabricated
attention/call reports with authenticated capture evidence; preserve the
explicit non-authorizing state until the human review and release process is
completed.
