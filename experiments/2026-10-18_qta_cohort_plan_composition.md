# QTA cohort plan composition and temporal-key audit (2026-10-18)

## Purpose

Connect the existing fixed-three-row review/clock/processor/baseline structural
join to the QTA nine-branch plan aggregator. This is an engineering
reproducibility check, not a real-data experiment or authorization to score.
An independent audit found that the previous plan aggregator checked the
temporal group count and selected evidence/control group indices but did not
compare the plan's token positions for each temporal group with the processor
mapping. It also accepted Python booleans where integer schema/count fields
were expected.

## Changes

- `qta_prescore_processor_row_adapter.py` now projects the mapper's ordered
  per-temporal-group visual token positions into the processor row, while
  excluding its private tensor snapshot.
- `qta_prescore_mapping_binding.py` requires those positions in the expected
  processor-group pins, checks them against the processor row, and includes
  them in each row digest. It requires nonempty sorted equal-size groups that
  form one contiguous video-token block.
- `qta_three_row_plan_aggregate.py` requires the pinned temporal positions and
  checks that every plan uses those exact groups. It fails the fixed cohort on
  malformed geometry or a mismatch. It also enforces the selected primary
  treatment's fixed decoder layers 5–16 and visual-key bias `ln(2)` across the
  complete cohort; a cohort-wide alternate setting is rejected.
- `qta_nine_branch_plan.py` strictly checks integer schema/count fields and
  hash value types, preventing `True` from passing as integer `1` and rejecting
  non-string hash values.
- `qta_synthetic_cohort_plan_preflight.py` composes the structural row join
  and plan aggregator while keeping all provenance pins caller-supplied and
  unauthenticated, and explicitly returns no scoring/execution authorization.

## Checks and observed results

- Focused structural tests: **30/30 passed** after the geometry hardening.
- Final full scripts suite passed **1,583/1,583** in **35.476 s**. Expected
  negative-path test diagnostics and two Transformers deprecation warnings
  appeared; the suite exited 0.
- `compileall` and `git diff --check` passed after the final geometry edits.
- Independent adversarial audit found and prompted the exact temporal-position
  binding fix and strict integer checks. A follow-up audit verified the new
  invariants against `qwen_review_interval_token_groups.py`: its output groups
  are equal-size slices of one contiguous ordered video-token block. The audit
  also identified and drove direct tests for unequal group sizes and a gap.
  A separate protocol audit checked layer endpoints and one-ULP deviations on
  the bias and confirmed the frozen treatment check matches the selection log.
- No GPU run was performed for this change: it modifies CPU-only structural
  validation, and a model forward would not test the changed contract.

## Faithfulness progress

**Direct evidence: `未実施`.** This change measures no faithfulness outcome.
The preregistered QTA cohort remains 0/3 scored rows; human claim review is
0/3, prediction-link adjudication is 0/3, and verified player-clock to decoder
frame correspondence is 0/15. There is no real-data `DeltaF`, confidence
interval, hypothesis test, or multiplicity adjustment. The previous synthetic
GPU contract smoke is engineering-only and does not change these counts.

This structural pass is only over synthetic fixtures and caller-supplied
digests. It does not authenticate source videos, reviewer identity or truth,
clock custody, prediction-link statements, processor provenance, or model
execution. Rights status remains unverified and protected data remain local.

## Next action

Run the final full scripts test suite and independently inspect the diff and
hashes. Then use this composed preflight only after authentic review, prediction
link, jointly frozen clock and frame-correspondence materials exist. Keep model
load and treatment scoring denied until those separate checks are implemented
and independently audited. Do not resume head/layer/alpha search.

## Source hashes

Final SHA-256 values:

- `qta_nine_branch_plan.py`: `29c7b77f9e0ae9d75670a71bbfdc82b1518dd9de14e631a647de857ced869206`
- `qta_prescore_mapping_binding.py`: `9f8adde229851260f248e635a5b7eaaf31fd69fa9b6e6901f2913acf6a108524`
- `qta_prescore_processor_row_adapter.py`: `20665f330395376ec2509f9e5008adb9e34beb714253b920dafdbfec3644b55d`
- `qta_three_row_plan_aggregate.py`: `7b3a21926ce51956dd1344f463710b442ce8adc6e74d54b62abc63c7ac179e2e`
- `qta_synthetic_cohort_plan_preflight.py`: `035fb4221e79c6af2d1558e62e3efd70f87e71df4715dc6d5b65e048f84afc61`
- Five focused test files: mapping `fafeeea13e52e534b8a5e305be4dbfedaf820686d60b21ab5c4eb9c98e41472d`, processor adapter `2c736b432848b1216e0f14902cdbd5f91b82cde32384d7f06879bdc058ee444b`, mapping preflight `d5d2ad0a5a5bddcee89daac7b073a8ff886337d9fb02c73a616064cf99370d49`, plan aggregate `c0300130a6c9fc4e353c7cd2ad451770c304c3480707303ba3a714d3cf84b4a5`, composed preflight `2a7e61f1044b8af55ce11dd398c004fdd1e912eb669a64e1e3a920a4a4dcfb6a`.
