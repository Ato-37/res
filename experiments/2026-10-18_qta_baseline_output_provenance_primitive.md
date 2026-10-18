# Synthetic QTA baseline fixture calculation (2026-10-18)

## Purpose

Test the fixed-choice p/q arithmetic and structural row/input contract without
model execution. An independent audit found that the first implementation still
accepted and invoked an arbitrary model-like callable despite its
`synthetic_only` label, and permissively collapsed logits shapes. That version
was replaced. The current helper is a pure synthetic fixture calculator; it
does not claim score provenance and does not change any human-review, trust,
clock/frame, or phase gate.

## Implementation

The current `scripts/qta_baseline_output_provenance.py` accepts exactly three
ordered rows containing explicit `synthetic_logits_1x1_vocab_v1` fixtures and
freezer-produced `FrozenBatch` snapshots. It computes A–E probabilities only
from fixture logits with exact shape `[1, 1, vocabulary]`, then binds p/q, gold
margin, prompt/input/batch hashes, score position, row order, and supplied run
digest labels. The API has no model or callable argument and performs no model
forward. It marks fixture logits as synthetic and input rows as unverified; it
does not assert that prompts or batches are synthetic. All three row schemas
and declared input IDs against the frozen batch's actual CPU `input_ids` are
checked before fixture calculation.

The verifier checks the three-row order, run-digest pins, inner and outer
non-authorizing scope fields, and payload digest. Prompt text is hashed, but
its association with token IDs is
unverified: no tokenizer is run. Fixture math and structural hashes provide no
evidence that any model executed or produced the supplied fixture logits. A
caller can replace fixture values and recompute the digest; the digest is an
integrity check, not authentication.

## Checks and observed results

- Focused synthetic regression suite: **8/8 passed** with
  `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_qta_baseline_output_provenance -v`.
- Covered A–E fixture calculation and p/q, wrong row order, changed run pins,
  payload mutation, exact fixture shape, CPU tensor input-ID binding,
  row-three mismatch/invalid-schema rejection, API rejection of a model
  callable, and rejection of contradictory outer authorization/scope fields.
- No protected review material, real video, Qwen model, GPU, or direct research
  scoring was used. No efficacy result or real score-provenance evidence was
  produced. Rights status remains unverified.

## Interpretation, limitations, and next action

This is only a pure arithmetic and structural regression fixture. It does not
close score provenance and must not be described as forward-output binding. A
later production design still needs a frozen pre-baseline receipt and
independent gate, a trusted model/runtime code path, exact frozen configuration
and source pins, and a baseline-only runner that derives values from the real
forward. It must preserve all human, trust, prediction-link, and 15/15
clock/frame prerequisites before any protected-data scoring.

## Provenance

- `scripts/qta_baseline_output_provenance.py` SHA-256:
  `862728d8aeda24f5beb6360b97944b377957e50f4ec4820c222edd1b7d50586c`
- `scripts/test_qta_baseline_output_provenance.py` SHA-256:
  `d3c7cc6265a8796258a86c92be9d6812ebb500ebddd55e7b13fba921dd3b2fd5`
- The fixtures and result objects exist only in the unit test process. No run
  receipt or raw output artifact was created.
