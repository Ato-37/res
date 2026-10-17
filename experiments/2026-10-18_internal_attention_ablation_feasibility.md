# Internal visual attention-access ablation feasibility (2026-10-18)

## Purpose

Prototype a model-internal evidence/control ablation compatible with the
fixed-prediction margin contrast in
`docs/internal_intervention_grounding_faithfulness_plan.md`. The existing six
condition GPU smoke uses black-masked video inputs and therefore does not
implement the planned internal ablation. This implementation is a CPU-tested
mechanism primitive only.

## Operation and setup

`scripts/qwen_internal_attention_ablation.py` clones the Qwen eager 4-D
additive attention mask and sets caller-supplied key positions to `-inf` on one
fixed causal answer-query row, leaving all other rows and positions unchanged.
The operation is broadcast across attention heads by the `(batch, 1, query,
key)` mask. It rejects non-square masks, malformed positions, selected keys
that were already masked or biased, NaN/+infinity anywhere, and masks with no
zero-valued causal key remaining. Hook setup freezes the position list into a
tuple and provides cleanup.

This follows the installed Transformers 5.18 Qwen2.5-VL eager implementation:
the 4-D additive mask is added to query-key logits before softmax; valid entries
are zero and invalid entries use `finfo(dtype).min`. An independent code audit
confirmed this source behavior and reviewed the helper/tests separately.

## Verified results

- Focused tests: **5/5 passed** after the final audit repair.
- Tests cover exact fixed-query masking, non-target preservation, already
  masked and biased selected keys, NaN/+infinity rejection, non-square cache
  shape rejection, `finfo.min` all-masked detection, list freezing, and hook
  removal.
- `py_compile` passed for the helper and tests. No GPU and no dataset were used
  for this primitive.

## Interpretation and limitations

This operation measures only direct access from the selected answer-query row
to selected visual key/value positions in selected layers. It does not remove
information already propagated through other visual/text states and cannot
establish full evidence necessity or explanation faithfulness. The helper does
not itself map a reviewed time interval to Qwen temporal-grid token groups,
authenticate a reviewer, freeze equal-budget evidence/control sets, authenticate
model/tokenizer provenance, or authorize scoring. The caller must use
`use_cache=False`; a square mask alone cannot prove cache state. `-inf` behavior
is tied to the tested eager backend and must be revalidated for any other
attention implementation.

No treatment effect, real-data score, or causal efficacy result exists from
this work. Human review remains incomplete, so direct faithfulness progress is
still 0/3 independently reviewed and 0/3 scored, with no real `ΔF` estimate,
interval, or test.

## Next action

Build a separate no-model contract that maps contiguous processor temporal-grid
groups to reviewed PTS intervals, freezes equal-count evidence/control groups
without using outcomes, and supplies identical groups to baseline and treatment
branches. Validate that mapping against the exact installed processor and
decoder implementation. Then create a synthetic Qwen integration test that
combines this ablation with the candidate internal attention-bias hook and
verifies finite output plus exact masking. Do not interpret it as a faithfulness
result or run real-data scoring until independent claim review and
browser-clock-to-decoder-PTS correspondence pass.
