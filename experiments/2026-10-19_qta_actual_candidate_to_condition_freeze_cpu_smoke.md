# Actual-Qwen candidate-to-condition-freeze composition smoke (synthetic CPU)

## Purpose

Close the specific untested adapter seam between the actual-processor full-video
candidate rows and the existing 3 x 3 condition-batch/freeze path. Existing
`test_qta_synthetic_adapter_composition.py` covers the broader condition-builder,
mapper, binding, freezer, and fake-scorer composition with a deterministic fake
processor. This smoke instead checks actual cached-Qwen processor batch identity
for synthetic clips and hands the resulting nine condition batches to the
existing atomic freezer. It is an engineering check, not a faithfulness test.

## Setup

- Date: 2026-10-19 (Asia/Tokyo).
- Runtime: repository `.venv`, CPU; PyTorch CUDA was not initialized.
- Processor: locally cached Qwen2.5-VL-7B-Instruct snapshot
  `cc594898137f460bfe9f0759e9844b3ce807cfb5` via `AutoProcessor` and
  `qwen_vl_utils.process_vision_info`; model weights were not loaded.
- Input: 3 generated RGB clips, 32 frames each at 56x56, synthetic question and
  five choices, synthetic claim, and interval [2, 6] seconds. Candidate adapter
  and condition builder receive the same frame objects, PTS grid, interval,
  QA text, and 2 FPS sampling value.
- Atomic-freeze callbacks are explicit local stubs returning `True`; they do
  not verify source, review, custody, clock, or authorization.
- Runner: `scripts/smoke_qta_candidate_to_frozen_conditions_qwen_cpu.py`.

## Command and observed result

```bash
.venv/bin/python scripts/smoke_qta_candidate_to_frozen_conditions_qwen_cpu.py
```

Observed: exit 0; **3/3** ordered candidate rows and **9/9** full/evidence-mask/
control-mask batches constructed. Each candidate's typed full-batch SHA256
matched the condition builder's corresponding `full` batch. The nine condition
batches passed `freeze_all_batches` with only synthetic callback stubs. Output
reported `model_weights_loaded=false`, `model_forwards=0`,
`model_load_permitted=false`, `scoring_authorized=false`,
`faithfulness_scored=false`, and `execution_authorization=not_granted`.

`py_compile` and `git diff --check` passed. The normal `python` command name is
not installed in this shell; checks used the repository `.venv/bin/python`.

Source SHA256 values at run time:

- Runner: `9223de69c4460e713d243155f0ac8ca271f885e5a977b6c2921c0b128cfa0817`
- Candidate adapter: `7bfb346d6e18a50791e03cdd81f4ccfeb89f0adc0af022e9d7132ef9269d8c65`
- Condition builder: `7486c6e959e507add3190b2af91b92ddf3d9dc84035a11cd44b3c0f36cee14e7`
- Atomic freezer: `b6e5c60bc51cdf913c91ec6a2d8acfdee2c4f715eadaf851d7d9ce7f1d569c92`

## Interpretation and limitations

This supplies actual cached-Qwen CPU evidence that the candidate adapter's
full-video processor batches are byte/type-identical (under the freezer's
typed digest) to the condition builder's `full` branch when all synthetic
inputs are shared, and that those rows can be passed to the existing freezer.
The evidence does not authenticate any pins, real source frames, reviewer
judgments, interval correctness, clock mapping, or gate callback behavior beyond
the stub interface. It loads no model and produces no model logits. No direct
faithfulness estimate or human-review progress is implied. It does not repeat
the existing fake-processor mutation/order/fake-scorer test coverage.

The nine expected condition hashes are derived from the batches in this same
runner immediately before freezing. Therefore this smoke verifies snapshot and
handoff consistency, not comparison against an independently frozen condition
manifest. Independent mutation/order rejection remains covered by the existing
synthetic adapter-composition and freezer unit tests, which use fake processor
inputs rather than this cached-Qwen path.

## Next action

Use the same row-order/hash join only after real review and clock/source gates
are independently validated. Until those inputs exist, this synthetic smoke
does not justify a real-data model forward or GPU scoring. Direct faithfulness
counts remain unchanged.
