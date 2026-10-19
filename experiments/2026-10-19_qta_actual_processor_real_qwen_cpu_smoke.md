# QTA actual processor composition smoke (synthetic CPU)

## Purpose

Verify that the non-authorizing three-row candidate adapter composes with the
actual cached Qwen2.5-VL processor and `qwen-vl-utils`, without loading model
weights or using protected videos. This is an engineering compatibility check,
not a faithfulness experiment.

## Setup

- Date: 2026-10-19 (Asia/Tokyo).
- Runtime: repository `.venv`; Python 3.12; PyTorch 2.11.0+cu128; Transformers
  5.18.0; execution was CPU-only (`torch.cuda.is_available() == false` in the
  normal sandbox).
- Processor: local cached Qwen2.5-VL-7B-Instruct snapshot revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; no model class or weights loaded.
- Input: three synthetic rows, each with 32 generated 56×56 RGB frames, 32
  endpoint-grid timestamps at 2 fps, a synthetic QA item and claim, and a
  synthetic review interval. No real dataset, protected video, or human review
  input was read.
- Processor resize limits were set to 56×56 minimum and four 56×56 pixels per
  frame maximum to keep this smoke small.
- Runner: `scripts/smoke_qta_actual_processor_qwen_cpu.py`.

## Command

```bash
.venv/bin/python scripts/smoke_qta_actual_processor_qwen_cpu.py
```

The runner defaults to the pinned cache revision and accepts
--model-snapshot when it lives elsewhere. An
initial invocation had a mistyped directory name and exited before processor
loading; it did not access inputs or change state.

## Verified result

- Requested/built rows: **3/3**.
- A–E continuation IDs: **5 per row**.
- Processor grids: **[16, 8, 8]** for all three rows; this represents 32 input
  frames with temporal patch size 2 and spatial merge size 2.
- Candidate external-pin validation: passed for all three rows.
- Elapsed time reported by the final primary run: **0.471 s** after processor
  load began; this is a small CPU smoke, not a throughput estimate.
- Independent audit rerun elapsed time: **0.446 s**.
- Pinned processor-input aggregate SHA256 (preprocessor config, tokenizer JSON
  and config, chat template, merges, vocab):
  `a1e3e98d2b7bc3f563115a890fe900bacc4927fd14324f887f66eec8fa9a4931`.
- Runner SHA256:
  `803c7de724bb2380105c4928c9f90c2d15c1c7b180e77379fe7e3a35f4f35725`.
- Adapter SHA256:
  `7bfb346d6e18a50791e03cdd81f4ccfeb89f0adc0af022e9d7132ef9269d8c65`.
- Candidate unit-test SHA256:
  `cac279ef63100deb7b919190146146a0f55535b5923ad5c03c9d5e3fe430fafa`.
- Source commit: `a3e0e3131ec78e202f52bd743a987f49ab72df23`.

- Output asserted `model_weights_loaded=false`, `model_load_permitted=false`,
  `scoring_authorized=false`, and `faithfulness_scored=false`.

The successful runner emits only aggregate grid shapes, candidate counts, and
non-authorizing status. It does not write a candidate bundle.

## Interpretation and limitations

This verifies that the revised adapter can process a synthetic 3-row batch with
the pinned Qwen processor/tokenizer implementation and map the 32-frame timing
grid into processor groups. It does not verify actual Next-GQA video decoding,
sampler provenance, claim support, reviewer identity or judgment, clock
calibration, baseline p/q, model logits, or any direct faithfulness endpoint.
The processor fingerprint in this adapter is an expected caller pin; this smoke
does not implement the upstream authenticated checkpoint/processor inventory
gate. The video grid result is a preprocessing property, not evidence that the
explanation follows video evidence.

## Faithfulness status and next action

Direct faithfulness is unchanged: claim review **0/3**, player-clock/decoder
frame correspondence **0/15**, Stage-2 prediction-link review **0/3**, direct
score **0/3**, and no measured real-data `ΔF`, interval, or test. The simple
nickname-to-questions artifact remains a preliminary impression check and no
response JSON has been found. The next useful action is to use that brief local
artifact, then continue only with the scoped review/custody inputs needed for a
valid claim-linked real-data scorer. GPU scoring remains gated; this CPU smoke
does not change that status.
