# QTA actual-source processor-row candidate adapter (CPU only)

## Purpose

Connect an already supplied, ordered three-row QA/source candidate to the
repository's fixed QA prompt and existing processor/timing mapping contracts.
This is a local candidate builder only. It does not read protected files,
authenticate media lineage or human review, load a checkpoint, call a model, or
authorize scoring.

## Implementation

Added `scripts/qta_actual_processor_row_candidate.py` and its focused CPU test.
For each row the adapter checks caller-supplied source text/hash and review
interval pins, distinct ordered review tokens and video hashes, and the
processor revision/fingerprint, tokenizer digest, duration, guard, and
sampler-row digest against an independently supplied ordered `expected_rows`
list. Sampler rows are a separate required argument; their source hash,
endpoint-uniform indices, PTS sequence, 32 RGB frame digests, frame count, and
effective FPS are validated and consumed by the mapper. Before the first
processor call, each RGB digest is recomputed from the exact supplied PIL
frame using `frame.convert("RGB").tobytes()` and mismatches are rejected. The
adapter then calls the existing
`build_qa_prompt` from `run_nextgqa_gold_interval_margin_pilot.py`.
`build_choice_token_binding` derives the distinct A–E continuation IDs from
that exact rendered prompt. The same prompt is sent through
`process_qa_video`; its CPU batch is passed to
`map_review_interval_from_processor_batch`, then projected by
`processor_binding_row`. Question positions are derived from that prompt and
the resulting processed input IDs. Candidate rows explicitly carry tokenizer,
processor, QA-text, sampler, prompt, and batch digests. It verifies the
tokenizer's `encode` IDs agree with its offset-tokenization IDs, then checks
that the processor's complete input ID sequence is exactly that rendered
prompt with only the video placeholder expanded. This verifies the prompt
suffix without another processor pass.

Before any processor/template call, the builder requires
`processor.tokenizer is tokenizer`. This blocks deriving continuation IDs
with a separate tokenizer instance. It does not independently inspect the
processor object's files/configuration to prove the supplied revision or
fingerprint; those are caller-provided values compared to external pins. Treat
them as caller assertions until an upstream processor-fingerprint verifier
binds them to the actual installed processor.

The current mapper return omits `sampled_frame_indices` and `video_grid_thw`,
although `processor_binding_row` expects them. The adapter projects the exact
sampler indices already consumed by the mapper and reads the grid directly
from the same processor batch whose digest the mapper computed. Neither step
authenticates the upstream sampler or processor.

The prompt builder renders the question and ordered A–E choices, but not the
claim text. The claim is source-pinned and carried as QA-source context only;
it does not alter this fixed QA prompt. The caller supplies decoded sampled
frames. The adapter does not prove those frames came from the claimed source
file or that timestamps match a browser review.

`verify_actual_processor_row_candidate` requires external ordered expected
pins and compares every candidate pin in row order. The separately named
`verify_actual_processor_row_candidate_self_consistency` only recomputes
internal commitments and batch hashes; it is not an external provenance
validator. The ordered internal commitment is an integrity check, not a
signature.

## Verification

Command:

```bash
cd scripts && ../.venv/bin/python -m unittest -v test_qta_actual_processor_row_candidate
```

Observed: **13/13 CPU fake-processor tests passed**. They cover exactly three
processor calls, distinct A–E continuation IDs, non-authorizing flags,
pre-processor rejection of source, interval, token, row-order, duplicate-source,
and source-text mutations, processor/tokenizer identity mismatch, sampler-row
splice, RGB-frame splice, duration/guard mismatch, post-build batch tampering,
external-pin tampering, authority-field tampering, reordered candidate rows, and a row-2 processor
exception that returns no candidate. Fake PIL frames are generated in memory;
no model forward, GPU, or protected file access occurred.

## Actual cached Qwen processor smoke

After the independent remediation, a second CPU smoke used three synthetic
32-frame PIL clips with the locally cached Qwen2.5-VL-7B-Instruct processor
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`. The real Qwen tokenizer
and `qwen-vl-utils` created the processor batches. All **3/3** candidates passed
the external-pin verifier; each produced five A–E IDs and grid `[16, 8, 8]`.
Runtime reported by the runner was 0.471 s. No model weights or protected data
were loaded. The runnable script and setup are documented in
[`qta_actual_processor_real_qwen_cpu_smoke`](2026-10-19_qta_actual_processor_real_qwen_cpu_smoke.md).

## Interpretation and limits

The unit tests establish API composition and structural integrity handling for
the fake fixture, while the separate synthetic smoke verifies basic field and
token-sequence compatibility with the cached Qwen processor. Neither establishes
authentic sampler provenance, processor object fingerprint authenticity, human
identity, review correctness, source lineage, clock calibration, or a valid
baseline. Output remains
`candidate_only`, `model_load_permitted=false`, `scoring_authorized=false`,
`execution_authorization=not_granted`, `human_identity_verified=false`,
`semantic_review_authenticated=false`, and `faithfulness_scored=false`.

## Next action

Independent audit confirmed the final source-frame and prompt-token bindings
and reran the then-current 12-test suite and cached-Qwen CPU smoke. A real-data row
builder still requires independently validated source, sampler, Stage-1 review,
clock/frame correspondence, calibration, processor fingerprint, and baseline
pins. This candidate adapter does not authenticate those gates or permit model
loading. Once the brief local human check is returned, use it only for its stated
impression-check purpose; it does not replace formal claim-support or
prediction-link review.
