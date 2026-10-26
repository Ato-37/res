# Prompt/answer-token reconstruction in the direct-faithfulness preload

## Purpose

Close a fail-open provenance gap before future direct claim-faithfulness scoring:
the signed phase-1 binding had declared prompt-token hash, A–E continuation IDs,
and choice-binding digest, but the atomic preload did not independently derive
those values from source prompt inputs and the processor/tokenizer candidate.
This is local structural readiness work; it is not a scored research result.

## Implementation

`qta_prompt_answer_token_bindings_v1.py` now reconstructs the fixed QA prompt
from ephemeral ordered question/options, frames and FPS using the supplied
processor's chat template, then recomputes prompt text hash, prompt token-ID
digest, A–E continuation IDs, choice-binding digest, and live processor input-ID
digest. It checks the reconstructed question/options against source pins, the
frame RGB hashes/FPS/source and sampler-row digest, and the exact video-token
expansion in the live `FrozenBatch.input_ids`. Frames are hashed before and
after rendering. The tokenizer must be the processor's actual `.tokenizer`
object; absence or identity mismatch fails closed. Missing ephemeral material
also blocks the atomic result. Raw question/options, prompt text and frames do
not enter the verifier report or binding receipt.

The atomic wrapper requires this verifier for its structural composition path.
Its success report explicitly remains `direct_faithfulness_ready=false`,
`model_load_permitted=false`, `scoring_authorized=false`, and
`execution_authorization=not_granted`. It does not authenticate the processor or
tokenizer asset origin, renderer code origin, external root-key origin, human
identity/semantics, or clock-to-frame truth.

Repository search found no non-test caller of
`preflight_qta_direct_faithfulness_atomic`; this is now a stronger atomic API,
but it is not yet wired into a production/real-data execution command. Do not
describe the repository as ready to run the actual direct-scoring experiment.

## Checks and results

A synthetic signed phase-1 receipt was composed through the atomic wrapper and
the real (unmocked) prompt/token verifier. Its QA question/options came from the
synthetic source CSV in row order, and their hashes matched Stage-1 pins on all
3/3 rows. This is a source-backed synthetic integration check, not a real
reviewed cohort. The validator explicitly rejects mutated signed declarations,
row order changes, source/frame/FPS/sampler mismatch, live input-ID mismatch,
renderer frame mutation, missing transient inputs, and a renderer without a
`.tokenizer` identity.

Independent audit initially found and reproduced acceptance of a renderer
without a `.tokenizer` attribute. The check was tightened to require a present,
identical tokenizer, and a regression test was added. Independent re-audit
confirmed the fix and source CSV hash alignment.

Primary-agent regression command:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest \
  test_qta_prompt_answer_token_bindings_v1 \
  test_qta_direct_faithfulness_atomic_preload \
  test_qta_prebaseline_transaction_preflight \
  test_qta_prebaseline_live_batch_binding \
  test_qta_processor_batch_digest_manifest_v1 \
  test_qta_actual_processor_row_candidate \
  test_qta_actual_candidate_input_bridge \
  test_qta_stage1_source_pin_adapter \
  test_qta_phase1_processor_artifact_adapter -q
```

An initial run with the first seven modules passed **149 tests** in 55.5
seconds. The expanded run adding the candidate-input bridge and Stage-1 source
pin adapter passed **186 tests** in 70.6 seconds. The focused independent audit ran
48 tests successfully (including duplicate collection of imported prompt test
cases). `py_compile` passed on all four touched Python files; a direct trailing-
whitespace scan found none. SHA-256 hashes:

- `qta_prompt_answer_token_bindings_v1.py`:
  `605e2267398edc5169106d1f08c201906bbac6160ce4ae58bcb5d45d43c876c6`
- `qta_direct_faithfulness_atomic_preload.py`:
  `bb9a3efb343ba483b8b9ce84c366f70e2aa17fdc3b6b27873b6ac311ba1e1ed8`
- `test_qta_prompt_answer_token_bindings_v1.py`:
  `6b5a5deaf2eb80d7c644972a87fe9615ada0aed4b0647a9902eb258bd801b96e`
- `test_qta_direct_faithfulness_atomic_preload.py`:
  `0823b5063cd0b31e0c7e3ad1a6f98368e13f9941b96a798f8ec9492d299ee98d`

No protected production response or media was opened for this work. No Qwen
model was loaded and no GPU forward or faithfulness score was produced. Formal
direct status remains claim review **0/3**, formal review inventory **0/25**,
clock/frame verification **0/15**, prediction-link review **0/3**, and direct
intervention scores **0/3**. Rights remain unverified; protected artifacts
remain local.

## Interpretation and next action

The readiness gap was reduced: signed token declarations are now checked against
source-bound ephemeral inputs, the supplied tokenizer and the exact live batch
in the synthetic integration path. This does not create evidence that an
explanation claim is visually supported, nor does it authorize a production
score. The next direct-faithfulness progress still depends on an actual review
response plus validated reviewer/custody and clock/frame gates. Once those
inputs exist, run the atomic preflight with the real processor candidate and
independently bound source material before considering any bounded GPU scoring.
Also implement and test a local-only caller that supplies the ephemeral inputs;
the current repository has no non-test caller for the strengthened API.
