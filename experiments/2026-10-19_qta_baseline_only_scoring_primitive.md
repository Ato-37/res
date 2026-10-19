# QTA three-row baseline-only scorer primitive

## Purpose and implementation

Added `scripts/nextgqa_baseline_only_scoring.py` and
`scripts/test_nextgqa_baseline_only_scoring.py`. The function accepts exactly
three prebuilt, ordered `FrozenBatch` full-video rows and five distinct
in-vocabulary answer token IDs per row. It requires the caller-supplied model
to be in eval mode with eager attention configured, including every hooked
`self_attn` module. It performs one unchanged
root forward per row, checked by `ModelCallBoundary`, and returns full-vocabulary
next-token log-probabilities for A–E, deterministic P/Q, score positions and
provenance hashes.

The same baseline forward captures question-query attention at Qwen layers
5–16. It checks each layer fired once and attention rows are finite and
normalized, aggregates attention over the pinned temporal groups, then records
the protocol's `max(1, ceil(T*0.10))` top groups (earliest-index tie break),
target token positions, and deterministic sham groups/token positions using
`qta-sham-seed-20261018-v1` and the pinned opaque review token. This defines
the output fields needed by the next planning stage; it does not authorize that
stage.

Pre-forward checks cover the complete three-row inventory, row order/source
and digest pins (including distinct source hashes and pinned A–E token IDs),
pinned question positions and temporal groups, exact full-video keys, absence
of ambiguous image-only input, batch-size-one fully unpadded attention masks,
integer positive grid values, floating rank-2 pixel payload with matching
pixel-row/grid counts, one positive finite `second_per_grid_ts` value, and
temporal-group/grid alignment. The one-value timing shape follows the existing
single-video processor contract. This scorer checks output layout only; the
upstream processor mapping/gate remains authoritative for sampling, FPS/PTS,
reviewed intervals, and source-to-processor temporal geometry.
Runtime abort, non-finite scores, hook failures, extra root calls, or incomplete
three-row scoring return no partial rows and remain non-authorizing.

## Verification

Command:

```bash
cd scripts && ../.venv/bin/python -m unittest -v test_nextgqa_baseline_only_scoring
```

Observed: **15/15 CPU fake-model tests passed** (0.055 seconds). They cover
exactly three root calls, A–E/PQ and selector/sham output, row-count/order/source
and batch tampering, condition rejection, invalid/unpinned answer tokens,
duplicate sources, missing or ambiguous video input, mask/pixel/grid/timing
shape and dtype errors, train mode and non-eager attention rejection, pinned
question/group positions, ranked versus sorted selector positions, NaN, forward
abort, missing attention-hook layer, and reentrant extra calls. `completed_forwards`
counts root calls that returned according to `ModelCallBoundary`, even when
subsequent attention post-capture validation fails; partial score rows remain
discarded in that case.
Pre-call failures invoke no model forward; post-start failures discard all
partial records. No real model, checkpoint, video, protected data, or GPU was
used.

Independent source audit also checked the hook contract against the locally
installed Qwen2.5-VL eager-attention implementation and verified the failure
count/atomicity behavior. It found and the primary repaired the one-video
`second_per_grid_ts` shape, expected A–E token pin, duplicate-source rejection,
and ranked-versus-sorted selector output contract. A final independent rerun
passed all **15/15** focused tests. This audit still does not establish
real-checkpoint runtime compatibility or source-media provenance.

## Interpretation and limitations

This establishes CPU control-flow and schema behavior only. It is not a Qwen
compatibility test and gives no baseline score or efficacy evidence. Required
video tensor keys, shapes, timings, and hashes do not prove that tensors came
from the claimed original decoded RGB video, or that timestamps/frame sampling
are semantically faithful. The full existing processor-mapping contract is not
duplicated here. A separately validated source-bound gate must bind
the exact source media, decoder output, sampler, processor payload, token/group
mapping, opaque review token, and expected row pins before any caller loads or
uses a model. The primitive neither validates that gate nor loads models.

Attention hook capture is an execution diagnostic, not authenticated evidence
of the model, checkpoint, or source. Fake attention scores cannot support a
faithfulness claim. No experiment outcome, human review, or permission to run
inference is established. Current direct faithfulness evidence remains **0/3
reviewed rows, 0/3 adjudications, 0/15 frame correspondences, and 0/3 scores**;
there is no empirical effect estimate or uncertainty interval. The primitive
is non-authorizing (`model_load_permitted=false`, `scoring_authorized=false`,
`execution_authorization=not_granted`, `faithfulness_result=false`).

## Next action

Complete and independently validate the source-bound pre-load gate and its
three-row processor adapter. Only after that gate passes may a separate caller
load the pinned model and invoke this primitive. Keep the resulting artifact
local unless its derived fields pass the repository's privacy review.
