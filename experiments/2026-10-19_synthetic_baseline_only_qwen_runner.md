# Synthetic baseline-only Qwen compatibility runner

## Purpose

Add `scripts/smoke_nextgqa_baseline_only_qwen_gpu.py` as a bounded integration
runner for the accepted three-row baseline scorer. It constructs three
deterministic, synthetic 32-frame RGB clips, passes only each full-video batch
through the locally cached pinned Qwen processor, validates the processor
batch/temporal mapping, and either performs a CPU-only preflight (`--dry-run`)
or loads the pinned local model and asks the scorer for exactly three forwards.
No protected dataset, network download, real media, treatment input, or human
review is involved.

The output is an exclusive mode-0600 JSON file under `results/`. The writer
rejects overwrite and paths outside that directory. A recursive field check
rejects raw question/answer text, prompt/message strings, frames/pixels,
tokenized input IDs, and per-record scorer outputs. A successful GPU receipt
contains only aggregate P/Q log-probability-margin summaries, call/capture
counts, hashes, versions, elapsed times, and VRAM counters. It always sets
`faithfulness_result=false`, `scoring_authorized=false`, and
`execution_authorization=not_granted`.

## Validation

Runner safety suite:

```bash
cd scripts && ../.venv/bin/python -m unittest -v test_smoke_nextgqa_baseline_only_qwen_gpu
```

Observed: **7/7 tests passed**. These test required arguments, dry-run parsing,
output containment and overwrite rejection, symlink escape rejection,
mode-0600 exclusive output, and rejection of sensitive nested fields. A
tokenizer stub also verifies A–E are resolved as one-token continuations of
the exact rendered `Answer:` prompt using
`qta_choice_token_binding.resolve_choice_continuation_ids`, rather than as
standalone label encodings.

Combined CPU suite:

```bash
cd scripts && ../.venv/bin/python -m unittest -v \
  test_smoke_nextgqa_baseline_only_qwen_gpu test_nextgqa_baseline_only_scoring
```

Observed before the GPU trials: **23/23 tests passed** in the latest combined
CPU suite (runner 7, scorer 16; 0.065 seconds). All four touched runner/scorer
and test modules passed `py_compile`, and `git diff --check` passed.

CPU processor dry-run:

```bash
.venv/bin/python scripts/smoke_nextgqa_baseline_only_qwen_gpu.py --dry-run \
  --output results/synthetic_baseline_only_qwen_cpu_preflight_20261019_04.json
```

Observed in the latest run: exit 0, 3.92 seconds, status
`preflight_passed_no_model_forward`;
three distinct synthetic source hashes, 3 clips × 32 frames, 16 processor
temporal groups per clip, 5 answer tokens, and zero attempted/completed model
forwards. The aggregate receipt records runtime/source hashes and processor
fingerprint but correctly leaves checkpoint content hash as
`not_computed_dry_run`. Transformers 5.18 emitted a deprecation warning for
`min_pixels`/`max_pixels`; processing completed. The dry-run artifact contains
no raw video, QA text, per-row records, or token IDs.

## GPU compatibility run and repair

The first GPU attempt loaded the pinned checkpoint but rejected the model before
any forward (`attempted_forwards=0`). A local reproduction isolated the cause:
the actual Qwen2.5-VL config exposes `vocab_size` under
`model.config.text_config.vocab_size`, while the scorer only read a root-level
field. The fake-model tests had exposed only the root-level layout. The scorer
now supports both layouts, and a regression test exercises the nested form.
An independent audit then identified a possible CUDA-sync counter loss and
missing source-code hashes; the runner was hardened to persist counts before
synchronization and hash the row-construction and selection dependencies.
The additional source-field name initially tripped the safe-output guard on a
separate GPU retry; after renaming it and adding a filter regression test, the
latest combined suite passed **23/23 tests**.

The corrected GPU run used three synthetic 32-frame clips and completed exactly
**3/3 forwards** and **3/3 attention captures** on layers 5–16. The model-call
boundary observed 3 entries, verified 3, and completed 3 calls. No failure was
recorded. End-to-end runtime was **37.69 s** (model load 8.39 s; three scorer
calls 12.03 s). Peak allocation was **10.72 GiB** and peak reservation **12.44
GiB** of 15.89 GiB; free VRAM after scoring, while the model was still resident,
was 1.89 GiB. The process exit releases its model allocation. This confirms a
working local integration path for this exact synthetic input, pinned model,
processor, runtime, and GPU; it does not justify concurrent large GPU jobs.

The aggregate-only P/Q log-probability margin summary was n=3, mean
**+0.7917 nat**, range **[+0.7500,+0.8750]**. These are synthetic diagnostic
rows, included only to verify serialization and scorer plumbing; no inferential
statistic is appropriate and they are not faithfulness evidence. The receipt
keeps `faithfulness_result=false`, `scoring_authorized=false`, and
`execution_authorization=not_granted`. Its mode is 0600 and it contains no raw
question, answers, frames, token IDs, or row-level results.

An intermediate rerun completed its model calls but failed while writing the
receipt: one source-hash inventory label contained `prompt`, which the
aggregate-field guard intentionally rejects. No receipt was written, so its
forward counts are not used as verified results. The label was renamed to
`qa_video_batch_builder`, and a regression test now checks that source-hash
field names pass the aggregate guard. Independent audit confirmed the nested
vocabulary fix, that forward counters are copied before CUDA synchronization,
that code hashes cover the synthetic input builder, QA/video batch builder,
token binding, selector, batch freeze, processor and temporal mappings,
attention capture and contract/sidecars, call boundary, and protocol, and that
the combined CPU suite passes 23/23.

The final rerun with this complete source inventory is
`results/synthetic_baseline_only_qwen_gpu_20261019_04.json`. It records status
`passed`, 3 requested/attempted/completed forwards, 3/3 verified root calls,
and 3/3 attention-capture rows. Runtime was **38.65 s**; peak allocated VRAM
was **10.72 GiB** and peak reserved VRAM was **12.44 GiB**. The receipt has 14
source hashes, mode 0600, and aggregate-only fields.

## Resource estimate and GPU command

The local checkpoint snapshot contains 16 files totaling **16.60 GB** by file
metadata. The GPU path hashes these bytes before model load, then loads 4-bit
weights with BF16 compute and eager attention. The closest prior run used the
same 3×32-frame synthetic cohort and completed **9 forwards in 30.11 seconds**;
it observed 10.74 GiB peak allocated / 12.47 GiB peak reserved on an RTX 5070
Ti with 15.89 GiB total VRAM. The present runner requests 3 forwards but also
captures question-query attention at layers 5–16. The latest measured end-to-end
time was **38.65 seconds**, including checkpoint hashing, model load, and three
forwards. Peak VRAM reservation was **12.44 GiB** of 15.89 GiB. The runner refuses
to load unless at least
12 GiB VRAM is free before load and 3 GiB before the first forward. Existing
matching peak reservations leave limited margin, so the primary should inspect
live free VRAM before executing. The final runner hashed all 16.60 GB of local
checkpoint files; this is recorded by the GPU receipt.

Reproduction command (the output path must be new; `_04` already exists, so use
`_05` or another unused suffix):

```bash
.venv/bin/python scripts/smoke_nextgqa_baseline_only_qwen_gpu.py \
  --output results/synthetic_baseline_only_qwen_gpu_20261019_05.json
```

The output name must remain unused; the runner enforces this. Expected scope is
exactly 3 synthetic full-video forwards. A passed smoke only establishes the
runner/scorer compatibility and resource behavior for this synthetic cohort,
checkpoint revision, processor, runtime, and hardware. It is not evidence of
faithfulness, benchmark accuracy, source-media provenance, or human-review
validity.

## Limits and next action

The CPU preflight alone verified local processing and group layout. The final
source-complete synthetic GPU receipt is
`results/synthetic_baseline_only_qwen_gpu_20261019_04.json`. Earlier local
receipts remain as history: `_01` is a pre-forward validation failure and `_02`
is an earlier successful run without the expanded source inventory. The
intermediate `_03` run failed the receipt field filter and produced no result
artifact. Keep this work synthetic-only. The next research step still requires
separate review and source/timing gates before any protected-real-video
inference; the present runner deliberately cannot authorize that work.
