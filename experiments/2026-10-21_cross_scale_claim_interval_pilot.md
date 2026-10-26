# Cross-scale claim-interval pilot (2026-10-21, local exploratory)

## Question and fixed endpoint

This post-hoc exploratory follow-up asks whether Qwen2.5-VL-7B's fixed QA
answer margin depends more on a candidate-claim interval localized by a pinned
Qwen2.5-VL-3B checkpoint than on an equal-count temporal-rank control. The
contrast is the finite-cohort mean of `margin(control black mask) -
margin(3B-localized candidate-interval black mask)`, in nats. Each margin is
gold-choice next-token log probability minus the best of four foil choices.
Positive values indicate greater 7B margin reduction under the 3B-selected
interval mask. The estimand is chosen before observing this run's scores.

The cohort remains exactly the three de-duplicated validation records in the
two local claim-generation packets used by the preceding pilots; no item
replacement, subgroup, or sample-size expansion is planned. Each candidate
description is localized once by 3B, then the exact same timestamps are mapped
to 32 decoded frames and scored by 7B under full, interval-mask, and equal-count
temporal-control conditions (nine 7B forwards). All three rows and all nine
forwards are required for an estimate. A localization, geometry, or runtime
failure means no mean; no surviving-subset average is allowed.

## Why this is a distinct, bounded diagnostic

The same-model 7B locator failed interval validation on item 3 before scoring.
The separate benchmark-interval diagnostic completed, but its QA annotations do
not establish where a generated claim is supported. This branch therefore uses
the cached Qwen2.5-VL-3B-Instruct revision
`66285546d2b821cf421d4f5eb2576359d3770cd3` for localization and the pinned
Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` for scoring. Both are Qwen2.5-VL
family checkpoints, so this reduces exact-checkpoint reuse but does not provide
independent human validation or eliminate shared-family bias. Candidate-claim
truth, frame support, presentation-time calibration, and causal relevance
remain unverified. Treat any completed result as a machine-generated
claim-to-time proxy, not direct or confirmatory explanation faithfulness.

This is one bounded three-item run, not a repeat of the failed same-model
localizer and not a continuation of the already-completed benchmark-QA interval
endpoint. It tests a different interval-construction source and model pairing.
The metric, cohort, video-frame count, and interval assignment rule are frozen.
Pre-score smoke on row 1 exposed response-format failures, so the current
prompt/parser are a post-hoc feasibility procedure. Row 1 was used during
prompt development and remains in the later run; any resulting three-row
contrast is a development/engineering description only, not a confirmatory or
unbiased efficacy estimate. The final instruction has no numerical timestamp
example to avoid temporal anchoring; its SHA-256 is
`3a4fad514128b46eebd40d3540fa674465b3007706e9a2bb853b2cb5494cbd9a`. Freeze
this prompt/parser before the final smoke and do not make score-driven changes.
Do not report a p-value or population interval for this fixed `n=3` cohort.

## Resource and privacy plan

Run offline from the cached local checkpoints, 4-bit quantization with BF16
compute, eager attention, seed 20261021, and 32 frames/video. Load 3B, generate
three timestamp intervals, release it, then load 7B and run nine scoring
forwards. Require at least 12 GiB free VRAM and 8 GiB WSL available memory
before model load; retain 2 GiB VRAM and 8 GiB host-memory guards before each
device transfer/model call. Expected GPU workload is two sequential model loads,
three short generation calls, and nine one-row scoring forwards (roughly
2–5 minutes after load). First run CPU regression/media-decode checks and one
preselected-item GPU pipeline smoke; the smoke output is not part of the result.

Output is a new exclusive mode-0600 file under `data/private_nextgqa/`. Keep
videos, QA text, candidate descriptions, detailed receipts, and derived
three-item scores local. Rights status remains unverified under the user's
local-only authorization. Human-review collection was waived by the user for
this exploratory branch; `review_actually_performed` remains false.

## Pre-score smoke iteration

CPU preflight decoded all three fixed videos into 32 frames and independently
matched each source digest to the input packet. The first single-item 3B GPU
smoke loaded the checkpoint and generated output, but the strict JSON parser
rejected it before any score. A same-input deterministic diagnostic reported
212 characters, one opening brace, no closing brace, and a JSON decode failure;
generated text was not saved or printed. No faithfulness score or interval was
produced. A tighter prompt with a concrete 1–2 second example and a 128-token
limit yielded one balanced brace object but still failed strict/structured
parsing. That numerical example was removed after independent review flagged
possible temporal anchoring; the frozen instruction hash above has no numeric
example. One later smoke command failed in its own inline Python f-string before
generation and is not counted as model output. The parser now accepts one
complete embedded interval object, with JSON, literal-only Python mapping, or
strict two-key numeric mapping syntax; it rejects ambiguity, extra fields,
truncation, and common refusal/disclaimer text, then checks unique keys,
numeric types, finiteness, and bounds. The current focused test file passes
**11/11** after adding duplicate-JSON-key rejection. These format-compliance
iterations occurred before scored forwards and do not validate semantic
localization. An initial invocation of the final smoke failed before inference
because its temporary script did not include the repository import path; that
was an invocation error and did not load a model. The corrected one-item smoke
loaded the pinned 3B model, decoded the preselected video, and completed a
generation, but the frozen parser rejected the response with
`localization_single_json_object_required`. The raw response was neither saved
nor printed. No timestamp interval was accepted and no 7B scoring forward ran.
Per the frozen stop rule, this branch is closed without another prompt/parser
retry and without a three-item run.

## Verified result

The final corrected smoke ended in `failed_no_valid_interval`; it yielded
**0/1 valid localizations, 0/3 scored rows, 0/9 scoring forwards, and no
effect estimate**. No full-run receipt exists, and no separate machine-readable
receipt was retained for this corrected one-item smoke. The counts and failure
code above are supported by the recorded command/tool output, not by a
persisted run artifact; the previous same-model receipt does not support this
cross-scale smoke. The cross-scale branch is closed because the frozen
one-item feasibility gate failed. The import-path invocation error is not
counted as a model failure. The smoke ran locally on the RTX 5070 Ti using the
pinned cached 3B checkpoint and the existing BF16/CUDA `.venv`; the tool stream
shows model loading completed before generation, while no artifact retained
the generated text or timing output. The branch result is a pipeline-format
failure, not evidence that the candidate interval lacks visual support. Its
run-level provenance is weaker than the adjacent benchmark-interval result,
whose private receipt and hashes were independently verified.

## Next action

Do not retry this localized-interval branch or infer an effect from the failed
smoke. The completed benchmark-QA-interval diagnostic remains a separate
auxiliary result and must not be presented as generated-claim faithfulness.
Continue only with a distinct, preregistered measurement that can use valid
inputs without substituting an outcome-selected interval. Human review was
waived for exploratory work, but the factual review counts remain 0/3 claims,
0/15 clock/frame correspondences, 0/3 prediction-link reviews, and 0/3 direct
scores. This run added no direct faithfulness estimate.
