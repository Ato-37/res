# Real-video QTA attention capture smoke (2026-10-18)

## Purpose

Connect the QTA selector to selected question-query rows captured from the
actual Qwen attention modules, and check that the capture's row-sum validation
matches the model's activation dtype. This is an engineering/wiring check on
three local NExT-GQA test videos. It does not review claims, score QA
predictions, evaluate faithfulness, or authorize treatment inference. Rights
status is unverified; the private JSON outputs remain local.

## Protocol repair

The initial GPU attempt loaded the pinned model and stopped on its first
baseline forward: the capture contract required row-sum error `<=2e-4`, but
the selected Q rows exceeded that threshold. A second diagnostic run completed
3/3 videos and recorded maximum deviations **0.0026866**, **0.0027470**, and
**0.0028209**. It also showed all selected tensors had dtype `bfloat16` after
conversion to float32 for canonical aggregation. The installed Transformers
5.18.0 Qwen eager-attention implementation computes softmax in float32 and
casts attention weights to `query.dtype`; this run loads query activations in
bfloat16. Thus the original float32 tolerance was inconsistent with the
returned tensor representation.

The dtype-specific row-sum bound is now explicit: **2e-4 for float32** and
**0.0041 for bfloat16**. The latter allows bfloat16 rounding error (one unit
roundoff is 2^-8 = 0.00390625, with small float32 softmax-sum slack). The
capture fails closed on values outside the declared dtype bound, mixed source
dtypes across layers, nonfinite/negative values, layer/module aliasing,
duplicate/missing layer calls, malformed attention shapes, and incomplete
hook registration. The report records source dtype, tolerance, maximum
deviation, layer paths/classes and call counts. It still labels the capture as
unauthenticated. The binary-sidecar schema carries the source dtype and applies
the same bound; this dtype is a signed assertion, not authenticated runtime
provenance.

The legacy hex capture contract still uses its original strict float32
tolerance because it does not carry a source-dtype field. BF16 captures must
use the new binary-sidecar contract. The real-video smoke currently exercises
the raw capture and selector path but does not write raw sidecars or bind them
to a phase-2 plan.

## Verified GPU result

The final private run used the pinned Qwen2.5-VL-7B snapshot
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, RTX 5070 Ti, PyTorch
2.11.0+cu128, Transformers 5.18.0, CUDA 12.8, and 4-bit weights with BF16
compute/activations. It completed **3/3** local test videos and **6/6**
forwards (one baseline capture and one attention-bias treatment per video) in
**38.25 s**. Each of the 12 hooks (layers 5–16) fired exactly once per
baseline. All captured sources were `bfloat16`; maximum row-sum deviations
were **0.0026866**, **0.0027470**, and **0.0028209**, all within 0.0041. The
captured shapes were `[12,28,22,2161]`, `[12,28,13,2056]`, and
`[12,28,13,2056]` (layers, heads, question queries, input tokens). Peak VRAM
was **7.2399 GiB allocated / 8.0781 GiB reserved**, from **14.6572 GiB** free
before model load.

The diagnostic intervention produced mean per-video RMS logit change
**1.14009** and changed the top vocabulary token on **0/3** videos. A repeat
using the same deterministic cohort before dtype recording matched all three
videos' selector sizes, group counts, query positions, max/RMS logit changes,
and top-token-change indicators exactly. These quantities verify wiring and
repeatability only; they do not measure answer correctness or explanation
faithfulness. No effect estimate, confidence interval, significance test, or
population claim is available.

The three private outputs are mode 0600 under `data/private_nextgqa/`:

- `qta_real_capture_smoke_20261018_01.json`: expected fail-closed stop at the
  first baseline due to the overly strict dtype-agnostic threshold.
- `qta_real_capture_smoke_20261018_02.json`: diagnostic run that measured the
  BF16 row-sum deviation but did not record tensor dtype in its report.
- `qta_real_capture_smoke_20261018_03.json`: final dtype-explicit run reported
  above.

An independent auditor opened only the final JSON and current source files,
not the source videos or annotations. It independently reproduced the counts,
dtype/tolerance/deviations, resource figures, and runtime, and recomputed all
12 source hashes with exact matches. It confirmed mode 0600 and the no-answer-
text contract. It also noted that source-video, sampled PTS/RGB, annotation,
and manifest hashes are linkable pseudonymous metadata; this report stays
local and is not suitable for publication.

## Reproducibility and limits

The final output binds hashes for the runner, capture helper, selector,
candidate selector, intervention helper, batch hasher, decoder, and installed
Qwen/processor implementations. The capture and selection arithmetic match:
the new captured-row reducer is regression-tested for exact equality with the
existing full-attention reducer on fixed synthetic inputs. The final focused
capture/sidecar/receipt tests passed **68/68**. The full repository suite
passed **1,993/1,993** in **70.847 s**, including the BF16
accept-within-bound/reject-outside-bound cases.

The capture path is local instrumentation, not cryptographic execution
attestation. Its report's forward count records one invocation through this
runner callback and one call per hooked module; it cannot prove a trusted model
binary or operator identity. The three-video cohort is a wiring smoke selected
by the existing deterministic local-video procedure. The question was used as
model input, but answer text, gold answer, claim review, or reviewed interval
was not used as an outcome. `forward_source_authenticated=false`,
`attention_capture_authenticated=false`, `scoring_authorized=false`, and
`faithfulness_result=false` remain in force. Direct faithfulness progress is
still **0/3** independent claim reviews, **0/3** prediction-link judgments,
**0/15** verified browser/player-to-decoder correspondences, and **0/3**
claim-linked scores. The separate human package remains candidate-only; no
trust-root pin, witnessed unopened handoff, or human judgment has been
completed.

## Follow-up: dtype-label binding audit

An independent code audit found that the raw-byte sidecar builder allowed a
caller-provided `attention_source_dtype` to choose the row-sum tolerance, but
the signed phase-2 plan did not independently fix that expected dtype. The
source flag itself remains an unauthenticated assertion, so this did not
change the GPU smoke result or create a faithfulness estimate. The phase-2
manifest is now version 2 and binds an expected dtype; the verifier rejects a
reindexed dtype label that conflicts with that plan. Synthetic regression
tests cover a deliberately relabeled row and a re-signed phase-2 index.
Focused capture-contract/phase-freeze tests pass **57/57**. Actual model dtype
authentication remains unresolved and all authentication/scoring flags stay
false. The full repository suite passes **1,995/1,995** in **71.756 s** after
this repair; an independent source audit confirmed the intended-plan dtype
comparison is reached by the tamper tests.

## Next action

Independently audit the run artifact and update the phase-2 sidecar
implementation log with the BF16 dtype-bound repair. Next, connect a
local capture output to the exact signed baseline/selection rows and add an
explicit runtime/model-source binding before considering a phase-2 candidate.
Continue human-free reproducibility checks while named independent reviewers,
trusted roster inputs, and the witnessed review handoff remain unavailable.
