# QTA-biased QA-evidence interval generation pilot (2026-10-21)

## Research question and scope

Can a fixed question-conditioned temporal visual-key bias change the accuracy
of a Qwen2.5-VL generated QA-evidence interval relative to the NExT-GQA QA
time annotation? This is the closest executable grounding endpoint available
without inventing claim-support labels. It evaluates QA temporal localization,
not whether a separate generated natural-language explanation is true or
supported. The finite cohort is the previously frozen three-record
`claim_likelihood_claims_remaining_20261021_30.json` test packet; the sample is
development/exposure-known and results are exploratory only.

For each record, hold question, five choices, video frames, prompt, model,
greedy decoding, and interval parser fixed. First obtain an untreated eager
attention pass, aggregate question-token attention over temporal visual groups
in layers 5–16, and select the top `ceil(10%)` groups. Generate one interval
under each of three conditions: no intervention; a fixed `+ln(2)` bias on
selected video keys in layers 5–16 throughout answer-token decoding; and an
equal-size, deterministic hash-ranked disjoint temporal-group sham. The primary
finite-cohort contrast is selected-bias minus baseline per-record max tIoU
against the row's QA interval(s). Selected-bias minus sham is a specificity
diagnostic. Invalid outputs receive IoU 0 and remain in the denominator. No
p-value, population confidence interval, or confirmation claim is planned.

## Implementation and validation before GPU

Added `scripts/run_qta_intervened_interval_generation.py` and focused CPU tests.
Inputs and source dependencies are hashed into an owner-only receipt; exact
source video bytes must match the existing frozen packet. Outputs stay under
`data/private_nextgqa/` with mode 0600. The runner requires explicit
`--user-authorized-local-only`, uses the pinned 7B checkpoint, 4-bit weights,
BF16 compute, eager attention, and offline local assets. `--mode full` requires
a successful source/input-matched smoke receipt.

An independent code review found that the first attention-selector draft asked
the model to return `output_attentions=True`, which could retain full attention
maps from every layer and create unnecessary VRAM pressure. It is corrected
to `output_attentions=False`; the pinned eager self-attention forward hook
receives the current layer's attention matrix and reduces only selected
question/group values to CPU. This is a static-code expectation and must be
verified by the real-video smoke.

The autoregressive key-bias hook has a stateful DynamicCache-only check: square
prefill followed by Q=1 decode with K incrementing by one at every step across
selected layers. This closes a discovered ambiguity where K−Q alone cannot
distinguish real prefix positions from static-cache capacity or a cropped key
axis. The runner also records and verifies actual per-layer shape traces.
Independent code review checked this against the pinned Transformers 5.18
Qwen implementation: eager attention returns the current layer's map to the
module hook even with output-attention collection disabled, and the decoder
does not retain every layer map in the model output. The smoke is still needed
to verify this real-video path and actual DynamicCache traces. Model shard
bytes and processor-config files are not hashed by this pilot; the model is
bound by its pinned revision and local config/generation-config hashes.
Focused CPU tests passed **17/17** across the new runner and generation hook;
`py_compile`, CLI help, and scoped `git diff --check` passed. No GPU run for
this interval-generation intervention has yet occurred.

## First real-video GPU smoke and repair

The first smoke, `interval_generation_cohort30_smoke.json` (mode 0600, 6,055
bytes), took **314.5 s** including one 7B model load. Preflight and the
untreated selector forward passed; baseline interval generation completed
1/1. The selected-bias generation failed before completing because all Python
loop-created layer hooks shared the final loop index, so the second selected
layer was incorrectly diagnosed as a duplicate within one generation step.
The receipt records 0 effective rows, baseline 1/1, treatment 0/1 completed,
sham 0/1, and failure code
`generation_layer_called_twice_before_step_complete`. No condition-triplet or
tIoU estimate is accepted from this partial run.

Fixed the layer-index closure binding in the hook installer and added a
two-layer, two-step regression. Post-repair focused CPU validation passed
**18/18** across the runner and generation-hook tests; `py_compile` and scoped
whitespace checks passed. The failed receipt is retained and will not be
overwritten or included in any estimate. The retry uses the same model,
cohort, prompt, metric, selector, bias dose, and parsing rule.

## GPU smoke gate and workload

The next command performs the fixed first-record smoke: one question-attention
selector forward and three greedy interval generations (baseline, selected
bias, hash sham) after one 4-bit BF16 model load. Prior local logs show pinned
7B weight loading alone took about 4.5 minutes once, while total 3-row
benchmark-mask execution was about 6 minutes; runtime varies with disk/cache
state. Peak VRAM should remain below the device's 16 GiB; the runner enforces
12 GiB free before load and 2 GiB before each generation. Run one exact
elevated command using the existing `.venv`; no parallel GPU job is permitted.

```bash
.venv/bin/python scripts/run_qta_intervened_interval_generation.py \
  --mode smoke \
  --output data/private_nextgqa/interval_generation_cohort30_smoke_retry01.json \
  --user-authorized-local-only
```

Only after verifying the private smoke receipt, its row arithmetic, source
hashes, all three intervention hooks, and DynamicCache traces should the full
three-row run proceed. The commands/results have not been published and no
raw video, question/choice text, or claim text may be shared. Rights remain
unverified under the user's local-only authorization. Human review remains
factually unperformed, even though its collection was waived for this
exploratory local run.

## Interpretation boundary

Even a positive tIoU difference would show only that this fixed question-
attention bias improved the pinned model's generated QA interval on this known
three-record cohort, relative to baseline and hash sham. NExT-GQA QA intervals
are not gold support spans for the independently generated explanation claim.
Clock/player-to-decoder alignment was not independently calibrated. The test
cannot establish explanation validity, claim truth, or direct faithfulness.
Claim-specific explanation grounding remains unmeasured without claim-to-frame
support labels. Report valid-N, invalid generations, every paired value, and
whether the sham behaves similarly; do not tune the prompt, top fraction,
layers, or bias dose using these scores.

## Repaired real-video smoke outcome

The repaired smoke completed on the first fixed row at 2026-10-21 UTC using
the existing 7B 4-bit/BF16 checkpoint and RTX 5070 Ti. The owner-only receipt
is `data/private_nextgqa/interval_generation_cohort30_smoke_retry01.json`
(mode 0600, 15,979 bytes, SHA-256
`07dafc36276c82f02a814ece2eec82d67a42f60fa14a3cb412d202c394a1e839`). All three
generation branches completed (3/3), each with 29 generated tokens, and all
selected layers recorded the expected square prefill followed by 28
incremental `Q=1, K+1` DynamicCache calls. The run took 337.66 seconds and
peaked at 7.34 GiB allocated VRAM.

However, baseline, selected-bias, and hash-sham decoded to exactly the same
token string (identical SHA-256 in the receipt); all three failed the strict
JSON parser and therefore scored IoU=0. The fixed-row result is a
protocol/behavioral null with zero valid intervals, not evidence that the
intervention improves or worsens temporal grounding. The cache-hook execution
path is verified, but this receipt does not log modified attention-mask values
or logits, so it cannot establish that the intervention changed internal
attention or model scores. Do not run the three-row full cohort under this
nonfunctional generation protocol.

The next research use of this result is a protocol repair: inspect the exact
generated string locally (do not add it to a publishable artifact), replace
the brittle free-form JSON request with a predeclared constrained interval
format/parser, and add per-generation intervention-value/logit diagnostics.
Then repeat only this fixed smoke; require a valid interval and an auditable
treatment-vs-sham internal effect before considering a full run. This
QA-localization pilot remains secondary to the thesis-relevant question of
whether the internal intervention improves claim-specific selective
prediction dependence. Across prior fixed-prediction QTA cohorts the mean
`DeltaF` has been near zero, changes sign, and is not reliably larger than
matched-sham changes; that negative exploratory pattern is a stronger current
thesis lead than a claimed grounding gain. A defensible thesis question is
whether question-attention-selected temporal-key bias improves claim-evidence
selective `DeltaF` over equal-budget sham on a fresh outcome-blind cohort;
current data do not support an efficacy conclusion.

## Constrained protocol amendment and completed GPU experiment

The original free-form JSON smoke was a failed protocol (3/3 branches emitted
the same 29-token string, strict parses 0/3); its scores are excluded. Before
seeing any constrained-protocol outcome, the endpoint was amended to select
one inclusive interval from all 136 intervals over the 16 sampled frames using
the exact two-digit `SS,EE` grammar. Each code maps to contiguous seconds using
midpoint cells between the actual decoded frame PTS, clipped to media duration.
This quantizes localization to 16 bins and changes the measurement resolution;
it does not preserve the earlier timestamp-generation endpoint. All eligible
codes are grammar-valid by construction, so syntactic validity is not a model
quality result. The sampling resolution is coarse, the prompt asks for QA-answer
evidence, and NExT-GQA intervals are not claim-specific support labels.

The final implementation binds all 136 suffix token sequences to the rendered
text prompt while excluding processor-expanded video tokens from the text-only
tokenizer boundary check. It constrains the generated continuation, explicitly
uses the tokenizer's chat EOS token despite the model's two-EOS generation
configuration, and validates Transformers' one-dimensional prefix callback.
The mask audit compares selected-cell changes with the bias representable in
the mask dtype (BF16 value `0.69140625` for requested `ln(2)`), while recording
the requested dose. Receipt validation requires generated token count,
forward-call count, and the full shape trace to agree. Video metadata tolerates
only 1e-7 absolute/relative floating-point error in frame count; exact sampled
indices and FPS remain checked.

Three actual GPU attempts exposed and repaired implementation defects: (1)
processor-expanded multimodal IDs cannot be compared directly with text-only
tokenizer IDs; (2) this Transformers checkpoint configures two EOS IDs; (3) its
prefix callback supplies a one-dimensional sequence; (4) BF16 mask addition
rounds `ln(2)`; and (5) one video reports a floating frame count
`15.999999999999998`. Those failed attempts are preserved in private receipts
and are excluded from estimates. Final focused CPU validation passed **26/26**
tests, `py_compile`, CLI help, and scoped `git diff --check`.

The final smoke completed one real frozen video and all three generation
conditions. Receipt SHA-256:
`e7a30efeb015e045455095983c9e890a3a14570c83d25b3f7195342079e69194`;
mode 0600, 3/3 branches, 7.35 GiB peak allocated VRAM, exact source and packet
binding, and five checkpoint weight-shard hashes. The three outputs were
identical and each condition scored tIoU `0.1863`. For both selected and sham,
all 12 layers over 6 generated-token forwards recorded their dtype-realizable
mask change; baseline recorded no intervention hook.

The final full run completed the frozen three-row cohort, 9/9 generations,
with no invalid generations. Owner-only receipt:
`data/private_nextgqa/interval_generation_cohort30_constrained_full_20261021_02.json`,
mode 0600, 30,020 bytes, SHA-256
`cb029dcdddd2bdd5b60e38224456982a754ecdd6bb1e9f6806838f33e6bb6fb7`;
runtime 86.04 s, peak allocated VRAM 7.48 GiB. Source/model/packet provenance,
smoke binding, row arithmetic, branch counts, token/forward counts, and every
intervention-mask audit were independently revalidated. Mean tIoU was **0.08300**
for baseline, selected bias, and hash sham. Selected-minus-baseline was
**[0, 0, 0]**; selected-minus-sham was **[0, 0, 0]**. Each row's three output
codes were identical. No inferential test or population interval is
appropriate for this fixed `n=3` pilot.

This is a valid null outcome for the declared endpoint: the requested internal
mask changes occurred, yet greedy QA-interval predictions and tIoU did not
change on these three videos. The audit proves that the additive mask changed,
not that the downstream attention output or logits changed; that mechanistic
link remains unmeasured. This result cannot establish improvement or
non-improvement in natural-language explanation faithfulness. Do not tune the
bias or selector using these outcomes. A thesis experiment should freeze a new
cohort and human-verified claim-support intervals, then compare claim-specific
fixed-prediction evidence-ablation dependence under support-key intervention
versus matched sham. If that review is unavailable, this reproducible
three-video null pilot remains usable as a feasibility/measurement chapter,
with direct claim-faithfulness explicitly untested. Rights remain unverified;
all receipts and source videos remain local.

## Post-hoc raw-logit mechanism diagnostic (exploratory)

After the first complete result showed identical intervals in all arms, I added
a post-outcome diagnostic of the actual raw LM
logits at each generated code/EOS token. The full vocabulary logits are reduced
in memory to chosen-token log probability, raw top-1 ID, top-1/top-2 gap and
entropy; full arrays are not stored. For each row, arm contrasts are computed
only across the common emitted token prefix. This is explicitly a post-hoc
mechanistic diagnostic, not an endpoint for efficacy or faithfulness, and it
does not repair the exploratory status of the n=3 cohort.

With the current source and updated schema, a new smoke plus full run completed.
The current-source-bound full receipt is
`data/private_nextgqa/interval_generation_cohort30_logits_full_20261021_01.json`,
mode 0600, 52,899 bytes, SHA-256
`807c77719878165ac95832a6e1fa44d260b55084d6360876452f12dc7c28fb41`;
3/3 videos, 9/9 generation branches, current validator passed, 77.90 s and
7.48 GiB peak allocated VRAM. Its smoke gate is
`interval_generation_cohort30_logits_smoke_20261021_01.json` (0600; SHA-256
`9fd6bc4cd1d344c4449e05426c2f5e76814f558ef94036fc8766df87e4c4329e`).
Current code, packet, and pinned model metadata/weight hashes were rebound;
no raw logits or video content were exported.

In all three videos, all arms emitted the same constrained code and all QA tIoU
values remained equal. Mean tIoU was `0.08300`; selected-minus-baseline and
selected-minus-sham remained `[0,0,0]`. Yet the chosen-token log probability
shifted on the same generated paths. Per-video means across the six code/EOS
steps for selected-minus-baseline were **+0.00583, +0.01054, +0.02475 nat**;
selected-minus-sham were **−0.00115, −0.00003, +0.02375 nat**. Averaging those
three video-level means descriptively gives +0.01371 and +0.00753 nat/token,
respectively. Mean absolute step change was 0.02871 nat versus baseline and
0.02392 versus sham. These 18 token steps are nested within only three videos
and are not independent samples; no test or CI is reported. Raw unconstrained
top-1 changed at one of six steps in one video, while the prefix-constrained
interval code remained the same. The emitted constrained token matched raw
vocabulary top-1 at 5/6 steps in each baseline and sham row; selected bias
matched it at 5/6, 5/6, and 6/6 steps. This shows a measurable local
distribution shift under intervention in this pilot, not improved evidence
grounding.

The thesis lead is now sharper and more defensible: **internal temporal-key
steering can change token-level confidence while leaving the selected QA
interval and its tIoU unchanged** on this frozen pilot. A worthwhile study is
to test, on a new human-reviewed cohort, whether such confidence movement
tracks claim-specific support ablation rather than generic confidence drift or
matched-sham effects. Until that direct measurement exists, do not call this a
faithfulness improvement. The earlier constrained full receipt
`..._full_20261021_02.json` was produced by an earlier source revision; its
counts/arithmetic were independently recomputed, but the current stricter
validator requires the subsequently added raw-logit fields. The new
`...logits_full...` receipt is the reproducible current-schema artifact.
