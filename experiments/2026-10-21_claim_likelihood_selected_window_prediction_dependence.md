# Claim-likelihood selected-window prediction dependence (2026-10-21)

## Purpose and status

This post-hoc exploratory design tests whether a temporal window selected by a
smaller video-language model's likelihood for a saved claim is associated with
the 7B model's answer margin. A failed free-form 3B interval-generation smoke
motivated avoiding generated JSON timestamps; this is a different, discrete
window-ranking procedure, not a retry of that branch. It is not preregistered
or confirmatory, and its development history must remain visible.

The user waived human review as a prerequisite for local exploratory work.
This is recorded as a collection waiver only: actual human review remains
unperformed. No machine-selected window will be represented as a human-verified
evidence span.

## Cohort and selection

Use a new test-split cohort of exactly three distinct eligible videos, frozen
before claim generation and window scoring with the repository's deterministic
candidate loader and conservative union of source exclusions plus all locally
discoverable scored-media digests. Bind lock assignments, selected and batch
pair-filter hashes, generation packet hashes, source-video bytes, and source
code. Do not reuse the just-scored frame-mean cohort. If three eligible rows
cannot be frozen, preserve the shortfall and do not substitute rows.

Generate one saved claim per frozen row with the existing pinned 7B generator
and fixed 16-frame prompt. Decode the same exact video into 32 uniform frames.
Partition those frames, in display order, into four non-overlapping windows of
eight frames. The pinned Qwen2.5-VL-3B revision scores each saved claim by
teacher-forced mean token log likelihood under the unchanged question/options
prompt and each window. All candidate windows have the same frame count,
resolution policy, target text, and temporal sampling step. Select the highest
mean log-likelihood window; break ties by earliest window. Store all four
scores and the selected window in the private receipt. This is adaptive
selection over four candidates and is subject to winner's-curse bias.

The fixed 7B answer endpoint has three conditions: full video, black-mask the
selected eight-frame window, and black-mask a deterministic equal-count
temporal-rank control disjoint from the selected window. Score the same
gold-choice minus best-foil next-token log-probability margin in each condition.
Primary descriptive value is `margin(control mask) - margin(selected-window
mask)` in nats. Preserve full-video accuracy and gold/foil components to show
whether the contrast reflects gold movement or foil movement. Require all
three rows, 12/12 3B window-scoring forwards, and 9/9 7B scoring forwards; any
invalid window, missing matched control, or failed forward yields no cohort
estimate. Do not replace rows or compute an incomplete-cohort mean.

## Meaning, limits, and gates

A positive contrast means the 7B answer margin decreased more when masking a
window selected by the 3B model's claim likelihood than when masking the
matched control. It does not establish that the claim is true, that the 3B
window semantically supports the claim, that either model causally used the
claim's evidence, or that explanation faithfulness improved. Both checkpoints
are in the Qwen2.5-VL family; this reduces exact-checkpoint reuse but is not
independent human validation. The answer margin is not accuracy; masking can
be out-of-distribution; and selection among four windows inflates the chosen
likelihood by construction. The data-derived endpoint remains exploratory.

Before GPU scoring: validate four-window partition/tie behavior and
target-logprob arithmetic on CPU; bind each packet row to the lock/filter and
video bytes; smoke all four 3B windows and all three 7B conditions on the first
frozen row; check at least 12 GiB free VRAM and 8 GiB host memory before each
model load, and existing per-transfer/forward guards. Use the existing local
`.venv`, pinned local checkpoints, 4-bit weights and BF16 compute. Keep all
protected inputs and detailed results local; rights remain unverified.

## Execution result (2026-10-21)

The test cohort was frozen before claim generation/scoring after scanning 20
candidates. The broadened local exclusion set contained 43 previously scored
media digests; selected-media overlap with that set was zero. Exactly three
distinct rows were frozen and claims were generated for all three. This remains
an exploratory test-split cohort, not a fresh confirmatory sample.

The pinned Qwen2.5-VL-3B model scored each claim against four equal chunks of
eight of the same 32 uniformly sampled frames. It selected chunk indices 0, 2,
and 1 across the three rows. All 12 forwards completed. The selector smoke ran
as 4/4 windows on one row; the other two rows ran as 8/8. The receipts were
combined only after requiring identical cohort/filter/source hashes, three
unique rows, and 12 total windows.

For downstream QA scoring, exact PTS binding and frame assignment were checked
on CPU. A first draft converted endpoint seconds to decimals and masked only 7
of 8 frames on two videos due to floating-point boundary rounding. The runner
now uses exact rational PTS from the decoder. The repaired path matched the
selector PTS hash and masked the exact selected 8/8 frames, with an equal-count
8-frame temporal control, on all three rows.

The Qwen2.5-VL-7B scorer completed a one-row 3-forward smoke and the full
9-forward run. All full-video gold-choice margins were positive (3/3 gold
choices ranked above every foil). Per-row values, in run order:

| Row | Full-video margin | Selected-window mask margin | Control mask margin | Control − selected mask |
|---:|---:|---:|---:|---:|
| 1 | 4.750 | 2.625 | 3.000 | +0.375 |
| 2 | 2.500 | 2.500 | 2.500 | approximately 0.000 |
| 3 | 3.500 | 1.000 | 3.625 | +2.625 |

The finite-cohort descriptive mean of `control − selected-window mask` is
`+1.000 nat`; median `+0.375`; range approximately `0.000` to `+2.625` (one
tiny negative value, −1.34e−7, is floating-point noise). No CI, significance
test, or population estimate is reported for n=3. GPU was an RTX 5070 Ti, CUDA
12.8; scorer peak allocated VRAM was 11.05 GiB and the final full run took 68.3 s.
The scorer used the pinned Qwen2.5-VL-7B revision, 4-bit weights and BF16
compute.

Two outcome-component audits were then run without further model inference.
Recomputing each saved margin from the benchmark gold-choice and the best foil
matched the recorded values for all three rows. For the primary contrast, the
gold-choice component contributed `+0.022`, `−0.008`, and `+0.295` nat; the
best-foil component contributed `−0.353`, `−0.008`, and `−2.330` nat,
respectively. Thus the two positive row contrasts were driven mainly by a
higher best-foil log probability under the selected-window mask, with a smaller
gold-choice shift. This is model answer sensitivity, not specifically loss of
gold evidence.

As a separate post-hoc annotation proxy, each of the four candidate chunks was
compared with benchmark QA time interval(s), without using overlap in selection.
Selected-chunk temporal IoU averaged `0.0167`, versus `0.0941` averaged over all
twelve candidate chunks; the selected chunk had the best QA-interval IoU in
`0/3` rows. These annotations target benchmark QA events, not independently
reviewed support for the generated claims, so this does not prove the selected
chunks are unfaithful. It does weaken a claim that this selector reliably finds
QA-annotated temporal evidence. Both audits are post-hoc and descriptive at n=3.

## Interpretation and remaining limitations

Observed: in this fixed three-item exploratory set, masking the 3B likelihood-
selected chunk reduced the 7B QA margin more than an equal-count temporal
control on two rows and had essentially no difference on one row. This is a
small, outcome-selected association, not a causal intervention on the 3B
localizer or proof that the explanation claim is true or supported by those
frames. Selecting the maximum from four likelihoods creates winner's-curse
bias; the 3B and 7B checkpoints share Qwen-family priors; black masking can be
out-of-distribution; and the 7B answer margin is not calibrated correctness.
Human claim/evidence review remains 0/3.

The reproducible selector receipt is
`claim_likelihood_window_selection_all_rerun_20261021_03.json`; the final 7B
score is `claim_likelihood_selected_window_7b_final_r2_20261021_03.json`. The
earlier one-row smoke is `claim_likelihood_selected_window_7b_smoke_20261021_03.json`.
These contain private paths and QA-linked records and must remain local. The
derived local audits are `claim_likelihood_window_gold_overlap_20261021_03.json`
and `claim_likelihood_window_answer_components_20261021_03.json`. No protected
output was staged or published.

## Separate frozen-cohort feasibility follow-up

After the initial result and post-hoc audits, a separate test-split cohort was
frozen outcome-blind for a possible exploratory replication. Its freezer scanned
23 candidates and selected 3 unique rows after excluding all 46 locally scored
claim-media digests in the frozen exclusion set; overlap was 0. This follow-up
is post-result exploratory cohort construction, not a confirmatory independent
replication. The pinned 7B generator produced valid structured claims for 1/1
smoke row and 1/2 remaining rows. The other row failed the fixed prompt's
`claim_sentence_shape_invalid` check. Thus only 2/3 locked rows are scoreable;
no rows were replaced, and no 3B window or 7B answer-margin scoring was run for
this cohort. Its incomplete receipts remain local. This is generation attrition,
not a null faithfulness result, and the successful two rows are not averaged.

## Next action

Independent receipt audit and exact-PTS/partial-smoke tests passed. No
Qwen-family alternative checkpoint is present in the local HF cache, so another
run with the same family would not provide independent localization. The current
localization-by-likelihood method has weak annotation overlap (0/3 selected
chunks were best against benchmark QA intervals) and its margin contrast is
mainly foil movement. Do not pool the failed follow-up's two surviving rows or
retune this selector from outcomes. A new experiment needs a prospectively
defined independent localizer or actual evidence review. No CI, p-value, or
population claim is appropriate at n=3.
