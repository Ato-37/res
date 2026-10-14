# Claim-linked evidence-frame protocol design (revised 2026-10-14)

## Purpose and status

This is an exploratory pre-execution protocol design, revised to match the user's selection:
fixed-answer log-margin, a claim-linked evidence-frame mask, all eligible videos
as primary, and baseline-correct videos as a secondary cohort. It reports no
claim-mask efficacy run. The existing research plan's Holm correction within
each predeclared secondary family is adopted; the secondary families are
baseline-correctness strata and question-type strata, with family membership
frozen before scoring. The primary is one contrast and has no within-study
multiplicity adjustment. The SESOI remains unfixed, so schema v4 rejects a
confirmatory stage. NOAA's required human review is incomplete and
NExT-GQA item-level rights are unresolved; real-data execution remains `no_go`.

## Frozen estimand

Each predeclared eligible video contributes one video-level QA unit. Before
mask scoring, freeze (a) the annotated gold answer label, (b) the five-choice
set and prompt, and (c) one independently reviewed answer-relevant claim with
its mapped temporal evidence interval. The same gold label and choices are
scored in every condition. The five answer labels are single tokens in the
existing metric path. For condition `z`,

`S_i(z) = log P(gold label | prompt, video under z) - max_{foil != gold} log P(foil | prompt, video under z)`.

The foil is the highest-scoring of the four fixed alternatives in each
condition, matching the existing five-choice margin implementation. This is a
gold-answer choice margin, not the likelihood of the claim text. Define

`d_i = [S_i(full) - S_i(claim-linked-frame-mask)] - [S_i(full) - S_i(control-frame-mask)]`.

The sole primary population is all predeclared eligible videos, independent of
baseline correctness. Baseline-correct means the unmasked greedy answer equals
the gold answer; baseline-correct and baseline-incorrect results are secondary
strata. A positive `d_i` indicates a larger gold-margin drop under the selected
claim-linked mask than under the matched temporal control. It is operational
selective dependence under this intervention, not proof of claim truth,
causality, or a faithful natural-language explanation.

## Mask mapping and control

The selected condition masks sampled frames with timestamp `t_i=frame_index /
decoded_source_fps` inside the independently reviewed claim interval, using
inclusive bounds `start <= t_i <= end`. The frozen interval determines the
evidence mask and resulting count `k`; observed model scores cannot alter this
mapping. For controls, sort evidence ranks `e_1..e_k` and candidate control
ranks `c_1..c_k`, then choose a `k`-frame subset outside the evidence set
minimizing `sum_j |e_j-c_j|`; ties resolve to the lexicographically lower
ordered control-rank tuple. If fewer than `k` controls exist, mark the item
pre-scoring ineligible; do not substitute after outcomes. Full,
selected, and control conditions share frame sampling, fill, preprocessing,
model, precision, prompt, gold target, and execution settings.

## Analysis and gates

Report requested/effective video counts, pre-run exclusions, failures, all-video
primary effect and uncertainty interval, baseline-correct/incorrect secondary
effects, and Holm-adjusted p-values within each predeclared secondary family.
Do not select or replace
the primary cohort based on correctness. Do not transfer the old head-search
`+0.25 nat` threshold without a separate pre-outcome justification. Mask OOD,
temporal-control content mismatch, claim-to-interval validity, and one QA per
video are limitations to retain.

The CPU-only validator contract is in
[`docs/claim_mask_protocol_contract.md`](../docs/claim_mask_protocol_contract.md).
Structural validation is not reviewer authentication, rights verification, or
execution authorization. No GPU inference or efficacy sampling is allowed
until the independent review and dataset-rights gates are actually resolved.

## Next action

Keep the real-data run blocked. Independently validate the schema and synthetic
checks, then resolve review provenance and item-level rights before preparing a
new smoke. Do not reuse gated local untracked sample candidates as if their
status had changed.
