# QTA v4 annotation-mask specificity diagnostic — frozen design

## Purpose and status

The completed v4 result is a test-associated, exploratory annotation-union
prediction-margin dependence proxy. A serious alternative explanation is a
generic response to temporal key ablation and its out-of-distribution effects.
This follow-up is designed to compare the observed annotation-union contrast
with a same-cardinality, same-run-shape translated temporal mask. It is a
post-result exploratory diagnostic, not a confirmatory test and not a direct
measure of generated-claim faithfulness.

No control geometry, GPU score, or output for this follow-up has been produced
yet. The protocol and selection rule are fixed before generating its outcomes.

## Frozen comparison

Keep the same ordered 147-row v4 cohort, processor batches, baseline-full
answer/foil, and QTA rule. The annotation mask is the set of annotation-covered
temporal groups. For a mask with group set `M` in a video with `T` temporal
groups, enumerate every nonzero integer translation `s` for which
`{m+s: m in M}` stays in `[0,T)`. Retain only translations that preserve both
the number of groups and the exact number of processor visual-token positions
masked. This preserves the mask's contiguous-run shape and group/token
cardinality; it does not control location-dependent content or OOD effects.

Choose one eligible translation without model outputs: minimize SHA-256 of the
UTF-8 bytes of the compact JSON array
`["qta-v4-mask-specificity-v1", "20261018", video_id, qid, str(offset)]`
(`ensure_ascii=False`, separators `(",", ":")`) over the eligible integer
offsets; break a hash tie by numeric offset. Do not wrap around, resample, or
choose based on scores. The masked-position count is the length of the exact
processor visual-token-position tuple used by the existing scorer, and the
translated mask must have identical cardinality. If there is no eligible
nonidentity translation (including full temporal coverage), retain the row in
the 147-row accounting and mark its specificity contrast structurally
unavailable. The specificity cohort is all and only rows with a geometry-valid
translation; freeze its size in the CPU preflight before any new model forward.
Run the GPU experiment only if at least one row is eligible; otherwise record
a predeclared geometry-only no-go and do not load the model.

The old v4 receipt does not store QTA-selected group identities, so old full
and annotation-mask cells cannot be paired to a rerun by margin agreement
alone. Recompute all four v4 baseline-full, baseline-annotation-mask,
QTA-full, and QTA-annotation-mask branches on the same batch for every one of
the 147 rows. Freeze the answer/foil from that fresh baseline-full result and
derive QTA groups from its captured attention. For each eligible row, add
baseline-placebo-mask and QTA-placebo-mask on that same batch and same freshly
selected QTA groups. Ineligible rows still receive the four v4 branches and
remain in the 147-row denominator, but have no specificity contrast. Define
`D(mask) = (margin_QTA_full - margin_QTA_mask) -
(margin_base_full - margin_base_mask)` and primary diagnostic
`specificity = D(annotation_union) - D(placebo_translation)`.
Positive specificity means the v4 QTA-vs-baseline margin-drop contrast is
larger for the benchmark annotation union than for one fixed translated mask.
For every row, the fresh baseline-full answer and foil indices must exactly
match the frozen v4 receipt; otherwise fail closed before scoring its other
branches and do not silently redefine the fixed prediction.

## Execution and analysis rules

- CPU preflight must bind exact v4 plan/census/preflight, source/data hashes,
  model processor manifest, the private v4 score receipt hash, row geometry,
  fixed answer/foil, candidate offset set, chosen offset, and mask-position
  count. It writes an owner-only aggregate/row-binding receipt and performs no
  model forward.
- Run a six-forward GPU smoke first on the first eligible row so it exercises
  all four annotation conditions and both placebo branches. It must reproduce
  the frozen baseline-full answer/foil indices and margin, bind the exact CPU
  preflight and prior v4 receipt, and pass the existing CUDA/BF16/free-VRAM/
  checkpoint gates.
- Only after independent audit of the preflight, smoke, and code may the full
  all-or-none eligible cohort run. No substitutions are allowed. Report all
  147 rows and failures, plus eligible-cohort finite-cohort mean, sample SD,
  range, and positive/negative/zero counts for `D(annotation_union)`,
  `D(placebo_translation)`, and their difference. Do not report p-values,
  population intervals, or use subgroup/threshold searches.
- On any required-branch failure, preserve exact requested/attempted/completed
  counts and per-row failure codes, but emit no completed eligible-cohort
  paired summary. Keep the 147-row denominator and structural availability
  table explicit.
- The existing v4 result was already observed. Treat this as exploratory and
  require an independent future sample/protocol before any confirmatory claim.

The prior full v4 run used 588 forwards in 2,539 seconds. The follow-up needs
four forwards for every row and two extra placebo forwards for every eligible
row. If 143/147 are eligible, that is 874 total forwards and a linear planning
estimate of about 3,775 seconds (63 minutes) of GPU time, plus a six-forward
one-row smoke and CPU geometry preflight. This is an estimate, not a guarantee.
Rights remain unverified; all media, individual scores, and receipts stay
local. No raw media or row-level results may be published.

## Limits

The placebo may overlap genuine evidence or another annotation. A translated
mask controls group count, run lengths, and masked-token count but not visual
content, semantic relevance, boundary position, or all distribution shift.
The v4 QID is processor-geometry conditioned and the official test-associated
cohort has unknown prior-exposure history. The result cannot establish claim
support, prediction use of reviewed evidence, causal faithfulness, or
generalization.

## Next action

Implement a separate local-only geometry-preflight/smoke/full runner and
synthetic tests for translation eligibility, exact mask cardinality, stable
selection, full-coverage specificity unavailability, zero-eligible no-run, and
finite-cohort arithmetic. Audit the code
independently, run the outcome-blind CPU preflight, and only then assess whether
the GPU smoke/full gates remain justified.
