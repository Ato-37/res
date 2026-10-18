# QTA evidence-retention contrast: prospective design addendum

## Status and purpose

This is a design proposal for a future claim-reviewed, independently clock-
verified video-disjoint cohort. It is not preregistered, implemented, or
authorized for scoring. It does not change the primary removal contrast in the
existing v1 draft or reinterpret the completed 147-row diagnostic.

The proposal adds a retention-side contrast because recent VLM work evaluates
both removal-based comprehensiveness and retention-based sufficiency. The
published image/document study reports heterogeneous modes and a gap between
human regions and model-attention regions, but its results do not validate this
temporal video protocol: Song et al., arXiv:2609.00830v1,
https://arxiv.org/abs/2609.00830. VideoQA causal-scene work by Wei et al. uses a
separately trained model, so it motivates temporal evidence questions without
validating this frozen-model explanation metric:
https://arxiv.org/abs/2305.04224.

## Proposed secondary estimand

Keep the existing fixed baseline-full option pair `p/q`, answer-query position,
layers, QTA dose, and primary removal contrast unchanged. For operation
`g ∈ {baseline, QTA-Bias}` define the same fixed log margin
`m_g(z)=log P_g(p|z)-log P_g(q|z)`.

Let `E-only` keep the reviewed evidence-group visual keys available at the
answer-query intervention layers and ablate the other visual-group keys. Let
`C-only` keep the equal-count matched control-group keys available and ablate
the other visual-group keys. Keep all nonvisual keys and the processor batch
unchanged; if the intervention hook cannot isolate these positions exactly,
fail closed for the whole cohort.

Define a comparative retention contrast:

`R_g = m_g(E-only) - m_g(C-only)`

`ΔR_QTA = R_QTA-Bias - R_baseline`.

Positive `ΔR_QTA` means the QTA operation increased the fixed `p/q` margin
advantage of retaining reviewed evidence keys over retaining an equal-count
control set. It is a comparative, intervention-specific retention descriptor,
not the standard probability-based sufficiency metric and not proof that the
claim is true or that all relevant evidence is present. Interpret it together
with the unchanged removal primary `ΔF_QTA`; do not replace one with the other
or combine them into a single post-hoc score.

## Workload and gates

The existing v1 layout uses baseline, QTA, and sham operations across full,
evidence-ablation, and control-ablation conditions. Adding `E-only` and `C-only`
for each operation produces 15 branches per row, with baseline-full shared with
the frozen selector and `p/q` stage. This adds six forwards per row over v1. A
future smoke must verify exact keep/ablate position sets and demonstrate that
the selected evidence/control key counts are equal, disjoint, causal, and
otherwise leave the query mask unchanged. Run only after the pre-load human,
clock, cohort, scorer, and phase-receipt gates pass. The feasibility cohort
must remain descriptive; any efficacy claim requires a disjoint cohort and a
pre-outcome precision/SESOI, sample-size, uncertainty, and multiplicity plan.

The first CPU-only synthetic geometry check is implemented in
`scripts/test_qta_retention_contrast_geometry.py` and passes. It uses four
synthetic two-token groups, equal-size disjoint evidence/control sets, and the
existing fixed-query ablation helper. It verifies that the retained visual
positions are exactly the requested set, all other visual positions are
ablated, and unrelated query rows remain unchanged. This is helper-level
software validation only; it does not run a model or validate real temporal
groups.

## Known alternatives and failure conditions

Retaining only selected keys can be severely out of distribution; text tokens
may still carry indirect visual information, and key ablation at chosen layers
does not remove information from every pathway. Equal key counts do not ensure
equal semantic content or temporal coverage. Therefore report both raw paired
margins and mask sizes, retain full-video baselines, include failure counts,
and avoid describing this as complete evidence sufficiency. A mismatch in
token positions, unequal evidence/control cardinality, full visual mask,
nonfinite scores, or incomplete branch counts invalidates the whole cohort.

## Next action

Use this addendum only during prospective protocol review. First complete the
review-only human gate and the production phase/scorer integration; then freeze
or reject this secondary contrast before any baseline or treatment outcome from
the new cohort is inspected. Current human-reviewed claim count remains 0/3.
