# QTA paired-video sample-size scenarios (2026-10-18)

## Purpose and limits

Quantify how much the eventual video-disjoint confirmation size depends on
assumed paired-video ΔF and SD, without treating the current 30-row
benchmark-annotation proxy as an estimate of the direct claim-linked outcome.
No primary SESOI, direct outcome variance, or source-cluster distribution is
currently established; this is scenario planning, not a registered design or
sample-size recommendation.

The existing planner models one independent source-video-cluster mean of
paired ΔF per unit and exact two-sided one-sample noncentral-t power under iid
Normal differences, `alpha=0.05`. It does not model higher-level source
dependence, unusable rows, multiplicity, selection, or within-video question
aggregation. Its assumptions and limitations remain in
`scripts/plan_qta_paired_sample_size.py`.

## Recomputed scenarios

The deterministic planning function was evaluated at assumed means
`0.10, 0.25, 0.40` nat and assumed SDs `0.20, 0.40, 0.60` nat. The smallest
video-cluster counts achieving 80% / 90% power were:

| Assumed mean ΔF (nat) | Assumed SD (nat) | n at 80% power | n at 90% power |
|---:|---:|---:|---:|
| 0.10 | 0.20 | 34 | 44 |
| 0.10 | 0.40 | 128 | 171 |
| 0.10 | 0.60 | 285 | 381 |
| 0.25 | 0.20 | 8 | 9 |
| 0.25 | 0.40 | 23 | 29 |
| 0.25 | 0.60 | 48 | 63 |
| 0.40 | 0.20 | 5 | 5 |
| 0.40 | 0.40 | 10 | 13 |
| 0.40 | 0.60 | 20 | 26 |

For the current three-record feasibility cohort, even three effects all in the
same nonzero direction have a minimum two-sided exact sign-test p-value of
`0.25`. The sign test targets direction among nonzero effects, not the primary
mean ΔF estimand.

## Interpretation and next action

These numbers are sensitive to assumptions and are **not** a recommended
cohort size. In particular, `0.25` nat is shown only as one scenario; no
evidence establishes it as the practical SESOI for QTA. The proxy's observed
SD `0.14284` nat is not substituted here because its estimand is dependence on
one benchmark-annotation anchor, not dependence on independently reviewed
claim evidence. Direct claim-linked coverage remains 0/3 reviewed, 0/3
prediction-link judgments, 0/15 verified browser-clock/decoder correspondences,
and 0/3 scored.

Before any confirmatory sample-size choice, freeze a scientifically justified
SESOI, eligible source-video cohort/cluster unit, within-video aggregation,
missingness rule, and inferential/multiplicity plan. Then obtain a direct
claim-linked pilot variance or use an explicitly conservative external basis.
Until then, retain these scenarios as sensitivity analysis only.

The source used for calculation was
`scripts/plan_qta_paired_sample_size.py`, SHA-256
`6bde0e5b065d360607493f3d4481a61f7e0fae63719582f980139b43b15afdb6`.
