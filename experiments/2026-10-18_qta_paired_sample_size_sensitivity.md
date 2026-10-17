# QTA paired-sample planning sensitivity (2026-10-18)

## Purpose

The QTA study's three-video set is a descriptive feasibility cohort, while the
thesis objective asks whether the fixed internal intervention increases
prediction dependence on reviewed evidence. The protocol requires a later
video-disjoint cohort and a justified precision/SESOI plan, but no direct QTA
effect or paired-difference variance exists yet. This outcome-blind calculation
shows how strongly a future sample size depends on those unknown quantities. It
does not choose the SESOI, estimate an effect, preregister a cohort, or license
real-data scoring.

## Method and assumptions

Added `scripts/plan_qta_paired_sample_size.py`. For each assumed mean paired
change `DeltaF` and paired-difference SD, it calculates exact power from the
noncentral t distribution for a two-sided one-sample t test at alpha 0.05, and
searches for the smallest `n` reaching 80% or 90% power. Distributional
assumptions are independent video-cluster means with Normal paired differences.
`n` therefore counts independent source videos, not QA rows. If several
questions come from one source video, the future design must define its
video-level aggregation before outcomes are observed; the aggregation target is
not currently frozen. The 0.10/0.25/0.40 nat and 0.20/0.40/0.60 nat SD grid is a
scenario grid only. In particular, 0.25 nat is not inherited as a QTA threshold
from the closed head-search branch.

The exact command was:

```bash
PYTHONPATH=scripts ./.venv/bin/python scripts/plan_qta_paired_sample_size.py
```

It evaluated nine effect/SD combinations. Selected observed calculations from
that deterministic output are:

| Assumed mean `DeltaF` | Assumed paired SD | Power at `n=3` | `n` for 80% | `n` for 90% |
|---:|---:|---:|---:|---:|
| 0.10 nat | 0.40 nat | 0.059 | 128 videos | 171 videos |
| 0.25 nat | 0.20 nat | 0.244 | 8 videos | 9 videos |
| 0.25 nat | 0.40 nat | 0.103 | 23 videos | 29 videos |
| 0.25 nat | 0.60 nat | 0.074 | 48 videos | 63 videos |
| 0.40 nat | 0.40 nat | 0.179 | 10 videos | 13 videos |

Focused tests passed **8/8** after the audit-driven fixes. They check null
rejection at alpha, monotonicity, minimality of the selected `n`, sensitivity
labeling, invalid-input rejection, generator-valued scenario grids, and
disallow target power at or below alpha. The full `scripts` suite then passed
**1,610/1,610** in **38.974 seconds**; expected negative-path fixture messages
and Transformers deprecation warnings appeared. `git diff --check` passed. No
media, model, GPU, or private review output was read. The environment's
installed SciPy version was 1.18.1.

## Interpretation and limits

The calculations show that three videos cannot give useful confirmatory power
across most of this illustrative grid. A nominal 80%-power design for a
0.25-nat mean requires 8–48 independent videos as the assumed SD ranges from
0.20–0.60 nat; this is conditional arithmetic, not evidence that a 0.25-nat
effect is meaningful or attainable. Without an outcome-independent SESOI,
empirical SD estimate from an allowed pilot, and a frozen source-video
aggregation/cohort rule, no single sample-size value is justified. Normality,
independence, attrition, and source-cluster homogeneity are unverified. No
multiplicity adjustment is included because this is a single-primary-test
scenario calculation; any secondary families need their own plan.

## Faithfulness progress

- `直接評価済み`: **0/3** reviewed claims, **0/3** prediction-link judgments,
  **0/15** verified browser-player/decoder-frame correspondences, **0/3** real
  QTA scores; no empirical `DeltaF`, uncertainty interval, or test.
- `補助診断のみ`: nine analytic sample-size scenarios, **8/8** focused tests,
  and **1,610/1,610** full-suite tests.
  These are not treatment-effect or faithfulness evidence.
- Human review and trust/timing/phase gates remain incomplete. Protected-data
  rights remain unverified and material remains local.

## Next action

An outcome-blind researcher must set a scientifically defensible QTA SESOI and
video-cluster aggregation rule, or the three-row study remains explicitly
descriptive. The human review and 15-pair clock evidence are independent
prerequisites to the three baseline forwards; they are still the immediate
blocking step for any direct real-data score. Do not use this planning table to
alter thresholds after observing QTA outcomes.

## Provenance

- `scripts/plan_qta_paired_sample_size.py` SHA-256:
  `c1f52e8fe9572100148c503f49ece2e1988f401e1d6be191b095f954f3b7be8c`
- `scripts/test_plan_qta_paired_sample_size.py` SHA-256:
  `8e5e875e265ec1e432d4c82826c920ae01761bbc03a77c9858485a1ab62cdb63`
- Runtime: Python 3.12.14, SciPy 1.18.1; analytic noncentral-t power.
