# QTA video-level exact sign-test resolution (2026-10-18)

## Purpose

The planned first QTA cohort contains three independent source videos and is
explicitly a descriptive feasibility pilot. Quantify what a distribution-free
two-sided test of video-level effect direction could resolve at this sample
size, without using outcome data or changing the primary mean-`DeltaF`
estimand.

## Method and reproducible output

Extended `scripts/plan_qta_paired_sample_size.py` with the exact binomial sign
test under an assumed independent probability that a source-video-level
`DeltaF` is positive conditional on `DeltaF != 0`. The power table treats n as
the number of independent nonzero video effects and enumerates every possible
positive-sign count using the exact two-sided `scipy.stats.binomtest` p-value
at alpha 0.05. The zero-effect rate is not modeled. The scenario inputs
`Pr(DeltaF > 0) = 0.7, 0.8, 0.9` are illustrative assumptions, not measured
effects or chosen SESOIs. No dataset, model, GPU, or private review data was
read.

Command:

```bash
PYTHONPATH=scripts .venv/bin/python scripts/plan_qta_paired_sample_size.py
```

Observed deterministic output:

| Assumed `Pr(DeltaF>0 | DeltaF!=0)` | Power at 3 nonzero effects | n for 80% power | n for 90% power |
|---:|---:|---:|---:|
| 0.70 | 0 | 49 | 65 |
| 0.80 | 0 | 20 | 28 |
| 0.90 | 0 | 12 | 15 |

At n=3 nonzero effects, even three positive effects yield an exact two-sided
`p=0.25`; there is no attainable rejection at alpha 0.05. The smallest n at
which any outcome can reject is 6, where six effects in one direction give
`p=0.03125`.

## Interpretation and limitations

This establishes that the planned n=3 cohort cannot support confirmatory
inference using this exact directional test. It does not establish that the
primary mean effect is zero, and it is not a power calculation for the
primary video-level mean `DeltaF` t-test. That separate parametric sensitivity
calculation remains in
[`2026-10-18_qta_paired_sample_size_sensitivity.md`](2026-10-18_qta_paired_sample_size_sensitivity.md).
The sign test addresses median/directional consistency and discards effect
magnitude; its illustrative power depends strongly on the assumed positive
sign probability. Independence across source videos and handling of tied zero
effects must be frozen before a confirmatory cohort is selected. Required n is
an effective nonzero-sign count, not a total recruitment target; a nonzero
tie rate requires additional videos and an outcome-blind model.

Focused tests passed **14/14**. The complete `scripts/` unit suite passed
**1,658/1,658** in **40.466 s**. An independent auditor reimplemented the
binomial enumeration with exact rational arithmetic and confirmed the key
values, including the nonzero-effect n interpretation. `git diff --check`
passed. A first focused run
exposed one incorrect expected arithmetic value in a new test; the independently
enumerated exact binomial probability corrected it before the passing rerun.

## Faithfulness progress

- `直接評価済み`: **0/3** independently reviewed claims, **0/3** prediction-link
  judgments, **0/15** verified browser-player/decoder-frame correspondences,
  **0/3** claim-linked QTA scores; no empirical `DeltaF`, confidence interval,
  hypothesis test, or multiplicity-adjusted result.
- `補助診断のみ`: exact sign-test resolution and scenario power calculations;
  14 focused tests and 1,658 full-suite tests. These are design calculations
  and engineering verification, not faithfulness evidence.
- Human, witness/trust, timing/calibration, and production scoring gates remain
  incomplete. Rights remain unverified; this calculation did not access
  protected inputs.

## Next action

Keep n=3 labelled descriptive. Before any confirmatory QTA study, set the
primary estimand's video-level aggregation, a defensible SESOI, usable-video
count and inference method without access to treatment outcomes. Direct
evidence still first requires the independently reviewed claim intervals,
15/15 clock/frame correspondence, and two-stage phase gate; no real score or
GPU run is authorized by this planning calculation.

## Provenance

- Planner SHA-256: `6bde0e5b065d360607493f3d4481a61f7e0fae63719582f980139b43b15afdb6`
- Test SHA-256: `8c8ec4f80557ab82c3a20ae87997303d781bbe40b7846cf81e9f6f73093c29b4`
- Environment: Python 3.12.14, SciPy 1.18.1; no random seed required for exact enumeration.
