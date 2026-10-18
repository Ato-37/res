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
used in the power calculation itself. In the later cohort-frame reconciliation
below, local candidate videos were streamed for SHA-256 only, and local private
exposure/run-lock JSON was parsed in memory; no video was decoded or viewed,
and no per-record identifiers or hashes were emitted.

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

Cross-reading the independent local-media audit gives a practical constraint:
the NExT-GQA test split has 990 mapped video IDs, but only 500 mapped media
files are currently local. Of those 500, 85 matched the supplied partial
prior-exposure index; the remaining 415 are not certified clean. A stricter
outcome-blind audit of one local-tree cohort method found **zero** eligible
test videos after the union of known path, video-ID, video/question-pair, and
media-hash exclusions; validation also had zero survivors. That audit covered
3,132 structurally eligible test rows across 500 videos before exclusions.
However, a separate ZIP-based builder later reported 490 unique media after
its own exclusion rules. This checkpoint independently rehashed those 490
extracted files (**1,990,584,365 bytes**), confirmed 490 distinct byte hashes,
and found **0/500** exact-byte overlap with the separate local mapped media
tree. The candidate-pool summary's test-CSV, gsub, video-map, prior-index,
HMAC-key, and 14-manifest-set fingerprints all match current local inputs; the
archived source ZIP itself was not independently rehashed in this check. The
two prior counts therefore refer to different media inventories.

The 490-file ZIP pool had been created under the older `_03` ledger. Rechecking
its HMAC exclusions confirmed no file matched that ledger. I then matched
local owner-only GPU run locks to the pool: the lock associated with the 16f
pilot smoke matched one file, the lock associated with the 16f 10-video run
matched ten, and the lock associated with the 32f smoke matched one file that
overlapped a video in that 10-video lock. Frame counts come from the linked
run records; the lock schema itself stores media/question hashes only. Across
these locks, **11 distinct pool videos** are known to have prior model
exposure; **479/490** files remain unmatched to these known HMAC records within
this ZIP pool. This does not establish that those 479 are independently
unexposed: the historical ledger may be incomplete. This is an aggregate-only
local byte/lock reconciliation. It cannot authenticate historical
completeness, detect re-encoded or same-event source clips, or establish
reviewed-claim eligibility. Thus the scenario table is only a planning
sensitivity; a video-disjoint confirmatory frame would need the 479-file
working pool checked against complete exposure history and source/event
identity records. Source/event clustering, review eligibility, and zero-effect
attrition also remain open; see
[`2026-10-18_qta_local_media_byte_diversity.md`](2026-10-18_qta_local_media_byte_diversity.md).
The full exclusions and scope qualification are recorded in
[`2026-10-17_pdca_open_until_stop.md`](2026-10-17_pdca_open_until_stop.md).

Focused sample-size tests passed **14/14**. Before the candidate helper was
added, the complete `scripts/` unit suite passed **1,658/1,658** in **40.466 s**.
After the helper revisions, the current complete suite passed **1,663/1,663**
in **40.892 s**. An independent auditor reimplemented the
binomial enumeration with exact rational arithmetic and confirmed the key
values, including the nonzero-effect n interpretation. `git diff --check`
passed. A first focused run
exposed one incorrect expected arithmetic value in a new test; the independently
enumerated exact binomial probability corrected it before the passing rerun.

A separate non-authorizing baseline-p/q candidate contract was added in
`scripts/qta_baseline_pq_candidate.py`. It checks a fixed three-row input,
recomputes A–E p/q with stable tie-breaking, binds caller-pinned source/run and
token-position digests, and emits candidate manifest bytes accepted by the
existing review renderer. Its six synthetic tests passed. This is only
structural validation: caller pins and reported model outputs are unauthenticated,
and it grants no model load or scoring permission. The helper has not been
connected to a trusted forward runner or used on protected review data. An
independent audit found the candidate initially accepted an arbitrary formatted
review-receipt digest and impossible log-probability vectors. The helper now
requires an externally pinned receipt digest, finite float log-probabilities
at most zero, and an A–E probability sum no greater than one (with `1e-12`
roundoff tolerance); integer scores are rejected to avoid precision-loss ties.
Regression tests cover those adversarial cases. These checks still cannot
authenticate the caller pins or prove that scores came from the stated model.
Per-record `supports_claim` decisions must also match caller-supplied frozen
decision pins; the helper does not revalidate stage-1 receipt bytes, so pin
provenance remains an unresolved prerequisite for any real use.

## Faithfulness progress

- `直接評価済み`: **0/3** independently reviewed claims, **0/3** prediction-link
  judgments, **0/15** verified browser-player/decoder-frame correspondences,
  **0/3** claim-linked QTA scores; no empirical `DeltaF`, confidence interval,
  hypothesis test, or multiplicity-adjusted result.
- `補助診断のみ`: exact sign-test resolution and scenario power calculations;
  490-file local media/exposure-lock reconciliation; 14 focused sample-size
  tests, 1,663 current full-suite tests, and six synthetic baseline-candidate tests.
  These are planning/data-integrity diagnostics and engineering verification,
  not faithfulness evidence.
- Human, witness/trust, timing/calibration, and production scoring gates remain
  incomplete. Rights remain unverified; protected media and derived run-lock
  data remained local.

## Next action

Keep n=3 labelled descriptive. Before any confirmatory QTA study, set the
primary estimand's video-level aggregation, a defensible SESOI, usable-video
count and inference method without access to treatment outcomes. The
reconciled 479-file working pool is not a verified independent sample frame:
first establish historical exposure coverage and source/event clusters, then
bind claim eligibility and review receipts. Direct evidence still first requires the
independently reviewed claim intervals,
15/15 clock/frame correspondence, and two-stage phase gate; no real score or
GPU run is authorized by this planning calculation.

## Provenance

- Planner SHA-256: `6bde0e5b065d360607493f3d4481a61f7e0fae63719582f980139b43b15afdb6`
- Test SHA-256: `8c8ec4f80557ab82c3a20ae87997303d781bbe40b7846cf81e9f6f73093c29b4`
- Baseline candidate helper SHA-256: `9eec6cb7c99afa8a017e91a865648f151a3eca659eed5a3fb7ba3d321984edbb`
- Baseline candidate test SHA-256: `6e79608490245f7aad947f6c14bfc0555b14ade3438d28c8ac272351aaba4585`
- Environment: Python 3.12.14, SciPy 1.18.1; no random seed required for exact enumeration.
