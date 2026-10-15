# Annotation-clock to decoder-PTS calibration worksheet (private working copy)

Use this blank form only after the protocol, eligible cohort, item-level media
rights, and reviewer procedure are frozen. Keep completed copies, video or
frame data, item identifiers, source locators, evidence bundles, and all
calibration measurements in the restricted local research area. Never commit a
completed copy. This form records a human calibration; it does not authenticate
the evidence, establish legal permission, validate a decoder, or authorize
inference. A structurally complete form cannot by itself clear a gate.

Do not select or revise the tolerance after inspecting outcomes. Freeze the
calibration method, timestamp precision, endpoint rule, and acceptance rule in
the protocol before opening any outcome data. If calibration needs a new
threshold or a different mapping rule, stop and version the protocol before
scoring.

## Frozen procedure references

| Field | Record privately |
|---|---|
| Protocol identifier/version and exact-byte SHA-256 | |
| Frozen eligible-manifest SHA-256 and declared item count | |
| Sampling/mapping code revision and SHA-256 | |
| Annotation timestamp unit, precision, and endpoint convention | |
| Decoder PTS time base representation (exact rational) | |
| Calibration method and mapping-function specification/hash | |
| Tolerance and its pre-outcome justification/evidence hash | |
| Acceptance rule covering offset, drift, residual, and uncertainty | |
| Required fit-anchor count/coverage rule and pre-outcome rationale | |
| Required held-out validation-anchor count/coverage rule and rationale | |
| Mapping direction, units, fit algorithm, and uncertainty calculation | |
| Calibration analyst and separate validation reviewer roles | |
| Validation-value withholding/freeze procedure | |

## Per-item private calibration record

Create one record for every item required by the frozen cohort. Keep actual
identifiers, paths, locator strings, timestamps, anchor values, and frame
images private. Bind each row to private evidence hashes; avoid a hash cycle by
having the evidence index reference the completed row only after that row is
frozen. Before measuring any anchor, freeze disjoint calibration-fit and
held-out validation-anchor sets (or deterministic, outcome-blind selection rules) and
their temporal coverage. The calibration analyst receives only fit pairs and
freezes the mapping function and coefficients first. A separate validator
then records or releases the held-out pairs and evaluates residuals without
refitting. Do not use a good fit residual as a substitute for held-out
validation: an affine offset/drift fit can make its own fit anchors look exact,
especially when only two anchors determine its two coefficients. The minimum
counts, spacing rule, fit algorithm, uncertainty method, role separation, and
acceptance threshold must be justified and frozen before measurements; this
worksheet does not choose them.
Held-out means excluded from fitting; it does not imply that anchor errors
within one video are statistically independent. Do not apply an independent,
identically distributed uncertainty model without a separate justification.
The optional CPU helper `scripts/annotation_clock_calibration.py` implements
exact-rational ordinary least-squares affine mapping with positive drift from
PTS seconds to annotation seconds and evaluates caller-supplied held-out anchors without
refitting. Use it only if that exact method was already frozen in the protocol;
it does not select a tolerance, validate anchor provenance or temporal coverage,
calculate mapping uncertainty, or clear this worksheet's gate.

| Field | Record privately |
|---|---|
| Eligible item ID and private source-lineage record reference/hash | |
| Evidence that planned local use is permitted; private rights-record reference/hash | |
| Annotation clock definition, origin, and source/version hash | |
| Annotation frame-ID/time-map exact-byte hash (if used) | |
| Local media identity hash and source identity comparison method/result | |
| Decoder name/version/build hash and stream identity | |
| Display-order frame count, PTS sequence hash, and exact rational time base | |
| PTS origin conversion to the declared media/annotation timeline | |
| Sampled source-frame indices and ordered pixel-buffer hashes | |
| Binding from each measured anchor to the exact decoded frame/pixel hash | |
| Prespecified visual/synchronization anchor method and private evidence reference/hash | |
| Fit-anchor table/reference/hash, coverage, and count | |
| Held-out validation-anchor table/reference/hash, coverage, and count | |
| Fit/validation role-separation and freeze-order evidence reference/hash | |
| Fitted mapping rule and coefficients, calculated from fit anchors only | |
| Fit residuals (diagnostic only; not acceptance evidence) | |
| Held-out validation residuals, maximum absolute error, and frozen tolerance comparison | |
| Prespecified uncertainty calculation and its validation result | |
| Outcome data, gold answers, and model scores not accessed before freeze? | yes / no / unknown |
| Independent reviewer role, review time (UTC), and review-record hash | |
| Item result | pass / fail / unresolved |
| Failure reason and protocol-owner disposition | |

## Cohort-level decision

| Field | Record |
|---|---|
| Frozen denominator | |
| Completed / failed / unresolved item counts | |
| Did every eligible item pass the frozen calibration rule? | yes / no |
| Completed cohort worksheet/index exact-byte SHA-256 | |
| Independent audit record reference/hash | |
| Protocol version and decision timestamp (UTC) | |

Any failed, missing, ambiguous, or unresolved item blocks the full frozen
cohort. Do not drop it, replace it, or compute a reduced-cohort result after
seeing calibration values. A changed cohort or restricted target requires a
separately specified pre-outcome estimand and protocol. After calibration,
bind the private completed index and its evidence references in the protocol
without exposing measurements or locators publicly.
