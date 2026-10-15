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
| Required independent reviewer role and blinding procedure | |

## Per-item private calibration record

Create one record for every item required by the frozen cohort. Keep actual
identifiers, paths, locator strings, timestamps, anchor values, and frame
images private. Bind each row to private evidence hashes; avoid a hash cycle by
having the evidence index reference the completed row only after that row is
frozen.

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
| Paired annotation-clock and decoder-time anchor measurements (private table/reference) | |
| Fitted mapping rule, offset, drift, residuals, and uncertainty (private values) | |
| Maximum absolute mapping error and frozen tolerance comparison | |
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
