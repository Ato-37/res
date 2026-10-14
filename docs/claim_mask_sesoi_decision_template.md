# Claim-mask SESOI decision record template

Complete this record before viewing any claim-mask outcome scores, including
development-pilot scores. This worksheet intentionally uses a stricter
freeze-before-any-outcome rule; after the threshold is fixed, a same-estimand
pilot may inform variance/runtime-based sample planning but not change the
threshold based on its effect estimate. This is a human decision aid, not an
automatic authorization. Keep reviewer identity, supporting private files,
and any item-level data out of the public repository; store only hashes in the
eventual protocol.

## Frozen target

- Protocol revision/hash:
- Primary-estimand hash (schema v12 canonical hash):
- Inference-plan hash (schema v12 canonical hash):
- Mask-transform specification/hash: specify exact replacement, parameters,
  color space, operation order, and whether the whole sampled frame is replaced.
  Keep the specification private and versioned; use the identical transform
  for selected and control masks:
- Outcome-score access started? `no / yes / unknown`
- If yes or unknown, date and scope (do not call a later choice pre-outcome):
- Primary unit: one eligible video, one fixed gold choice, all eligible videos:
- Target population and relation to the finite eligible cohort (census target,
  probability sample, or explicitly model-based superpopulation):
- Cohort construction/sampling provenance and frame of reference:
- Available creator/event/series/source-lineage cluster fields and pre-outcome
  dependence assessment:
- Inference method justified for that target and cluster count; if a cluster-aware
  method is proposed, state the number of independent clusters and its limits:
- Distributional assumption for video-level paired differences (iid normal or
  iid finite variance), supporting pre-outcome rationale hash, and frozen
  tail-sensitivity plan hash; UTC review timestamp, outcome-access status and
  access-audit record hash. This distribution review must occur no later than
  independent SESOI review, which must be no later than the SESOI freeze:
- Primary contrast: fixed-answer log-margin drop under claim-linked evidence-frame
  mask minus the matched temporal-control margin drop:
- Score unit: natural-log units (nat):
- Success rule: lower bound of two-sided 95% video-level paired-difference
  confidence interval exceeds the fixed SESOI:

The contrast is operational evidence dependence under this mask/control pair.
It does not establish that the claim is true, that the explanation is complete,
or that the masked frames are a unique causal mechanism. Masking may create
out-of-distribution inputs.

## Proposed threshold and practical interpretation

- Proposed SESOI, `δ` in nat (leave blank while unresolved):
- `exp(δ)`, ratio of control-mask to evidence-mask gold-versus-best-foil
  log-probability ratios:
- State in plain language what result just below this threshold would mean for
  the research claim, and what a result just above it would change:
- Why is that distinction meaningful for this thesis question and the tested
  population/model/intervention?
- What evidence supports this interpretation, with private artifact hashes and
  dates:

For video `i`, `exp(d_i) = exp(S_i(control)) / exp(S_i(evidence-mask))` under
the fixed gold answer versus that condition's highest-scoring foil. Therefore
`exp(mean(d))` is the geometric mean of the per-video ratios, not their
arithmetic mean. This is an interpretation of log-margin contrast; it is not a
change in answer probability, percent accuracy, or the probability that a
claim is true. A numerical value chosen only because it is convenient for
power or because a prior result exceeded it has no practical-threshold
justification.

Do not transfer a threshold from the Correlational Counterfactual Test (CCT):
that work measures standardized association between intervention TVD and
explanation mention, and its mention threshold depends on the observed TVD
distribution. It does not specify a natural-log-unit minimum important
difference for this per-video gold-versus-foil margin contrast.

## Evidence considered

For each item, distinguish evidence actually available from a proposal for
future measurement. Do not use a result from another estimand as a substitute.

| Basis | Evidence and hash | What it can support | Limit or unresolved issue |
|---|---|---|---|
| Practical consequence for the thesis claim | | | |
| Human annotation/evidence interval resolution | | | |
| Same-estimand score reproducibility or measurement resolution | | | |
| Independent methodological or domain judgment | | | |

The prior head-ablation `+0.25 nat` threshold and its observed variance belong
to a different intervention/estimand. They must not be transferred to this
claim-linked frame-mask contrast. Same-estimand pilot variance may support
sample-size planning only after `δ` has been frozen under this worksheet; it
does not by itself justify `δ`, and pilot outcome direction/significance must
not determine it.

## Independent pre-outcome review

- Reviewer role and independence basis (keep identity in the private record):
- Reviewed artifacts and hashes:
- Review timestamp (UTC):
- Confirmed no claim-mask outcome scores were accessed before this decision?
- Outcome-score access audit record/hash and coverage period:
- Reviewer agrees the proposed value follows from the practical interpretation
  above, not the observed pilot effect or a desired power/result?
- Reviewer confirms the selected value and decision rule are unambiguous?
- Private review-record hash:

## Decision

Choose one and complete only its fields:

- `unresolved`: no threshold is defensible yet; leave SESOI value/rationale,
  fixed time, review/access fields, and inference-plan hash null in schema v12.
  Do not label the protocol confirmatory.
- `preoutcome_fixed`: set `δ > 0`; bind the exact canonical primary-estimand
  and inference-plan hashes and rationale evidence; record a UTC freeze time
  and a separate completed, independent review record made before scoring. Record the review
  timestamp no later than the freeze and provide a private access-audit record
  supporting the declaration that outcome scores had not been accessed.
  Schema validity
  checks only structure and declarations, not evidence authenticity or truth.

Decision date (UTC):
Decision rationale and any dissent:
Protocol version/hash after recording this decision:

Even a completed decision does not clear human claim review, item-level data
rights, model/runtime, or power/sample-size gates. The validator always reports
`execution_authorization: not_granted`.
