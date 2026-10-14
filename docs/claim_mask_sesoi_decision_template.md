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
- Primary-estimand hash (schema v5 canonical hash):
- Outcome-score access started? `no / yes / unknown`
- If yes or unknown, date and scope (do not call a later choice pre-outcome):
- Primary unit: one eligible video, one fixed gold choice, all eligible videos:
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
- Reviewer agrees the proposed value follows from the practical interpretation
  above, not the observed pilot effect or a desired power/result?
- Reviewer confirms the selected value and decision rule are unambiguous?
- Private review-record hash:

## Decision

Choose one and complete only its fields:

- `unresolved`: no threshold is defensible yet; leave SESOI value/rationale,
  fixed time, and review fields null in schema v5. Do not label the protocol
  confirmatory.
- `preoutcome_fixed`: set `δ > 0`; bind the exact canonical primary-estimand
  hash and rationale evidence; record a UTC freeze time and a separate
  completed, independent review record made before scoring. Schema validity
  checks only structure and declarations, not evidence authenticity or truth.

Decision date (UTC):
Decision rationale and any dissent:
Protocol version/hash after recording this decision:

Even a completed decision does not clear human claim review, item-level data
rights, model/runtime, or power/sample-size gates. The validator always reports
`execution_authorization: not_granted`.
