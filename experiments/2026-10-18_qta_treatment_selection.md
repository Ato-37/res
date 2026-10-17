# Outcome-independent QTA treatment selection (2026-10-18)

## Purpose

Resolve the treatment-selection branch of the user's chosen thesis direction:
whether a fixed prediction depends selectively on explanation-indicated video
evidence, and whether an internal intervention can increase that dependence.
This is an outcome-blind design decision, not a registered experiment or a
claim-linked result.

## Decision and verified implementation

Select the existing Qwen Question-Temporal Attention Bias (QTA-Bias) as the
sole primary-treatment candidate for the next claim-linked feasibility run.
It chooses groups from untreated question-token attention, independently of
generated claim text, reviewed intervals, answer labels, gold labels, and
outcome logits. The fixed candidate is:

- average question-query attention over heads and decoder layers 5–16;
- score complete processor temporal groups, then select
  `k=max(1, ceil(0.10*T))`, ties to earlier group;
- at the fixed final answer-score query in layers 5–16, add `ln(2)` to each
  selected visual-key logit, freezing the selection for every branch;
- compare baseline and QTA-Bias across full, reviewed-evidence-key ablation,
  and equal-key-budget temporal-control-key ablation; run the matched
  deterministic sham as a secondary diagnostic, for nine total forwards per
  record including baseline full.

The reviewed interval remains an evaluation target, not the intervention
selector. A direct bias on the reviewed evidence keys is excluded as a primary
treatment: because the same keys are then masked in the evidence-ablation arm
but retained in the control-ablation arm, that treatment can mechanically
inflate the interaction it is intended to measure. It could only be described
as a separate positive-control manipulation. The QTA selector can still
overlap reviewed evidence or control groups; record exact overlap for every
row and condition, keep it descriptive, and do not exclude rows or define
subgroups from overlap after outcome access. The random-key sham can overlap
either group and is not a pure generic-bias control.

Primary score remains the fixed baseline-full A–E prediction/foil margin:
`F_g = m_g(control-ablation) - m_g(evidence-ablation)` and
`DeltaF = F_QTA-Bias - F_baseline`. A positive value supports only increased
selective score dependence under this exact QTA/access-ablation operation; it
does not show that explanations became truer or naturally more faithful.

Root source review confirmed that the existing synthetic runner implements
question-token selection, layers 5–16, the `ceil(10%)` budget, `ln(2)` bias,
fixed selector reuse, evidence/control ablations, and ablation precedence.
Focused runner tests pass **16/16**. Independent method review recommended QTA
over direct reviewed-evidence bias for the same non-circularity reason; the
primary agent independently checked that rationale against the draft estimand
and source implementation. The full scripts suite also passes **1,559/1,559**.

## Limits and direct faithfulness status

This selects treatment semantics only. The 3-record cohort still has **0/3**
independent claim reviews, **0/3** prediction-link adjudications, **0/15**
verified player-clock/decoder-frame correspondences, and **0/3** direct scores.
There is no empirical `DeltaF`, confidence interval, or hypothesis test.
Human/trust/timing gates, production claim-mapped scorer and cohort-wide
preflight, confirmatory SESOI/sample-size/uncertainty plan, and treatment
efficacy remain unresolved. Existing GPU results are synthetic engineering or
selector-wiring checks and do not count as faithfulness progress. Do not launch
another GPU job until a separately audited run can use authentic reviewed
evidence/control mappings and the fixed treatment without relaxing these gates.
Rights remain unverified; protected inputs and review outputs stay local.

## Next action

Complete the external review, trust-root/witness, and frame-correspondence
steps. In parallel, complete and independently audit the production adapter
that binds the actual QTA selector output, reviewed evidence/control groups,
fixed `p/q` scorer, nine branch hashes, and source/calibration receipts before
any cohort branch forward. Preserve the 3-row run as a descriptive feasibility
pilot only; any efficacy confirmation requires a disjoint, preregistered cohort
and a defensible precision/SESOI plan.

## Provenance

- Protocol draft SHA-256: `b997f900d83813db825aa03c35cf5d599eda043ac055e14ff4864d1c5b3aea6f`
- Nine-branch synthetic runner SHA-256:
  `8b1be5ec0193e27892b50a478aabbe625f7466ab86ab32aa80b49987e95961d2`
- QTA selector runner SHA-256:
  `37fc0fd7a05887f12ccf256758e6b102901f4f9137911bba9e78532275860be2`
- QTA selector helper SHA-256:
  `3f56fe1e992ffca5a035864ced7472cbe1f12af9978c1559d1b82e0b938712f8`
- Fixed-prediction scorer SHA-256:
  `84ba18241668fd659ddce0ab25a4e3dfa2cacc656ef43fb494fdc056a5c21744`
