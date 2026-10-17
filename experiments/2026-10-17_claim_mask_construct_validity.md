# Claim-mask v2: construct-validity audit

## Question

Does the current finite-cohort v2 protocol measure faithfulness of the model's
generated natural-language explanation, or a narrower property?

## Verified design facts

The v2 scoring prompt is constructed from the dataset question and five answer
options. It does not contain the generated claim text. The claim selects the
video intervals that define the `claim_evidence_black` intervention. The
measured outcome is the gold-versus-best-foil next-token margin for the QA
answer. The primary finite-cohort contrast subtracts the claim-interval mask
margin from an equal-count, rank-distance-matched temporal-control mask margin.
The protocol correctly forbids population inference, confidence intervals,
p-values, significance claims, and an efficacy threshold for its three items.

Therefore, if every upstream review and score gate later passes, this design
can describe whether the QA answer margin depends differently on frames
selected by a human-validated claim than on the specified temporal control,
for these exact three items and this model. It does not measure whether the
model used the natural-language claim/explanation in producing its answer,
whether the explanation reflects internal reasoning, or whether the explanation
is faithful or true. It is claim-aligned visual-input dependence, not a direct
test of explanation faithfulness.

The interpretation is further limited because a positive contrast may arise
from a gold-log-probability change, a best-foil change, or both; the best foil
is recomputed in each condition. Black fills can be out of distribution, and
the temporal control matches count and sampled-rank distance rather than visual
information or semantic content. These caveats already appear in the v2
protocol and scorer and must remain attached to any result.

## Literature cross-check

Atanasova et al. (ACL 2023) frame NLE faithfulness as whether generated
explanations reflect the model's decision process and propose tests that either
insert counterfactual reasons absent from an NLE or reconstruct inputs from
reasons stated in the NLE. This supports treating the explanation content as
part of the measured object, rather than only using it to choose an input
region. [Paper](https://aclanthology.org/2023.acl-short.25/).

The discussion of causal versus explanatory faithfulness in Atanasova et al.
(ACL 2024) distinguishes a prediction's causal dependence on an input from
whether the explanation text usefully represents the factors behind the
prediction. The paper warns that causal dependence alone does not ensure an
explanation is useful or faithful. The current v2 intervention is even
narrower: it measures QA output dependence on video frames selected by a
reviewed claim, while the claim is absent from the scoring prompt.
[Paper](https://aclanthology.org/2024.acl-short.49/).

## Research decision before any outcomes

Keep v2 unchanged as a finite descriptive **claim-aligned visual-evidence
dependence feasibility study** if its existing human and technical gates are
completed. Do not use its result as a direct explanation-faithfulness finding.
If the thesis requires an explanation-faithfulness claim, version a separate
pre-outcome protocol that specifies the explanation object and a discriminating
intervention on the explanation's content or its claimed evidence, with
outcome-blind controls and a scope appropriate to the available sample. Do not
retrofit that claim onto v2 results, and do not reopen the closed head search.

No outcomes have been inspected for v2, so this is a pre-outcome construct
validity correction, not a post-hoc response to a result. A concrete v3
estimand is not fixed here: that choice should follow the thesis's intended
target (textual rationale dependence, visual evidence attribution, or both)
and cannot be inferred from the current QA-margin metric alone.

## Faithfulness status and next step

Direct generated-explanation faithfulness: **not measured**. Current direct
claim review is 0/3; claim-mask scoring is 0/3; no estimate, interval, or test
exists. Auxiliary engineering checks and old closed head-study results do not
change this status. The immediate research task is to settle the intended
faithfulness construct before freezing a revised outcome protocol. Independently,
the current human-review branch remains blocked by absent trusted role/root
artifacts and by 0/15 actual screenshot-to-decoder correspondences.
