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

## Follow-up literature audit: what a more direct test would require

An outcome-blind review of rationale-faithfulness methods reinforces that
simply inserting the generated claim into the QA prompt would measure
**claim-conditioned answer sensitivity**, not whether the original prediction
was actually produced from that rationale. Prompt-following, answer leakage,
lexical cues, and changes in prompt length/position remain alternative causes.
Such a result can be a behavior diagnostic, but should not be called semantic
or internal-process faithfulness without a causal link to the original answer
generation and independent validation of the claim's meaning.

The more direct visual direction is an explanation-driven counterfactual test:
identify a visual concept asserted as relevant by the explanation, intervene
on that concept while preserving other evidence, and check the answer and
explanation against the verified edit. Atanasova et al. (ACL 2023) provide
textual NLE tests based on counterfactual reasons and reconstructed inputs;
Siegel et al. (ACL 2024) motivate distribution-shift-sensitive counterfactual
metrics rather than binary answer flips alone. A 2026 arXiv preprint, EDCT-
Bench, adapts this idea to VLMs through verified localized edits and reports
paired human validation of 128 concept-level interventions. This is a recent
preprint, not a peer-reviewed standard, and its image-editing pipeline does
not directly solve temporal video editing.
[Atanasova et al. (ACL 2023)](https://aclanthology.org/2023.acl-short.25/),
[Siegel et al. (ACL 2024)](https://aclanthology.org/2024.acl-short.49/),
[EDCT-Bench (arXiv, September 2026)](https://arxiv.org/abs/2609.17953).

For this project, the practical consequence is unchanged: the existing
proposed claim-interval mask, if independent human review validates its
intervals, would be closer to visual evidence attribution than claim-text
sensitivity. It remains only a three-video proxy feasibility protocol and has
not run. A genuine explanation-faithfulness extension would
need independently reviewed claim propositions and localized, validity-checked
video counterfactuals. That design is materially larger than the current
pipeline; do not spend GPU time on a prompt-only contrast and relabel it as
faithfulness. Direct generated-explanation faithfulness remains not measured
(0/3 reviewed, 0/3 scored, no estimate/interval/test).

There is also an existing, separate proposal in
[`internal_intervention_grounding_faithfulness_plan.md`](../docs/internal_intervention_grounding_faithfulness_plan.md): generate a fixed
answer/claim/timestamp output, then compare teacher-forced answer/claim scores
under an intervention to a model-selected temporal chunk versus a matched
control chunk. It avoids editing source videos and is a plausible lower-cost
research branch, but it measures internal chunk dependence under a particular
intervention. It would still need a pre-outcome fixed intervention and
selection rule, verified processor-token-to-time mapping, independent review
of claim/support correspondence, and a non-head control. It must not turn into
a search over layers, strengths, or chunk policies, and a score decrease alone
would not validate the claim's semantic truth. This proposal is separate from
v2 and has not been implemented or run as a study.

Accordingly, the decision is not simply between current v2 and visual editing:
the temporal hidden-state intervention is a third candidate. Its relative
merit depends on whether the thesis endpoint is evidence attribution or
faithfulness of a free-text explanation, while the actual direct-claim and
frame-review gates stay outstanding.

## Follow-up: thesis target selected, internal-intervention metric remains draft

On 2026-10-17, the user selected the thesis target as **selective prediction
dependence on video evidence indicated by the explanation**, and selected
**improvement under an internal intervention** as the main metric candidate.
This rules out treating the current v2 QA-margin mask as the thesis endpoint:
the generated claim is used only to choose frames and is absent from the scored
QA prompt.

The internal-intervention plan now records a candidate difference-in-differences
estimand. For each fixed prediction, compare the drop in its fixed gold-versus-
foil margin after ablating claim-mapped evidence tokens with the drop after an
equal-count matched-control token ablation; then compare this selectivity between
baseline and the prespecified intervention. A positive treatment contrast would
mean increased operational dependence on the reviewed evidence tokens relative
to that control, under this model and setup. It would not establish claim truth,
complete internal-process faithfulness, or generalization beyond the tested
cohort.

This estimand and intervention are still **not frozen**. The internal treatment,
fixed foil rule, ablation/control construction, inclusion rule, sample size, and
smallest effect of interest need outcome-blind specification. The close prior
work also limits novelty: Yeo et al. study activation-patching attribution for
textual NLE faithfulness, while Cai et al. study temporal-aware activation
engineering to reduce VideoLLM hallucination. A possible contribution must be
the narrowly defined combination and evaluation, not either broad idea alone.
[Yeo et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.529/),
[Cai et al. (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html).

An independent design review proposed directly amplifying the reviewed
claim-span visual-token residuals. This was rejected as the primary treatment:
because the treatment and selective-ablation outcome target the same tokens, a
positive effect could be built into the measurement by construction. A
question-conditioned internal selector, independent of reviewer/gold spans,
would be a stronger candidate, but its selection signal, intervention point,
strength, and global-shift control are not yet justified. The `ΔF` metric
therefore remains a draft and no efficacy experiment is authorized by this
design work.

Implementation work added a fail-closed local cohort-fingerprint auditor and
processor-grid-to-source-frame temporal-token mapping diagnostics. A synthetic
end-to-end test now exercises the auditor's full metadata/hash/evidence/control
recomputation path; mapper tests cover valid and malformed temporal layouts.
Full script test discovery passed **1,370/1,370**. A fresh local-only
candidate inventory bound 65 required roles with no missing slot; independent
file-hash recomputation found 0 mismatches. The inventory remains non-authorizing
and marks review incomplete, scoring unavailable, and rights unverified. These
are provenance and geometry checks, not faithfulness outcomes; protected
diagnostic artifacts remain local.

Direct generated-explanation faithfulness remains **0/3 independently reviewed,
0/3 scored**, with no direct effect estimate, interval, or test. Actual
browser-player-to-decoder-frame correspondences remain **0/15**. No efficacy
inference follows from the implementation checks, and no activation-treatment
score has been run. The next valid research step is to specify one intervention
and all controls before outcomes, then complete the external human-review,
trusted-release, and clock-correspondence steps needed to attach reviewed claims
to the mapped tokens.
