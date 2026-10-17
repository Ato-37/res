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

## Follow-up: human handoff readiness and GPU wiring feasibility

An independent runbook audit confirmed that the pre-review gate remains
external: no separately trusted root pin, real signed role roster, witnessed
unopened owner-to-custodian handoff, or actual reviewer evidence is present.
The latest private candidate inventory is not a release artifact and must be
rebuilt from current bytes immediately before any freeze. The runbook now has
an ordered pre-opening checklist with stop conditions. This is preparation
only; it does not open the package or establish review completion.

Direct faithfulness evidence remains **0/3 independently reviewed, 0/3
scored**, with no direct estimate, interval, or test; actual browser screenshot
to decoder correspondence remains **0/15**. These are the latest local
verified counts and are not changed by the documentation work.

An independent GPU feasibility review found that the existing synthetic Qwen
smoke validates nine-forward scoring capacity but does not test internal
intervention wiring. A new synthetic-only wiring smoke is justified as a
bounded engineering check: it will map one predeclared synthetic temporal
token group and test baseline, zero-strength, and fixed nonzero hook passes at
language-model layer 8. The group and gain are wiring fixtures, not tuned
intervention settings. KFP is not yet reproducibly specifiable because its
public method description does not give top-k; no efficacy run or KFP
replication is authorized by this check. Expected workload, if the smoke is
implemented and the current GPU preflight passes, is three model forwards plus
load, roughly 40–60 seconds and up to about 10.8 GiB allocated VRAM based on
prior synthetic runs; enforce at least 12 GiB free before load and 512 MiB
before each forward. These are estimates, not measurements of this run.

**Engineering smoke result (2026-10-17).** The independently audited,
synthetic-only GPU wiring check completed 32/32 frames and 3/3 forwards at
Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, BF16 with 4-bit BF16 compute, on
an RTX 5070 Ti. The processor produced 16 temporal groups; the predeclared
synthetic group 0 contained 256 video-token positions. At decoder layer 8
(zero-based), the zero-strength hook reproduced baseline logits exactly, the
fixed 0.1 wiring perturbation changed the selected hidden activations and
touched the intended positions, and output logits remained finite. Runtime was
20.22 seconds. Free VRAM was 14.66 GiB before load and 1.95 GiB minimum before
a forward. The measured 10.67 GiB allocated / 12.38 GiB reserved peaks cover
only the post-load, post-batch-transfer measurement window; they exclude model
loading and transfer.

The first attempt exposed a real implementation mismatch: this installed
Transformers decoder layer returns a tensor rather than a tuple. The hook was
repaired to support the verified tensor return contract; CPU coverage is now
5/5. The failed initial attempt stopped before completing the hook checks and
did not produce efficacy evidence. The final result is saved in
[`results/synthetic_temporal_hook_wiring_20261017.json`](../results/synthetic_temporal_hook_wiring_20261017.json).
The smoke establishes token-position/hook wiring and finite numerics only. It
does not validate claim relevance, treatment quality, prediction evidence
dependence, or faithfulness improvement. No treatment is frozen.

**Verification:** the independent code review found no wiring-scope defect;
the full repository CPU suite passed 1,375/1,375 after registering the new
model loader in the closed-branch classifier. The research gate remains actual
human-reviewed evidence mapping plus a pre-outcome, non-circular
intervention/estimand; no direct faithfulness conclusion follows from this
engineering result.

## Primary-source check: KFP treatment remains underspecified

The arXiv full text for VERHallu §IV-B says to rank attention scores and select
top-k key frames, but the inspected paper does not state the value of `k` or a
complete operational definition of the attention score/aggregation. It does
state layers 8–15, Gaussian `sigma=1`, window `m=5`, and `beta=0.6`. The
authors' linked implementation repository currently exposes only README and
LICENSE, not runnable method code. The same paper reports a Qwen2.5-VL-7B
CFQA decrease of 6.0 percentage points under KFP while other task scores can
improve, so an assumed uniform benefit is particularly unsafe. This confirms
that KFP cannot yet be reproduced as a single frozen treatment for this
faithfulness estimand. No KFP settings were tuned on project outcomes.
Sources: [VERHallu paper, §IV-B and results](https://arxiv.org/abs/2601.10010),
[official repository](https://github.com/zefanZhang-cn/VERHallu) (checked
2026-10-17).

The smoke's layer-8 hook success resolves only the software feasibility item;
it does not supply KFP's missing selector or its propagation/MLP definition.
Next, design a separate, fully specified question-conditioned selector and
matched internal control from independent methodological grounds, or obtain
clarification/source code for KFP. Do not label a locally filled-in `k` as a
KFP replication, and do not start efficacy scoring until the treatment,
primary estimand, and human-review evidence mapping are frozen.

## Follow-up: thesis target selected, internal-intervention metric remains draft

On 2026-10-17, the user selected the thesis target as **selective prediction
dependence on video evidence indicated by the explanation**, and selected
**improvement under an internal intervention** as the main metric candidate.
This rules out treating the current v2 QA-margin mask as the thesis endpoint:
the generated claim is used only to choose frames and is absent from the scored
QA prompt.

The internal-intervention plan now records a candidate difference-in-differences
estimand. For each fixed baseline-predicted option, compare the drop in its
fixed predicted-option-versus-foil margin after ablating claim-mapped evidence
tokens with the drop after an equal-count matched-control token ablation; then
compare this selectivity between baseline and the prespecified intervention. A
positive treatment contrast would mean increased operational dependence on
the reviewed evidence tokens relative to that control, under this model and
setup. It would not establish claim truth, complete internal-process
faithfulness, or generalization beyond the tested cohort.

An independent audit found the draft had used “fixed original prediction” in
the internal-intervention plan but “gold-versus-foil” in this audit, which
changes the target when the model is wrong. The working convention is now the
unmodified baseline model's highest-scoring A–E option, with alphabetical
tie-breaking; the foil is its highest-scoring competitor from that same full
baseline distribution, also alphabetically tie-broken. Both labels remain
fixed across all intervention and ablation conditions. Gold margin/accuracy
are secondary outcomes. This is a pre-outcome construct-alignment decision,
not a score freeze; the ablation operator, selector, treatment, cohort size,
and inference plan remain unresolved.

The candidate measurement fixes the baseline-generated answer, claim, and time
span across intervention conditions. It therefore tests changed dependence of
the same claim under internal scoring, not whether the intervention produces a
better free-generation claim. Any post-intervention generation quality or
grounding must remain a separate secondary outcome. If the treatment uses an
internal key-frame selector, its selection must also be fixed from one ordinary
full-video pass and reused under evidence/control ablations; reselecting after
each ablation would mix the ablation effect with selector changes.

The proposed contrast simplifies algebraically to
`F_g = m_g(control-ablation) - m_g(evidence-ablation)`: the full-input term
cancels. Thus `ΔF` is an intervention-by-ablation interaction and is invariant
to a margin offset added equally to both ablation branches, provided answer and
foil stay fixed. Condition-specific shifts, changing foils, or mismatched token
positions are not canceled and remain alternative explanations to control.

This estimand and intervention are still **not frozen**. The internal treatment,
fixed foil rule, ablation/control construction, inclusion rule, sample size, and
smallest effect of interest need outcome-blind specification. The close prior
work also limits novelty: Yeo et al. study activation-patching attribution for
textual NLE faithfulness, while Cai et al. study temporal-aware activation
engineering to reduce VideoLLM hallucination. A possible contribution must be
the narrowly defined combination and evaluation, not either broad idea alone.
[Yeo et al. (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.529/),
[Cai et al. (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5683583b88da51c79d2eb263f295286e-Abstract-Conference.html).

The source audit also verified that the existing v2 scorer computes gold minus
the best non-gold option separately in each condition. It cannot be reused as
the proposed fixed baseline-prediction/foil scorer: both the target answer and
foil rule differ, and the current foil may change by condition. The next scorer
must fix both label IDs from the untouched baseline full-video distribution,
preserve the same next-token positions across all conditions, and keep gold
margin as a secondary performance outcome. The exact internal ablation
operator, application order relative to the selector treatment, and equal-count
temporal control remain unspecified. The current 0/3 review and 0/15
browser-to-decoder correspondence gates also remain open.

An independent design review supplied one fully operational **hypothesis**,
not a treatment decision: on an untouched full-video pass, select the temporal
group whose mean layer-8 visual-token vector has the greatest cosine similarity
to the final prompt-token vector (earliest group breaks ties); freeze that
selection, multiply its layer-8 hidden rows by 1.1, and compare with an
equal-size group selected by a fixed-seed uniform rule that does not inspect
the question or claim. Apply the same assignment in every scoring branch and
leave human-reviewed claim spans evaluation-only. This is non-circular at the
assignment step, but cosine similarity, layer 8, top-1, and the 1.1 multiplier
lack an independent rationale. The smoke tested only a fixed group-0 hook; it
did not exercise this selector. Therefore this proposal is not frozen or
efficacy-justified and must not be represented as an established faithfulness
improvement. A second independent audit likewise concluded that a
question-conditioned attention selector is plausible but still a hypothesis;
published KFP does not specify its key selection sufficiently, and TA-AE's
learned offsets/module classifier are not a drop-in treatment for this setup.

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

## Follow-up: fixed prediction/foil measurement component

To close the label-rule portion of the scorer gap, a small CPU-only draft
helper now freezes the baseline A–E prediction/foil using alphabetical
tie-breaking, scores subsequent conditions with those same label IDs, and
computes `ΔF` from full/evidence/control margins. Four focused tests cover
fixed labels despite condition argmax changes, tie rules, the interaction
algebra, and invalid/non-finite inputs. The helper does not choose the
internal-ablation operation, selector, cohort, or treatment; it performs no
model inference or actual-data scoring. It is a measurement component, not a
frozen or validated study protocol.

Verification after this addition: focused helper tests passed 4/4; repository
test discovery passed **1,379/1,379**; `compileall` and `git diff --check`
passed. Direct faithfulness evidence is unchanged at **0/3 reviewed, 0/3
scored, no effect estimate/interval/test**, and **0/15** actual
browser-to-decoder correspondences. The workflow remains gated on a frozen
treatment and ablation/control protocol as well as the external human-review
release and frame-correspondence evidence.

Follow-up input-contract audit tightened the helper to reject booleans and
numeric strings for answer log-probabilities and condition margins. Focused
tests remain **4/4**, all repository tests remain **1,379/1,379**, Python
compilation and `git diff --check` pass. This is measurement input hygiene;
it does not alter the estimand or add any actual-data evidence.

A second independent boundary audit found huge-integer float conversion could
raise an uncaught `OverflowError`; conversion now fails closed with `ValueError`,
and tests cover huge integers, overflow in derived contrasts, and common-offset
invariance. The plan now distinguishes frozen A–E class indices from the
answer-token IDs/score positions that a future inference scorer must bind and
reuse across every condition. Re-run results for these follow-up edits are
focused helper tests passed **4/4**, repository discovery passed
**1,379/1,379**, Python compilation passed, and `git diff --check` passed. No
actual-data score was run.

An independent source-only intervention audit found no minimally specified,
prior-supported treatment for improving this `ΔF` endpoint, based on the
references then examined. It rejected choosing KFP top-k/strength by search and
noted that layer 8/gain 1.1 are only hook-smoke settings. It suggested
evidence/control zero-ablation as a possible *measurement* design, with OOD
and incomplete semantic matching limitations. The subsequent STEAR search
below supersedes the earlier literature conclusion and adds a closer supported
candidate; it does not itself establish benefit on this `ΔF` endpoint. The
synthetic hook smoke remains wiring-only. Human review remains 0/3 and actual
browser-to-decoder correspondence remains 0/15, so even the baseline evidence
dependence score is not yet available.

### Literature-driven update: STEAR is now the leading implementation audit

A further primary-source search found STEAR, a closer prior than the previously
considered KFP candidate. The paper explicitly evaluates Qwen2.5-VL-7B and
reports NExT-QA accuracy of 59.63% for baseline and 60.87% for STEAR (+1.24
percentage points); its method uses uncertainty-triggered, middle-layer
key-evidence selection and reinjection. Reported settings include risk layers
5–16, normalized-entropy threshold 0.85, top 10% attention-salient patches,
temporal homogenization coefficient 0.80, and contrastive strength 0.75. These
are the paper's accuracy results, not an estimate of our claim-specific
selective-dependence `ΔF`; the article reports no uncertainty interval in the
table, and the result cannot establish the target construct. The paper says
code is in supplementary material, but the main article leaves the trigger
neighborhood and some implementation details unspecified. Therefore STEAR is
now the first treatment-reproducibility audit, not yet a selected or validated
treatment. Next inspect the official supplement/code and check whether the
reported mechanism can be implemented faithfully in the pinned Qwen checkout
without substituting a different intervention. If unavailable/incompatible,
record the exact missing details before considering a clearly labeled
adaptation. No GPU run or real-data score was run for this literature update.

Local source inspection of the pinned environment (`transformers` 5.18.0)
found `Qwen2_5_VLDecoderLayer` exposes `self_attn`, `mlp`, and norms, with no
separate cross-attention submodule. STEAR's text-query-to-visual-key scores
could potentially be extracted from multimodal self-attention, but that is an
architecture mapping that needs a CPU synthetic contract test. Its reinjection
equations also introduce query/key/value/output transforms and a visual-memory
projection; they are not equivalent to the current simple hidden-row multiplier
smoke. This strengthens the need to obtain and audit the supplement before
implementation. The STEAR table prints `±0.00` for Qwen results but does not
define that notation as a confidence interval or report an uncertainty method
there, so the +1.24 percentage-point accuracy difference is descriptive only.

### Synthetic attention-routing GPU feasibility smoke

The initial 32-frame/all-attention attempt failed with CUDA OOM during the
forward. An 8-frame reduced smoke captured only the final prompt-query row at
paper layers 5–16. The first retry found a selector shape-contract mismatch;
subsequent audit also found that duplicate layer indices and batch sizes other
than one were not rejected. The selector now rejects both cases, with dedicated
CPU tests.

**Final source-matched run (2026-10-17).** On the RTX 5070 Ti, Qwen2.5-VL-7B-
Instruct revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, Transformers
5.18.0, PyTorch 2.11.0+cu128, CUDA 12.8, BF16 and 4-bit BF16 compute, one
synthetic 8-frame forward completed in 17.94 s. It captured 12 layers over
sequence length 1,068; 1,024 positions were identified as synthetic visual
tokens, and the deterministic top-10%-ceil selector returned 103 finite-score
positions. Free VRAM was 14.657 GiB before model load, 8.683 GiB before
forward, and 7.560 GiB in the runner's post-forward, pre-cleanup measurement;
peak allocated/reserved after model load were 6.213/7.033 GiB. The peak
counter was reset before forward and excludes model loading and batch transfer.
The runner source SHA-256 recorded in the artifact is
`061fa5dc96f1ac685953f4c0316873a4a0118c2db326668e76741f8e6802f161`.
Artifact: `results/synthetic_qwen_stear_attention_feasibility_8frames_auditfix_20261017.json`.

Repository CPU test discovery passed **1,383/1,383**, including the new
duplicate-layer and batch-size failures. This remains engineering evidence
only: synthetic input, one forward, no generated answer or score. The 32-frame
OOM remains a capacity boundary. Earlier attempt artifacts are retained for
failure provenance; the successful retry2 artifact used a prior source hash and
is superseded for reproducibility by the source-matched auditfix artifact.

### STEAR primary-source audit: implementation is only partly specified

The official arXiv HTML gives more detail than the initial scan: trigger risk
is maximum normalized entropy over layers 5–16 with threshold 0.85; the KES
neighborhood is centered on the maximum-entropy layer; attention is averaged
over that neighborhood; the top patch ratio is 10%; reinjection uses query/key/
value/output transforms over selected projected visual memory; and the
temporal-negative branch uses either patch temporal shuffling or homogenization
with gamma 0.80, followed by contrastive strength alpha 0.75. The paper says
code is in supplementary material, but the paper body still does not give the
neighborhood width, the concrete Qwen2.5-VL module mapping, or a complete
operational account of those on-the-fly transforms. Its Qwen2.5-VL-7B NExT-QA
accuracy is 59.63% baseline and 60.87% STEAR (+1.24 percentage points), an
accuracy result rather than our claim-conditioned internal selective-dependence
`ΔF`; its table does not explain the printed ±0.00 as uncertainty. Source:
[STEAR primary paper](https://arxiv.org/html/2604.03045).

**Next action:** retrieve and inspect the official supplementary archive and
determine whether its code implements the missing choices for the pinned Qwen
architecture. If it does not, document a distinct, fully specified adaptation
before coding. Do not call such an adaptation STEAR reproduction or run
efficacy scoring yet. Direct faithfulness remains **0/3 reviewed, 0/3 scored,
no estimate/interval/test**; actual browser-to-decoder correspondence remains
**0/15**. The human review and trusted-release gates remain open.
