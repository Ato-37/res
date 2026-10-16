# NExT-GQA claim-mask scale-up preregistration skeleton

Status: **planning worksheet only; not a frozen protocol, sample-size plan,
or scoring authorization**. Fill and hash-freeze every required field before
the first score forward in any new scale-up. Keep source identities and
question text in the private manifest only. The three-record
`sentence_exploratory` candidate membership is a finite feasibility pilot
proposal; its v2 protocol remains a draft and is not scoreable. It is excluded,
by default, from any later scale-up's primary/confirmatory cohort. If reused,
report it as a separate exploratory finite description and never pool it into
the confirmatory primary. Fix this disposition before generating or opening
any scores from the three-item pilot.

This worksheet concerns the current proposed operational question: whether
Qwen2.5-VL's fixed-choice answer margin changes differently when masking a
human-reviewed interval supporting one of its generated claims than when
masking an equal-count temporal control. It does not test whether the claim is
true in general, establish internal reasoning, or by itself show a causal
explanation. This is distinct from the closed head-intervention study.

Treat this as a **masking-based necessity probe**, not a sufficiency test: it
removes a reviewed interval but does not test whether that interval alone
preserves the answer. Black frames may create out-of-distribution inputs or
temporal discontinuities. Equal-count temporal controls match frame count and
rank distance under the frozen rule, not visual content or information value;
human review supports interval relevance to a claim but does not establish that
the model used that interval. Any result is scoped to the fixed model, supplied
choice set, interval policy, and masking procedure.

The current candidate score and contrast should align with the fuller
definition in [`claim_mask_protocol_contract.md`](claim_mask_protocol_contract.md):
fixed gold choice versus the highest-scoring foil in each condition, and
control-minus-claim-linked-mask margin. Treat this as a candidate specification,
not a frozen estimand: target population, claim-support conditioning, contrast
unit, denominator/failure rule, uncertainty target, source clustering, and
practical threshold still need resolution and hash-freeze before score access.
Record full/evidence/control margins and both condition changes from full so the
primary contrast cannot conceal either condition's direction. The fixed-choice
log-margin is not claim-text likelihood or an automatic measure of a generated
claim's truth.

These cautions follow the intervention dependence of perturbation tests and
the distinction between removal-based comprehensiveness and retained-input
sufficiency; they are methodological guidance from image/NLP studies, not
validation of this video-QA setup ([Fong & Vedaldi, 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Fong_Interpretable_Explanations_of_ICCV_2017_paper.html);
[DeYoung et al., 2020](https://aclanthology.org/2020.acl-main.408/)). A
choice-score margin retains graded score information that a binary answer-flip
indicator discards, but a target-versus-best-foil margin still collapses shifts
among other choices. It does not remove the video-domain validation gap
([Siegel et al., 2024](https://aclanthology.org/2024.acl-short.49/)), whose
correlational counterfactual test measures changes across the full label
distribution in NLP.

## Freeze before scoring

Record a value, artifact path, and SHA-256 for each item. `TBD` means the
scale-up is not ready to score.

| Decision | Required frozen value or artifact | Current status |
| --- | --- | --- |
| Target population and scope | Benchmark split, eligible video/question frame, target of inference (finite census, probability sample of that frame, or external superpopulation), and permitted generalization | TBD |
| Candidate construction | Inclusion/exclusion rules, prior-use exclusion, per-video QA selection, seed, and complete screened denominator | TBD |
| Prior exposure audit | Canonical dataset video IDs, video-byte hashes, and video/QID pairs compared against every prior candidate, screening, review, pilot, score, and manifest; hash the resulting denylist/report | Latest `_03` local partial index covers 45 unique source contents: 30 explicit-schema and 15 media/custody sources with provenance unverified. Global sets contain 212 video-ID, 118 media-byte, and 225 video/QID HMAC tokens. The three candidates overlap 3/3 with both the 118-entry legacy list and review-custody package hashes; neither establishes actual viewing or full history. Keep them quarantined. The 10-row answer-margin lock has 0/3 overlap; QID-only hashes are opaque and unsafe. See the 18:52 checkpoint in the dated log |
| Claim-generation procedure | Model/checkpoint, exact prompt/template, decoding settings, output parsing, and claim acceptance rule | TBD |
| Review | Frozen policy, independent reviewer qualifications, assignment/blinding, disagreement/adjudication, interval and support rules | Current v5 is for the three-item feasibility packet only |
| Primary estimand | Population/finite-cohort quantity, exact score, contrast direction, unit, and conditioning on supportable claims | Candidate operation: one fixed gold-vs-best-foil next-token log-margin per video, with `control margin − claim-linked-mask margin`; best foil is selected within each condition. One predeclared claim/interval and one equal-count control per video. Full/evidence/control margins are all recorded. This matches the contract's proposed contrast but is not frozen: finite-cohort target, reviewed-claim eligibility, sampling frame, source dependence, and inference scope remain unresolved |
| Intervention and control | Frame sampler, decoder PTS mapping, mask fill, evidence-frame rule, control selection, tie-break, and source hashes | Three-item draft only; scale-up freeze TBD |
| Denominator and attrition | Counts at screened, sampled, attempted review, reviewed, supported, target-ready, and scored stages; separate the screening-frame counts from the eligible-manifest analysis denominator and post-freeze technical failures | Candidate strict rule for the schema-v13 all-eligible estimand: report review/support/target-ready counts against the full sampled frame, but do not remove unsupported, ambiguous, unreviewed, or zero-overlap cases from a future frozen eligible manifest. Any such case blocks that primary run before scoring; any post-freeze technical failure also stays in the denominator and blocks the complete-cohort primary. The separate three-record v2 pilot draft likewise proposes no row replacement but is not scoreable until the full protocol is jointly frozen. A supportable-claims-only target would require a separately defined and frozen estimand, not attrition relabeling |
| Dependence structure | Video sampling unit and source-cluster audit (e.g. shared source/event/series); analysis unit and cluster method | TBD |
| Uncertainty target | Finite-cohort description or a named superpopulation/design-based estimand with assumptions | TBD |
| Sample-size/precision rationale | If inferential sampling is intended: target half-width or relevant contrast, alpha/power, variance source/range, cluster/finite-population adjustment, and executable calculation. For a complete finite census, state census size and attrition without inventing a power target. | TBD; current n=3 feasibility packet does not supply an effect variance |
| Multiplicity | One primary contrast and any secondary families with adjustment rule | TBD |
| Interpretation/stopping | Practical interpretation, null/inconclusive language, sequential looks, and stopping/failure rules | TBD |
| Pilot inclusion | Whether feasibility-pilot records enter scale-up, with a rule fixed before any pilot scores are generated or opened | Default: exclude this three-item cohort from confirmatory primary |
| Execution freeze | Timestamped cohort, claim packet, target manifest, clock plan/results, model/processor/tokenizer, scoring and analysis code hashes; custodian outcome-access log and unlock rule | TBD; freeze before the first scale-up score forward |

## Required denominator ledger

Preserve counts and sanitized failure codes at every transition:

`eligible frame → sampled → review attempted → review complete → claim
supported → clock/source verified → target and mask/control assigned → model
forward attempted → complete score triplet`.

Do not silently replace a failed record or average only successful records.
Report unsupported, ambiguous, unreviewed, and zero-overlap cases as stage
counts and reasons against the screened/sampled frame. Under the schema-v13
all-eligible candidate estimand, do not remove them from the future frozen
eligible manifest: any such case blocks the primary score run. Excluding them
would define a different, supportable-claim target and requires a separate
protocol frozen before outcomes. After cohort freeze, decoder, clock,
target-preflight, and forward failures remain in the original analysis denominator and block the
complete-cohort primary rather than forming a convenient valid subset. Always
report support and completion rates against the original screened/sampled
frame as well as the frozen eligible-manifest denominator.

## Sample-size decision rules

First choose the target. A complete finite benchmark census can be described
exactly without a population confidence interval. A probability sample for a
finite benchmark frame needs its selection probabilities and design variance.
Claims about an external superpopulation need a separately justified sampling
frame and dependence model. Do not use an independent-video formula until
source clustering has been checked.

For an inferential paired-contrast design, document either a precision target
or a minimum relevant contrast before outcome access. An independent-pair planning
approximation may be shown as a sensitivity calculation, but the final count
must use the actual design's t, finite-population, or cluster-aware simulation
as appropriate. Base the variance range on an independent development source
or conservative prespecified values; the current n=3 feasibility cohort is
not a reliable variance estimator. Inflate the number screened using a
predeclared conservative review/supportability rate, and do not adapt cohort
membership to observed score results.

## Gates that feasibility work must measure

## Production scoring integration status

`nextgqa_claim_mask_conditions.py` is currently a standalone condition
constructor and is called only by its synthetic tests. It does not read videos,
authenticate reviewer/clock provenance, verify playable duration, prepare the
Qwen processor inputs, or call a scorer. The existing NExT-GQA gold-interval
margin runner evaluates benchmark QA evidence intervals, not reviewed
model-generated claims; the timing preflight and synthetic scorer gate also do
not form a production path. Therefore no claim-specific scoring CLI is ready.

The next implementation must bind the proposed three-record membership only
after the full v2 protocol is jointly frozen, plus staged and source media
hashes, verified media duration, completed blinded review,
plan-bound HTML5/decoder clock calibration, and the fixed model/prompt/processor
hashes. It must construct all three conditions for every record and pass an
all-record atomic preflight before model loading. If the v2 protocol is
selected and jointly frozen, the feasibility run would make exactly **9
one-token score forwards** (3 conditions × 3 records); any preflight failure
means zero forwards, and any post-freeze failure means an incomplete cohort
result, never dropping a failed record. This describes a
required future implementation, not current capability or scoring
authorization.

Before scale-up scoring, the development stage should quantify operational
completion and cost: reviewer completion/disagreement and time; supportable
claim rate; exact media/source binding; actual browser-clock held-out residual
and its predeclared frame-classification guard; mask/control assignment rate;
production target-preflight failures; end-to-end runtime and peak memory. If
development scores are collected to estimate variance, keep that cohort
separate from the scale-up confirmation and freeze every method choice before
opening its outcomes.

The current prerequisites and their evidence are tracked in
[`2026-10-16_pdca_6h_restart.md`](../experiments/2026-10-16_pdca_6h_restart.md).
