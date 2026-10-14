# Claim-linked evidence-frame mask protocol contract (schema v12)

`scripts/validate_claim_mask_protocol.py` is a CPU-only, fail-closed structural
validator for the exploratory design stage. The secondary-family multiplicity
policy is inherited from the research plan: Holm correction within each
predeclared family (baseline-correctness strata and question-type strata), with
one unadjusted primary contrast. Its uncertainty interval is a two-sided 95%
Student-t interval over one contrast per video, assuming independent
video-level contrasts, with sample SD and `df=n-1`; the primary decision rule
compares the lower bound with the separately fixed SESOI. Family membership
must be frozen before scoring. The SESOI is still
not fixed, so schema v12 rejects a `confirmatory` stage.
Structural validity neither authenticates evidence nor authorizes inference.

Schema v12 adds a distributional plan to the hashed inference declaration. A
resolved Student-t method requires a reviewed declaration of either iid normal
video-level differences or iid finite-variance differences, plus hashes for a
pre-outcome rationale and tail-sensitivity plan. The t interval is exact under
the normal model and an approximation under non-normal finite-variance models.
The synthetic stress test recorded in the dated PDCA log found material
distribution sensitivity under iid right-skewed differences; that does not
establish the shape of project outcomes. The schema also requires a UTC review
timestamp and an outcome-access audit record declaration, and prevents a fixed
SESOI timestamp from preceding the distributional review. These caller-supplied
timestamps and hashes do not authenticate chronology or the underlying records.
The independent SESOI review must occur at or after the distributional review
and no later than the SESOI freeze. The review and sensitivity plan must be fixed
before outcome access. Switching methods in response to observed
outcomes makes the analysis exploratory for those data.

The target population and sampling mechanism for these eligible videos are not
yet established. “All eligible videos” defines the finite cohort mean only
after the eligibility manifest is frozen; it does not by itself make that
cohort a probability sample from a broader video population. A Student-t
interval is therefore not a design-based generalization to videos in general.
Such a generalization needs an explicit sampling/model target and assumptions
about how videos were selected. If the frozen eligible cohort itself is the
target, its observed mean is the cohort description; any interval must state
the additional stochastic population or measurement interpretation it uses.
Unless probability sampling and its design are documented, do not call the
interval design-based sampling error or use it alone to claim population
representativeness; probability sampling is needed for design-based sampling
error estimation ([Statistics Canada](https://www150.statcan.gc.ca/n1/edu/power-pouvoir/ch6/sampling-echantillonage/5214807-eng.htm)).
The usual one-sample t interval is motivated by independent observations from
a population model (see [NIST's confidence-limit discussion](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm)).

For the candidate NExT-GQA test cohort, the official dataset README describes
its test/validation rows as a subset of NExT-QA and says global-content and
descriptive questions are excluded. The NExT-QA dataset page reports a released
1,000-video test split, but neither source describes probability sampling of
videos from a broader real-world population ([NExT-GQA README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md),
[NExT-QA dataset page](https://doc-doc.github.io/docs/nextqa.html)). These
documents define benchmark provenance, not a probability-sampling frame. Until
the target is fixed, claims must not generalize from the benchmark cohort to
VideoQA videos generally.

One row per unique `video_id` prevents counting multiple questions from the
same video as independent units, but does not establish independence across
different videos. Videos may share a creator, recording event, series, or
source lineage; whether such clusters occur in this cohort is unknown until
appropriate metadata are audited. Schema v12 requires explicit
`inference_design` and `dependence_plan` objects, including unresolved states,
and binds target/cohort-selection into the primary-estimand hash and
uncertainty/method declarations into a separate
`analysis.inference_plan_sha256`. Unresolved values do not clear the gate.
`inference_design` records the target kind and definition reference, selection
design, sampling-frame reference, and selection-provenance reference.
`dependence_plan` separately records the uncertainty target, independent unit,
cluster-audit status/unit/count/reference, distributional status/assumption,
rationale and tail-sensitivity hashes, review timestamp and outcome-access
record declaration, and whether the fixed video-level t
method is still unresolved. These declarations are all bound by the inference
plan hash. A finite-cohort descriptive target is paired with
that same uncertainty scope; a video-superpopulation target requires a
model-based or probability-design-based uncertainty target. A resolved t method
requires an identified target/sampling design, a model-based video
superpopulation target, and a completed review with no known cluster fields.
That review is not proof that contrasts are independent. If a pre-score audit
identifies clusters, this version fails closed and a new cluster-aware
method/version is required before scoring. All hashes are caller-provided
references and do not authenticate the underlying documents.
Before any claim-mask outcome access, record
the target population, cohort construction/sampling provenance, available
cluster fields, and whether any cluster structure is material. Keep the
current t method only with its independence/model assumptions stated; if
pre-outcome review finds material clustering, version and freeze an appropriate
cluster-aware method or narrow the inferential claim before scoring. Cluster
methods also require enough independent clusters to support their approximation
(see [Cameron and Miller's review](https://cameron.econ.ucdavis.edu/research/cluster_10-6.pdf)).
After outcome access, a changed method is exploratory for these data; a
confirmatory claim requires a fresh sample under the revised protocol. No
cross-video dependence has been observed or established by this note.

The `analysis` object must declare one unadjusted primary contrast and
`primary_ci_method=two_sided_student_t_df_n_minus_1`, the two
secondary families (`baseline_correctness_strata` and `question_type_strata`),
Holm adjustment within each family, and a SHA-256 reference to a frozen manifest
listing each secondary hypothesis/test and its family membership. That manifest
must bind grouping definitions, contrast directions, test/statistic, and family
size; the validator checks only the hash shape, not the file contents or when it
was frozen. The current design records SESOI as unresolved, with the value and
supporting fields null. A future `preoutcome_fixed` declaration must supply a
positive threshold in nat, the primary CI decision rule, both the
primary-estimand and inference-plan hashes, a rationale category and hash, a
UTC fixed time, and a separate independent review record hash with
completed/before-scoring declarations and review timestamp no later than the
fixed time. It also requires an explicit `not_accessed_at_freeze` outcome-score
access declaration and a hash for its private access-audit record. These remain
caller declarations; schema validity does not prove access history, reviewer
independence, or that referenced records are authentic. The primary-estimand
and uncertainty target plus CI method must already be resolved before fixing
the SESOI; an unresolved target/method remains invalid even with both hash
strings present. The primary-estimand
SHA-256 is recomputed from compact,
sorted-key UTF-8 JSON (`ensure_ascii=false`) containing the model identity,
eligible-manifest hash and primary population, the explicit target/sampling
design declaration, sampling-config hash, claim,
mapping and human-review record hashes, answer-candidate, gold-choice, and
full/selected/control triplet hashes, score/baseline/target/margin definitions,
and all intervention/control definition fields and hashes, including the
`mask_transform_sha256` and `minimum_selected_frame_count`. This
stops a threshold for another estimand from being structurally substituted by
unit alone and catches changes to those declared inputs if the hash is stale.
The separate inference-plan hash binds the uncertainty target, independence
unit, cluster-audit declaration, selected CI method, and multiplicity plan;
the SESOI record must bind both hashes. These hashes distinguish a changed
estimand from changed uncertainty assumptions but do not validate the truth of
either declaration.
The validator cannot authenticate referenced artifacts, reviewer independence,
or that the declaration predates outcome access. A pass remains exploratory
and does not authorize inference.

Use [the SESOI decision record template](claim_mask_sesoi_decision_template.md)
to document the practical interpretation, estimand-specific support, and
independent pre-outcome review. It deliberately supplies no default threshold.

## Proposed estimand

Before scoring, freeze the annotated gold answer choice and one independently
reviewed, answer-relevant claim mapped to a temporal evidence interval. Keep the
gold label and four foil labels fixed in every condition; the foil with the
largest score may differ by condition, matching the existing five-choice margin
implementation. For condition `z`, `S_i(z)` is the teacher-forced next-token
log-probability of the gold label minus the largest log-probability among the
four foil labels, under the same question, five choices, prompt, tokenization,
and processor settings. This is a gold-answer choice margin, not claim-text
likelihood. Baseline prediction correctness is a secondary grouping variable.

`d_i = [S_i(full) - S_i(claim-linked-frame-mask)] - [S_i(full) - S_i(control-frame-mask)]`.
Algebraically, the shared full condition cancels, so the primary contrast is
exactly `d_i = S_i(control-frame-mask) - S_i(claim-linked-frame-mask)`.
`S_i(full)` is still collected to determine the pre-mask greedy-correctness
secondary stratum; it does not contribute numerically to the primary contrast.

The primary mean includes every predeclared eligible video, regardless of
baseline correctness. Determine correctness before masking by taking the
highest next-token log-probability among the five fixed A–E label tokens in the
unmasked condition; break exact ties by the fixed A-to-E label order. This is a
restricted-choice prediction, not unrestricted free-generation exact match.
Correct and incorrect strata are secondary. A positive `d_i` means the
gold-answer margin dropped more under the
selected mask than the matched control. This is an operational dependence
contrast, not proof that a claim is true or causally explained.

Interpret this as a post-hoc, input-intervention test of explanatory
faithfulness/evidence dependence: the model first produces an answer and its
claim, then the evaluated answer margin is measured after altering frames.
It does not test causal faithfulness of an explanation generated before and
used to produce the answer, nor identify an internal reasoning mechanism.
Siegel et al. distinguish explanatory-faithfulness tests that intervene on
inputs from causal-faithfulness claims about a prediction causally depending
on a generated reasoning trace; the present post-hoc design belongs to the
former operational family. Its margin-difference contrast is narrower than
their distribution-shift-based CCT, but the broad idea of relating
counterfactual input impact to free-text explanations is established prior
work. Video intervention itself is also established: [MACD (ICML 2026)](https://proceedings.mlr.press/v306/xiao26w.html)
constructs object-level counterfactual video inputs at inference time to
improve hallucination-mitigating contrastive decoding. The proposed design is
an evaluation contrast over a model-generated claim's reviewed temporal span
against an equal-count temporal control, not a decoding intervention or a
test that the explanation caused the answer. These method distinctions do not
establish priority. See also [The Probabilities Also Matter (ACL 2024)](https://aclanthology.org/2024.acl-short.49/).

For this current all-eligible-video estimand, the independent pre-score review
must establish a supportable claim-to-interval mapping for every eligible
video. A contradicted, unsupported, ambiguous, unlocalizable, or unreviewed
claim does not authorize silently dropping that video, substituting another
claim/video, or continuing with a reduced denominator. Any such case is a
no-go for the planned full-cohort claim-mask run; record the gate failure and
retain the frozen cohort. This preserves the stated population but makes
feasibility contingent on every mapping. A future study may instead define a
supportable-claim population before outcome access, but that is a different,
restricted estimand: it must freeze the eligibility rule and screened
denominator in a new protocol version, report screening attrition, and limit
claims to that population. Unsupported-claim rates may be reported as a
separate descriptive grounding outcome; they must not be converted into
margin-drop values without a separately justified estimand.

## Existing-runner compatibility

[`necessity_choice_margin_test.py`](../scripts/necessity_choice_margin_test.py)
is a legacy exploratory runner, not an implementation of schema v12. It shares
the gold-versus-best-foil next-token margin, per-condition best-foil selection,
pre-mask restricted-choice correctness, inclusive timestamp membership, and
equal-count nearest temporal-rank control. It uses the existing grounding
interval directly; it does not bind an independently reviewed answer-relevant
claim-to-interval mapping. Its empty-evidence/insufficient-control cases fail
before that sample's score calls, but are recorded as generic runtime failures
rather than frozen pre-scoring exclusions in the cohort denominator. It
currently declares the baseline-correct cohort as primary, reports bootstrap
intervals and unadjusted sign-flip tests, and has no question-type secondary
family. Preserve its historical outputs as pre-v12 legacy results; do not
describe them as protocol-v12 results or use this runner unchanged for a future
schema-v12 run. A future runner must score all
eligible videos for the primary population, use the fixed Student-t primary
interval and frozen Holm secondary families, establish the independent claim
mapping, record zero-overlap exclusions before scoring, and enforce/hash the
current contract's mask-transform and nonempty-mask rules in a schema-v12-
compatible runner.

The legacy scorer and CPU preflight share one endpoint-inclusive rounded
uniform sampler, `round(i * (N - 1) / max(F - 1, 1))`, where `N` is decoded
frame count and `F` is requested frame count. For clips shorter than `F`, the
helper deliberately preserves repeated indices for legacy parity; the
preflight deduplicates and now explicitly rejects a clip when fewer than the
requested number of distinct frames remain, before model scoring. The old
scoring path may still fail when it cannot materialize the requested distinct
frames. Any future schema-v12-compatible runner must freeze its sampling rule
in the sampling-config hash and enforce the same requirement before any model
score call. This shared helper does not by itself establish timestamp precision
or contract compatibility. Preflight also requires a finite, positive source FPS from the
video metadata; it does not substitute a conventional FPS when metadata is zero,
missing, or non-finite, because that would silently change the time-to-frame
mapping used for evidence intervals.

`necessity_choice_margin_utils.assign_evidence_control_frames` now composes
the sampler, distinct-frame check, inclusive interval overlap, and equal-count
matched-control selection into one deterministic CPU assignment with explicit
pre-score statuses and matched rank-distance. It validates finite
nonnegative-duration endpoints and canonicalizes interval order/duplicates for
coverage, whose union membership does not depend on either; the private claim
ledger separately requires ordered, unique, positive-duration reviewed
intervals. The legacy preflight passes its
existing QA grounding intervals to this generic helper; that does not turn
those QA-level intervals into independently reviewed model-claim mappings.
Preflight also requires an exact nonempty `video_id` and rejects every
otherwise-preflightable row sharing a `video_id` before decoding, preserving
the one-video analysis unit instead of selecting a first row from duplicates.
The helper also cannot authenticate source FPS or referenced review records.

## Cohort, mapping, and control

`candidate_manifest_sha256` binds the pre-screen universe;
`eligible_manifest_sha256` binds the frozen one-video/one-QA eligibility set;
`exclusion_manifest_sha256` binds pre-scoring exclusions and reason codes. The
candidate count must reconcile with eligible count plus pre-run exclusions;
the eligible count is the analysis denominator. The primary-estimand hash
binds the final eligible set, while the candidate,
exclusion, and overlap-audit hashes remain separately required provenance for
the screening and selection process; changes to that process are not encoded in
the primary-estimand hash itself. The answer-candidate protocol hash binds the
prompt, five choices, label tokenization, and scoring
implementation. `gold_choice_manifest_sha256` binds per-video gold labels;
`triplet_manifest_sha256` binds full/selected/control assignments. Claim,
mapping, review, and rights hashes refer to separate artifacts. These are
declared references; the validator does not resolve or authenticate them.

For NExT-GQA, the supplied temporal interval is a QA-pair-level answer-evidence
annotation. It is not, by itself, a mapping for a later Qwen-generated
natural-language claim. If such an interval is used to seed review, the
independent reviewer must inspect the frozen generated claim and video and
record whether the reviewed interval supports that specific claim. The paper's
grounding criterion accepts any moment sufficient to reveal the answer; its
appendix asks annotators to label all relevant moments when the same evidence
recurs. A future claim-level protocol must state whether it maps one sufficient
occurrence or all relevant occurrences, and reviewers must follow that rule.
The schema-v12 hash declarations do not enforce these semantic requirements;
the independent human inspection of claim and video remains necessary, and a
structural pass alone cannot establish the mapping or authorize inference.
Use the blank [private human-review worksheet](claim_mask_human_review_worksheet.md)
to record the review procedure and decisions; it guides a local private record
but does not replace the frozen claim policy or independent human review.

For independence from QA-span anchoring, the primary mapper sees the video,
question, and frozen generated claim, but not the QA interval seed, gold answer,
or outcome scores. The mapper records and freezes claim support and intervals
before any seed exposure. Only after that freeze may an auditor compare the QA
seed with the mapping; that comparison is descriptive and must never edit or
expand the primary intervals. `qa_seed_hidden_until_mapping_frozen` must be
declared `true` in the private review record. This is a self-declaration: the
structural checker cannot prove what the reviewer saw, when the freeze occurred,
or whether the referenced record is authentic. A seed-derived review that
cannot establish this ordering does not pass the independent-mapping gate.

[`scripts/audit_claim_mapping_manifest.py`](../scripts/audit_claim_mapping_manifest.py)
provides a separate private, CPU-only consistency check for this record. Its
ledger binds one frozen model-generated claim per eligible video to one or more
positive-duration reviewed intervals and a human-review record; it declares
whether a NExT-GQA interval was not used or was used only as a seed, and freezes
whether the review covers one sufficient occurrence or all relevant repeated
occurrences.
It rejects an explicit `qa_span_copy`, mutable model-revision aliases, missing
claim/review fields, mismatched claim/interval hashes declared in the review
row, parsed/raw-byte mismatches, non-finite numeric input, whitespace-padded
IDs, invalid occurrence-policy types, float-collapsed endpoints, and review
timestamps at or after a supplied scoring start. This remains a declaration
checker: it cannot authenticate the model run or reviewer, inspect the linked
video or review record, establish semantic support or exhaustive occurrence
coverage, or authorize execution. Its summary omits identifiers, claims,
interval values, locators, and per-item hashes.

Identical QA-seed and reviewed interval coordinates are not automatically
rejected: a human may independently confirm that the same interval supports
the specific claim. The ledger's `seed_only`, `independent_claim_review`, and
`qa_seed_hidden_until_mapping_frozen` fields remain self-declarations, so a
dishonest ledger can still label a copied span as independently reviewed.
Structural acceptance is only a
precondition to human audit, never proof that the review occurred.
If a `claim_support_decision` field is present and is not `supports_claim`,
the checker emits the generic `claim_support_not_established` failure code,
even if another review field is missing. It does not disclose which
non-support category was selected or which item failed.

Run it only against the private eligible manifest and private claim ledger, for
example:

```sh
.venv/bin/python scripts/audit_claim_mapping_manifest.py \
  --eligible-manifest /private/path/eligible.json \
  --claim-ledger /private/path/claim_mapping_ledger.json
```

The version-2 ledger binds the eligible-manifest hash, claim-review policy
hash, occurrence policy, and optional scoring-start timestamp. Each claim row
contains the video and claim IDs, claim-text hash, frozen model origin/revision
and generation-run hash, mapping method, QA-seed usage and source intervals,
the reviewed evidence intervals, and a review declaration binding the seen
claim/interval hashes, independent-human role, record hash/time, video-review
conditions, answer/gold blindness, seed-hidden-until-freeze declaration,
support decision, and applied occurrence policy. Keep this
ledger private; its row-level identifiers, timestamps, and hashes can reveal
or link sensitive research records.

Before scoring, a runner must verify that the five distinct A–E labels each
map to one token and that the exact rendered chat-generation prefix remains
stable when each label is appended as the assistant target, with the target's
first contextual token equal to its standalone label token. For video inputs,
also verify prefix stability and target-token identity after the production
video processor expands sampled frames into model inputs. Verify the selected
and control mask transforms preserve input token IDs, video-grid metadata, and
pixel tensor shape while changing pixel content; otherwise, freeze an explicit
score-position rule before outcomes. Bind the pinned tokenizer, chat template,
prompt builder, video processor, and these boundary-audit results in the
answer-candidate protocol provenance. A CPU-only synthetic boundary audit
does not validate any model score or real item.

The selected mask contains the fixed sampled-frame indices overlapping the
reviewed claim-evidence interval. Under the current implementation convention,
sampled time is original frame index divided by decoded source FPS and interval
endpoints are inclusive (`start <= time <= end`); the mapping hash must bind
source timestamps and mapping code. This reviewed interval determines the
evidence mask and its size `k`; observed model scores cannot alter it. Require
`k >= 1`: if no sampled frame overlaps the reviewed interval (`k=0`), record a
pre-scoring exclusion and do not score the item. Schema v9 fixes
`minimum_selected_frame_count=1`. For control, sort evidence ranks
`e_1..e_k` and candidate control ranks `c_1..c_k`,
then choose a subset of `k` non-evidence frames minimizing
`sum_j |e_j-c_j|`. Resolve equal-cost ties with the lexicographically lower
ordered control-rank tuple. If fewer than `k` control frames exist, mark the
item ineligible before scoring; do not substitute after observing outcomes.
The structural validator does not decode videos or calculate per-item overlap
or mask cardinality. Any future scoring runner must enforce the minimum before
scoring and record zero-overlap exclusions in the frozen exclusion manifest.
Full, selected, and control conditions share frame
sampling, fill, preprocessing, model, precision, prompt, gold target, and
execution settings. The required `mask_transform_sha256` must bind a private,
versioned specification of the exact transform used on every selected/control
frame, including replacement (for example black or a precisely parameterized
blur), color space, operation order relative to resize/normalization, and
whether the entire sampled frame is replaced. The same transform is applied
to both masks. This field binds the transform choice to the primary estimand;
the validator checks only hash syntax and consistency, not the specification
or implementation. No transform choice is made by this schema. Masking can
create out-of-distribution input.

## Gates and limits

The rights declaration binds a separate item-rights manifest and reconciles
covered plus unresolved eligible-video counts to the frozen eligible cohort.
“Covered” means item-specific rights evidence for the intended use has been
reviewed and recorded; a video-ID mapping alone does not count. Declaring
`verified` requires all eligible videos covered and zero unresolved. The
validator checks only the declared counts and hash shape, not the evidence or
whether a video license permits the intended use.

The private item-rights manifest should contain exactly one row per eligible
video and a local-only crosswalk to the eligible manifest. Record the
NExT-GQA-to-source-video lineage, item-specific source license identifier and
evidence locator/hash/date, reviewer and review date, conditions, and distinct
decisions for local decoding/inference, retention/cache, derived aggregate
publication, and redistribution if proposed. Assess annotation terms separately
from original video-media terms. Missing, duplicate, ambiguous, or unresolved
rows count as unresolved for the declared intended-use scope. Conditional terms
remain unresolved until their applicability and fulfillment for that use are
reviewed and recorded. Keep item identifiers, source URLs, and evidence
private. A matching validator could check row/key/count reconciliation, but a
human must assess evidence authenticity, reviewer authority, and license scope.
If obtaining or identifying the source items also invokes a dataset-level
access or metadata agreement, review that separately from annotation and
media licenses. The current v2 rights ledger has no general dataset-access-
agreement record; keep any applicable agreement and its use-scope decision in a
separate private record until the schema is deliberately extended. In this
protocol, `aggregate_publication` means privacy-safe derived numerical
summaries only. It does not authorize publishing source media, identifiable
frames, item-level records, or raw model outputs.
The blank [human-review worksheet](claim_mask_human_review_worksheet.md) also
provides a private evidence checklist for claim and rights reviewers.

For a private draft ledger, `scripts/audit_item_rights_manifest.py` checks the
eligible-manifest byte hash, one-to-one video-ID row coverage, separate
annotation and per-video media evidence references, every source-lineage mapping
transition, review declarations,
resolved-conditions declarations, and explicit decisions for each declared use.
The eligible manifest is a nonempty JSON array of objects with a unique
`video_id` string. Schema version 2 adds a top-level `eligible_dataset_id` for
that identifier namespace. The ledger's top-level `intended_uses` must include
`local_inference` and may also include `local_retention`,
`aggregate_publication`, and `redistribution`. An `intended_use_scope` string
records the human-declared purpose and constraints for those use categories.
`annotation_rights` contains `dataset_id`, `license_id`, an `evidence_locator`,
evidence and review-record SHA-256 values, `reviewer_role`, `review_status`,
`reviewed_at_utc`, and decisions keyed by every intended use. Each `items` row
contains `video_id`, `source_dataset_id`, `source_video_id`, `media_license_id`,
`media_evidence_locator`, media-evidence, review-record, and conditions-record
SHA-256 values, `media_reviewer_role`, media review status/time,
`conditions_status`, and decisions keyed by every intended use.
Each item also carries a nonempty ordered `source_lineage` list. Every
transition records its from/to dataset IDs and record IDs plus a private mapping
evidence locator and SHA-256. The first transition must start at the eligible
item ID, adjacent transitions must join exactly, and the final transition must
end at the media-license source dataset/video ID. This records cross-dataset
mapping provenance separately from the source-media license evidence.
Locators, roles, and notes stay in the private ledger. Decisions are exactly
`permitted`, `not_permitted`, or `unresolved`; review statuses must be completed
and conditions must be declared `reviewed_resolved` for every eligible item.
Keep both inputs local and private. The output contains no item identifiers or
evidence text; it reports hashes, counts, structural-status fields, and failure
codes. The callable audit also reparses both hash-bound raw byte inputs and
requires canonical parsed-object equality, rejecting duplicate JSON keys and
non-finite numbers; the CLI summary records only generic mismatch codes. Video,
dataset, and lineage record identifiers must match exactly and cannot rely on
whitespace trimming for cohort reconciliation.
Because the hashes are unsalted, they can reveal equality and may be guessable
for predictable manifests; do not treat the output as absolutely private.
Example:

```sh
.venv/bin/python scripts/audit_item_rights_manifest.py \
  --eligible-manifest /private/path/eligible_videos.json \
  --rights-ledger /private/path/item_rights_ledger.json
```

The ledger shape is defined by the validator's synthetic tests. `permitted`,
`not_permitted`, or `unresolved` are human declarations; a completed review
record, a hash, a resolved-conditions label, or a structurally valid result
does not establish truth or legal scope. Any missing, extra, duplicate, or
unresolved item/use keeps the rights gate closed, and execution authorization
is always `not_granted`.

The protocol input's `runtime.execution_authorization` must remain boolean
`false`; the item-rights audit summary reports the string `not_granted`.
Review timing and data-rights status are separate declarations, but a structural
pass does not make either gate true. The claim-mask validator does not
authenticate signatures, validate hash targets,
establish claim/interval semantics, verify rights, or validate model behavior.
NOAA human review remains incomplete and NExT-GQA item-level rights unresolved;
the real-data branch remains `no_go`. No efficacy result comes from this
contract or its synthetic tests.

CPU check:

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest scripts/test_validate_claim_mask_protocol.py -v
```
