# Claim-linked evidence-frame mask protocol contract (schema v9)

`scripts/validate_claim_mask_protocol.py` is a CPU-only, fail-closed structural
validator for the exploratory design stage. The secondary-family multiplicity
policy is inherited from the research plan: Holm correction within each
predeclared family (baseline-correctness strata and question-type strata), with
one unadjusted primary contrast. Its uncertainty interval is a two-sided 95%
Student-t interval over one contrast per video, assuming independent
video-level contrasts, with sample SD and `df=n-1`; the primary decision rule
compares the lower bound with the separately fixed SESOI. Family membership
must be frozen before scoring. The SESOI is still
not fixed, so schema v9 rejects a `confirmatory` stage.
Structural validity neither authenticates evidence nor authorizes inference.

The `analysis` object must declare one unadjusted primary contrast and
`primary_ci_method=two_sided_student_t_df_n_minus_1`, the two
secondary families (`baseline_correctness_strata` and `question_type_strata`),
Holm adjustment within each family, and a SHA-256 reference to a frozen manifest
listing each secondary hypothesis/test and its family membership. That manifest
must bind grouping definitions, contrast directions, test/statistic, and family
size; the validator checks only the hash shape, not the file contents or when it
was frozen. The current design records SESOI as unresolved, with the value and
supporting fields null. A future `preoutcome_fixed` declaration must supply a
positive threshold in nat, the primary CI decision rule, a rationale category
and hash, a UTC fixed time, and a separate independent review record hash with
completed/before-scoring declarations and review timestamp no later than the
fixed time. It also requires an explicit `not_accessed_at_freeze` outcome-score
access declaration and a hash for its private access-audit record. These remain
caller declarations; schema validity does not prove access history, reviewer
independence, or that referenced records are authentic. The primary-estimand
SHA-256 is recomputed from compact,
sorted-key UTF-8 JSON (`ensure_ascii=false`) containing the model identity,
eligible-manifest hash and primary population, sampling-config hash, claim,
mapping and human-review record hashes, answer-candidate, gold-choice, and
full/selected/control triplet hashes, score/baseline/target/margin definitions,
and all intervention/control definition fields and hashes, including the
`mask_transform_sha256` and `minimum_selected_frame_count`. This
stops a threshold for another estimand from being structurally substituted by
unit alone and catches changes to those declared inputs if the hash is stale.
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

## Existing-runner compatibility

[`necessity_choice_margin_test.py`](../scripts/necessity_choice_margin_test.py)
is a legacy exploratory runner, not an implementation of schema v9. It shares
the gold-versus-best-foil next-token margin, per-condition best-foil selection,
pre-mask restricted-choice correctness, inclusive timestamp membership, and
equal-count nearest temporal-rank control. It uses the existing grounding
interval directly; it does not bind an independently reviewed answer-relevant
claim-to-interval mapping. Its empty-evidence/insufficient-control cases fail
before that sample's score calls, but are recorded as generic runtime failures
rather than frozen pre-scoring exclusions in the cohort denominator. It
currently declares the baseline-correct cohort as primary, reports bootstrap
intervals and unadjusted sign-flip tests, and has no question-type secondary
family. Preserve its historical outputs as such; do not treat them as v9
results or use it unchanged for a future v9 run. A future runner must score all
eligible videos for the primary population, use the fixed Student-t primary
interval and frozen Holm secondary families, establish the independent claim
mapping, record zero-overlap exclusions before scoring, and enforce/hash the
v9 mask-transform and nonempty-mask rules.

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
codes.
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
