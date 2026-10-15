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
The frozen primary cohort must contain at least two eligible videos because
the declared sample SD and `df=n-1` interval are undefined at `n=1`; the
validator rejects smaller cohorts. This is only a mathematical minimum, not a
claim that two videos provide adequate precision or support the population
assumptions.

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
claim, a zero-overlap interval, or an insufficient-control assignment does not
authorize silently dropping that video, substituting another claim/video, or
continuing with a reduced denominator. Any such case is a no-go for the planned
full-cohort claim-mask run; record the gate failure and retain the frozen
cohort. This preserves the stated population but makes feasibility contingent
on every mapping and assignment. A future study may instead define a
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
mapping, and enforce/hash the current contract's mask-transform and
nonempty-mask rules in a schema-v12-compatible runner. Before any model score
call, it must compute assignments for the entire frozen eligible cohort. If
any reviewed claim has zero sampled overlap or lacks an equal-count control,
the runner must record a fixed-code cohort-level gate failure, score none of
the cohort, and preserve its denominator. A subset run requires a separately
versioned estimand and cohort frozen before outcome access.

The legacy scorer and CPU preflight share one endpoint-inclusive rounded
uniform sampler, `round(i * (N - 1) / max(F - 1, 1))`, where `N` is decoded
frame count and `F` is requested frame count. For clips shorter than `F`, the
helper deliberately preserves repeated indices for legacy parity; the
preflight deduplicates and now explicitly rejects a clip when fewer than the
requested number of distinct frames remain, before model scoring. The old
scoring path may still fail when it cannot materialize the requested distinct
frames. Schema v12 requires `F >= 2` so a sampled temporal rate and a distinct
equal-count control can be defined. Any future schema-v12-compatible runner
must freeze one cohort-wide `F` in the sampling-config hash and enforce the
same requirement before any model score call. This shared helper does not by
itself establish timestamp precision or contract compatibility. Preflight also
requires a finite, positive source FPS from the video metadata; it does not
substitute a conventional FPS when metadata is zero, missing, or non-finite,
because that would silently change the time-to-frame
mapping used for evidence intervals. A finite nominal FPS alone does not prove
that decoded frame index divided by FPS equals media presentation time. Before
any schema-v12 scoring run, the runner must obtain a validated presentation-time
map for the exact decoded sequence and exact frames passed to the model. The
current sampler/control objective is based on sampled-frame rank, which
corresponds to elapsed media time only when sampled PTS fit a uniform temporal
grid within a tolerance frozen before outcomes. The runner must check this for
every eligible video and pass an explicit effective `sample_fps` to the
processor that agrees with the validated grid. A PTS map alone does not make an
irregular VFR sequence compatible: the current processor and rank-distance
control do not consume an arbitrary per-frame time vector.

This is confirmed for the active local path: `qwen_vl_utils` 0.0.14's
frame-list `fetch_video` defaults `sample_fps` to 2.0 and synthesizes uniform
frame indices. However, `activation_patching_sweep.build_inputs` calls
`process_vision_info` without requesting video metadata/kwargs and then passes
only its video tensor to Transformers; the 2.0-FPS metadata is discarded.
Transformers 5.17.0 creates fallback metadata with `fps=None`, whose
`sampled_fps` property returns 24, and the Qwen2.5-VL processor derives one
`second_per_grid_ts` value per video from that `sampled_fps` and
`temporal_patch_size`, rather than accepting per-frame PTS. With the local
temporal patch size of 2, missing metadata therefore yields 1/12 second per
temporal grid. A synthetic four-frame CPU path confirmed fallback 24 FPS and
that grid interval; it did not load a model or score an item. The separate
`fetch_video` unit smoke confirmed its own 2.0-FPS default, but that value does
not reach this caller's processor. Preserve legacy results as outputs of that
historical input pipeline; do not treat these checks as a retroactive
correction to their scores.

The active local Transformers 5.17.0 Qwen model implementation adds another
timing gate: `get_rope_index` calculates the temporal position step as
`tokens_per_second * int(second_per_grid_ts)`. An interval below one second is
therefore truncated to zero before scaling. With temporal patch size 2 and the
fallback 24 FPS, the processor's 1/12-second interval produces zero temporal
spacing in this installed source path. A call to the installed method on a
four-step synthetic grid returned `[0, 0, 0, 0]`. Transformers 5.18.0 changes
this calculation to preserve the fractional interval until final integer
position quantization; the same CPU fixture returned `[0, 0, 0, 1]`. The
upstream regression test deliberately includes repeated adjacent coordinates
for fractional intervals, so the protocol must compare against that version's
defined quantization rather than require every timestamp to have a unique
integer ID. Pin at least the fixed implementation, bind its source/version and
checkpoint timing config, and test expected position IDs on the exact processor
path before scoring. This is a verified source-level compatibility issue, not
a model-output efficacy result, and does not retroactively alter historical
scores ([Transformers 5.18 release notes](https://newreleases.io/project/pypi/transformers/release/5.18.0),
[upstream Qwen fractional-position regression test](https://github.com/huggingface/transformers/blob/main/tests/models/qwen2_5_vl/test_modeling_qwen2_5_vl.py)).

For the current rank-matched design, a future schema-v12 runner must fail the
entire frozen cohort before scoring if any selected PTS sequence exceeds the
predeclared uniform-grid tolerance, lacks complete/strictly ordered timestamps,
or cannot be bound to the actual input pixels. It must derive and record the
effective sample rate from those same PTS and bind the decoder/source, exact
frame ordering, time origin, time base, sample-rate conversion, processor/model
versions, tolerance, preprocessing, and mapping code in private provenance.
The exact same temporal metadata must be supplied to full, selected, and
control conditions. An arbitrary VFR cohort requires a new protocol version
with an elapsed-time control rule and an input representation that preserves
its timing; do not silently reuse the rank-matched estimand. The current
OpenCV index/FPS helper has not passed this validation and remains geometry
only. The structural protocol validator does not inspect media timestamps or
enforce this gate.

`scripts/presentation_time_utils.py` implements exact arithmetic for a
decoder-provided display-order PTS sequence and rational time base, including
inclusive interval membership and the existing sampled-rank control assignment.
Its synthetic tests cover irregular positive PTS intervals and malformed or
missing timing. It is not a decoder adapter and does not prove that the PTS map
belongs to the exact pixel sequence supplied to the model or that a zero-based
first-frame origin matches the annotation player's timeline. A production
adapter must bind the decoder/library version, stream and frame ordering,
timestamp origin conversion, sampled-frame pixel identity, and mapping-code
hash; validate this correspondence on the exact path used for model inputs;
and freeze the annotation-time calibration and endpoint precision before any
outcome access. Until that exists, the PTS helper does not clear the timing gate.
`scripts/preflight_claim_mask_timing.py` adds a synthetic-only whole-cohort
contract check for complete monotonic PTS, fixed-sampler indices, nonempty
evidence/control assignment, pixel-identity witnesses, derived FPS, and equal
timing declarations across full/selected/control conditions. Its tests show
that each record must match one cohort-level frozen frame count, that one
failing member blocks the declared denominator, and the helper itself has no
scoring call path. It is not connected to the actual cross-gate
preflight or scorer; the hashes in these synthetic fixtures are caller-provided
identity witnesses, not proof of production pixel provenance. It cannot clear
the production PTS, pixel-binding, or annotation-clock gates.
Use the blank private
[annotation-clock calibration worksheet](annotation_clock_calibration_worksheet.md)
to prepare the human, outcome-blind clock mapping record. It does not supply
measurements or clear the gate until all frozen eligible items pass the
prespecified calibration rule. The worksheet separates anchors used to fit an
offset/drift mapping from disjoint, preselected held-out validation anchors: fit
residuals alone cannot establish out-of-sample mapping accuracy. Freeze the
anchor counts and temporal coverage, mapping direction/function, fit and
uncertainty methods, and validation tolerance before measuring; evaluate held-
out residuals without refitting. Held-out anchors are not necessarily
statistically independent within a video; any uncertainty model must justify
its dependence assumptions. This is a protocol safeguard, not a claim that any
existing video has passed calibration.

The direct choice-margin scorer and its publisher currently refuse every
preflight report before model loading, including a hand-written
`ready`/`granted` JSON. The schema-v12 cross-gate deliberately emits
`blocked`/`not_granted` and cannot authorize execution. A future gate must use
a revised, authenticated report schema that binds the exact eligible/sample
manifests and model, sampling, and intervention configuration; it must require
human claim review, item and dataset rights, and production timing/pixel
calibration, and a prespecified inferential target and SESOI. The consumer must
verify those bindings and record the accepted report hash in private run
provenance before it can allow a model load. Such a gate is not implemented,
so a report's self-declared status cannot clear it.
The NExT-GQA README represents its annotated spans in seconds and separately
lists per-video duration/FPS and a frame-ID-to-time map, but does not specify
how that map's clock origin is calibrated against a decoded stream's PTS
([official NExT-GQA README](https://github.com/doc-doc/NExT-GQA/blob/main/datasets/nextgqa/README.md)).
The README therefore establishes the annotation representation, not the
annotation-to-decoder timing correspondence required above.

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
candidate count must reconcile with eligible count plus pre-run exclusions.
The freeze order is: freeze the candidate manifest, complete the prespecified,
outcome-blind eligibility screen, record every exclusion and reason, then
freeze the eligible and exclusion manifests before claim review and scoring.
The eligible count is the analysis denominator. Never remove a row from an
already-frozen eligible manifest to repair claim coverage or control geometry;
such a case blocks the full-cohort run. In this protocol the claim-input
manifest and human review are bound to the frozen eligible manifest, so those
claim-dependent geometry failures are only known after cohort freeze. The
structural validator checks count reconciliation and hash syntax, not whether
the exclusions were justified or whether the review really preceded scoring.
Pre-freeze exclusions cannot be used to remove an unsupported or unreviewed
claim from the current all-eligible cohort; the full-cohort no-go above
applies. A supportable-claim-only cohort requires the separately defined
restricted estimand described above. The primary-estimand hash
binds the final eligible set, while the candidate,
exclusion, and overlap-audit hashes remain separately required provenance for
the screening and selection process; changes to that process are not encoded in
the primary-estimand hash itself. The answer-candidate protocol hash binds the
prompt, five choices, label tokenization, and scoring
implementation. `gold_choice_manifest_sha256` binds per-video gold labels;
`triplet_manifest_sha256` binds full/selected/control assignments. Claim,
mapping, review, and rights hashes refer to separate artifacts. These are
declared references; the validator does not resolve or authenticate them.
For claim review, `claim_manifest_sha256` refers to the exact frozen pre-review
claim-input manifest bytes, including each private frozen claim text;
`claim_text_sha256` is computed over exact UTF-8 bytes without normalization.
`mapping_sha256` refers to the exact completed v2
claim-mapping ledger bytes, and `review_record_sha256` to the exact bytes of a
private index that lists each ledger row's per-row review-record hash and
private reference. The private worksheet specifies the two v1 record shapes;
`scripts/audit_claim_review_evidence_index.py` checks their byte bindings and
row joins. This checker complements, but does not replace, the protocol and
claim-ledger validators or the human semantic review. It always returns
`execution_authorization=not_granted`.

`scripts/preflight_claim_mask_gates.py` is a CPU-only cross-gate diagnostic for
the protocol, eligible manifest, claim-input manifest, claim ledger, claim
review index, item-rights ledger, and dataset-rights evidence index. It reads
each file once, runs the structural checks against those exact byte snapshots,
and emits only component statuses and fixed failure codes. Even if every
structural component passes, it reports `overall_status=blocked` and
`execution_authorization=not_granted`: schema v12 only permits exploratory
protocols, and the claim-mask SESOI is unresolved. This tool does not initialize
a model or video processor, authenticate human/legal evidence, or authorize a
future run. Do not use its process exit code alone as an execution permission.
It exits nonzero for every result, including when all structural components
pass, because `overall_status` is always blocked under schema v12.

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
`k >= 1`. If no sampled frame overlaps the reviewed interval (`k=0`), record a
fixed-code cohort-level gate failure before scoring any item, stop the planned
full-cohort run, and retain its eligible denominator.
Schema v9 fixes
`minimum_selected_frame_count=1`. For control, sort evidence ranks
`e_1..e_k` and candidate control ranks `c_1..c_k`,
then choose a subset of `k` non-evidence frames minimizing
`sum_j |e_j-c_j|`. Resolve equal-cost ties with the lexicographically lower
ordered control-rank tuple. If fewer than `k` control frames exist for any
eligible video, record a fixed-code cohort-level gate failure before scoring
any item, stop the full-cohort run, and retain its frozen denominator. Never
substitute after observing outcomes.
The structural validator does not decode videos or calculate per-item overlap
or mask cardinality. Any future scoring runner must enforce the minimum over
the entire frozen eligible cohort before scoring; a failure blocks that run
and does not alter the frozen exclusion manifest or denominator.
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
Declare `verified` only after a human has also reviewed every applicable
dataset-level access/metadata agreement for the same planned-use scope; if none
applies, the private evidence index must preserve the substantiated
non-applicability decision. The validator cannot enforce this condition.

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
For the protocol's `data_rights` object, use `item_rights_manifest_sha256` for
the exact v2 item-rights ledger and `evidence_sha256` for a versioned private
evidence-index JSON that lists the exact-byte hashes and private references for
that ledger, annotation terms, item-level media/conditions evidence, and any
dataset-level agreement applicability/use review (including substantiation of
non-applicability when relevant). The validator checks only hash shape and
does not resolve the index or its entries. The blank worksheet specifies the
minimum private review record; it does not substitute for evidence inspection.
The CPU-only `scripts/audit_data_rights_evidence_index.py` can separately
cross-check the exact-byte hashes of the eligible manifest and v2 ledger, and
compare the annotation evidence hash declared in the index with the hash
declared in the ledger. It also checks declared scope/count consistency across
the protocol, eligible manifest, ledger, and evidence index. It checks
the media/conditions and agreement hash/locator references for shape and
coverage/declarations but does not resolve those references or compare their
target bytes with the ledger rows. A human must inspect those private artifacts
and verify the references before declaring rights verified. Run the checker
alongside the existing structural protocol and item-rights audits. This component is not
a launch orchestrator, does not authenticate private evidence or legal
judgments, and always reports `execution_authorization=not_granted`; a passing
structural check is not permission to load a model or score data.
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
The summary field `declared_item_and_annotation_use_gates_structurally_complete`
is limited to the ledger's per-item media and annotation declarations; it does
not cover any dataset-level access or metadata agreement. The summary marks
`dataset_access_agreement_gate` as `not_machine_verified`; assess any applicable
agreement from its separate private record before treating planned use as
eligible. Neither field grants permission.

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
