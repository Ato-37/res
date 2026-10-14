# Claim-mask human review worksheet (private working copy)

This worksheet helps a reviewer prepare the records required by the claim-mask
contract. It is not a rights opinion, annotation, eligibility decision, or
execution authorization. Keep completed copies, identities, item identifiers,
source URLs, video files, evidence documents, and notes in the restricted local
research area. Do not commit completed records. The public repository should
contain only this blank worksheet and privacy-safe aggregate summaries.

Before review begins, freeze and record the protocol revision, eligible-manifest
hash, **pre-review frozen model-claim input manifest hash**, model/tokenizer
revision, claim-review policy hash, and the occurrence policy
(`one_sufficient_occurrence` or `all_relevant_occurrences`). This input-manifest
hash is not the completed claim-mapping ledger hash: the latter can be computed
only after intervals and review records are frozen. Record both artifacts under
distinct names. The occurrence policy must come from the protocol; a reviewer
must not choose a policy per video. Bind the display procedure and permitted
evidence modalities, the claim-support decision rule, and the timestamp
precision/boundary convention to the claim-review policy hash. Because the
primary intervention masks video frames only, every primary evidence interval
must be supported by visual content in the source video frames. Audio,
subtitles, and metadata must not establish primary support, even when they are
present in the model input; a multimodal evidence intervention requires its own
separately frozen protocol. Use the frozen procedure consistently. If the
protocol has not fixed these inputs, stop before opening items and return the
decision to the protocol owner.

## Claim-to-evidence review

For each eligible video, create a private record with these fields:

| Field | Record |
|---|---|
| Eligible video ID and claim ID | |
| Frozen claim-text SHA-256 | |
| Frozen claim-generation run and model-revision references | |
| Reviewer role and review-record reference | |
| Review timestamp (UTC, before any scoring) | |
| Source-video identity verified against the private lineage record? | yes / no / unresolved |
| Full source video reviewed using the frozen display procedure? | yes / no |
| Question and frozen claim were the only answer-context fields shown? | yes / no |
| Gold answer and QA interval seed were hidden until the mapping was frozen? | yes / no |
| Outcome scores were not accessed? | yes / no / unknown |
| Frozen occurrence policy applied | one sufficient occurrence / all relevant occurrences |
| Claim support decision | supports claim / partial / contradicted / unsupported / ambiguous / unlocalizable |
| Evidence intervals in source-video seconds, inclusive endpoints | |
| Brief private rationale for each interval and each material claim component | |
| Reviewer notes or reason for a non-support decision | |
| Final review status | completed / incomplete |

Confirm that the reviewed video is the exact source item linked to the eligible
video, not merely a similarly named or visually similar file. Review the entire
video needed to assess the frozen claim. Do not consult the QA interval seed,
gold answer, model scores, or intervention outcomes while selecting intervals.
Record whether the claim's material factual content is supported by temporal
evidence in the video. If only part is supported, the item is contradicted,
unsupported, ambiguous, or cannot be localized, do not label it
`supports_claim` and do not manufacture an interval to make it pass.

For a supported claim, record source-video start/end seconds with positive
duration and inclusive endpoints. The interval must show the evidence that
supports the claim, not merely a scene correlated with it. Apply the frozen
occurrence policy exactly: one sufficient occurrence means one such interval;
all relevant repeated occurrences means record every relevant occurrence under
the fixed policy. Keep a short private rationale tied to each interval. Do not
copy QA-span coordinates into the primary mapping. Only after the primary
mapping is frozen may a separately assigned auditor compare any declared QA
seed; that comparison is descriptive and must never edit the frozen mapping.

The current estimand requires one supportable mapping for every video in the
frozen eligible cohort. Any non-support, unresolved identity, incomplete
review, missing interval, or unreviewed eligible video is a **no-go for the
full-cohort claim-mask run**. Record the gate failure and retain the original
cohort. Do not silently exclude the item, substitute another claim/video, or
continue with a reduced denominator. A supportable-claim-only population would
require a separately frozen eligibility rule, denominator, and protocol before
outcome access.

After completion, keep the pre-review input-manifest hash in the private
protocol record; the v2 claim-mapping ledger has no field for that whole-file
hash. Bind each claim-text hash, generation-run reference, frozen interval
hash, review-record hash, and review timestamp to the completed private ledger
using its per-row fields, then record the final ledger hash in the private
protocol record. The machine audit checks structural declarations only; the human reviewer remains
responsible for semantic support, source identity, blinding, and the truth of
the review record.

When transcribing a completed supported row into the claim-mapping ledger, use
the exact contract keys: `claim_origin=model_generated_frozen`,
`mapping_method=independent_claim_review`, the frozen model revision and
generation-run hash, and the protocol's `occurrence_policy`; set
`claim_policy_sha256` to the frozen policy artifact hash. In `review`, set
`status=completed`, `reviewer_role=independent_human`, the private
`review_record_sha256` and UTC timestamp, `claim_text_sha256_seen` and
`evidence_intervals_sha256_seen`, `video_reviewed=true`,
`answer_and_gold_hidden=true`, `qa_seed_hidden_until_mapping_frozen=true`,
`outcome_scores_seen=false`, `claim_support_decision=supports_claim`, and
`occurrence_policy_applied` to the frozen policy. Declare `qa_seed.usage` as
`not_used` with an empty interval list, or `seed_only` if the QA interval was
consulted only after the primary mapping was frozen. A row that cannot truthfully
meet these declarations does not pass the gate.

## Item-specific rights review

Use `scripts/create_item_rights_review_template.py` to create a private,
unresolved ledger from the frozen eligible manifest, then complete it alongside
a separate private evidence bundle. Review annotation rights and original
media rights separately. For each item, the evidence bundle should let an
independent reviewer reproduce the decision without exposing materials in the
public repository.

| Evidence to record privately | Record |
|---|---|
| Eligible ID → benchmark row → source dataset/video lineage | |
| Local media file identity check and method/result (for example, byte hash plus source metadata comparison) | |
| Source record or license-document version, locator, retrieval/review date, and SHA-256 | |
| Exact license terms or clauses relevant to this use; conditions and attribution duties | |
| Applicability of each condition and evidence that it is fulfilled | |
| Reviewer role, authority basis, review timestamp (UTC), and review-record hash | |
| Annotation dataset terms reviewed separately? evidence reference | |
| Dataset-level access/metadata agreement applicability, evidence, and review record | |

For every use declared in the ledger, record a separate decision and its
evidence: `local_inference`, `local_retention`, `aggregate_publication`, and
`redistribution` only when that use is actually proposed. In this protocol,
`aggregate_publication` means privacy-safe derived numerical summaries; it
does not permit source media, identifiable frames, item-level records, or raw
model outputs. Do not infer permission for one use from permission for another.
Treat missing, ambiguous, conditional-but-unfulfilled, or unverified terms as
unresolved. A `not_permitted` or unresolved decision blocks the corresponding
planned use. If the local media cannot be tied confidently to the source item,
keep the item unresolved.

### Dataset-level access or metadata agreement

Assess agreements that govern access to the benchmark, source media, metadata,
or download/service account separately from annotation and media licenses.
Keep this record private. Record the agreement name and issuer, version or
effective date, private evidence locator and retrieved-copy SHA-256, retrieval
date, affected dataset/service and access route, and any project/account
acceptance evidence and date. State whether the agreement applies to the frozen
cohort and why; use `applicable`, `not_applicable`, or `unresolved`. For an
applicable agreement, decide separately for each proposed use—local inference,
local retention/cache, privacy-safe numerical aggregate publication, and
redistribution—using `permitted`, `not_permitted`, or `unresolved`. For each
decision, cite the clause and record required conditions and evidence of their
fulfillment. Missing, ambiguous, conditional-but-unfulfilled, or unverified
terms remain unresolved. Record the reviewer's role, authority basis, review
time (UTC), and private review-record hash. A `not_applicable` decision also
needs evidence and a written basis; it is not the default when no agreement has
been located.

Maintain a versioned private evidence-index JSON containing the exact-byte
SHA-256 and private record reference for the item-rights ledger, annotation
terms, item-level media and conditions evidence, and dataset-level agreement
assessment (including evidence for non-applicability when relevant). Set the
protocol's `data_rights.evidence_sha256` to the SHA-256 of the exact index-file
bytes; keep `data_rights.item_rights_manifest_sha256` bound separately to the
v2 item-rights ledger. The structural validator checks only hash syntax and
does not read this index or authenticate any evidence.

For the optional CPU cross-checker `scripts/audit_data_rights_evidence_index.py`,
use a closed schema with root keys `schema_version=1`, `eligible_manifest`,
`item_rights_manifest`, `annotation_terms`, `item_media_conditions`, and
`dataset_access_agreement`. The first three evidence entries contain
`sha256` and `private_locator`; `eligible_manifest` also contains
`eligible_video_count`. `item_media_conditions` contains `sha256`,
`private_locator`, and `covered_video_count`. The agreement entry contains
`applicability` (`applicable`, `not_applicable`, or `unresolved`), evidence
`sha256`, `private_locator`, `review_record_sha256`, `reviewer_role`,
`reviewed_at_utc`, `planned_use_scope`, `decisions`, and
`non_applicability_basis_sha256`. When applicable, `decisions` has exactly one
`permitted`/`not_permitted`/`unresolved` value per ledger intended use and the
non-applicability basis is null. When not applicable, preserve a basis hash
equal to the entry's `sha256` and set `decisions` to null. For an applicable
agreement, `sha256` identifies the reviewed agreement evidence; for a
substantiated non-applicability decision, it identifies that basis evidence.
Unresolved applicability or any uncleared planned use cannot pass the
structural cross-check. Bind ledger and eligible-manifest
entries to their exact bytes, annotation terms to the ledger's declared
annotation-evidence hash, media/conditions to the restricted evidence bundle,
and the agreement entry to the reviewed agreement or non-applicability
evidence. The checker also cross-checks the exact intended-use scope and
eligible count/hash across protocol, ledger, and index. Run it in addition to
the protocol and item-rights validators; its pass means only that declarations
and byte references are structurally consistent. It cannot authenticate
evidence, decide rights, establish reviewer authority, or grant execution.

The v2 item-rights ledger records hashes and declarations, not the full legal
reasoning. Preserve the private evidence bundle and any applicable
dataset-level agreement in a restricted project record and link its hash from
the private review record. The validator cannot determine license scope,
reviewer authority, source identity, or whether conditions were actually met.
Escalate uncertain terms to the project owner or appropriate rights contact;
do not mark them permitted by default.

Transcribe each item into the ledger's exact fields: source dataset/video IDs;
media license ID, evidence locator/hash, reviewer role/status/time, and review
record hash; `conditions_record_sha256` and
`conditions_status=reviewed_resolved`; and one `permitted`, `not_permitted`, or
`unresolved` decision for every declared use. In the top-level
`annotation_rights` object, record annotation `dataset_id`, `license_id`,
evidence locator/hash, review-record hash, reviewer role/status/time, and
one decision for each declared use. For each source-lineage edge,
record `from_dataset_id`, `from_record_id`, `to_dataset_id`, `to_record_id`,
`mapping_evidence_locator`, and `mapping_evidence_sha256`; edges must connect
from the eligible video to the exact source-media item. Annotation rights use
their separate top-level record and use-by-use decisions. Keep the complete
terms, clause analysis, retrieval dates, condition fulfillment evidence, and
any dataset-access agreement in the linked private evidence bundle.
Do not declare the protocol's `data_rights.status=verified` unless the private
evidence index also records completed review of every applicable
dataset-level agreement for the planned-use scope, or substantiates that none
applies; leave the status unresolved if this review is incomplete.
