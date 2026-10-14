# Claim-mask human review worksheet (private working copy)

This worksheet helps a reviewer prepare the records required by the claim-mask
contract. It is not a rights opinion, annotation, eligibility decision, or
execution authorization. Keep completed copies, identities, item identifiers,
source URLs, video files, evidence documents, and notes in the restricted local
research area. Do not commit completed records. The public repository should
contain only this blank worksheet and privacy-safe aggregate summaries.

Before review begins, freeze and record the protocol revision, eligible-manifest
hash, **pre-review frozen model-claim input manifest hash**, model/tokenizer
revision, and the occurrence policy (`one_sufficient_occurrence` or
`all_relevant_occurrences`). This input-manifest hash is not the completed
claim-mapping ledger hash: the latter can be computed only after intervals and
review records are frozen. Record both artifacts under distinct names. The
occurrence policy must come from the protocol; a reviewer must not choose a
policy per video. Record the approved review display and evidence modalities
and use them consistently. If the protocol has not fixed these inputs, stop
before opening items and return the decision to the protocol owner.

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
generation-run hash, and the protocol's `occurrence_policy`. In `review`, set
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
| Dataset-level access/metadata agreement applicable? separate record and use-scope decision | |

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
