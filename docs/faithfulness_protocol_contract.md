# Video faithfulness protocol contract (schema v1)

`scripts/validate_faithfulness_protocol.py` checks whether a proposed protocol
records a small set of required design choices before any future runner uses it.
It validates JSON shape, controlled terms, counts, and hash-string formats only.
It does not read the files named by hashes, authenticate a person, verify a video
split, establish claim–interval meaning, prove that a control is adequate, or
validate model/cache behavior. Every result says
`semantic_validation=not_machine_verifiable` and
`execution_authorization=not_granted`, including structurally valid results.

## Illustrative synthetic input

The values below are invented placeholders to show the schema. They are not
real source hashes, a cohort approval, a human attestation, or an authorization
to run an experiment. Do not copy them into an actual protocol.

```json
{
  "schema_version": 1,
  "protocol_stage": "confirmatory",
  "model_revision": "example-pinned-model-revision",
  "tokenizer_revision": "example-pinned-tokenizer-revision",
  "cohort": {
    "primary_unit": "video_id",
    "video_count": 2,
    "record_count": 2,
    "analysis_unit_count": 2,
    "records_aggregated_within_video": false,
    "split_manifest_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
    "overlap_audit_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
    "split_overlap_count": 0
  },
  "annotation": {
    "locator_kind": "interval",
    "source_manifest_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
    "time_unit": "seconds",
    "time_origin": "video_start",
    "evaluation_metric": "interval_iou",
    "semantic_review_record_sha256": "3333333333333333333333333333333333333333333333333333333333333333",
    "point_conversion": null
  },
  "sampling": {
    "policy": "whole_video_fixed",
    "config_sha256": "4444444444444444444444444444444444444444444444444444444444444444",
    "uses_annotation_for_selection": false
  },
  "estimand": {
    "primary_estimand": "claim_selective_dependence",
    "primary_score_target": "claim",
    "score_kind": "teacher_forced_logprob",
    "target_text_sha256": "5555555555555555555555555555555555555555555555555555555555555555",
    "gold_role": "evaluation_only",
    "claim_intervention_mapping_sha256": "6666666666666666666666666666666666666666666666666666666666666666"
  },
  "intervention": {
    "kind": "internal_activation",
    "selector_sha256": "7777777777777777777777777777777777777777777777777777777777777777",
    "selector_gold_access": false,
    "control_protocol_sha256": "8888888888888888888888888888888888888888888888888888888888888888",
    "control_match_dimensions": ["layer", "token_count", "magnitude"]
  },
  "cache": {
    "policy": "no_cache_full_sequence",
    "use_cache": false,
    "decode_position_policy": null,
    "runtime_evidence_sha256": null,
    "runtime_source_version": null
  }
}
```

## Structural rules

- The primary analysis unit is `video_id`. Counts must declare one analysis
  unit per video; repeated records require explicit within-video aggregation.
  Confirmatory protocols must declare zero split overlap. The validator checks
  those declarations and hash syntax, not the private IDs or the underlying audit.
- An annotation is explicitly a `point` or `interval`. Point error can be
  scored directly. An interval metric cannot silently consume a point: a
  point-to-interval conversion must reference a fixed-rule hash, derived
  interval-artifact hash, and review-record hash. These references are not proof
  that the rule or review is sound. Interval metrics also require a semantic
  review-record reference.
- Confirmatory sampling cannot declare use of annotation bounds for frame or
  chunk selection. Annotation-conditioned sampling must be labeled as such;
  exploratory use does not become an unbiased comparison by passing validation.
- Estimands distinguish claim dependence, answer dependence, grounding accuracy,
  and QA accuracy. An answer log-probability cannot be declared claim-selective
  dependence. Claim dependence requires a target-text hash and a separately
  hash-referenced claim-to-intervention mapping. The hashes are pointers only;
  the validator cannot inspect or approve that mapping.
- The current protocol contract permits `internal_activation` only. Its claim
  control must declare matching dimensions for layer, token count, and magnitude.
  The selector must declare no gold access. These declarations do not prove the
  implementation followed them or that the control is scientifically sufficient.
- `no_cache_full_sequence` must explicitly disable cache. A cached prefill/decode
  route must provide a decode-position policy, runtime-evidence hash, and source
  version. This records which technical check is cited; it does not validate its
  conclusion.
- Unknown keys and malformed JSON fail closed. Duplicate JSON object keys and
  non-finite values (`NaN`/`Infinity`) are rejected. Output contains only a
  structural-validity flag, generic failure codes, fixed scope statements, and
  the input-file SHA-256 for readable, parseable JSON. Unreadable or malformed
  JSON emits generic failure fields without an input hash. It omits the input
  path and all protocol values.

## Use and gates

Run against a private, reviewable protocol file:

```sh
.venv/bin/python scripts/validate_faithfulness_protocol.py /private/path/protocol.json
```

Exit code 0 means only that the declared schema is structurally complete. Exit
code 2 means invalid/unreadable input. Neither result authorizes inference,
human-data use, or a faithfulness claim. Follow applicable rights, annotation,
human-review, runtime, and study-design gates separately.

The NOAA pilot remains `no_go` for QA inference until its actual human
attestation exists. This protocol validator neither reads nor changes
`scripts/audit_noaa_qa_gate.py`; passing an illustrative or real protocol file
does not satisfy that gate, establish NOAA annotation validity, or upgrade the
three-video/six-QA feasibility cohort to confirmatory evidence.

## Standalone first-pass structure check

`scripts/noaa_first_pass_contract.py` is a separate pure CPU helper; it is not
integrated into the existing NOAA attestation gate. It checks an unsigned
schema-1 first-pass export against caller-verified canonical group/source
hashes, nonempty QID sets with exactly five choices, and expected silent-review
media metadata. Every expected QID must occur once with a strict integer index
0–4 and matching A–E label. It returns aggregate counts and generic failure
codes, never the reviewer choices, QIDs or media metadata.

`structurally_valid` and `review_media_matches_expected_metadata` indicate
consistency with the supplied expectations only. The helper does not
independently authenticate those expectations, compare choices to gold,
establish reviewer identity or chronology, or grant execution authorization.
The current private packet and v1 gate are unchanged; mandatory integration
would require a separately versioned contract. User-reported human review
remains incomplete.

CPU reproduction (synthetic records; no private review or inference):

```sh
PYTHONPATH=scripts .venv/bin/python -m unittest scripts/test_noaa_first_pass_contract.py -v
```
