import contextlib, copy, hashlib, io, json, tempfile, unittest
from pathlib import Path
from validate_claim_mask_protocol import compute_primary_estimand_sha256, main, validate

H = "0" * 64
G = "0" * 40
def fixture():
    value = {
        "schema_version": 9,
        "protocol_stage": "exploratory",
        "model": {"revision": G, "tokenizer_revision": G, "quant": "4bit", "dtype": "bfloat16", "attention_implementation": "sdpa"},
        "cohort": {"candidate_manifest_sha256": H, "eligible_manifest_sha256": H, "exclusion_manifest_sha256": H, "overlap_audit_sha256": H, "candidate_video_count": 2, "eligible_video_count": 2, "prerun_exclusion_count": 0, "analysis_unit_count": 2, "split_overlap_count": 0, "primary_population": "all_eligible_videos", "secondary_population": "baseline_correctness_strata"},
        "claim_review": {"claim_manifest_sha256": H, "mapping_sha256": H, "review_record_sha256": H, "review_status": "completed", "review_timing": "before_scoring", "mapping_unit": "claim_to_evidence_frames"},
        "data_rights": {"scope": "future chosen dataset split", "status": "unresolved", "evidence_sha256": H, "item_rights_manifest_sha256": H, "covered_eligible_video_count": 0, "unresolved_eligible_video_count": 2},
        "sampling": {"policy": "whole_video_fixed", "config_sha256": H, "uses_claim_for_selection": False, "num_frames": 16, "frame_size": 1024, "max_pixels": 200704, "cap_pixels_per_frame": True},
        "estimand": {"answer_candidate_protocol_sha256": H, "gold_choice_manifest_sha256": H, "primary_estimand_sha256": H, "score_kind": "teacher_forced_gold_vs_best_foil_logprob_margin", "baseline_correction": "selected_minus_control_margin_drop", "triplet_manifest_sha256": H, "target_definition": "fixed_gold_choice_from_annotation", "margin_definition": "gold_label_logprob_minus_max_foil_label_logprob_same_prompt"},
        "intervention": {"kind": "claim_linked_evidence_frame_mask", "selector_sha256": H, "control_sha256": H, "mask_transform_sha256": H, "minimum_selected_frame_count": 1, "selected_condition": "claim-linked evidence frames masked", "control_condition": "matched temporal control frames masked", "selected_unit": "sampled_frames_overlapping_reviewed_claim_evidence_interval", "control_matching": "equal_count_minimum_total_temporal_rank_distance_lexicographic_tie_break"},
        "analysis": {"primary_contrast_count": 1, "primary_multiplicity": "none_single_primary_contrast", "primary_ci_method": "two_sided_student_t_df_n_minus_1", "secondary_families": ["baseline_correctness_strata", "question_type_strata"], "secondary_adjustment": "holm_within_each_predeclared_family", "secondary_hypothesis_manifest_sha256": H, "sesoi": {"status": "unresolved", "value_nat": None, "decision_rule": None, "estimand_sha256": None, "rationale_category": None, "rationale_sha256": None, "fixed_at_utc": None, "review_status": None, "review_timing": None, "independent_review_sha256": None, "independent_reviewed_at_utc": None, "outcome_access_status": None, "outcome_access_record_sha256": None}},
        "runtime": {"seed": 1, "source_commit": "0123456789abcdef0123456789abcdef01234567", "execution_authorization": False},
    }
    value["estimand"]["primary_estimand_sha256"] = compute_primary_estimand_sha256(value)
    return value

class TestClaimMaskProtocol(unittest.TestCase):
    def test_valid_is_structural_only(self):
        r=validate(fixture(),H); self.assertTrue(r["structurally_valid"]); self.assertEqual(r["execution_authorization"],"not_granted")

    def test_invalid_caller_supplied_input_digest_is_not_echoed(self):
        r=validate(fixture(), "not-a-hash")
        self.assertFalse(r["structurally_valid"])
        self.assertIn("input_sha256_invalid",r["failure_codes"])
        self.assertNotIn("input_sha256",r)
    def test_incomplete_review_fails_closed(self):
        x=fixture(); x["claim_review"]["review_status"]="incomplete"; r=validate(x); self.assertIn("claim_review_incomplete",r["failure_codes"])
    def test_annotation_sampling_fails(self):
        x=fixture(); x["sampling"]["uses_claim_for_selection"]=True; r=validate(x); self.assertIn("sampling_fields_invalid",r["failure_codes"])

    def test_boolean_schema_version_is_not_integer_one(self):
        x=fixture(); x["schema_version"]=True; r=validate(x)
        self.assertFalse(r["structurally_valid"]); self.assertIn("schema_version_invalid",r["failure_codes"])

    def test_unhashable_protocol_stage_returns_failure_instead_of_raising(self):
        for bad in ([], {}, None):
            with self.subTest(protocol_stage=bad):
                x=fixture(); x["protocol_stage"]=bad; r=validate(x)
                self.assertFalse(r["structurally_valid"])
                self.assertIn("protocol_stage_invalid",r["failure_codes"])

    def test_model_and_tokenizer_revisions_require_full_immutable_ids(self):
        for field, bad in (("revision", "main"), ("tokenizer_revision", "abc"),
                           ("revision", "g" * 40), ("tokenizer_revision", "0" * 39)):
            with self.subTest(field=field, value=bad):
                x=fixture(); x["model"][field]=bad; r=validate(x)
                self.assertIn("model_field_missing",r["failure_codes"])

    def test_runtime_requires_full_git_commit_hash(self):
        for bad in ("abc", "g" * 40, "0" * 39):
            with self.subTest(source_commit=bad):
                x=fixture(); x["runtime"]["source_commit"]=bad; r=validate(x)
                self.assertIn("runtime_fields_invalid",r["failure_codes"])

    def test_overlap_count_cannot_exceed_video_count(self):
        x=fixture(); x["cohort"]["split_overlap_count"]=3; r=validate(x)
        self.assertIn("split_overlap_exceeds_video_count",r["failure_codes"])

    def test_selected_and_control_conditions_must_be_distinct(self):
        x=fixture(); x["intervention"]["control_condition"]=x["intervention"]["selected_condition"]
        r=validate(x); self.assertIn("intervention_fields_invalid",r["failure_codes"])

    def test_primary_population_is_all_eligible_and_correctness_is_secondary(self):
        x=fixture(); x["cohort"]["primary_population"]="baseline_correct_only"
        r=validate(x); self.assertIn("cohort_fields_invalid",r["failure_codes"])

    def test_confirmatory_stage_is_blocked_until_statistics_are_frozen(self):
        x=fixture(); x["protocol_stage"]="confirmatory"
        self.assertIn("protocol_stage_invalid",validate(x)["failure_codes"])

    def test_schema_v8_is_rejected_after_v9_change(self):
        x=fixture(); x["schema_version"]=8
        self.assertIn("schema_version_invalid",validate(x)["failure_codes"])

    def test_mask_transform_hash_is_required_and_part_of_primary_estimand(self):
        x=fixture(); x["intervention"].pop("mask_transform_sha256")
        self.assertIn("intervention_schema_invalid",validate(x)["failure_codes"])
        x=fixture(); x["intervention"]["mask_transform_sha256"]="a" * 64
        x["estimand"]["primary_estimand_sha256"] = compute_primary_estimand_sha256(x)
        self.assertTrue(validate(x)["structurally_valid"])
        x["intervention"]["mask_transform_sha256"]="b" * 64
        self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])

    def test_empty_selected_mask_is_not_eligible(self):
        x=fixture(); x["intervention"]["minimum_selected_frame_count"]=0
        x["estimand"]["primary_estimand_sha256"] = compute_primary_estimand_sha256(x)
        self.assertIn("intervention_fields_invalid",validate(x)["failure_codes"])

    def test_primary_ci_method_is_frozen(self):
        x=fixture(); x["analysis"]["primary_ci_method"]="bootstrap"
        self.assertIn("analysis_fields_invalid",validate(x)["failure_codes"])

    def test_eligible_denominator_must_equal_video_level_analysis_units(self):
        x=fixture(); x["cohort"]["analysis_unit_count"]=1
        self.assertIn("analysis_unit_not_video_level",validate(x)["failure_codes"])

    def test_candidate_eligible_and_prerun_exclusion_counts_reconcile(self):
        x=fixture(); x["cohort"]["prerun_exclusion_count"]=1
        self.assertIn("cohort_fields_invalid",validate(x)["failure_codes"])

    def test_review_must_precede_scoring(self):
        x=fixture(); x["claim_review"]["review_timing"]="after_scoring"
        self.assertIn("claim_review_timing_invalid",validate(x)["failure_codes"])

    def test_rights_status_is_recorded_but_does_not_authorize_execution(self):
        r=validate(fixture())
        self.assertTrue(r["structurally_valid"])
        self.assertEqual(r["execution_authorization"],"not_granted")
        x=fixture(); x["data_rights"]["status"]=[]
        self.assertIn("data_rights_fields_invalid",validate(x)["failure_codes"])

    def test_rights_coverage_reconciles_to_eligible_videos(self):
        x=fixture(); x["data_rights"]["unresolved_eligible_video_count"]=1
        self.assertIn("data_rights_coverage_count_invalid",validate(x)["failure_codes"])
        x=fixture(); x["data_rights"].update(status="verified", covered_eligible_video_count=1, unresolved_eligible_video_count=1)
        self.assertIn("verified_rights_incomplete",validate(x)["failure_codes"])
        x=fixture(); x["data_rights"].update(status="verified", covered_eligible_video_count=2, unresolved_eligible_video_count=0)
        self.assertEqual(validate(x)["execution_authorization"],"not_granted")

    def test_review_mapping_must_end_in_evidence_frames(self):
        x=fixture(); x["claim_review"]["mapping_unit"]="claim_tokens"
        r=validate(x); self.assertIn("claim_review_mapping_unit_invalid",r["failure_codes"])

    def test_score_must_be_fixed_answer_margin(self):
        x=fixture(); x["estimand"]["score_kind"]="teacher_forced_logprob"
        r=validate(x); self.assertIn("estimand_fields_invalid",r["failure_codes"])

    def test_secondary_multiplicity_must_match_predeclared_plan(self):
        x=fixture(); x["analysis"]["secondary_adjustment"]="none"
        self.assertIn("analysis_fields_invalid",validate(x)["failure_codes"])

    def test_secondary_hypotheses_require_a_frozen_manifest_hash(self):
        x=fixture(); x["analysis"]["secondary_hypothesis_manifest_sha256"]="not-a-hash"
        self.assertIn("analysis_fields_invalid",validate(x)["failure_codes"])

    def test_primary_is_one_unadjusted_contrast_and_sesoi_status_is_explicit(self):
        x=fixture(); x["analysis"]["primary_contrast_count"]=2
        self.assertIn("analysis_fields_invalid",validate(x)["failure_codes"])
        x=fixture(); x["analysis"]["sesoi"]["status"]="picked_after_results"
        self.assertIn("sesoi_status_invalid",validate(x)["failure_codes"])

    def test_unresolved_sesoi_cannot_smuggle_in_a_threshold(self):
        x=fixture(); x["analysis"]["sesoi"]["value_nat"]=0.25
        self.assertIn("sesoi_unresolved_fields_not_null",validate(x)["failure_codes"])

    def test_fixed_sesoi_requires_estimand_bound_independent_record(self):
        bound_hash=compute_primary_estimand_sha256(fixture())
        x=fixture(); x["analysis"]["sesoi"].update(
            status="preoutcome_fixed", value_nat=0.2,
            decision_rule="primary_95pct_ci_lower_gt_sesoi", estimand_sha256=bound_hash,
            rationale_category="measurement_interpretability", rationale_sha256=H,
            fixed_at_utc="2026-10-14T13:00:00Z", review_status="completed",
            review_timing="before_scoring", independent_review_sha256=H,
            independent_reviewed_at_utc="2026-10-14T12:00:00Z",
            outcome_access_status="not_accessed_at_freeze", outcome_access_record_sha256=H)
        self.assertTrue(validate(x)["structurally_valid"])
        x["analysis"]["sesoi"]["estimand_sha256"]="1"*64
        self.assertIn("sesoi_fixed_fields_invalid",validate(x)["failure_codes"])

    def test_primary_estimand_hash_recomputes_from_definition_and_provenance(self):
        x=fixture(); x["estimand"]["answer_candidate_protocol_sha256"]="1"*64
        self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])
        x=fixture(); x["intervention"]["control_sha256"]="1"*64
        self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])
        x=fixture(); x["estimand"]["triplet_manifest_sha256"]="1"*64
        self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])
        x=fixture(); x["cohort"]["eligible_manifest_sha256"]="1"*64
        self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])
        for field in ("claim_manifest_sha256", "mapping_sha256", "review_record_sha256"):
            with self.subTest(claim_review_hash=field):
                x=fixture(); x["claim_review"][field]="1"*64
                self.assertIn("primary_estimand_hash_mismatch",validate(x)["failure_codes"])

    def test_primary_estimand_hash_rejects_non_utf8_surrogate_text(self):
        x=fixture(); x["model"]["quant"]="\ud800"
        result=validate(x)
        self.assertFalse(result["structurally_valid"])
        self.assertIn("primary_estimand_hash_mismatch",result["failure_codes"])

    def test_fixed_sesoi_rejects_nonfinite_boolean_nonpositive_and_bad_time(self):
        base={"status":"preoutcome_fixed", "value_nat":0.2,
              "decision_rule":"primary_95pct_ci_lower_gt_sesoi", "estimand_sha256":H,
              "rationale_category":"measurement_interpretability", "rationale_sha256":H,
              "fixed_at_utc":"2026-10-14T13:00:00Z", "review_status":"completed",
              "review_timing":"before_scoring", "independent_review_sha256":H,
              "independent_reviewed_at_utc":"2026-10-14T12:00:00Z",
              "outcome_access_status":"not_accessed_at_freeze", "outcome_access_record_sha256":H}
        for key,bad in (("value_nat",True),("value_nat",float("nan")),("value_nat",10**1000),("value_nat",0.0),
                        ("fixed_at_utc","2026-13-45T13:00:00Z"),
                        ("independent_reviewed_at_utc","2026-13-45T12:00:00Z"),
                        ("independent_reviewed_at_utc","2026-10-14T14:00:00Z"),
                        ("outcome_access_status","accessed_before_freeze"),
                        ("outcome_access_record_sha256","bad-hash"),
                        ("review_timing","after_scoring"),
                        ("rationale_category",[])):
            with self.subTest(key=key,bad=bad):
                x=fixture(); x["analysis"]["sesoi"]=dict(base); x["analysis"]["sesoi"][key]=bad
                self.assertIn("sesoi_fixed_fields_invalid",validate(x)["failure_codes"])

    def test_intervention_masks_frames_not_claim_tokens(self):
        x=fixture(); x["intervention"]["selected_unit"]="claim_tokens"
        r=validate(x); self.assertIn("intervention_fields_invalid",r["failure_codes"])

    def test_runtime_cannot_grant_execution_authorization(self):
        x=fixture(); x["runtime"]["execution_authorization"]=True; r=validate(x)
        self.assertIn("runtime_fields_invalid",r["failure_codes"])

    def test_duplicate_json_key_is_rejected_without_echoing_input(self):
        raw=json.dumps(fixture()).replace('"schema_version": 9', '"schema_version": 9, "schema_version": 9', 1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"duplicate.json"; path.write_text(raw, encoding="utf-8")
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
        self.assertEqual(code,2)
        result=json.loads(stdout.getvalue())
        self.assertFalse(result["structurally_valid"])
        self.assertIn("input_unreadable_or_malformed",result["failure_codes"])
        self.assertNotIn("claim tokens masked",stdout.getvalue())

    def test_cli_rejects_nonfinite_json_and_valid_input_never_authorizes_execution(self):
        valid=json.dumps(fixture())
        nonfinite=valid.replace('"seed": 1', '"seed": NaN')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"protocol.json"; stdout=io.StringIO()
            path.write_text(nonfinite, encoding="utf-8")
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
            self.assertEqual(code,2)
            failed=json.loads(stdout.getvalue())
            self.assertIn("input_unreadable_or_malformed",failed["failure_codes"])
            stdout=io.StringIO(); path.write_text(valid, encoding="utf-8")
            with contextlib.redirect_stdout(stdout): code=main([str(path)])
        self.assertEqual(code,0)
        accepted=json.loads(stdout.getvalue())
        self.assertTrue(accepted["structurally_valid"])
        self.assertEqual(accepted["execution_authorization"],"not_granted")
