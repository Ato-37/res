import hashlib
import json
import unittest

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from claim_review_assignment_plan import plan_claim_review_assignments


def sha(value):
    return hashlib.sha256(value).hexdigest()


def rebind_calibration(row):
    payload = {key: value for key, value in row.items()
               if key not in {"artifact_raw", "artifact_sha256"}}
    raw = json.dumps(payload, sort_keys=True).encode()
    row["artifact_sha256"] = sha(raw)
    row["artifact_raw"] = raw


def rebind_review(args):
    payload = json.loads(args["review_payload_raw"])
    payload["records"] = args["review_records"]
    args["review_payload_raw"] = json.dumps(payload, sort_keys=True).encode()
    args["review_payload_sha256"] = sha(args["review_payload_raw"])
    args["review_index"]["review_payload_sha256"] = args["review_payload_sha256"]
    args["review_index_raw"] = json.dumps(args["review_index"], sort_keys=True).encode()
    args["review_audit_report"]["review_payload_sha256"] = args["review_payload_sha256"]
    args["review_audit_report"]["review_index_sha256"] = sha(args["review_index_raw"])


def rebind_worksheet(args):
    prior = json.loads(args["worksheet_raw"])
    prior["records"] = args["worksheet_records"]
    args["worksheet_raw"] = json.dumps(prior, sort_keys=True).encode()
    args["review_audit_report"]["worksheet_sha256"] = sha(args["worksheet_raw"])


def fixture():
    videos = ["private-video-a", "private-video-b", "private-video-c"]
    worksheet, claim_rows, reviews, calibrations, timings = [], [], [], [], []
    for i, video_id in enumerate(videos):
        media_sha = sha(f"media-{i}".encode())
        token = f"reviewtoken{i:02d}" + "x" * 11
        claim_text = f"private claim {i}"
        claim_text_sha = sha(claim_text.encode())
        qid = f"private-qid-{i}"
        claim_id = "claim-" + sha((media_sha + "\0" + qid + "\0" + claim_text_sha).encode())
        worksheet.append({"video_id": video_id, "review_token": token,
                          "video_sha256": media_sha, "claim_text": claim_text,
                          "question": f"private question {i}",
                          "claim_id": claim_id, "claim_text_sha256": claim_text_sha})
        claim_rows.append({"video_id": video_id, "claim_id": claim_id,
                           "claim_text": claim_text, "claim_text_sha256": claim_text_sha,
                           "model_revision": "qwen@" + "a" * 40,
                           "generation_run_sha256": sha(f"run-{i}".encode())})
        reviews.append({"review_token": token, "review": {
            "full_source_video_reviewed": True,
            "question_and_claim_only_context_shown": True,
            "gold_and_seed_hidden_until_mapping_frozen": True,
            "outcome_scores_not_accessed": True, "media_bytes_verified": True,
            "occurrence_policy_applied": "all_relevant_occurrences",
            "support_decision": "supports_claim",
            "evidence_intervals_seconds": [[0.9, 1.1]],
            "interval_rationales": ["synthetic visible evidence"],
            "media_duration_seconds": 3.0,
            "reviewer_role": "independent_human",
            "display_player": "Synthetic HTML5 player",
            "first_played_at_utc": "2026-10-16T05:59:00Z",
            "reviewed_at_utc": "2026-10-16T06:00:00Z",
            "review_status": "completed",
            "claim_review_policy_sha256": FROZEN_POLICY_SHA256,
        }})
        calibration = {
            "video_id": video_id,
            "clock_scope": "html5_video_requestVideoFrameCallback_mediaTime_seconds",
            "empirical_residual_within_tolerance": True,
            # Synthetic fixture only: the planner checks required declarations
            # but cannot authenticate these evidence hashes or validation.
            "clock_validation_attestation": "independent_review_claimed_unverified",
            "capture_evidence_sha256": sha(f"capture-evidence-{i}".encode()),
            "frame_correspondence_evidence_sha256": sha(f"frame-match-evidence-{i}".encode()),
            "validation_scope": "empirical_heldout_residual_only",
            "guard_band_source": "max_observed_heldout_residual_pts_seconds",
            "uncertainty_status": "not_evaluated",
            "fit_anchor_count": 3, "heldout_anchor_count": 2,
            "source_media_sha256": media_sha,
            "slope": "1", "intercept": "0", "empirical_guard_band_pts_seconds": "1/100",
            "frozen_tolerance_seconds": "1/50",
        }
        rebind_calibration(calibration)
        calibrations.append(calibration)
        frame_sha = [sha(f"{video_id}-{j}".encode()) for j in range(4)]
        timings.append({"video_id": video_id, "source_media_sha256": media_sha,
                        "pts": list(range(7)), "time_base": [1, 2], "num_frames": 7,
                        "sampled_pixel_sha256": frame_sha,
                        "processor_pixel_sha256": frame_sha.copy()})
    review_payload = {"schema_version": 3, "purpose": "anonymous_local_claim_video_human_review",
                      "status": "human_review_recorded_unvalidated", "review_completed": True,
                      "execution_authorization": "not_granted",
                      "claim_review_policy_id": "nextgqa_claim_review_v5",
                      "claim_review_policy_sha256": FROZEN_POLICY_SHA256,
                      "human_reviewer_role": "independent_human",
                      "human_reviewer_private_id": "private reviewer",
                      "review_environment": {"browser_user_agent": "Synthetic",
                                             "offline_static_page": True,
                                             "network_fetch_available": False},
                      "independence_attestation": "reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk",
                      "records": reviews}
    review_raw = json.dumps(review_payload, sort_keys=True).encode()
    review_sha = sha(review_raw)
    index = {"schema_version": 1,
             "purpose": "anonymous_unvalidated_human_review_handoff",
             "review_payload_sha256": review_sha,
             "claim_review_policy_sha256": FROZEN_POLICY_SHA256,
             "review_status": "unvalidated_requires_custodian_audit",
             "created_at_utc": "2026-10-16T06:00:00Z",
             "execution_authorization": "not_granted"}
    source_packet_sha = sha(b"packet")
    cohort_sha = sha(b"cohort")
    dataset_inputs = {name: sha(name.encode()) for name in
                      ("test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json")}
    worksheet_raw = json.dumps({
        "schema_version": 1, "occurrence_policy": "all_relevant_occurrences",
        "claim_review_policy_id": "nextgqa_claim_review_v5",
        "claim_review_policy_sha256": FROZEN_POLICY_SHA256,
        "review_procedure_status": "frozen_pre_review",
        "source_packet_sha256": source_packet_sha, "cohort_sha256": cohort_sha,
        "dataset_inputs_sha256": dataset_inputs, "records": worksheet,
    }, sort_keys=True).encode()
    claim_manifest_raw = json.dumps({
        "schema_version": 1, "eligible_dataset_id": "synthetic-nextgqa",
        "eligible_manifest_sha256": sha(b"eligible"),
        "claim_policy_sha256": sha(b"claim-policy"),
        "occurrence_policy": "all_relevant_occurrences", "claims": claim_rows,
    }, sort_keys=True).encode()
    index_raw = json.dumps(index, sort_keys=True).encode()
    review_audit_report = {"status": "unvalidated_handoff_structurally_valid",
                           "failure_codes": [], "reviewed_records": 3,
                           "execution_authorization": "not_granted",
                           "semantic_review_authenticated": False,
                           "worksheet_sha256": sha(worksheet_raw),
                           "review_payload_sha256": review_sha,
                           "review_index_sha256": sha(index_raw),
                           "lineage_manifest_sha256": sha(b"lineage")}
    kwargs = {"expected_video_ids": videos, "worksheet_records": worksheet,
              "worksheet_raw": worksheet_raw,
              "claim_input_manifest_raw": claim_manifest_raw,
              "claim_input_manifest_sha256": sha(claim_manifest_raw),
              "review_records": reviews, "review_payload_raw": review_raw,
              "review_payload_sha256": review_sha,
              "review_audit_report": review_audit_report,
              "review_index": index, "review_index_raw": index_raw,
              "custody_failure_codes": [], "calibration_records": calibrations,
              "timing_records": timings, "expected_num_frames": 4}
    return kwargs


class ClaimReviewAssignmentPlanTests(unittest.TestCase):
    def test_complete_synthetic_cohort_assigns_equal_count_disjoint_controls(self):
        result = plan_claim_review_assignments(**fixture())
        self.assertEqual(result["status"], "assignment_plan_structurally_passed_unvalidated")
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["validated_videos"], 3)
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")
        for assignment in result["assignments"]:
            self.assertEqual(len(assignment["evidence_indices"]), 1)
            self.assertEqual(len(assignment["control_indices"]), 1)
            self.assertTrue(set(assignment["evidence_indices"]).isdisjoint(
                assignment["control_indices"]))
        encoded = json.dumps(result)
        self.assertNotIn("private-video", encoded)
        self.assertNotIn("private claim", encoded)
        self.assertNotIn("opaque-review-token", encoded)

    def test_fabricated_evidence_hashes_only_pass_structural_unvalidated_plan(self):
        args = fixture()
        for i, calibration in enumerate(args["calibration_records"]):
            calibration["clock_validation_attestation"] = "independent_review_claimed_unverified"
            calibration["capture_evidence_sha256"] = sha(f"fabricated-capture-{i}".encode())
            calibration["frame_correspondence_evidence_sha256"] = sha(
                f"fabricated-frame-match-{i}".encode())
            rebind_calibration(calibration)

        result = plan_claim_review_assignments(**args)

        # Structural acceptance is intentionally not evidence authentication.
        self.assertEqual(result["status"], "assignment_plan_structurally_passed_unvalidated")
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["failure_codes"], [])

    def test_review_payload_hash_is_verified(self):
        args = fixture(); args["review_payload_raw"] += b" "
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["audit_report_byte_binding_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_first_play_must_precede_item_completion_and_export(self):
        args = fixture()
        args["review_records"][0]["review"]["first_played_at_utc"] = "2026-10-16T06:01:00Z"
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["review_declaration_gate_failed"])
        self.assertEqual(result["score_calls"], 0)

    def test_structural_review_audit_does_not_claim_semantic_authentication(self):
        args = fixture(); args["review_audit_report"]["semantic_review_authenticated"] = True
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["review_audit_not_passed"])
        self.assertEqual(result["score_calls"], 0)

    def test_review_rows_cannot_be_swapped_under_a_valid_payload_hash(self):
        args = fixture()
        args["review_records"][0]["review"]["support_decision"] = "partial"
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["review_payload_rows_mismatch"])
        self.assertEqual(result["score_calls"], 0)

    def test_json_boolean_cannot_be_replaced_by_integer_one(self):
        args = fixture()
        args["review_records"][0]["review"]["full_source_video_reviewed"] = 1
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["review_payload_rows_mismatch"])
        self.assertEqual(result["score_calls"], 0)

    def test_late_unsupported_claim_blocks_full_denominator(self):
        args = fixture()
        args["review_records"][2]["review"]["support_decision"] = "partial"
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_codes"], ["claim_support_gate_failed"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["validated_videos"], 0)
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)

    def test_empirical_clock_guard_near_interval_boundary_blocks_all(self):
        args = fixture()
        # One frozen sample lies within 0.01 PTS seconds of the 1.0 endpoint.
        args["review_records"][0]["review"]["evidence_intervals_seconds"] = [[0.99, 1.1]]
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertIn("sample_classification_uncertain_at_boundary", result["failure_codes"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["score_calls"], 0)

    def test_zero_sampled_evidence_overlap_blocks_complete_cohort(self):
        args = fixture()
        # The reviewed interval is valid and away from all sampled PTS values,
        # so the frozen intervention would otherwise be a no-op mask.
        args["review_records"][1]["review"]["evidence_intervals_seconds"] = [[0.4, 0.6]]
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_codes"], ["zero_sampled_evidence_overlap"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["validated_videos"], 0)
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)
        self.assertEqual(result["execution_authorization"], "not_granted")

    def test_insufficient_equal_count_controls_block_complete_cohort(self):
        args = fixture()
        args["expected_num_frames"] = 7
        args["review_records"][2]["review"]["evidence_intervals_seconds"] = [[0.1, 2.9]]
        for timing in args["timing_records"]:
            hashes = [sha(f"sampled-pixel-{i}".encode()) for i in range(7)]
            timing["sampled_pixel_sha256"] = hashes
            timing["processor_pixel_sha256"] = hashes.copy()
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["failure_codes"], ["evidence_control_assignment_failed"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["validated_videos"], 0)
        self.assertEqual(result["assignments"], [])
        self.assertEqual(result["score_calls"], 0)

    def test_affine_clock_drift_maps_player_intervals_to_pts(self):
        args = fixture()
        for row in args["review_records"]:
            row["review"]["evidence_intervals_seconds"] = [[2.3, 2.7]]
        for row in args["calibration_records"]:
            row["slope"] = "2"
            row["intercept"] = "1/2"
            rebind_calibration(row)
        rebind_review(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["status"], "assignment_plan_structurally_passed_unvalidated")
        self.assertEqual(result["assignments"][0]["evidence_indices"], [2])

    def test_calibration_must_match_media_and_pass_heldout(self):
        for field, value, code in (
            ("source_media_sha256", "0" * 64, "clock_calibration_gate_failed"),
            ("empirical_residual_within_tolerance", False, "clock_calibration_gate_failed"),
            ("empirical_guard_band_pts_seconds", "1/10", "clock_calibration_values_invalid"),
        ):
            with self.subTest(field=field):
                args = fixture(); args["calibration_records"][1][field] = value
                rebind_calibration(args["calibration_records"][1])
                result = plan_claim_review_assignments(**args)
                self.assertEqual(result["failure_codes"], [code])
                self.assertEqual(result["score_calls"], 0)

    def test_empirical_calibration_alone_cannot_pass_clock_gate(self):
        args = fixture()
        for field in ("clock_validation_attestation", "capture_evidence_sha256",
                      "frame_correspondence_evidence_sha256"):
            args = fixture()
            args["calibration_records"][0].pop(field)
            rebind_calibration(args["calibration_records"][0])
            result = plan_claim_review_assignments(**args)
            self.assertEqual(result["failure_codes"], ["clock_calibration_gate_failed"])
            self.assertEqual(result["score_calls"], 0)

    def test_calibration_values_are_bound_to_exact_artifact_bytes(self):
        args = fixture()
        args["calibration_records"][0]["slope"] = "2"
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["calibration_artifact_content_mismatch"])
        self.assertEqual(result["score_calls"], 0)

    def test_review_worksheet_claim_must_match_frozen_input_manifest(self):
        args = fixture()
        args["worksheet_records"][0]["claim_id"] = "different-claim"
        rebind_worksheet(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["worksheet_claim_binding_mismatch"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["score_calls"], 0)

    def test_displayed_claim_text_cannot_be_rebound_without_matching_manifest(self):
        args = fixture()
        args["worksheet_records"][0]["claim_text"] = "tampered displayed claim"
        rebind_worksheet(args)
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["worksheet_claim_binding_mismatch"])
        self.assertEqual(result["score_calls"], 0)

    def test_occurrence_policy_must_match_frozen_review_policy(self):
        args = fixture()
        manifest = json.loads(args["claim_input_manifest_raw"])
        manifest["occurrence_policy"] = "one_sufficient_occurrence"
        args["claim_input_manifest_raw"] = json.dumps(manifest, sort_keys=True).encode()
        args["claim_input_manifest_sha256"] = sha(args["claim_input_manifest_raw"])
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["claim_input_manifest_schema_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_lone_surrogate_claim_text_fails_closed(self):
        args = fixture()
        manifest = json.loads(args["claim_input_manifest_raw"])
        manifest["claims"][0]["claim_text"] = chr(0xD800)
        args["claim_input_manifest_raw"] = json.dumps(manifest, sort_keys=True).encode()
        args["claim_input_manifest_sha256"] = sha(args["claim_input_manifest_raw"])
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["claim_input_row_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_worksheet_occurrence_policy_must_match_review_v5(self):
        args = fixture()
        worksheet = json.loads(args["worksheet_raw"])
        worksheet["occurrence_policy"] = "one_sufficient_occurrence"
        args["worksheet_raw"] = json.dumps(worksheet, sort_keys=True).encode()
        args["review_audit_report"]["worksheet_sha256"] = sha(args["worksheet_raw"])
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["worksheet_policy_binding_invalid"])
        self.assertEqual(result["score_calls"], 0)

    def test_empty_claim_id_and_invalid_model_revision_fail_closed(self):
        for index, field, value in ((0, "claim_id", ""), (1, "model_revision", "")):
            with self.subTest(field=field):
                args = fixture()
                manifest = json.loads(args["claim_input_manifest_raw"])
                manifest["claims"][index][field] = value
                if field == "claim_id":
                    args["worksheet_records"][index]["claim_id"] = value
                    rebind_worksheet(args)
                args["claim_input_manifest_raw"] = json.dumps(manifest, sort_keys=True).encode()
                args["claim_input_manifest_sha256"] = sha(args["claim_input_manifest_raw"])
                result = plan_claim_review_assignments(**args)
                self.assertEqual(result["failure_codes"], ["claim_input_row_invalid"])
                self.assertEqual(result["score_calls"], 0)

    def test_processor_pixel_mismatch_blocks_cohort(self):
        args = fixture(); args["timing_records"][1]["processor_pixel_sha256"][0] = "0" * 64
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["sampled_pixel_identity_invalid"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["score_calls"], 0)

    def test_custody_failure_blocks_without_reducing_denominator(self):
        args = fixture(); args["custody_failure_codes"] = ["custody_media_hash_mismatch"]
        result = plan_claim_review_assignments(**args)
        self.assertEqual(result["failure_codes"], ["custody_audit_failed"])
        self.assertEqual(result["denominator"], 3)
        self.assertEqual(result["score_calls"], 0)


if __name__ == "__main__":
    unittest.main()
