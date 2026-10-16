import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path

from build_nextgqa_claim_mapping_packet import apply_review_policy, build, main


H1 = "a" * 64
REV = "Qwen/Qwen2.5-VL-7B-Instruct@" + "b" * 40
COMMIT = "c" * 40
RUNNER = "d" * 64
def packet(records=3):
    rows = []
    for index in range(records):
        rows.append({
            "review_index": index,
            "local_video_path": f"/private/video-{index}.mp4",
            "video_sha256": f"{index + 1:064x}",
            "dataset_question_id": f"qid-{index}",
            "question": f"Synthetic question {index}?",
            "choices": ["secret choice"],
            "gold_answer": "secret answer",
            "gold_interval_seconds": [[0.0, 1.0]],
            "generated_raw_text": "RAW SECRET GENERATION",
            "generated_parsed": {
                "valid": False,
                "answer_letter": "A",
                "claim": f"Synthetic claim {index}.",
                "evidence_frame_indices": [],
            },
        })
    value = {
        "schema_version": 1,
        "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
        "status": "partial",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": REV,
        "source_commit": COMMIT,
        "runner_sha256": RUNNER,
        "pilot_runner_sha256": "e" * 64,
        "cohort_sha256": H1,
        "dataset_inputs_sha256": {
            "test.csv": "1" * 64,
            "gsub_test.json": "2" * 64,
            "map_vid_vidorID.json": "3" * 64,
            "frame2time_test.json": "4" * 64,
        },
        "requested_videos": records,
        "effective_videos": records,
        "records": rows,
    }
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()
    return value, raw


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def assert_exact_output_keys(test, value):
    top = {
        "schema_version", "purpose", "status", "created_at_utc",
        "source_packet_sha256", "cohort_sha256", "model_revision", "source_commit",
        "claim_packet_runner_sha256", "cohort_selector_runner_sha256",
        "dataset_inputs_sha256", "occurrence_policy", "review_procedure_status",
        "requested_records", "effective_records", "claim_records_available",
        "review_completed", "execution_authorization", "records",
    }
    policy_keys = {"claim_review_policy_id", "claim_review_policy_sha256"}
    record = {
        "video_path", "video_sha256", "video_id", "claim_id", "review_token", "question",
        "claim_text", "claim_text_sha256", "review",
    }
    review = {
        "source_identity_verified", "full_source_video_reviewed",
        "question_and_claim_only_context_shown", "gold_and_seed_hidden_until_mapping_frozen",
        "outcome_scores_not_accessed", "occurrence_policy_applied", "support_decision",
        "evidence_intervals_seconds", "rationale", "reviewer_role",
        "review_record_reference", "first_played_at_utc", "reviewed_at_utc", "review_status",
    }
    test.assertIn(set(value), (top, top | policy_keys))
    for row in value["records"]:
        test.assertEqual(set(row), record)
        test.assertEqual(set(row["review"]), review)


class BuildNextgqaClaimMappingPacketTests(unittest.TestCase):
    @staticmethod
    def policy():
        path = Path(__file__).resolve().parents[1] / "docs" / "nextgqa_claim_review_policy_v5.json"
        return path.read_bytes()

    def test_blinded_draft_contains_claims_and_blank_review_fields_only(self):
        source, raw = packet()
        self.assertNotIn("fully_structured_claim_records", source)
        built, encoded = build(source, raw)
        self.assertEqual(built["status"], "draft_not_reviewed")
        self.assertFalse(built["review_completed"])
        self.assertEqual(built["execution_authorization"], "not_granted")
        self.assertEqual(built["effective_records"], 3)
        self.assertEqual(built["claim_records_available"], 3)
        self.assertEqual(built["cohort_sha256"], H1)
        self.assertEqual(built["source_packet_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(built["model_revision"], REV)
        self.assertEqual(built["source_commit"], COMMIT)
        self.assertEqual(built["claim_packet_runner_sha256"], RUNNER)
        self.assertEqual(built["cohort_selector_runner_sha256"], "e" * 64)
        self.assertEqual(built["dataset_inputs_sha256"], source["dataset_inputs_sha256"])
        self.assertEqual(built["source_packet_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(built["occurrence_policy"], "unfrozen")
        self.assertEqual(built["review_procedure_status"], "unfrozen")
        self.assertFalse(built["review_completed"])
        assert_exact_output_keys(self, built)
        for row in built["records"]:
            self.assertEqual(row["review"], {
                "source_identity_verified": None,
                "full_source_video_reviewed": None,
                "question_and_claim_only_context_shown": None,
                "gold_and_seed_hidden_until_mapping_frozen": None,
                "outcome_scores_not_accessed": None,
                "occurrence_policy_applied": None,
                "support_decision": None,
                "evidence_intervals_seconds": None,
                "rationale": None,
                "reviewer_role": None,
                "review_record_reference": None,
                "first_played_at_utc": None,
                "reviewed_at_utc": None,
                "review_status": "not_started",
            })
        text = encoded.decode("utf-8")
        for forbidden in ("gold_answer", "choices", "gold_interval_seconds",
                          "evidence_frame_indices", "generated_raw_text", "RAW SECRET"):
            self.assertNotIn(forbidden, text)
        parsed = json.loads(encoded)
        self.assertEqual(parsed["records"][0]["claim_text"], "Synthetic claim 0.")
        self.assertEqual(parsed["records"][0]["video_sha256"], f"{1:064x}")
        self.assertNotIn("dataset_question_id", parsed["records"][0])

    def test_frozen_policy_is_hash_bound_but_does_not_grant_review_or_scoring(self):
        source, raw = packet()
        built, _ = build(source, raw)
        policy_raw = self.policy()
        policy = json.loads(policy_raw)
        result = apply_review_policy(built, policy, policy_raw)
        self.assertEqual(result["occurrence_policy"], "all_relevant_occurrences")
        self.assertEqual(result["review_procedure_status"], "frozen_pre_review")
        self.assertEqual(result["claim_review_policy_sha256"], hashlib.sha256(policy_raw).hexdigest())
        self.assertFalse(result["review_completed"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        assert_exact_output_keys(self, result)

    def test_claim_format_artifacts_are_rejected_before_review_packet_build(self):
        for claim, reason in (("A readable claim\ncontinued", "claim_multiline"),
                              ("A addCriterion fragment", "claim_template_or_token_artifact")):
            source, _ = packet()
            source["records"][1]["generated_parsed"]["claim"] = claim
            with self.subTest(reason=reason), self.assertRaisesRegex(
                    ValueError, "claim_packet_claim_text_not_reviewable_" + reason):
                build(source, encode(source))

    def test_policy_mutation_and_invalid_occurrence_fail_closed(self):
        source, raw = packet()
        built, _ = build(source, raw)
        policy_raw = self.policy()
        policy = json.loads(policy_raw)
        with self.assertRaisesRegex(ValueError, "review_policy_bytes_invalid"):
            apply_review_policy(built, policy, b"{}\n")
        invalid = dict(policy, occurrence_policy="one_sufficient_occurrence")
        with self.assertRaisesRegex(ValueError, "review_policy_values_invalid"):
            apply_review_policy(built, invalid, encode(invalid))

    def test_zero_to_two_record_packets_are_supported(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                value, raw = packet(count)
                built, _ = build(value, raw)
                self.assertEqual(built["effective_records"], count)

    def test_invalid_schema_missing_claim_or_bad_counts_fail_closed(self):
        value, raw = packet()
        malformed = dict(value, purpose="wrong")
        with self.assertRaisesRegex(ValueError, "claim_packet_schema_invalid"):
            build(malformed, encode(malformed))
        missing = json.loads(json.dumps(value))
        missing["records"][1]["generated_parsed"]["claim"] = None
        with self.assertRaisesRegex(ValueError, "claim_packet_required_claim_missing_or_invalid"):
            build(missing, encode(missing))
        for field, bad_value in (("requested_videos", 4), ("effective_videos", 2),
                                 ("requested_videos", True), ("effective_videos", True)):
            with self.subTest(field=field):
                count_mismatch = json.loads(json.dumps(value))
                count_mismatch[field] = bad_value
                with self.assertRaisesRegex(ValueError, "claim_packet_counts_invalid"):
                    build(count_mismatch, encode(count_mismatch))

    def test_duplicate_video_and_invalid_hash_fail_closed(self):
        value, raw = packet()
        duplicate = json.loads(json.dumps(value))
        duplicate["records"][1]["video_sha256"] = duplicate["records"][0]["video_sha256"]
        with self.assertRaisesRegex(ValueError, "claim_packet_video_duplicate"):
            build(duplicate, encode(duplicate))
        invalid_hash = json.loads(json.dumps(value))
        invalid_hash["cohort_sha256"] = "bad"
        with self.assertRaisesRegex(ValueError, "claim_packet_cohort_hash_invalid"):
            build(invalid_hash, encode(invalid_hash))
        invalid_dataset = json.loads(json.dumps(value))
        invalid_dataset["dataset_inputs_sha256"]["test.csv"] = "bad"
        with self.assertRaisesRegex(ValueError, "claim_packet_dataset_hashes_invalid"):
            build(invalid_dataset, encode(invalid_dataset))

    def test_cli_writes_mode_0600_without_echo_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            value, _ = packet(3)
            for index, row in enumerate(value["records"]):
                media = Path(directory) / f"private-video-{index}.bin"
                media_bytes = f"synthetic private media fixture {index}".encode()
                media.write_bytes(media_bytes)
                row["local_video_path"] = str(media)
                row["video_sha256"] = hashlib.sha256(media_bytes).hexdigest()
            raw = encode(value)
            source = Path(directory) / "source.json"
            policy_path = Path(directory) / "policy.json"
            output = Path(directory) / "worksheet.json"
            source.write_bytes(raw)
            policy_path.write_bytes(self.policy())
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["--input", str(source), "--policy", str(policy_path), "--output", str(output),
                               "--user-authorized-local-only"])
            self.assertEqual(result, 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertNotIn("Synthetic", stdout.getvalue())
            before = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--input", str(source), "--policy", str(policy_path), "--output", str(output),
                      "--user-authorized-local-only"])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)

    def test_cli_rejects_media_hash_mismatch_without_echoing_path(self):
        with tempfile.TemporaryDirectory() as directory:
            value, _ = packet(3)
            for index, row in enumerate(value["records"]):
                media = Path(directory) / f"private-video-{index}.bin"
                media_bytes = f"synthetic private media fixture {index}".encode()
                media.write_bytes(b"different synthetic media" if index == 0 else media_bytes)
                row["local_video_path"] = str(media)
                if index != 0:
                    row["video_sha256"] = hashlib.sha256(media_bytes).hexdigest()
            source = Path(directory) / "source.json"
            policy_path = Path(directory) / "policy.json"
            source.write_bytes(encode(value))
            policy_path.write_bytes(self.policy())
            output = Path(directory) / "worksheet.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                main(["--input", str(source), "--policy", str(policy_path), "--output", str(output),
                      "--user-authorized-local-only"])
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("source_video_hash_mismatch", stderr.getvalue())
            self.assertNotIn(str(media), stderr.getvalue())
            self.assertFalse(output.exists())

    def test_cli_requires_explicit_local_only_authorization(self):
        _, raw = packet(0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            policy_path = Path(directory) / "policy.json"
            output = Path(directory) / "worksheet.json"
            source.write_bytes(raw)
            policy_path.write_bytes(self.policy())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--input", str(source), "--policy", str(policy_path), "--output", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_cli_refuses_output_inside_repository(self):
        _, raw = packet(0)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            policy_path = Path(directory) / "policy.json"
            source.write_bytes(raw)
            policy_path.write_bytes(self.policy())
            repo_output = Path(__file__).resolve().parents[1] / "results" / "should_not_exist.json"
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                main(["--input", str(source), "--policy", str(policy_path), "--output", str(repo_output),
                      "--user-authorized-local-only"])
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(repo_output.exists())


if __name__ == "__main__":
    unittest.main()
