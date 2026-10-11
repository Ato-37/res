"""Tests for the publish-safe NOAA pre-inference gate."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_noaa_qa_gate import GROUPS, audit_dataset, _write_run_artifact


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NoaaQaGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.data_root = self.repo / "datasets" / "noaa_qa_pilot_20261011"
        self.expected_media = {}
        self.expected_rights = {}
        self.group_qids = {}
        for index, group in enumerate(GROUPS):
            group_root = self.data_root / group["directory"]
            private = group_root / "private"
            media = group_root / "media" / "source.mp4"
            private.mkdir(parents=True, exist_ok=True)
            media.parent.mkdir(parents=True, exist_ok=True)
            media.write_bytes(f"fixture-video-{index}".encode())
            media_hash = digest(media)
            qid = f"Q{index + 1:03d}"
            video_id = f"NOAA_TEST_DIVE{index:02d}"
            relative_media = Path("datasets/noaa_qa_pilot_20261011") / group["directory"] / "media" / "source.mp4"
            samples = [{
                "qid": qid,
                "video_id": video_id,
                "video": relative_media.as_posix(),
                "video_sha256": media_hash,
                "question": "Fixture prompt",
                "choices": [
                    {"id": label, "text": f"Choice {label}"}
                    for label in "ABCDE"
                ],
                "window_start_seconds": 1.0,
                "window_end_seconds": 2.0,
            }]
            gold = [{"qid": qid, "video_id": video_id, "answer_choice_index": 0}]
            audit = {
                "method": "independent AI sub-agent blind reviewer",
                "items": [{"qid": qid, "review_status": "accepted_after_blind_visual_review"}],
            }
            rights = {
                "video_id": video_id,
                "media_sha256": media_hash,
                "human_verified_manifest_status": "pending; attestation absent",
                "audio_used": False,
            }
            (private / "samples.json").write_text(json.dumps(samples), encoding="utf-8")
            (private / "gold.json").write_text(json.dumps(gold), encoding="utf-8")
            (private / group["audit_file"]).write_text(json.dumps(audit), encoding="utf-8")
            (private / "rights_review_v2.json").write_text(json.dumps(rights), encoding="utf-8")
            self.expected_media[group["group_id"]] = media_hash
            self.expected_rights[group["group_id"]] = digest(private / "rights_review_v2.json")
            self.group_qids[group["group_id"]] = [qid]

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, *, attestation=None, supplied=False):
        return audit_dataset(
            self.data_root,
            attestation=attestation,
            attestation_supplied=supplied,
            expected_media_hashes=self.expected_media,
            expected_rights_hashes=self.expected_rights,
        )

    def valid_attestation(self, summary):
        groups = []
        for group in summary["groups"]:
            groups.append({
                "group_id": group["group_id"],
                "media_sha256": group["media_sha256"],
                "samples_sha256": group["samples_sha256"],
                "gold_sha256": group["gold_sha256"],
                "audit_sha256": group["audit_sha256"],
                "rights_sha256": group["rights_sha256"],
                "reviewed_qids": self.group_qids[group["group_id"]],
                "checks": {
                    "answer_checked_against_video": True,
                    "evidence_supports_answer": True,
                    "unique_best_choice": True,
                    "stem_leakage_checked": True,
                    "video_only_review": True,
                    "no_audio_used": True,
                    "rights_scope_confirmed": True,
                    "credit_acknowledged": True,
                },
                "rights_scope": {
                    "video_only_model_inference": True,
                    "audio_use": False,
                    "external_media_publication": False,
                    "required_credit": "NOAA Ocean Exploration",
                },
            })
        return {
            "schema_version": 1,
            "signed": True,
            "signed_at_utc": "2026-10-12T12:00:00Z",
            "attestor": {"name": "Fixture Human", "role": "annotator", "human": True},
            "groups": groups,
        }

    def test_missing_human_attestation_is_no_go_and_counts_only(self):
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertEqual(result["attestation_status"], "missing")
        self.assertEqual((result["samples"], result["gold_records"], result["joined_qids"]), (3, 3, 3))
        self.assertEqual(result["label_leak_count"], 0)
        self.assertEqual(result["ai_blind_review_items"], 3)
        serialized = json.dumps(result)
        self.assertNotIn("Q001", serialized)
        self.assertNotIn("Fixture prompt", serialized)
        self.assertNotIn("Fixture Human", serialized)
        self.assertNotIn(str(self.repo), serialized)

    def test_valid_human_attestation_opens_only_video_qa_inference_gate(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "go_for_video_only_qa_inference")
        self.assertEqual(result["attestation_status"], "valid")
        self.assertIn("does_not_establish_qa_efficacy", result["scope_limit"])

    def test_sample_answer_leak_is_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["correct_choice_id"] = "A"
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("label_leak_in_sample", result["failure_reasons"])

    def test_normalized_answer_label_aliases_are_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        aliases = ("correctAnswer", "answerIndex", "goldAnswer", "ground_truth", "labelId", "is_correct")
        for alias in aliases:
            with self.subTest(alias=alias):
                # Restore the fixture before each adversarial alias.
                self.setUp_samples()
                samples = json.loads(sample_path.read_text(encoding="utf-8"))
                samples[0][alias] = "hidden label"
                sample_path.write_text(json.dumps(samples), encoding="utf-8")
                result = self.audit()
                self.assertIn("label_leak_in_sample", result["failure_reasons"])
                self.assertIsNone(result["label_leak_count"])
                self.assertFalse(result["label_leak_count_complete"])

    def setUp_samples(self):
        """Restore canonical fixture sample records for a parameterized mutation."""
        for index, group in enumerate(GROUPS):
            group_root = self.data_root / group["directory"]
            private = group_root / "private"
            media = group_root / "media" / "source.mp4"
            relative_media = Path("datasets/noaa_qa_pilot_20261011") / group["directory"] / "media" / "source.mp4"
            sample = {
                "qid": f"Q{index + 1:03d}",
                "video_id": f"NOAA_TEST_DIVE{index:02d}",
                "video": relative_media.as_posix(),
                "video_sha256": digest(media),
                "question": "Fixture prompt",
                "choices": [{"id": label, "text": f"Choice {label}"} for label in "ABCDE"],
                "window_start_seconds": 1.0,
                "window_end_seconds": 2.0,
            }
            (private / "samples.json").write_text(json.dumps([sample]), encoding="utf-8")

    def test_unknown_sample_and_choice_fields_fail_closed(self):
        sample_path = self.data_root / "private" / "samples.json"
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["unrecognized_metadata"] = "extra"
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertIn("invalid_sample_schema", result["failure_reasons"])

        self.setUp_samples()
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["choices"][0]["unrecognized_metadata"] = "extra"
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertIn("invalid_choice_schema", result["failure_reasons"])

    def test_choice_label_alias_is_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["choices"][0]["correctAnswer"] = True
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertIn("label_leak_in_sample", result["failure_reasons"])

    def test_invalid_video_interval_is_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        for start, end in ((-0.1, 1.0), (2.0, 1.0), (1.0, float("nan")), (1.0, float("inf"))):
            with self.subTest(start=start, end=end):
                self.setUp_samples()
                samples = json.loads(sample_path.read_text(encoding="utf-8"))
                samples[0]["window_start_seconds"] = start
                samples[0]["window_end_seconds"] = end
                sample_path.write_text(json.dumps(samples), encoding="utf-8")
                result = self.audit()
                self.assertIn("invalid_video_interval", result["failure_reasons"])

    def test_boolean_qid_and_invalid_choice_permutation_are_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["qid"] = True
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertIn("invalid_qid", result["failure_reasons"])

        self.setUp_samples()
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["choice_order_original_indices"] = [0, 1, 2, 3, 3]
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertIn("invalid_choice_order", result["failure_reasons"])

    def test_gold_qid_mismatch_is_rejected(self):
        gold_path = self.data_root / "private" / "gold.json"
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        gold[0]["qid"] = "OTHER"
        gold_path.write_text(json.dumps(gold), encoding="utf-8")
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("gold_qid_join_mismatch", result["failure_reasons"])

    def test_wrong_media_hash_is_rejected(self):
        media_path = self.data_root / "media" / "source.mp4"
        media_path.write_bytes(b"changed-video")
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("media_hash_mismatch", result["failure_reasons"])

    def test_wrong_fixed_rights_hash_is_rejected(self):
        self.expected_rights["dive03"] = "0" * 64
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("rights_hash_mismatch", result["failure_reasons"])

    def test_invalid_choice_schema_is_rejected(self):
        sample_path = self.data_root / "private" / "samples.json"
        samples = json.loads(sample_path.read_text(encoding="utf-8"))
        samples[0]["choices"] = samples[0]["choices"][:4]
        sample_path.write_text(json.dumps(samples), encoding="utf-8")
        result = self.audit()
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("choice_count_mismatch", result["failure_reasons"])

    def test_ai_or_unsigned_attestation_is_rejected(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        attestation["attestor"]["human"] = False
        attestation["signed"] = False
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "no_go")
        self.assertEqual(result["attestation_status"], "invalid")
        self.assertIn("attestation_human_missing", result["failure_reasons"])
        self.assertIn("attestation_not_signed_v1", result["failure_reasons"])

    def test_ai_detection_uses_explicit_tokens_only(self):
        audit_path = self.data_root / "private" / GROUPS[0]["audit_file"]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["method"] = "training validation by chair"
        audit["items"][0]["independent_review"] = {"reviewer_role": "chair"}
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        result = self.audit()
        first = next(group for group in result["groups"] if group["group_id"] == "dive03")
        self.assertEqual(first["ai_blind_review_item_count"], 0)

        audit["method"] = "independent AI reviewer"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        result = self.audit()
        first = next(group for group in result["groups"] if group["group_id"] == "dive03")
        self.assertEqual(first["ai_blind_review_item_count"], 1)

    def test_boolean_schema_version_is_rejected(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        attestation["schema_version"] = True
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("attestation_not_signed_v1", result["failure_reasons"])

    def test_duplicate_reviewed_qids_are_rejected(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        attestation["groups"][0]["reviewed_qids"] *= 2
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("attestation_reviewed_qids_mismatch", result["failure_reasons"])

    def test_result_artifact_is_provenanced_and_never_overwritten(self):
        result = self.audit()
        output = self.repo / "results" / "qa-gate.json"
        artifact = _write_run_artifact(result, output, repo_root=self.repo)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(payload["run"]["runner_sha256"], digest(Path(__file__).with_name("audit_noaa_qa_gate.py")))
        self.assertRegex(payload["run"]["created_at_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertIn("checkout_head", payload["run"])
        with self.assertRaises(FileExistsError):
            _write_run_artifact(result, output, repo_root=self.repo)
        with self.assertRaises(ValueError):
            _write_run_artifact(result, self.repo / "elsewhere" / "qa.json", repo_root=self.repo)

    def test_attestation_requires_answer_and_evidence_review(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        attestation["groups"][0]["checks"]["evidence_supports_answer"] = False
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("attestation_checks_incomplete", result["failure_reasons"])

    def test_attestation_must_match_all_reviewed_items_and_rights_scope(self):
        preliminary = self.audit()
        attestation = self.valid_attestation(preliminary)
        attestation["groups"][0]["reviewed_qids"] = []
        attestation["groups"][1]["rights_scope"]["audio_use"] = True
        result = self.audit(attestation=attestation, supplied=True)
        self.assertEqual(result["decision"], "no_go")
        self.assertIn("attestation_reviewed_qids_mismatch", result["failure_reasons"])
        self.assertIn("attestation_rights_scope_invalid", result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
