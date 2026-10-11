"""Synthetic tests for the private first-pass record's structural contract."""

from __future__ import annotations

import copy
import json
import unittest

from noaa_first_pass_contract import audit_first_pass_structure


class FirstPassContractTests(unittest.TestCase):
    def setUp(self):
        self.hashes = {
            "group_a": {
                "media_sha256": "a" * 64,
                "samples_sha256": "b" * 64,
                "gold_sha256": "c" * 64,
                "audit_sha256": "d" * 64,
                "rights_sha256": "e" * 64,
            },
            "group_b": {
                "media_sha256": "1" * 64,
                "samples_sha256": "2" * 64,
                "gold_sha256": "3" * 64,
                "audit_sha256": "4" * 64,
                "rights_sha256": "5" * 64,
            },
        }
        self.qids = {"group_a": {"QA1", "QA2"}, "group_b": {"QB1"}}
        self.choice_counts = {
            "group_a": {"QA1": 5, "QA2": 5},
            "group_b": {"QB1": 5},
        }
        self.review_media = {
            group: {
                "group_id": group,
                "original_media_sha256": self.hashes[group]["media_sha256"],
                "silent_copy_sha256": ("f" if group == "group_a" else "6") * 64,
                "source_video_packet_sha256": ("7" if group == "group_a" else "8") * 64,
                "copy_video_packet_sha256": ("7" if group == "group_a" else "8") * 64,
                "source_video_start_time": 0.0,
                "copy_video_start_time": 0.0,
                "source_video_duration": 300.033067,
                "copy_video_duration": 300.033067,
                "source_video_nb_frames": 8992,
                "copy_video_nb_frames": 8992,
                "video_pts_count": 8992,
                "source_audio_streams_removed": 2,
                "copy_audio_streams": 0,
            }
            for group in self.hashes
        }
        self.record = self.valid_record()

    def valid_record(self):
        return {
            "schema_version": 1,
            "signed": False,
            "exported_at_utc": "2026-10-12T12:34:56.789Z",
            "attestor": {"name": "", "role": "", "human": False},
            "source_sha256": copy.deepcopy(self.hashes),
            "review_media": [copy.deepcopy(self.review_media[group]) for group in self.hashes],
            "groups": [
                {
                    "group_id": group,
                    "source_sha256": copy.deepcopy(self.hashes[group]),
                    "choices": [
                        {"qid": qid, "choice_index": 0, "choice_label": "A"}
                        for qid in sorted(self.qids[group])
                    ],
                }
                for group in ("group_a", "group_b")
            ],
        }

    def audit(self, record=None):
        return audit_first_pass_structure(
            self.record if record is None else record,
            expected_hashes_by_group=self.hashes,
            expected_qids_by_group=self.qids,
            choice_counts_by_group=self.choice_counts,
            expected_review_media_by_group=self.review_media,
        )

    def assert_invalid(self, mutate, code):
        record = copy.deepcopy(self.record)
        mutate(record)
        result = self.audit(record)
        self.assertFalse(result["structurally_valid"])
        self.assertIn(code, result["failure_codes"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertFalse(result["semantic_correctness_verified"])
        self.assertFalse(result["chronology_verified"])
        self.assertFalse(result["reviewer_identity_verified"])
        self.assertNotIn("QA1", json.dumps(result))
        self.assertNotIn("group_a", json.dumps(result))

    def test_complete_record_is_structural_only_and_never_authorizes_execution(self):
        result = self.audit()
        self.assertEqual(result["validity"], "structural_only")
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["expected_groups"], 2)
        self.assertEqual(result["effective_groups"], 2)
        self.assertEqual(result["expected_questions"], 3)
        self.assertEqual(result["effective_choices"], 3)
        self.assertTrue(result["review_media_matches_expected_metadata"])
        self.assertEqual(result["failure_codes"], [])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertNotIn("QA1", json.dumps(result))

    def test_any_in_range_choice_is_structurally_valid_without_gold_comparison(self):
        record = copy.deepcopy(self.record)
        record["groups"][0]["choices"][0] = {
            "qid": "QA1", "choice_index": 4, "choice_label": "E"
        }
        result = self.audit(record)
        self.assertTrue(result["structurally_valid"])
        self.assertEqual(result["validity"], "structural_only")

    def test_missing_or_malformed_first_pass_is_rejected(self):
        result = audit_first_pass_structure(
            None,
            expected_hashes_by_group=self.hashes,
            expected_qids_by_group=self.qids,
            choice_counts_by_group=self.choice_counts,
            expected_review_media_by_group=self.review_media,
        )
        self.assertFalse(result["structurally_valid"])
        self.assertIn("first_pass_not_object", result["failure_codes"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assert_invalid(lambda record: record.pop("groups"), "first_pass_fields_invalid")

    def test_schema_signed_timestamp_and_attestor_template_are_strict(self):
        self.assert_invalid(lambda record: record.update(schema_version=True), "first_pass_schema_version_invalid")
        self.assert_invalid(lambda record: record.update(signed=True), "first_pass_signed_flag_invalid")
        self.assert_invalid(lambda record: record.update(exported_at_utc="2026-10-12T12:34:56Z"), "first_pass_timestamp_invalid")
        self.assert_invalid(lambda record: record.update(exported_at_utc="2026-02-30T12:34:56.789Z"), "first_pass_timestamp_invalid")
        self.assert_invalid(lambda record: record["attestor"].update(human=True), "first_pass_attestor_template_invalid")

    def test_source_hashes_and_group_set_are_bound_and_unique(self):
        self.assert_invalid(
            lambda record: record["source_sha256"]["group_a"].update(media_sha256="0" * 64),
            "first_pass_source_hashes_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0]["source_sha256"].update(rights_sha256="0" * 64),
            "first_pass_group_hash_mismatch",
        )
        self.assert_invalid(
            lambda record: record["groups"].append(copy.deepcopy(record["groups"][0])),
            "first_pass_group_set_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0].update(group_id="unknown"),
            "first_pass_group_set_invalid",
        )
        self.assert_invalid(
            lambda record: record["review_media"].pop(),
            "first_pass_review_media_invalid",
        )

    def test_review_media_is_exactly_bound_to_caller_supplied_metadata(self):
        def wrong_silent_copy(record):
            record["review_media"][0]["silent_copy_sha256"] = "0" * 64

        self.assert_invalid(wrong_silent_copy, "first_pass_review_media_mismatch")

        def wrong_original_media(record):
            record["review_media"][0]["original_media_sha256"] = "0" * 64

        self.assert_invalid(wrong_original_media, "first_pass_review_media_mismatch")

        def extra_media_field(record):
            record["review_media"][0]["unverified_extra"] = "not accepted"

        self.assert_invalid(extra_media_field, "first_pass_review_media_mismatch")

    def test_choice_qids_must_be_complete_unique_and_known(self):
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"].pop(),
            "first_pass_choice_qid_set_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"].append(copy.deepcopy(record["groups"][0]["choices"][0])),
            "first_pass_choice_qid_set_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"][0].update(qid=True),
            "first_pass_choice_qid_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"][0].update(qid="UNKNOWN"),
            "first_pass_choice_qid_invalid",
        )

    def test_choice_index_is_strict_integer_in_range_and_label_consistent(self):
        for value in (True, 1.0, "1", -1, 5):
            with self.subTest(value=value):
                self.assert_invalid(
                    lambda record, choice=value: record["groups"][0]["choices"][0].update(choice_index=choice),
                    "first_pass_choice_index_invalid",
                )
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"][0].update(choice_label="B"),
            "first_pass_choice_label_invalid",
        )
        self.assert_invalid(
            lambda record: record["groups"][0]["choices"][0].pop("choice_label"),
            "first_pass_choice_fields_invalid",
        )

    def test_unknown_fields_fail_closed_without_echoing_values(self):
        self.assert_invalid(
            lambda record: record.update(private_note="never echo this"),
            "first_pass_fields_invalid",
        )
        result = self.audit({**self.record, "private_note": "never echo this"})
        self.assertNotIn("never echo this", json.dumps(result))

    def test_expected_input_requires_nonempty_qids_and_exact_five_choices(self):
        with self.assertRaisesRegex(ValueError, "at least one QID"):
            audit_first_pass_structure(
                self.record,
                expected_hashes_by_group=self.hashes,
                expected_qids_by_group={"group_a": set(), "group_b": {"QB1"}},
                choice_counts_by_group={"group_a": {}, "group_b": {"QB1": 5}},
                expected_review_media_by_group=self.review_media,
            )
        for invalid_count in (4, 6, 26, True):
            with self.subTest(invalid_count=invalid_count):
                bad_counts = copy.deepcopy(self.choice_counts)
                bad_counts["group_a"]["QA1"] = invalid_count
                with self.assertRaisesRegex(ValueError, "exactly five choices"):
                    audit_first_pass_structure(
                        self.record,
                        expected_hashes_by_group=self.hashes,
                        expected_qids_by_group=self.qids,
                        choice_counts_by_group=bad_counts,
                        expected_review_media_by_group=self.review_media,
                    )


if __name__ == "__main__":
    unittest.main()
