from __future__ import annotations

import csv
import io
import json
import unittest

from audit_nextgqa_frozen_candidate_exposure import audit_frozen_candidate_exposure
from build_nextgqa_prior_exposure_manifest import _token
from build_nextgqa_zip_gold_interval_candidate_pool import candidate_pair_sha256


KEY = b"synthetic-frozen-cohort-key-0123456789abcdef"


def _split(rows):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["video_id", "qid"])
    writer.writerows(rows)
    return out.getvalue().encode()


def _index(video=(), pair=(), media=()):
    return json.dumps({
        "schema_version": 1,
        "purpose": "outcome_blind_prior_exposure_index",
        "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
        "sources": [{"identity_namespace": "nextgqa"}],
        "identities": {
            "dataset_video_hmac_sha256": sorted(video),
            "media_sha256_hmac_sha256": sorted(media),
            "video_question_pair_hmac_sha256": sorted(pair),
        },
    }).encode()


def _inputs():
    rows = [("vid-a", "shared-qid"), ("vid-b", "shared-qid"),
            ("vid-b", "second-qid")]
    selected = [("vid-b", "shared-qid"), ("vid-b", "second-qid")]
    media = ["a" * 64, "b" * 64]
    pair_hashes = sorted(candidate_pair_sha256(*pair) for pair in selected)
    frozen = json.dumps({"schema_version": 1, "pair_sha256": pair_hashes}).encode()
    packet = json.dumps({
        "dataset_split": "val",
        "records": [
            {"dataset_question_id": "shared-qid", "video_sha256": media[0]},
            {"dataset_question_id": "second-qid", "video_sha256": media[1]},
        ],
    }).encode()
    prior = _index(
        video=[_token(KEY, "dataset-video-id", ("nextgqa", "vid-b"))],
        pair=[_token(KEY, "video-question-pair",
                     ("nextgqa", "dataset-video-id", "nextgqa", "vid-b", "shared-qid"))],
        media=[_token(KEY, "media-sha256", (media[0],))],
    )
    return _split(rows), frozen, packet, prior


class FrozenCandidateExposureAuditTests(unittest.TestCase):
    def test_pair_digest_resolves_reused_qid_and_counts_three_identity_kinds(self):
        split, frozen, packet, prior = _inputs()
        report = audit_frozen_candidate_exposure(
            split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
            index_raws=[prior], hmac_key=KEY)
        self.assertEqual(report["requested_frozen_pairs"], 2)
        self.assertEqual(report["mapped_frozen_pairs"], 2)
        self.assertEqual(report["packet_records_mapped"], 2)
        self.assertEqual(report["known_video_id_matches"], 1)
        self.assertEqual(report["known_video_qid_pair_matches"], 1)
        self.assertEqual(report["known_packet_claimed_media_sha256_matches"], 1)
        self.assertFalse(report["packet_pair_digest_binding_present"])
        self.assertFalse(report["packet_pair_digest_binding_authenticated"])
        self.assertFalse(report["expected_pair_to_media_crosswalk_verified"])
        self.assertTrue(report["packet_media_matches_are_packet_claims_only"])
        self.assertFalse(report["hmac_key_to_index_binding_authenticated"])
        self.assertFalse(report["historical_completeness_authenticated"])
        self.assertEqual(report["freshness_conclusion"], "not_made")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_wrong_key_zero_match_is_not_freshness(self):
        split, frozen, packet, prior = _inputs()
        report = audit_frozen_candidate_exposure(
            split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
            index_raws=[prior], hmac_key=b"different-synthetic-key-0123456789abc")
        self.assertEqual(report["known_video_id_matches"], 0)
        self.assertEqual(report["known_video_qid_pair_matches"], 0)
        self.assertEqual(report["known_packet_claimed_media_sha256_matches"], 0)
        self.assertEqual(report["freshness_conclusion"], "not_made")

    def test_missing_frozen_candidate_rejects(self):
        split, _frozen, packet, prior = _inputs()
        frozen = json.dumps({"schema_version": 1, "pair_sha256": ["0" * 64]}).encode()
        with self.assertRaisesRegex(ValueError, "frozen_pair_not_in_split"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
                index_raws=[prior], hmac_key=KEY)

    def test_packet_qid_set_must_match_exact_frozen_pairs(self):
        split, frozen, packet, prior = _inputs()
        doc = json.loads(packet)
        doc["records"][0]["dataset_question_id"] = "unrelated-qid"
        with self.assertRaisesRegex(ValueError, "packet_to_frozen_pair_join_invalid"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen,
                packet_raw=json.dumps(doc).encode(), index_raws=[prior], hmac_key=KEY)

    def test_aggregate_never_emits_video_ids_or_qids(self):
        split, frozen, packet, prior = _inputs()
        report = audit_frozen_candidate_exposure(
            split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
            index_raws=[prior], hmac_key=KEY)
        encoded = json.dumps(report)
        for private in ("vid-a", "vid-b", "shared-qid", "second-qid"):
            self.assertNotIn(private, encoded)
        self.assertNotIn("/home/", encoded)

    def test_duplicate_packet_qids_reject(self):
        split, frozen, packet, prior = _inputs()
        doc = json.loads(packet)
        doc["records"][1]["dataset_question_id"] = "shared-qid"
        with self.assertRaisesRegex(ValueError, "packet_to_frozen_pair_join_invalid"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen,
                packet_raw=json.dumps(doc).encode(), index_raws=[prior], hmac_key=KEY)

    def test_explicit_pair_digests_support_selected_reused_qid(self):
        rows = [("vid-a", "shared-qid"), ("vid-b", "shared-qid")]
        selected = [candidate_pair_sha256(*row) for row in rows]
        split = _split(rows)
        frozen = json.dumps({"schema_version": 1, "pair_sha256": sorted(selected)}).encode()
        packet = json.dumps({"dataset_split": "val", "records": [
            {"dataset_question_id": "shared-qid", "candidate_pair_sha256": digest,
             "video_sha256": sha}
            for digest, sha in zip(selected, ("a" * 64, "b" * 64))
        ]}).encode()
        report = audit_frozen_candidate_exposure(
            split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
            index_raws=[_index()], hmac_key=KEY)
        self.assertEqual(report["packet_records_mapped"], 2)
        self.assertTrue(report["packet_pair_digest_binding_present"])
        self.assertFalse(report["packet_pair_digest_binding_authenticated"])

    def test_swapped_packet_media_hashes_remain_explicitly_unverified(self):
        split, frozen, packet, prior = _inputs()
        doc = json.loads(packet)
        doc["records"][0]["video_sha256"], doc["records"][1]["video_sha256"] = (
            doc["records"][1]["video_sha256"], doc["records"][0]["video_sha256"])
        report = audit_frozen_candidate_exposure(
            split_raw=split, frozen_pairs_raw=frozen,
            packet_raw=json.dumps(doc).encode(), index_raws=[prior], hmac_key=KEY)
        self.assertEqual(report["known_packet_claimed_media_sha256_matches"], 1)
        self.assertTrue(report["packet_media_matches_are_packet_claims_only"])
        self.assertFalse(report["expected_pair_to_media_crosswalk_verified"])

    def test_legacy_packet_rejects_selected_reused_qid(self):
        rows = [("vid-a", "shared-qid"), ("vid-b", "shared-qid")]
        split = _split(rows)
        frozen = json.dumps({"schema_version": 1, "pair_sha256": sorted(
            candidate_pair_sha256(*row) for row in rows)}).encode()
        packet = json.dumps({"dataset_split": "val", "records": [
            {"dataset_question_id": "shared-qid", "video_sha256": "a" * 64},
            {"dataset_question_id": "shared-qid", "video_sha256": "b" * 64},
        ]}).encode()
        with self.assertRaisesRegex(ValueError, "packet_to_frozen_pair_join_invalid"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
                index_raws=[_index()], hmac_key=KEY)

    def test_duplicate_json_keys_reject_even_when_nested(self):
        split, frozen, packet, prior = _inputs()
        bad_packet = packet.replace(b'"dataset_split": "val"',
                                    b'"dataset_split": "train", "dataset_split": "val"')
        with self.assertRaisesRegex(ValueError, "run_packet_invalid"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen, packet_raw=bad_packet,
                index_raws=[prior], hmac_key=KEY)
        nested = _index().replace(b'"schema_version": 1',
                                  b'"schema_version": 1, "schema_version": 1')
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            audit_frozen_candidate_exposure(
                split_raw=split, frozen_pairs_raw=frozen, packet_raw=packet,
                index_raws=[nested], hmac_key=KEY)

    def test_duplicate_split_pair_rejects(self):
        split, frozen, packet, prior = _inputs()
        duplicate = split + b"vid-b,shared-qid\r\n"
        with self.assertRaisesRegex(ValueError, "split_csv_duplicate_pair"):
            audit_frozen_candidate_exposure(
                split_raw=duplicate, frozen_pairs_raw=frozen, packet_raw=packet,
                index_raws=[prior], hmac_key=KEY)


if __name__ == "__main__":
    unittest.main()
