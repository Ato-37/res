"""Synthetic checks for exact-ID prior-exposure auditing."""
from __future__ import annotations

import csv
import io
import json
import unittest

from audit_nextgqa_split_prior_identity import audit_split_identity_exposure
from build_nextgqa_prior_exposure_manifest import _token


KEY = b"synthetic-prior-exposure-key-0123456789abcdef"


def _index(video_ids=(), pairs=()):
    return json.dumps({
        "schema_version": 1,
        "purpose": "outcome_blind_prior_exposure_index",
        "canonicalization_version": "explicit-identifiers-v1-no-path-stems",
        "coverage": {"historical_completeness_authenticated": False},
        "identities": {
            "dataset_video_hmac_sha256": sorted(video_ids),
            "media_sha256_hmac_sha256": [],
            "video_question_pair_hmac_sha256": sorted(pairs),
        },
    }, separators=(",", ":")).encode()


def _token_video(value):
    return _token(KEY, "dataset-video-id", ("nextgqa", value))


def _token_pair(video_id, qid):
    return _token(KEY, "video-question-pair",
                  ("nextgqa", "dataset-video-id", "nextgqa", video_id, qid))


def _split(rows):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["video_id", "qid"])
    writer.writerows(rows)
    return out.getvalue().encode()


class SplitPriorIdentityAuditTests(unittest.TestCase):
    def test_counts_exact_video_id_and_video_qid_matches(self):
        rows = [("vid-a", "q1"), ("vid-a", "q2"), ("vid-b", "q3")]
        prior = _index([_token_video("vid-a")], [_token_pair("vid-a", "q1")])
        report = audit_split_identity_exposure(
            split_raw=_split(rows), index_raws=[prior], hmac_key=KEY)
        self.assertEqual(report["split_unique_video_ids"], 2)
        self.assertEqual(report["split_unique_video_qid_pairs"], 3)
        self.assertEqual(report["matched_video_ids_in_prior_index_union"], 1)
        self.assertEqual(report["matched_video_qid_pairs_in_prior_index_union"], 1)

    def test_unions_known_history_from_multiple_indexes(self):
        rows = [("vid-a", "q1"), ("vid-b", "q2")]
        first = _index([_token_video("vid-a")], [_token_pair("vid-a", "q1")])
        second = _index([_token_video("vid-b")], [_token_pair("vid-b", "q2")])
        report = audit_split_identity_exposure(
            split_raw=_split(rows), index_raws=[first, second], hmac_key=KEY)
        self.assertEqual(report["matched_video_ids_by_prior_index"], [1, 1])
        self.assertEqual(report["matched_video_qid_pairs_by_prior_index"], [1, 1])
        self.assertEqual(report["matched_video_ids_in_prior_index_union"], 2)
        self.assertEqual(report["matched_video_qid_pairs_in_prior_index_union"], 2)

    def test_qid_mismatch_does_not_count_as_pair_match(self):
        prior = _index([], [_token_pair("vid-a", "old-qid")])
        report = audit_split_identity_exposure(
            split_raw=_split([("vid-a", "new-qid")]), index_raws=[prior], hmac_key=KEY)
        self.assertEqual(report["matched_video_ids_in_prior_index_union"], 0)
        self.assertEqual(report["matched_video_qid_pairs_in_prior_index_union"], 0)

    def test_duplicate_split_rows_do_not_inflate_distinct_counts(self):
        report = audit_split_identity_exposure(
            split_raw=_split([("vid-a", "q1"), ("vid-a", "q1")]),
            index_raws=[_index()], hmac_key=KEY)
        self.assertEqual(report["split_unique_video_ids"], 1)
        self.assertEqual(report["split_unique_video_qid_pairs"], 1)

    def test_nonmatching_index_never_claims_freshness_or_authorizes_execution(self):
        report = audit_split_identity_exposure(
            split_raw=_split([("private-video", "private-qid")]),
            index_raws=[_index()], hmac_key=KEY)
        self.assertEqual(report["freshness_conclusion"], "not_made")
        self.assertFalse(report["hmac_key_to_index_binding_authenticated"])
        self.assertFalse(report["historical_completeness_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["faithfulness_scored"])
        self.assertFalse(report["gpu_used"])
        serialized = json.dumps(report)
        self.assertNotIn("private-video", serialized)
        self.assertNotIn("private-qid", serialized)

    def test_wrong_key_zero_match_is_explicitly_unverified(self):
        prior = _index([_token_video("vid-a")], [_token_pair("vid-a", "q1")])
        report = audit_split_identity_exposure(
            split_raw=_split([("vid-a", "q1")]), index_raws=[prior],
            hmac_key=b"another-synthetic-key-0123456789abcdef")
        self.assertEqual(report["matched_video_ids_in_prior_index_union"], 0)
        self.assertEqual(report["matched_video_qid_pairs_in_prior_index_union"], 0)
        self.assertFalse(report["hmac_key_to_index_binding_authenticated"])

    def test_accepts_utf8_bom_csv_without_echoing_values(self):
        report = audit_split_identity_exposure(
            split_raw=b"\xef\xbb\xbfvideo_id,qid\nvid-a,q1\n",
            index_raws=[_index()], hmac_key=KEY)
        self.assertEqual(report["split_unique_video_ids"], 1)

    def test_rejects_duplicate_csv_headers_and_invalid_utf8(self):
        for raw in (b"video_id,qid,qid\nvid-a,q1,q2\n", b"video_id,qid\n\xff,q1\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                audit_split_identity_exposure(
                    split_raw=raw, index_raws=[_index()], hmac_key=KEY)

    def test_rejects_authenticated_or_malformed_index_schema(self):
        malformed = json.loads(_index())
        malformed["coverage"]["historical_completeness_authenticated"] = True
        with self.assertRaisesRegex(ValueError, "prior_index_schema_invalid"):
            audit_split_identity_exposure(
                split_raw=_split([("vid-a", "q1")]),
                index_raws=[json.dumps(malformed).encode()], hmac_key=KEY)

    def test_rejects_malformed_token_and_duplicate_tokens(self):
        for values in (["bad-token"], [_token_video("vid-a")] * 2):
            malformed = json.loads(_index())
            malformed["identities"]["dataset_video_hmac_sha256"] = values
            with self.subTest(values=values), self.assertRaisesRegex(
                    ValueError, "prior_index_identity_lists_invalid"):
                audit_split_identity_exposure(
                    split_raw=_split([("vid-a", "q1")]),
                    index_raws=[json.dumps(malformed).encode()], hmac_key=KEY)

    def test_rejects_duplicate_json_keys_and_index_limit_overflow(self):
        for raw, indexes in ((b'{"schema_version":1,"schema_version":1}', [b"{}"]),
                             (_index(), [b"{}"] * 65)):
            with self.subTest(indexes=len(indexes)), self.assertRaises(ValueError):
                audit_split_identity_exposure(
                    split_raw=_split([("vid-a", "q1")]), index_raws=indexes,
                    hmac_key=KEY)

    def test_rejects_bad_csv_identity_or_row_width(self):
        for raw in (b"video_id,qid\nvid-a,\n", b"video_id,qid\nvid-a\n"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                audit_split_identity_exposure(
                    split_raw=raw, index_raws=[_index()], hmac_key=KEY)


if __name__ == "__main__":
    unittest.main(verbosity=2)
