import json
import unittest

from scripts.audit_nextqa_cohort_split import _decode, _read_split, audit


def raw(value):
    return json.dumps(value).encode()


def fixture():
    cohort = [{"video": "test-a.mp4"}, {"video": "test-b.mp4"}]
    splits = {
        "train": [{"video_id": "train-a"}],
        "val": [{"video_id": "val-a"}],
        "test": [{"video_id": "test-a"}, {"video_id": "test-b"}],
    }
    return cohort, splits


class NextQACohortSplitAuditTests(unittest.TestCase):
    def test_cohort_membership_and_disjoint_splits(self):
        cohort, splits = fixture()
        result = audit(cohort, raw(cohort), splits, {key: raw(value) for key, value in splits.items()})
        self.assertEqual(result["status"], "mapped_unambiguously")
        self.assertEqual(result["cohort_matches_by_split"], {"train": 0, "val": 0, "test": 2})
        self.assertEqual(result["source_split_pairwise_overlaps"],
                         {"train_val": 0, "train_test": 0, "val_test": 0})
        self.assertEqual(result["rights_or_inference_authorization"], "not_granted")
        encoded = json.dumps(result)
        for private_value in ("test-a", "train-a", "val-a"):
            self.assertNotIn(private_value, encoded)

    def test_multiple_split_membership_is_ambiguous(self):
        cohort, splits = fixture()
        splits["val"].append({"video_id": "test-a"})
        result = audit(cohort, raw(cohort), splits, {key: raw(value) for key, value in splits.items()})
        self.assertIn("source_split_membership_ambiguous", result["failure_codes"])

    def test_unmatched_cohort_fails_closed(self):
        cohort, splits = fixture()
        cohort.append({"video": "unknown.mp4"})
        result = audit(cohort, raw(cohort), splits, {key: raw(value) for key, value in splits.items()})
        self.assertEqual(result["cohort_ids_unmatched"], 1)
        self.assertIn("cohort_video_unmatched_to_source_splits", result["failure_codes"])

    def test_duplicate_json_keys_rejected(self):
        with self.assertRaises(ValueError):
            _decode(b'{"video":"a.mp4","video":"b.mp4"}')
        with self.assertRaises(ValueError):
            _decode(b'{"x":NaN}')

    def test_split_csv_rejects_ambiguous_headers_and_row_width(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split.csv"
            for content in (
                b"video_id,video_id\na,b\n",
                b"other_id\na\n",
                b"video_id,other\na\n",
                b"video_id,other\na,b,c\n",
                b'video_id\n"unclosed-id\n',
            ):
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    _read_split(path)


if __name__ == "__main__":
    unittest.main()
