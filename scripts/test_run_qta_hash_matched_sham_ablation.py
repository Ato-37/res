from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qta_hash_matched_sham_ablation import (  # noqa: E402
    _write_new_json, choose_overlap_matched_sham,
)


class HashMatchedShamTests(unittest.TestCase):
    def test_deterministic_and_matches_zero_zero_overlap(self):
        kwargs = dict(temporal_group_count=16, evidence_groups=(8, 9, 10, 11),
            control_groups=(4, 5, 6, 7), qta_groups=(12, 14), opaque_row_token="opaque")
        first = choose_overlap_matched_sham(**kwargs)
        self.assertEqual(first, choose_overlap_matched_sham(**kwargs))
        self.assertEqual(len(first), 2)
        self.assertEqual(len(set(first) & set(kwargs["evidence_groups"])), 0)
        self.assertEqual(len(set(first) & set(kwargs["control_groups"])), 0)

    def test_matches_two_control_group_overlap(self):
        sham = choose_overlap_matched_sham(temporal_group_count=16,
            evidence_groups=(8, 9, 10, 11), control_groups=(4, 5, 6, 7),
            qta_groups=(6, 7), opaque_row_token="opaque")
        self.assertEqual(len(sham), 2)
        self.assertEqual(len(set(sham) & {8, 9, 10, 11}), 0)
        self.assertEqual(len(set(sham) & {4, 5, 6, 7}), 2)

    def test_rejects_when_matching_overlap_is_impossible(self):
        with self.assertRaisesRegex(ValueError, "overlap_matched_sham_unavailable"):
            choose_overlap_matched_sham(temporal_group_count=2,
                evidence_groups=(0,), control_groups=(1,), qta_groups=(0,),
                opaque_row_token="opaque")

    def test_receipt_create_is_exclusive_and_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            _write_new_json(path, {"ok": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                _write_new_json(path, {"overwritten": True})
            self.assertEqual(path.read_text(), '{\n  "ok": true\n}\n')


if __name__ == "__main__":
    unittest.main()
