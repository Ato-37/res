from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_claim_support_qta_16frame_exploratory import (
    _read_bound_response,
    compute_selective_dependence,
    select_control_groups,
)


class ClaimSupportQtaExploratoryTests(unittest.TestCase):
    def test_control_is_equal_budget_disjoint_and_rank_nearest(self):
        control = select_control_groups((2,), (2,), 8)
        self.assertEqual(len(control), 1)
        self.assertNotIn(2, control)
        self.assertEqual(control, (1,))

    def test_control_excludes_multiple_support_and_qta_groups(self):
        control = select_control_groups((1, 2), (2,), 8)
        self.assertEqual(len(control), 2)
        self.assertFalse(set(control) & {1, 2})
        self.assertEqual(control, (0, 3))

    def test_control_failure_is_not_replaced_by_another_budget(self):
        with self.assertRaisesRegex(ValueError, "matched_control_unavailable"):
            select_control_groups((0, 1, 2, 3), (4,), 8)

    def test_delta_contrast_sign_matches_protocol(self):
        result = compute_selective_dependence(
            baseline_support=-2.0, baseline_control=-1.0,
            qta_support=-1.0, qta_control=-1.5)
        self.assertEqual(result["D_baseline_nats_per_token"], 1.0)
        self.assertEqual(result["D_qta_nats_per_token"], -0.5)
        self.assertEqual(result["delta_D_qta_minus_baseline_nats_per_token"], -1.5)

    def test_nonfinite_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "branch_score_nonfinite"):
            compute_selective_dependence(
                baseline_support=float("nan"), baseline_control=-1.0,
                qta_support=-1.0, qta_control=-1.5)

    def test_bound_response_accepts_download_style_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            path.write_bytes(b"{}\n")
            path.chmod(0o644)
            self.assertEqual(_read_bound_response(path), b"{}\n")

    def test_bound_response_rejects_symlink_and_oversize(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, link = root / "source", root / "link"
            source.write_bytes(b"{}")
            link.symlink_to(source)
            with self.assertRaises(OSError):
                _read_bound_response(link)
            with self.assertRaisesRegex(ValueError, "bound_response_file_contract_invalid"):
                _read_bound_response(source, max_bytes=1)


if __name__ == "__main__":
    unittest.main()
