from __future__ import annotations

import unittest
import os
from pathlib import Path
import tempfile
import hashlib

from audit_qta_v4_mask_specificity_result import (
    audit_records, read_private_score, validate_completed_metadata,
)
from run_nextgqa_qta_v4_mask_specificity import (
    ALL_BRANCHES, EXPECTED_N, V4_BRANCHES, build_summaries,
)


def synthetic_receipt_parts(eligible_n=140):
    records, preflight, old = [], [], []
    for index in range(EXPECTED_N):
        binding = f"{index+1:064x}"
        eligible = index < eligible_n
        video, qid, geom = f"v{index}", f"q{index}", f"{index+1:064x}"
        preflight.append({"video_id": video, "qid": qid, "geometry_sha256": geom,
                          "row_binding_sha256": binding, "eligible": eligible})
        old.append({"video_id": video, "qid": qid, "geometry_sha256": geom,
                    "row_binding_sha256": binding, "answer_index": 0, "foil_index": 1})
        margins = {"baseline_full": 1.0, "baseline_union_mask": 0.5,
                   "intervention_full": 1.2, "intervention_union_mask": 0.3}
        record = {"video_id": video, "qid": qid, "geometry_sha256": geom,
            "row_binding_sha256": binding, "answer_index": 0, "foil_index": 1,
            "specificity_eligible": eligible, "historical_answer_foil_match": True,
            "historical_baseline_full_margin_abs_error": 0.0,
            "margin_by_condition": margins, "D_annotation_union": 0.4}
        if eligible:
            record.update({"baseline_placebo_margin": 0.8,
                "intervention_placebo_margin": 0.8, "D_placebo_translation": 0.2,
                "specificity_difference": 0.2})
        records.append(record)
    return records, preflight, old


class SpecificityResultAuditTests(unittest.TestCase):
    def test_independent_row_arithmetic_and_summary_reproduction(self):
        records, preflight, old = synthetic_receipt_parts()
        branches = {name: {"requested": EXPECTED_N if name in V4_BRANCHES else 140,
                           "attempted": EXPECTED_N if name in V4_BRANCHES else 140,
                           "completed": EXPECTED_N if name in V4_BRANCHES else 140}
                    for name in ALL_BRANCHES}
        summary = build_summaries(records)
        result = audit_records(records, preflight_rows=preflight, old_rows=old,
            branch_counts=branches, requested=868, attempted=868, completed=868,
            published_summary=summary)
        self.assertEqual(result["eligible_rows"], 140)
        self.assertEqual(result["unavailable_rows"], 7)
        self.assertAlmostEqual(result["summary"]["eligible_specificity_difference"]["mean"], 0.2)

    def test_rejects_changed_d_contrast(self):
        records, preflight, old = synthetic_receipt_parts()
        records[0]["D_annotation_union"] = -0.4
        branches = {name: {"requested": EXPECTED_N if name in V4_BRANCHES else 140,
                           "attempted": EXPECTED_N if name in V4_BRANCHES else 140,
                           "completed": EXPECTED_N if name in V4_BRANCHES else 140}
                    for name in ALL_BRANCHES}
        with self.assertRaisesRegex(ValueError, "D_annotation_arithmetic_mismatch"):
            audit_records(records, preflight_rows=preflight, old_rows=old,
                branch_counts=branches, requested=868, attempted=868, completed=868)

    def test_rejects_forward_count_and_missing_placebo(self):
        records, preflight, old = synthetic_receipt_parts()
        branches = {name: {"requested": EXPECTED_N if name in V4_BRANCHES else 140,
                           "attempted": EXPECTED_N if name in V4_BRANCHES else 140,
                           "completed": EXPECTED_N if name in V4_BRANCHES else 140}
                    for name in ALL_BRANCHES}
        with self.assertRaisesRegex(ValueError, "denominator_or_branch"):
            audit_records(records, preflight_rows=preflight, old_rows=old,
                branch_counts=branches, requested=867, attempted=868, completed=868)
        records[0].pop("D_placebo_translation")
        with self.assertRaisesRegex(ValueError, "D_placebo_nonfinite"):
            audit_records(records, preflight_rows=preflight, old_rows=old,
                branch_counts=branches, requested=868, attempted=868, completed=868)

    def test_rejects_boolean_branch_counter(self):
        records, preflight, old = synthetic_receipt_parts()
        branches = {name: {"requested": EXPECTED_N if name in V4_BRANCHES else 140,
                           "attempted": EXPECTED_N if name in V4_BRANCHES else 140,
                           "completed": EXPECTED_N if name in V4_BRANCHES else 140}
                    for name in ALL_BRANCHES}
        branches[ALL_BRANCHES[0]]["requested"] = True
        with self.assertRaisesRegex(ValueError, "denominator_or_branch"):
            audit_records(records, preflight_rows=preflight, old_rows=old,
                branch_counts=branches, requested=868, attempted=868, completed=868)

    def test_rejects_negative_historical_absolute_error(self):
        records, preflight, old = synthetic_receipt_parts()
        records[0]["historical_baseline_full_margin_abs_error"] = -0.1
        branches = {name: {"requested": EXPECTED_N if name in V4_BRANCHES else 140,
                           "attempted": EXPECTED_N if name in V4_BRANCHES else 140,
                           "completed": EXPECTED_N if name in V4_BRANCHES else 140}
                    for name in ALL_BRANCHES}
        with self.assertRaisesRegex(ValueError, "historical_margin_tolerance"):
            audit_records(records, preflight_rows=preflight, old_rows=old,
                branch_counts=branches, requested=868, attempted=868, completed=868)

    def test_complete_metadata_requires_exact_denominators_and_summary(self):
        payload = {"status": "qta_v4_mask_specificity_exploratory_complete",
            "requested_rows": 147, "effective_rows": 147,
            "eligible_specificity_rows": 140, "structurally_unavailable_rows": 7,
            "summary": {}, "rights_status": "unverified_local_only",
            "faithfulness_result": False,
            "inferential_status": "finite cohort descriptive only; no p-value or population interval"}
        validate_completed_metadata(payload, eligible_n=140)
        del payload["summary"]
        with self.assertRaisesRegex(ValueError, "complete_metadata_invalid"):
            validate_completed_metadata(payload, eligible_n=140)
        payload["summary"] = {}
        payload["eligible_specificity_rows"] = 139
        with self.assertRaisesRegex(ValueError, "complete_metadata_invalid"):
            validate_completed_metadata(payload, eligible_n=140)

    def test_private_loader_rejects_symlink_and_wrong_mode(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            root.chmod(0o700)
            actual = root / "actual.json"
            actual.write_text("{}", encoding="utf-8")
            actual.chmod(0o600)
            loaded, digest, mode = read_private_score(actual, root)
            self.assertEqual(loaded, {})
            self.assertEqual(digest, hashlib.sha256(b"{}").hexdigest())
            self.assertEqual(mode, "0o600")
            link = root / "link.json"
            link.symlink_to(actual.name)
            with self.assertRaises(OSError):
                read_private_score(link, root)
            normal = root / "normal.json"
            normal.write_text("{}", encoding="utf-8")
            normal.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "mode_invalid"):
                read_private_score(normal, root)


if __name__ == "__main__":
    unittest.main()
