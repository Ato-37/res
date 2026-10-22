import unittest

from run_nextgqa_qta_v3_geometry_census import (
    make_private_plan, min_hash_qid, select_v3_qids,
    ensure_outputs_absent, validate_candidate_geometry, validate_census_structure,
)
import os
from pathlib import Path
import tempfile


class QtaV3GeometryCensusTests(unittest.TestCase):
    def fixture(self):
        return {"videos": [
            {"video_id": "v1", "valid_qid_count": 3, "eligible_qids": [
                {"qid": "q-a", "geometry_sha256": "a" * 64,
                 "evidence_group_index": 1, "control_group_index": 3,
                 "row_binding_sha256": "b" * 64},
                {"qid": "q-b", "geometry_sha256": "c" * 64,
                 "evidence_group_index": 2, "control_group_index": 0,
                 "row_binding_sha256": "d" * 64}],
             "candidate_results": [{"qid": "q-a", "status": "eligible"},
                                   {"qid": "q-b", "status": "eligible"},
                                   {"qid": "q-c", "status": "ineligible"}]},
            {"video_id": "v2", "valid_qid_count": 1, "eligible_qids": [
                {"qid": "q-z", "geometry_sha256": "e" * 64,
                 "evidence_group_index": 0, "control_group_index": 2,
                 "row_binding_sha256": "f" * 64}],
             "candidate_results": [{"qid": "q-z", "status": "eligible"}]},
        ]}

    def test_min_hash_is_deterministic_and_scoped_to_feasible_qids(self):
        census = self.fixture()
        selected = select_v3_qids(census)
        expected = min(("q-a", "q-b"), key=lambda q: min_hash_qid("v1", q))
        self.assertEqual(selected[0]["qid"], expected)
        self.assertEqual(selected, select_v3_qids(census))
        self.assertEqual([x["video_id"] for x in selected], ["v1", "v2"])

    def test_any_video_without_eligible_qid_fails_all_or_none(self):
        census = self.fixture()
        census["videos"][1]["eligible_qids"] = []
        with self.assertRaisesRegex(ValueError, "v3_all_or_none_no_feasible_qid"):
            select_v3_qids(census)

    def test_plan_preserves_exact_video_partition_and_marks_selection_conditioning(self):
        census = self.fixture()
        plan = {"calibration": [{"video_id": "cal"}],
                "evaluation": [{"video_id": "v1"}, {"video_id": "v2"}]}
        result = make_private_plan(plan_v2=plan, census=census,
            source_sha256={"source": "1" * 64}, data_sha256={"test": "2" * 64},
            v2_plan_sha256="3" * 64, preflight_receipt_sha256="4" * 64)
        self.assertEqual([x["video_id"] for x in result["evaluation"]], ["v1", "v2"])
        self.assertEqual(result["counts"]["evaluation_videos"], 2)
        self.assertTrue(result["counts"]["eligibility_conditioned_selection"])
        self.assertIn("not_confirmatory", result["status"])
        self.assertEqual(result["calibration"], plan["calibration"])

    def test_control_must_be_outside_all_annotation_covered_groups(self):
        valid = {"evidence_group_indices": [1], "control_group_indices": [3],
                 "annotation_covered_group_indices": [1, 2]}
        self.assertEqual(validate_candidate_geometry(valid, [0, 1, 2, 3]), (1, 3))
        invalid = {**valid, "control_group_indices": [2]}
        with self.assertRaisesRegex(ValueError, "single_anchor_geometry_contract_invalid"):
            validate_candidate_geometry(invalid, [0, 1, 2, 3])

    def test_census_preserves_video_and_candidate_attempted_completed_denominators(self):
        census = {"videos": [
            {"video_id": "v1", "valid_qid_count": 2,
             "candidate_results": [{"qid": "q1", "status": "eligible"},
                                   {"qid": "q2", "status": "ineligible"}],
             "eligible_qids": [{"qid": "q1"}]},
        ]}
        counters = {"videos_attempted": 1, "videos_completed": 1,
                    "candidate_qids_attempted": 2, "candidate_qids_completed": 2,
                    "eligible_qids": 1}
        validate_census_structure(census, counters, 1)
        counters["candidate_qids_completed"] = 1
        with self.assertRaisesRegex(ValueError, "census_all_or_none_denominator_invalid"):
            validate_census_structure(census, counters, 1)

    def test_existing_census_or_plan_target_fails_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); os.chmod(root, 0o700)
            ensure_outputs_absent(root, ["census.json", "plan.json"])
            (root / "plan.json").write_text("sentinel")
            with self.assertRaisesRegex(FileExistsError, "private_output_collision"):
                ensure_outputs_absent(root, ["census.json", "plan.json"])
            self.assertFalse((root / "census.json").exists())
            self.assertEqual((root / "plan.json").read_text(), "sentinel")


if __name__ == "__main__":
    unittest.main()
