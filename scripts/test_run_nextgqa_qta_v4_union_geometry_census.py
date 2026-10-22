import unittest

from run_nextgqa_qta_v4_union_geometry_census import (
    build_v4_plan, processor_geometry_provenance, selection_hash, validate_census,
    validate_full_output_names, validate_union_geometry, CENSUS_NAME, PLAN_NAME,
)
from run_nextgqa_qta_v3_geometry_census import ensure_outputs_absent
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch


class QtaV4UnionCensusTests(unittest.TestCase):
    def test_full_run_uses_new_version_and_refuses_stale_or_colliding_outputs(self):
        validate_full_output_names(CENSUS_NAME, PLAN_NAME)
        with self.assertRaisesRegex(ValueError, "full_census_requires_exact_v4_census_and_plan_names"):
            validate_full_output_names(CENSUS_NAME.replace("20261018_03", "20261018_01"),
                                       PLAN_NAME.replace("20261018_03", "20261018_01"))
        with TemporaryDirectory() as td:
            root = Path(td); (root / CENSUS_NAME).write_text("occupied")
            with self.assertRaises(FileExistsError):
                ensure_outputs_absent(root, [CENSUS_NAME, PLAN_NAME])

    def test_union_geometry_requires_exact_annotation_union_and_explicit_no_control(self):
        mapped = {"evidence_group_indices": [1, 2],
                  "annotation_covered_group_indices": [1, 2],
                  "control_group_indices": []}
        self.assertEqual(validate_union_geometry(mapped, [0, 1, 2, 3]), [1, 2])
        for bad in (
            {**mapped, "control_group_indices": [0]},
            {**mapped, "evidence_group_indices": [1]},
            {**mapped, "evidence_group_indices": [1, 1],
             "annotation_covered_group_indices": [1, 1]},
            {**mapped, "annotation_covered_group_indices": []},
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(
                    ValueError, "annotation_union_geometry_contract_invalid"):
                validate_union_geometry(bad, [0, 1, 2, 3])

    def test_census_requires_exact_video_and_candidate_denominators(self):
        census = {"videos": [{"video_id": "v1", "valid_qid_count": 2,
            "candidate_results": [{"qid": "q1", "status": "eligible"},
                                  {"qid": "q2", "status": "ineligible"}],
            "eligible_qids": [{"qid": "q1"}]}]}
        counters = {"videos_attempted": 1, "videos_completed": 1,
            "candidate_qids_attempted": 2, "candidate_qids_completed": 2,
            "eligible_qids": 1}
        validate_census(census, counters, 1)
        counters["candidate_qids_completed"] = 1
        with self.assertRaisesRegex(ValueError, "v4_census_all_or_none_denominator_invalid"):
            validate_census(census, counters, 1)

    def test_v4_plan_keeps_all_videos_and_freezes_estimand_before_outcomes(self):
        census = {"videos": [
            {"video_id": "v1", "valid_qid_count": 2, "eligible_qids": [
                {"qid": "q1", "geometry_sha256": "a"*64, "row_binding_sha256": "b"*64,
                 "temporal_group_count": 32, "covered_group_count": 3,
                 "covered_fraction": 3/32, "full_coverage": False},
                {"qid": "q2", "geometry_sha256": "c"*64, "row_binding_sha256": "d"*64,
                 "temporal_group_count": 32, "covered_group_count": 32,
                 "covered_fraction": 1.0, "full_coverage": True}]},
            {"video_id": "v2", "valid_qid_count": 1, "eligible_qids": [
                {"qid": "q3", "geometry_sha256": "e"*64, "row_binding_sha256": "f"*64,
                 "temporal_group_count": 32, "covered_group_count": 1,
                 "covered_fraction": 1/32, "full_coverage": False}]},
        ]}
        base = {"calibration": [{"video_id": "cal"}],
                "evaluation": [{"video_id": "v1"}, {"video_id": "v2"}]}
        gate = {"plan_sha256": "1"*64, "preflight_receipt_sha256": "2"*64}
        plan = build_v4_plan(base_plan=base, census=census,
            source_hashes={"source": "3"*64}, data_hashes={"data": "4"*64},
            input_gate=gate, processor_manifest={"tokenizer_sha256": "5"*64},
            library_versions={"torch": "test"}, expected_n=2)
        self.assertEqual([x["video_id"] for x in plan["evaluation"]], ["v1", "v2"])
        expected = min(("q1", "q2"), key=lambda q: selection_hash("v1", q))
        self.assertEqual(plan["evaluation"][0]["qid"], expected)
        row = plan["evaluation"][0]
        if expected == "q1":
            self.assertEqual(row["annotation_union_covered_group_count"], 3)
            self.assertFalse(row["annotation_union_full_coverage"])
        else:
            self.assertEqual(row["annotation_union_covered_group_count"], 32)
            self.assertTrue(row["annotation_union_full_coverage"])
        self.assertEqual(plan["estimand"]["primary"], "delta_union_dependence=D_intervention-D_baseline")
        self.assertEqual(len(plan["estimand"]["conditions"]), 4)
        census["videos"][1]["eligible_qids"] = []
        with self.assertRaisesRegex(ValueError, "v4_all_or_none_no_feasible_qid"):
            build_v4_plan(base_plan=base, census=census,
                source_hashes={}, data_hashes={}, input_gate=gate,
                processor_manifest={}, library_versions={}, expected_n=2)

    def test_processor_provenance_binds_separate_video_processor_config(self):
        class Video:
            def to_dict(self): return {"fps": 2, "sampling": "uniform"}
        class Processor:
            video_processor = Video()
        base = {"processor_fingerprint_sha256": "a" * 64}
        with patch("run_nextgqa_qta_v4_union_geometry_census.kernel.processor_provenance",
                   return_value=base):
            one = processor_geometry_provenance(Processor(), None)
            Processor.video_processor = type("Video2", (), {
                "to_dict": lambda self: {"fps": 4, "sampling": "uniform"}})()
            two = processor_geometry_provenance(Processor(), None)
        self.assertNotEqual(one["video_processor_config_sha256"],
                            two["video_processor_config_sha256"])
        self.assertNotEqual(one["processor_fingerprint_sha256"],
                            two["processor_fingerprint_sha256"])


if __name__ == "__main__":
    unittest.main()
