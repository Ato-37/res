import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_nextgqa_qta_annotation_proxy import analyze_receipt


def valid_receipt(repo: Path) -> dict:
    source = repo / "source.py"
    source.write_text("immutable source\n", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    names = (
        "baseline_control_ablation", "baseline_evidence_ablation", "baseline_full",
        "intervention_control_ablation", "intervention_evidence_ablation", "intervention_full",
    )
    rows = []
    for ordinal, delta in enumerate((-0.1, 0.2), start=1):
        rows.append({
            "ordinal": ordinal, "temporal_group_count": 4,
            "annotation_covered_group_indices": [0, 1],
            "annotation_covered_group_count": 2,
            "representative_annotation_group_index": 0,
            "evidence_group_indices": [0], "control_group_indices": [2],
            "evidence_group_count": 1, "control_group_count": 1,
            "selector_evidence_group_overlap": 1,
            "selector_control_group_overlap": 1,
            "delta_f": delta, "baseline_full_correct": int(ordinal == 1),
            "answer_index": 0, "foil_index": 1,
            "margin_by_condition": {
                "baseline_full": 1.0, "baseline_evidence_ablation": 0.0,
                "baseline_control_ablation": 0.0,
                "intervention_full": 1.0, "intervention_evidence_ablation": 0.0,
                "intervention_control_ablation": delta,
            },
            "representative_annotation_overlap_fraction": 0.5,
            "representative_group_position_fraction": 0.25,
        })
    return {
        "status": "completed_exploratory_single_anchor_finite_cohort",
        "scope": "NExT-GQA annotation-interval dependence proxy; not explanation faithfulness",
        "evidence_selection_policy": "single_overlap_anchor",
        "inference": "descriptive_only_no_p_value_or_population_CI",
        "requested_n": 2, "effective_n": 2, "failures": 0,
        "preflight_requested": 2, "preflight_attempted": 2, "preflight_completed": 2,
        "requested_forwards": 12, "attempted_forwards": 12, "completed_forwards": 12,
        "branch_counts": {name: {key: 2 for key in ("requested", "attempted", "completed", "effective")}
                          for name in names},
        "records": rows,
        "source_sha256": {"source.py": source_hash},
        "finite_cohort_delta_f": {"n": 2, "mean": 0.05, "sample_sd": 0.21213203435596426,
                                   "min": -0.1, "max": 0.2},
        "finite_cohort_baseline_full_restricted_choice_accuracy": {
            "n": 2, "correct": 1, "accuracy": 0.5,
        },
    }


class AnalyzeNextGqaQtaAnnotationProxyTests(unittest.TestCase):
    def test_recomputes_aggregate_from_complete_synthetic_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            summary = analyze_receipt(valid_receipt(repo), repo)
        self.assertEqual(summary["effective_n"], 2)
        self.assertEqual(summary["delta_f_descriptive_nat"]["positive"], 1)
        self.assertEqual(summary["delta_f_descriptive_nat"]["negative"], 1)
        self.assertEqual(summary["baseline_full_accuracy"]["correct"], 1)

    def test_rejects_control_inside_annotation_union(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            receipt = valid_receipt(repo)
            receipt["records"][0]["control_group_indices"] = [1]
            with self.assertRaisesRegex(ValueError, "annotation_anchor_geometry_invalid"):
                analyze_receipt(receipt, repo)

    def test_rejects_incomplete_forward_count(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            receipt = valid_receipt(repo)
            receipt["completed_forwards"] -= 1
            with self.assertRaisesRegex(ValueError, "all_or_none_denominator_or_forward_count_invalid"):
                analyze_receipt(receipt, repo)

    def test_rejects_source_hash_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            receipt = valid_receipt(repo)
            (repo / "source.py").write_text("changed source\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_manifest_hash_mismatch"):
                analyze_receipt(receipt, repo)

    def test_rejects_delta_f_not_reproduced_by_frozen_margin_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            receipt = valid_receipt(repo)
            receipt["records"][0]["margin_by_condition"]["intervention_control_ablation"] = 7.0
            with self.assertRaisesRegex(ValueError, "delta_f_margin_recomputation_mismatch"):
                analyze_receipt(receipt, repo)


if __name__ == "__main__":
    unittest.main()
