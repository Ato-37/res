from __future__ import annotations

import unittest

from run_nextgqa_qta_v4_mask_specificity import (
    ALL_BRANCHES, EXPECTED_N, _choice_logps, initial_branch_counts,
    validate_preflight, validate_preflight_bindings, validate_runtime_preflight_row,
    validate_smoke_receipt,
)


def fixture_rows():
    rows, items, old_rows = [], [], []
    for i in range(EXPECTED_N):
        video_id, qid = f"video-{i}", f"qid-{i}"
        binding = f"{i + 1:064x}"
        rows.append({
            "video_id": video_id, "qid": qid, "geometry_sha256": "a" * 64,
            "row_binding_sha256": binding, "group_count": 3,
            "annotation_group_indices": [1], "group_token_counts": [2, 2, 2],
            "annotation_position_count": 2, "annotation_position_sha256": "b" * 64,
            "eligible_offsets": [-1, 1], "chosen_offset": -1,
            "placebo_group_indices": [0], "placebo_position_count": 2,
            "placebo_position_sha256": "c" * 64, "eligible": True,
        })
        items.append({"video_id": video_id, "qid": qid,
                      "planned": {"selected_row_binding_sha256": binding}})
        old_rows.append({"video_id": video_id, "qid": qid,
                         "row_binding_sha256": binding})
    return rows, items, old_rows


class SpecificityRunnerContractTests(unittest.TestCase):
    def test_choice_logps_rejects_invalid_ids_before_indexing(self):
        import torch
        output = {"logits": torch.zeros((1, 1, 8))}
        for ids in ((-1, 1, 2, 3, 4), (True, 1, 2, 3, 4),
                    (1, 1, 2, 3, 4), (0, 1, 2, 3), (0, 1, 2, 3, 8)):
            with self.subTest(ids=ids):
                with self.assertRaisesRegex(ValueError, "specificity_answer_token"):
                    _choice_logps(output, ids, torch)

    def test_choice_logps_accepts_scored_single_position_logits(self):
        import torch
        logits = torch.tensor([[[0.0, 0.0, 10.0, 9.0, 0.0, 0.0]]])
        values = _choice_logps({"logits": logits}, (0, 1, 2, 3, 4), torch)
        self.assertEqual(len(values), 5)
        self.assertGreater(values[2], values[0])

    def test_preflight_accepts_complete_geometry_and_ordered_bindings(self):
        rows, items, old_rows = fixture_rows()
        saved = {"header": "expected", "rows": rows, "completed_rows": EXPECTED_N,
                 "eligible_specificity_rows": EXPECTED_N,
                 "structurally_unavailable_rows": 0, "run_allowed": True}
        validate_preflight(saved, {"header": "expected"})
        validate_preflight_bindings(rows, items, old_rows)

    def test_preflight_rejects_duplicate_binding_and_fabricated_identity(self):
        rows, items, old_rows = fixture_rows()
        rows[1]["row_binding_sha256"] = rows[0]["row_binding_sha256"]
        with self.assertRaisesRegex(ValueError, "order_or_binding"):
            validate_preflight_bindings(rows, items, old_rows)
        rows, items, old_rows = fixture_rows()
        rows[0]["video_id"] = "fabricated"
        with self.assertRaisesRegex(ValueError, "order_or_binding"):
            validate_preflight_bindings(rows, items, old_rows)

    def test_preflight_rejects_geometry_or_eligibility_tampering(self):
        rows, _, _ = fixture_rows()
        rows[0]["chosen_offset"] = 0
        saved = {"header": "expected", "rows": rows, "completed_rows": EXPECTED_N,
                 "eligible_specificity_rows": EXPECTED_N,
                 "structurally_unavailable_rows": 0, "run_allowed": True}
        with self.assertRaisesRegex(ValueError, "translation_invalid"):
            validate_preflight(saved, {"header": "expected"})

    def test_smoke_rejects_missing_branch_even_if_total_is_claimed_six(self):
        eligible = {"row_binding_sha256": "d" * 64}
        old = {"answer_index": 1, "foil_index": 2}
        counts = {name: {"requested": 1, "attempted": 1, "completed": 1}
                  for name in ALL_BRANCHES}
        smoke = {"status": "specificity_six_forward_smoke_complete",
            "preflight_sha256": "p", "v4_score_sha256": "v", "source_sha256": {},
            "data_sha256": {}, "row_binding_sha256": eligible["row_binding_sha256"],
            "requested_forwards": 6, "attempted_forwards": 6, "completed_forwards": 6,
            "branch_counts": counts, "records": [{"row_binding_sha256": eligible["row_binding_sha256"],
                "answer_index": 1, "foil_index": 2}], "baseline_answer_index": 1,
            "baseline_foil_index": 2, "baseline_full_margin_abs_error": 0.0}
        kwargs = {"preflight_sha256": "p", "old_sha": "v", "source_hashes": {},
                  "data_hashes": {}, "eligible_row": eligible, "old_row": old}
        validate_smoke_receipt(smoke, **kwargs)
        smoke["branch_counts"].pop(ALL_BRANCHES[-1])
        with self.assertRaisesRegex(ValueError, "smoke_receipt_invalid"):
            validate_smoke_receipt(smoke, **kwargs)
        smoke["branch_counts"] = {name: {"requested": 1, "attempted": 1, "completed": 1}
                                  for name in ALL_BRANCHES}
        smoke["branch_counts"][ALL_BRANCHES[0]]["requested"] = True
        with self.assertRaisesRegex(ValueError, "smoke_receipt_invalid"):
            validate_smoke_receipt(smoke, **kwargs)

    def test_requested_counts_exist_before_resource_gates(self):
        smoke = initial_branch_counts("smoke")
        self.assertTrue(all(row["requested"] == 1 for row in smoke.values()))
        full = initial_branch_counts("full", eligible_n=23)
        self.assertTrue(all(full[name]["requested"] == EXPECTED_N
                            for name in ALL_BRANCHES[:4]))
        self.assertTrue(all(full[name]["requested"] == 23
                            for name in ALL_BRANCHES[4:]))

    def test_json_list_preflight_arrays_match_recomputed_tuples(self):
        prepared = {"group_count": 3, "annotation_group_indices": (1,),
            "group_token_counts": (2, 2, 2), "annotation_position_count": 2,
            "annotation_position_sha256": "b" * 64, "eligible_offsets": (-1, 1),
            "chosen_offset": -1, "placebo_group_indices": (0,),
            "placebo_position_count": 2, "placebo_position_sha256": "c" * 64,
            "eligible": True}
        expected = {**prepared, "annotation_group_indices": [1],
            "group_token_counts": [2, 2, 2], "eligible_offsets": [-1, 1],
            "placebo_group_indices": [0]}
        validate_runtime_preflight_row(prepared, expected)
        expected["eligible_offsets"] = [-1]
        with self.assertRaisesRegex(ValueError, "runtime_preflight_mismatch"):
            validate_runtime_preflight_row(prepared, expected)


if __name__ == "__main__":
    unittest.main()
