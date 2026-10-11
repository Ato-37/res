from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

from audit_clevrer_humans_annotations import (
    audit,
    causal_edge_pairing_summary,
    normalize_scene_id,
    scan_time_keys,
    summarize_ceg,
)


class TinyGraph:
    def nodes(self, data: bool = False):
        assert data
        return [
            ("The ball moves at frame 12", {"description": "private text", "kind": "event", "start_frame": 3}),
            ("Cube hit at 00:04", {}),
        ]

    def edges(self, data: bool = False):
        assert data
        return [("event-1", "event-2", {"relation": "causes", "confidence": 1.0})]


class EdgeGraph:
    def __init__(self, edges):
        self._edges = edges

    def edges(self, data: bool = False):
        assert data
        return self._edges


class AnnotationAuditTests(unittest.TestCase):
    def test_causal_pairing_counts_shared_targets_without_exposing_labels(self) -> None:
        payload = {
            "00001": {
                "CEG_pos": EdgeGraph([
                    ("cause-private-a", "target-private", {}),
                    ("cause-private-b", "target-private", {}),
                    ("cause-private-c", "other-target", {}),
                ]),
                "CEG_neg": EdgeGraph([
                    ("control-private-a", "target-private", {}),
                    ("control-private-b", "other-target", {}),
                    ("control-private-c", "negative-only-target", {}),
                ]),
            }
        }

        result = causal_edge_pairing_summary(payload)
        encoded = json.dumps(result)

        self.assertEqual(result["scenes_analyzed"], 1)
        self.assertEqual(result["scenes_with_positive_and_negative_edges"], 1)
        self.assertEqual(result["scenes_with_a_target_having_both_edge_types"], 1)
        self.assertEqual(result["targets_with_both_edge_types"], 2)
        self.assertEqual(
            result["targets_with_exactly_one_positive_and_one_negative_incoming_edge"], 1
        )
        self.assertEqual(
            result[
                "scenes_with_target_having_exactly_one_positive_and_one_negative_incoming_edge"
            ],
            1,
        )
        self.assertEqual(result["same_target_positive_negative_pair_combinations"], 3)
        self.assertNotIn("target-private", encoded)
        self.assertNotIn("cause-private", encoded)

    def test_graph_schema_summary_counts_keys_and_types_without_values(self) -> None:
        result = summarize_ceg({"00001": {"CEG_full": TinyGraph()}})
        encoded = json.dumps(result)

        self.assertEqual(
            result["graph_attribute_key_occurrences"]["node"],
            {"description": 1, "kind": 1, "start_frame": 1},
        )
        self.assertEqual(
            result["graph_attribute_value_type_counts"]["node"]["description"],
            {"str": 1},
        )
        self.assertEqual(result["time_key_counts"], {"start_frame": 1})
        self.assertNotIn("private text", encoded)
        self.assertNotIn("The ball moves", encoded)
        self.assertNotIn("causes", encoded)
        self.assertEqual(
            result["graph_node_label_audit_by_field"]["CEG_full"][
                "explicit_time_locator_pattern_counts"
            ],
            {"clock_timestamp": 1, "frame_number": 1},
        )

    def test_normalizes_unpadded_clevrer_scene_ids(self) -> None:
        self.assertEqual(normalize_scene_id("3"), "00003")
        self.assertEqual(normalize_scene_id("10002"), "10002")
        self.assertEqual(normalize_scene_id("not-a-scene"), "not-a-scene")

    def test_time_key_scan_handles_nested_and_cyclic_containers(self) -> None:
        cyclic: list[object] = []
        payload = {"event": {"start": 4, "end_frame": 12}, "cycle": cyclic}
        cyclic.append(payload)

        result = scan_time_keys(payload)

        self.assertEqual(result["time_key_counts"], {"end_frame": 1, "start": 1})
        self.assertFalse(result["scan_truncated"])

    def test_audit_reports_json_and_official_pickle_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for filename in ("causal_cloze.json", "train_question.json", "valid_question.json"):
                (root / filename).write_text(json.dumps([{"question": "q", "answer": "a"}]))
            for filename in ("train_ceg_data.p", "valid_ceg_data.p"):
                with (root / filename).open("wb") as target:
                    pickle.dump({"00001": {"video_filename": "video_00001.mp4", "events": []}}, target)

            result = audit(root, trust_pickle=True)

            self.assertEqual(result["files"]["valid_question.json"]["record_count"], 1)
            self.assertEqual(result["files"]["valid_ceg_data.p"]["scene_count"], 1)
            self.assertEqual(result["files"]["valid_ceg_data.p"]["video_filename_count"], 1)
            self.assertEqual(result["report_schema_version"], 7)
            self.assertEqual(
                result["causal_edge_pairing_by_split"]["valid"]["scenes_analyzed"], 0
            )
            self.assertEqual(result["annotation_dir_name"], root.name)
            self.assertNotIn(str(root), json.dumps(result))

    def test_audit_reports_question_counts_and_split_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "causal_cloze.json").write_text(json.dumps({"1": {"questions": []}}))
            (root / "train_question.json").write_text(
                json.dumps({"1": {"questions": [{"question": "q", "choices": [
                                {"choice": "a", "answer": "correct"},
                                {"choice": "b", "answer": "wrong"},
                            ]}]},
                            "00002": {"questions": []}})
            )
            (root / "valid_question.json").write_text(
                json.dumps({"10001": {"questions": [{"question": "q"}]}})
            )
            for filename, payload in (
                ("train_ceg_data.p", {"00001": {"video_filename": "video_00001.mp4"}}),
                ("valid_ceg_data.p", {"10001": {"video_filename": "video_10001.mp4"}}),
            ):
                with (root / filename).open("wb") as target:
                    pickle.dump(payload, target)

            result = audit(root, trust_pickle=True)

            self.assertEqual(
                result["files"]["train_question.json"]["nested_question_count"], 1
            )
            self.assertEqual(result["files"]["train_question.json"]["nested_choice_count"], 2)
            self.assertEqual(
                result["files"]["train_question.json"]["choice_answer_labels"],
                {"correct": 1, "wrong": 1},
            )
            self.assertEqual(result["cross_file_alignment"]["train"]["matched_scene_count"], 1)
            self.assertEqual(
                result["cross_file_alignment"]["train"]["question_scene_without_graph_count"], 1
            )
            self.assertEqual(
                result["cross_file_alignment"]["causal_cloze"]["matched_train_or_valid_graph_count"],
                1,
            )

    def test_audit_rejects_missing_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, "Missing expected"):
                audit(Path(temp_dir), trust_pickle=False)


if __name__ == "__main__":
    unittest.main()
