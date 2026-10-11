"""Privacy and accounting tests for the CLEVRER counterfactual schema audit."""

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_clevrer_counterfactual_programs import (  # noqa: E402
    audit_payload,
    discover_json_files,
    format_summary,
)


class ClevrerCounterfactualAuditTests(unittest.TestCase):
    def test_counts_counterfactual_question_and_choice_program_coverage(self):
        payload = [{
            "scene_index": 7,
            "questions": [{
                "question": "private question text",
                "question_type": "counterfactual",
                "answer": "private answer",
                "choices": [
                    {"text": "private choice", "program": [{"op": "filter"}], "answer": True},
                    {"text": "other private choice", "program": [{"op": "query"}], "answer": False},
                ],
            }, {
                "question": "another private question",
                "question_type": "descriptive",
                "choices": ["not structured"],
            }],
        }]
        report = audit_payload(payload)
        self.assertEqual(report["question_count"], 2)
        self.assertEqual(report["counterfactual_count_by_type_field"], 1)
        self.assertEqual(report["counterfactual_with_answer"], 1)
        self.assertEqual(report["counterfactual_with_program"], 1)
        self.assertEqual(report["choice_count"], 3)
        self.assertEqual(report["choice_with_answer"], 2)
        self.assertEqual(report["choice_with_program"], 2)
        self.assertEqual(report["question_with_scene_link"], 2)

    def test_summary_never_prints_question_answer_or_program_content(self):
        path = Path("/private/machine/CLEVRER/train.json")
        report = audit_payload([{
            "question": "SECRET_QUESTION",
            "question_type": "counterfactual",
            "answer": "SECRET_ANSWER",
            "program": ["SECRET_PROGRAM"],
        }])
        rendered = format_summary([(path, report)])
        self.assertIn("train.json", rendered)
        self.assertIn("counterfactual-type=1", rendered)
        for secret in ("SECRET_QUESTION", "SECRET_ANSWER", "SECRET_PROGRAM", "/private/"):
            self.assertNotIn(secret, rendered)

    def test_counts_same_scene_counterfactual_pairability_without_exposing_programs(self):
        payload = [{
            "scene_index": 1,
            "questions": [{
                "question": "private cf one",
                "question_type": "counterfactual",
                "choices": [{"program": [{"op": "remove", "object": 1}]}],
            }, {
                "question": "private cf two",
                "question_type": "counterfactual",
                "choices": [{"program": [{"op": "remove", "object": 2}]}],
            }, {
                "question": "private descriptive",
                "question_type": "descriptive",
                "program": [{"op": "query"}],
            }],
        }, {
            "scene_index": 2,
            "questions": [{
                "question": "single private cf",
                "question_type": "counterfactual",
                "program": [{"op": "remove", "object": 3}],
            }],
        }]
        report = audit_payload(payload)
        self.assertEqual(report["counterfactual_scene_count"], 2)
        self.assertEqual(report["counterfactual_scenes_with_2plus_questions"], 1)
        self.assertEqual(report["counterfactual_scenes_with_2plus_program_signatures"], 1)
        self.assertEqual(report["max_counterfactual_questions_per_scene"], 2)
        rendered = format_summary([(Path("train.json"), report)])
        self.assertIn("1 with 2+ distinct program signatures", rendered)
        for secret in ("private cf", '"remove"', "object"):
            self.assertNotIn(secret, rendered)

    def test_pairability_is_unknown_for_non_scene_list_payload(self):
        report = audit_payload({"question": "private", "question_type": "counterfactual"})
        self.assertIsNone(report["counterfactual_scene_count"])
        self.assertIsNone(report["counterfactual_scenes_with_2plus_questions"])
        self.assertIsNone(report["counterfactual_scenes_with_2plus_program_signatures"])

    def test_compares_operator_traces_without_literal_values(self):
        payload = [{"questions": [{
            "question": "private cf one",
            "question_type": "counterfactual",
            "program": [{"type": "filter_collision", "value_inputs": [1]},
                        {"type": "query"}],
        }, {
            "question": "private cf two",
            "question_type": "counterfactual",
            "program": [{"type": "filter_collision", "value_inputs": [2]},
                        {"type": "query"}],
        }]}, {"questions": [{
            "question": "private cf three",
            "question_type": "counterfactual",
            "program": [{"type": "filter_collision"}, {"type": "query"}],
        }, {
            "question": "private cf four",
            "question_type": "counterfactual",
            "program": [{"type": "filter_before"}, {"type": "query"}],
        }]}]
        report = audit_payload(payload)
        self.assertEqual(report["paired_scenes_distinct_programs_same_operator_trace"], 1)
        self.assertEqual(report["paired_scenes_distinct_programs_different_operator_trace"], 1)
        self.assertEqual(report["paired_scenes_operator_trace_unavailable"], 0)

    def test_extracts_official_string_token_programs_using_executor_modules(self):
        payload = [{"questions": [{
            "question": "private cf one",
            "question_type": "counterfactual",
            "choices": [{"program": [
                "<START>", "objects", "red", "filter_color", "count", "<END>",
            ]}],
        }, {
            "question": "private cf two",
            "question_type": "counterfactual",
            "choices": [{"program": [
                "<START>", "objects", "blue", "filter_color", "count", "<END>",
            ]}],
        }]}, {"questions": [{
            "question": "private cf three",
            "question_type": "counterfactual",
            "choices": [{"program": ["objects", "filter_collision", "exist"]}],
        }, {
            "question": "private cf four",
            "question_type": "counterfactual",
            "choices": [{"program": ["objects", "filter_before", "exist"]}],
        }]}]
        report = audit_payload(payload)
        self.assertEqual(report["paired_scenes_distinct_programs_same_operator_trace"], 1)
        self.assertEqual(report["paired_scenes_distinct_programs_different_operator_trace"], 1)
        self.assertEqual(report["paired_scenes_operator_trace_unavailable"], 0)

    def test_same_trace_question_text_is_only_reported_as_private_distance_bins(self):
        payload = [{"questions": [{
            "question": "Would the red object collide?",
            "question_type": "counterfactual",
            "choices": [{"program": ["objects", "red", "filter_color", "exist"]}],
        }, {
            "question": "Would the blue object collide?",
            "question_type": "counterfactual",
            "choices": [{"program": ["objects", "blue", "filter_color", "exist"]}],
        }]}]
        report = audit_payload(payload)
        text_audit = report["same_operator_trace_pairs_question_token_edit_distance"]
        self.assertEqual(text_audit["candidate_pairs"], 1)
        self.assertEqual(text_audit["bins"]["1"], 1)
        rendered = str(text_audit)
        self.assertNotIn("red", rendered)
        self.assertNotIn("blue", rendered)

    def test_prefers_choice_programs_over_empty_question_program_placeholder(self):
        payload = [{"questions": [{
            "question": "private cf one",
            "question_type": "counterfactual",
            "program": [],
            "choices": [{"program": [{"type": "remove", "value_inputs": [1]}]}],
        }, {
            "question": "private cf two",
            "question_type": "counterfactual",
            "program": [],
            "choices": [{"program": [{"type": "remove", "value_inputs": [2]}]}],
        }]}]
        report = audit_payload(payload)
        self.assertEqual(report["paired_scenes_distinct_programs_same_operator_trace"], 1)
        self.assertEqual(report["counterfactual_program_schema"][
            "empty_question_program_with_choice_program_count"], 2)

    def test_program_schema_audit_reports_field_types_not_values(self):
        report = audit_payload([{"questions": [{
            "question": "PRIVATE QUESTION",
            "question_type": "counterfactual",
            "choices": [{
                "program": [{"module_name": "PRIVATE OPERATOR", "value_inputs": ["PRIVATE ARG"]}],
            }],
        }]}])
        schema = report["counterfactual_program_schema"]
        self.assertEqual(schema["choice_level"]["root_type_counts"], {"list": 1})
        self.assertEqual(schema["choice_level"]["field_type_counts"]["module_name:str"], 1)
        self.assertEqual(schema["choice_level"]["field_type_counts"]["value_inputs:list"], 1)
        self.assertEqual(schema["choice_level"]["dict_node_count"], 1)
        self.assertNotIn("PRIVATE OPERATOR", str(schema))
        self.assertNotIn("PRIVATE ARG", str(schema))

    def test_discovers_only_named_json_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "questions").mkdir()
            (root / "questions" / "train.json").write_text("[]", encoding="utf-8")
            (root / "questions" / "unrelated.json").write_text("[]", encoding="utf-8")
            matches = discover_json_files([root])
            self.assertEqual([path.name for path in matches], ["train.json"])


if __name__ == "__main__":
    unittest.main()
