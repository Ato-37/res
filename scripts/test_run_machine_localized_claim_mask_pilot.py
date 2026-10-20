import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_machine_localized_claim_mask_pilot import (
    _endpoint_record_view, _require_model_load_resources, _exact_selected_window_interval,
    parse_localized_interval, parse_single_embedded_interval,
    MIN_FREE_VRAM, MIN_HOST_AVAILABLE,
)


class LocalizedIntervalTests(unittest.TestCase):
    def test_selected_window_uses_exact_rational_pts_not_rounded_decimal_bounds(self):
        from fractions import Fraction
        from presentation_time_utils import assign_evidence_control_frames_from_times
        times=[Fraction(i,3) for i in range(32)]
        selected={"selected_sampled_frame_indices":list(range(8,16)),
                  "sampled_pts_seconds":[float(x) for x in times],
                  "decoded_pts_sha256":"pts-hash"}
        decoded={"decoded_pts_sequence_sha256":"pts-hash",
                 "sampled_frames":[{"relative_presentation_time":[x.numerator,x.denominator]} for x in times]}
        interval=_exact_selected_window_interval(selected,decoded)
        full=[Fraction(i,3) for i in range(32)]
        assignment=assign_evidence_control_frames_from_times(32,32,full,[interval])
        self.assertEqual(len(assignment["evidence_indices"]),8)
        self.assertEqual(assignment["evidence_indices"],list(range(8,16)))

    def test_selected_window_rejects_pts_hash_or_display_value_mismatch(self):
        selected={"selected_sampled_frame_indices":list(range(8)),
                  "sampled_pts_seconds":[i/3 for i in range(32)],
                  "decoded_pts_sha256":"expected"}
        decoded={"decoded_pts_sequence_sha256":"other",
                 "sampled_frames":[{"relative_presentation_time":[i,3]} for i in range(32)]}
        with self.assertRaisesRegex(ValueError,"selected_window_pts_binding_invalid"):
            _exact_selected_window_interval(selected,decoded)

    def test_accepts_exact_json_and_json_fence(self):
        self.assertEqual(parse_localized_interval('{"start_seconds":0.25,"end_seconds":1.5}', 2),
                         (0.25, 1.5))
        self.assertEqual(parse_localized_interval('```json\n{"start_seconds":0,"end_seconds":2}\n```', 2),
                         (0.0, 2.0))

    def test_rejects_extra_keys_and_non_json_text(self):
        for value in ('start 0 end 1', '{"start_seconds":0,"end_seconds":1,"comment":"x"}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_localized_interval(value, 2)

    def test_rejects_duplicate_json_keys_in_direct_and_embedded_paths(self):
        duplicate = '{"start_seconds":0,"start_seconds":0.5,"end_seconds":1}'
        with self.assertRaisesRegex(ValueError, "localization_duplicate_json_key"):
            parse_localized_interval(duplicate, 2)
        with self.assertRaises(ValueError):
            parse_single_embedded_interval("Final: " + duplicate, 2)

    def test_rejects_reversed_empty_nonfinite_and_out_of_bounds(self):
        for value in (
            '{"start_seconds":1,"end_seconds":0}',
            '{"start_seconds":1,"end_seconds":1}',
            '{"start_seconds":0,"end_seconds":3}',
            '{"start_seconds":NaN,"end_seconds":1}',
            '{"start_seconds":true,"end_seconds":1}',
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_localized_interval(value, 2)

    def test_rejects_invalid_duration(self):
        with self.assertRaises(ValueError):
            parse_localized_interval('{"start_seconds":0,"end_seconds":1}', float('nan'))

    def test_embedded_parser_accepts_one_exact_interval_object(self):
        response = 'Reasoning omitted. Final answer: {"start_seconds":0.5,"end_seconds":1.25}'
        self.assertEqual(parse_single_embedded_interval(response, 2), (0.5, 1.25))

    def test_embedded_parser_safely_recovers_python_style_mapping(self):
        response = "Final: {'start_seconds': 0.5, 'end_seconds': 1.25,}"
        self.assertEqual(parse_single_embedded_interval(response, 2), (0.5, 1.25))

    def test_embedded_parser_safely_recovers_exact_unquoted_numeric_mapping(self):
        response = "Final: {start_seconds: 0.5, end_seconds: 1.25}"
        self.assertEqual(parse_single_embedded_interval(response, 2), (0.5, 1.25))

    def test_embedded_parser_rejects_ambiguous_or_truncated_objects(self):
        for response in (
            '{"start_seconds":0,"end_seconds":1} {"start_seconds":1,"end_seconds":2}',
            'Result: {"start_seconds":0,"end_seconds":1',
            "{'start_seconds':0,'start_seconds':0.5,'end_seconds':1}",
            '{start_seconds:0,end_seconds:1,comment:"extra"}',
            'I cannot identify a supported interval: {"start_seconds":0,"end_seconds":1}',
            'No interval available',
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_single_embedded_interval(response, 2)

    def test_model_load_resource_floor_checks_each_resource(self):
        _require_model_load_resources(MIN_FREE_VRAM, MIN_HOST_AVAILABLE)
        for free, host in ((MIN_FREE_VRAM - 1, MIN_HOST_AVAILABLE),
                           (MIN_FREE_VRAM, MIN_HOST_AVAILABLE - 1)):
            with self.subTest(free=free, host=host), self.assertRaisesRegex(
                    ValueError, "pre_model_resource_guard_failed"):
                _require_model_load_resources(free, host)

    def test_cross_scale_receipt_names_candidate_interval_without_claim_evidence_alias(self):
        rows = [{"choice_log_probabilities_nats": {
            "full": [0.0], "claim_evidence_black": [-1.0],
            "temporal_control_black": [-0.5]},
            "gold_vs_best_foil_margins_nats": {
                "full": 2.0, "claim_evidence_black": 1.0,
                "temporal_control_black": 1.5},
            "best_foil_index_by_condition": {
                "full": 0, "claim_evidence_black": 1,
                "temporal_control_black": 1},
            "d_control_minus_claim_mask_nats": 0.5}]
        view, label, metric = _endpoint_record_view(rows, "independent_3b_localized")
        self.assertEqual(label, "candidate_interval_3b_black")
        self.assertEqual(metric, "D3_control_minus_3b_candidate_interval_nats")
        self.assertEqual(view[0]["gold_vs_best_foil_margins_nats"][label], 1.0)
        self.assertNotIn("claim_evidence_black", view[0]["gold_vs_best_foil_margins_nats"])
        self.assertEqual(view[0]["d_control_minus_3b_candidate_interval_nats"], 0.5)


if __name__ == "__main__":
    unittest.main()
