"""Strict synthetic CPU tests for discrete frame-index pilot helpers."""
import json
import unittest

from PIL import Image

from run_claim_frame_index_mask_pilot import (
    _read_private_json,
    _validate_packet,
    _write_private_json,
    allowed_next_tokens,
    build_contact_sheet,
    complete_triplets,
    interval_indices,
    interval_json_token_sequences,
    matched_control_indices,
    parse_frame_interval,
    validate_smoke_forward_counts,
    LOCALIZER_TEMPLATE,
)


class FrameIndexPilotTests(unittest.TestCase):
    def test_smoke_gate_requires_requested_attempted_and_completed_forwards(self):
        receipt = {"requested_scorer_forwards": 5,
                   "attempted_scorer_forwards": 5,
                   "completed_scorer_forwards": 5}
        validate_smoke_forward_counts(receipt)
        for key in receipt:
            bad = dict(receipt, **{key: 4})
            with self.subTest(key=key), self.assertRaisesRegex(
                    ValueError, "smoke_scorer_forward_counts_invalid"):
                validate_smoke_forward_counts(bad)

    def test_constrained_generation_prefixes_stay_in_finite_grammar(self):
        sequences = [[1, 2], [1, 3, 4], [5]]
        self.assertEqual(allowed_next_tokens(sequences, [], 99), [1, 5])
        self.assertEqual(allowed_next_tokens(sequences, [1], 99), [2, 3])
        self.assertEqual(allowed_next_tokens(sequences, [1, 2], 99), [99])
        self.assertEqual(allowed_next_tokens(sequences, [1, 3], 99), [4])
        self.assertEqual(allowed_next_tokens(sequences, [8], 99), [])

    def test_interval_json_grammar_covers_exact_valid_span_count(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                self.assert_false_special = add_special_tokens
                return [len(text), ord(text[0])]
        tokenizer = Tokenizer()
        sequences = interval_json_token_sequences(tokenizer)
        self.assertEqual(len(sequences), 58)
        self.assertTrue(all(len(sequence) == 2 for sequence in sequences))
        self.assertFalse(tokenizer.assert_false_special)

    def test_localizer_template_formats_literal_json_example(self):
        prompt = LOCALIZER_TEMPLATE.format(question="Q", claim="C")
        self.assertIn('{"start_frame_index": 0, "end_frame_index": 0}', prompt)
        self.assertIn("Question: Q", prompt)
        self.assertIn("Candidate claim: C", prompt)

    @staticmethod
    def packet():
        return {
            "status": "passed",
            "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
            "records": [{
                "generated_parsed": {"valid": True, "claim": f"candidate {i}"},
                "question": "synthetic question", "choices": ["a", "b", "c", "d", "e"],
                "gold_answer": "a", "video_sha256": f"{i:064x}",
                "local_video_path": f"/private/synthetic-{i}.mp4",
            } for i in range(3)],
        }

    def test_packet_contract_and_unique_source_binding(self):
        packet = self.packet()
        self.assertEqual(len(_validate_packet(packet, "smoke")), 1)
        self.assertEqual(len(_validate_packet(packet, "full")), 3)
        packet["records"][2]["video_sha256"] = packet["records"][1]["video_sha256"]
        with self.assertRaisesRegex(ValueError, "source_videos_not_unique"):
            _validate_packet(packet, "smoke")

    def test_private_io_requires_mode_0600_and_never_overwrites(self):
        import os
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            _write_private_json(path, {"status": "passed"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            value, digest = _read_private_json(path)
            self.assertEqual(value, {"status": "passed"})
            self.assertEqual(len(digest), 64)
            with self.assertRaises(FileExistsError):
                _write_private_json(path, {"status": "changed"})
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "private_input_permissions_invalid"):
                _read_private_json(path)

    def test_parser_accepts_only_exact_bounded_inclusive_json(self):
        self.assertEqual(parse_frame_interval('{"start_frame_index":3,"end_frame_index":6}'), (3, 6))
        self.assertEqual(parse_frame_interval('```json\n{"start_frame_index":1,"end_frame_index":1}\n```'), (1, 1))
        self.assertEqual(parse_frame_interval('Selected range: {"start_frame_index":2,"end_frame_index":3}.'), (2, 3))
        for text in (
            '{"start_frame_index":1,"end_frame_index":1} and {"start_frame_index":2,"end_frame_index":2}',
            '{"start_frame_index":1,"start_frame_index":1,"end_frame_index":1}',
            '{"start_frame_index":true,"end_frame_index":1}',
            '{"start_frame_index":0,"end_frame_index":4}',
            '{"start_frame_index":-1,"end_frame_index":0}',
            '{"start_frame_index":15,"end_frame_index":16}',
            '{"start_frame_index":4.0,"end_frame_index":4}',
            '{"start_frame_index":2,"end_frame_index":1}',
            '{"start_frame_index":2,"end_frame_index":2,"extra":0}',
        ):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_frame_interval(text)

    def test_interval_and_nearest_rank_control_geometry(self):
        self.assertEqual(interval_indices((5, 7)), [5, 6, 7])
        controls = matched_control_indices(5, 7)
        self.assertEqual(len(controls), 3)
        self.assertEqual(len(set(controls)), 3)
        self.assertFalse(set(controls) & {5, 6, 7})
        self.assertEqual(controls, matched_control_indices(5, 7))
        with self.assertRaises(ValueError):
            interval_indices((0, 4))

    def test_contact_sheet_is_fixed_4_by_4_and_rejects_wrong_count(self):
        frames = [Image.new("RGB", (32, 24), (i, 20, 30)) for i in range(16)]
        one, two = build_contact_sheet(frames), build_contact_sheet(frames)
        self.assertEqual(one.size, (1536, 1024))
        self.assertEqual(one.tobytes(), two.tobytes())
        with self.assertRaises(ValueError):
            build_contact_sheet(frames[:-1])

    def test_triplet_aggregation_is_all_or_none(self):
        conditions = {"full": 0.4, "claim_interval_black": 0.1,
                      "matched_control_black": 0.3,
                      "claim_interval_blur": 0.2, "matched_control_blur": 0.35}
        rows = [{"row_index": i, "margins": dict(conditions)} for i in range(3)]
        summary = complete_triplets(rows, 3)
        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(summary["mean_control_minus_claim_black"], 0.2)
        self.assertAlmostEqual(summary["mean_control_minus_claim_blur"], 0.15)
        for bad_rows, expected in ((rows[:2], 3),
                                   ([dict(rows[0], margins={"full": 0.1})], 1),
                                   ([{"row_index": 1, "margins": conditions}], 1),
                                   ([{"row_index": 0, "margins": dict(conditions, full=float("nan"))}], 1)):
            with self.subTest(bad_rows=bad_rows, expected=expected), self.assertRaises(ValueError):
                complete_triplets(bad_rows, expected)


if __name__ == "__main__":
    unittest.main()
