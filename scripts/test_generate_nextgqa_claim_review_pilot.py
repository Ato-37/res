import unittest

from generate_nextgqa_claim_review_pilot import parse_claim_output


class ClaimOutputParserTests(unittest.TestCase):
    def test_parses_fixed_labeled_claim_and_frame_indices(self):
        parsed = parse_claim_output(
            "A=B\nC=A person opens the cabinet.\nF=2, 5, 9", 16)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["answer_letter"], "B")
        self.assertEqual(parsed["evidence_frame_indices"], [2, 5, 9])

    def test_rejects_missing_or_out_of_range_frame_indices(self):
        parsed = parse_claim_output(
            "ANSWER_LETTER: A\nCLAIM: A car turns.\n"
            "FRAMES: 2, 16", 16)
        self.assertFalse(parsed["valid"])

    def test_preserves_unparseable_generation_as_invalid(self):
        parsed = parse_claim_output("A markdown answer without labels", 16)
        self.assertFalse(parsed["valid"])
        self.assertIsNone(parsed["claim"])


if __name__ == "__main__":
    unittest.main()
