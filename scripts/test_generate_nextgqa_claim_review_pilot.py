import unittest

from generate_nextgqa_claim_review_pilot import parse_claim_output


class ClaimOutputParserTests(unittest.TestCase):
    def test_parses_one_plain_claim_line_without_model_frame_selection(self):
        parsed = parse_claim_output("CLAIM=A person opens the cabinet.", 16)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["claim"], "A person opens the cabinet.")
        self.assertIsNone(parsed["answer_letter"])
        self.assertEqual(parsed["evidence_frame_indices"], [])

    def test_rejects_multiline_or_extra_output(self):
        for text in ("CLAIM=A person opens the cabinet.\nF=1,2",
                     "CLAIM=\nA person opens the cabinet."):
            parsed = parse_claim_output(text, 16)
            self.assertFalse(parsed["valid"])

    def test_rejects_template_and_special_token_artifacts(self):
        for text in ("CLAIM=A addCriterion fragment", "CLAIM=<|assistant|>event"):
            parsed = parse_claim_output(text, 16)
            self.assertFalse(parsed["valid"])
            self.assertEqual(parsed["claim_text_quality_failure"],
                             "claim_template_or_token_artifact")

    def test_empty_or_unlabeled_output_is_invalid(self):
        self.assertFalse(parse_claim_output("", 16)["valid"])
        self.assertTrue(parse_claim_output("A person opens the cabinet.", 16)["valid"])


if __name__ == "__main__":
    unittest.main()
