import sys
import unittest

sys.path.insert(0, "scripts")
from run_head_gold_answer_query_exploration import encode_gold_answer_ids


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        if text == "unencodable":
            return []
        return [1, 2]


class GoldAnswerQueryTests(unittest.TestCase):
    def test_nonempty_answer_is_tokenized_without_special_tokens(self):
        self.assertEqual(encode_gold_answer_ids({"answer": "yes"}, Tokenizer()), [1, 2])

    def test_missing_or_blank_answer_is_rejected(self):
        for row in ({}, {"answer": "  "}, {"answer": None}):
            with self.subTest(row=row), self.assertRaisesRegex(
                    ValueError, "gold_answer_missing_or_empty"):
                encode_gold_answer_ids(row, Tokenizer())

    def test_empty_tokenization_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gold_answer_tokenization_empty"):
            encode_gold_answer_ids({"answer": "unencodable"}, Tokenizer())


if __name__ == "__main__":
    unittest.main()
