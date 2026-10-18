from __future__ import annotations

import unittest

from qta_processor_choice_suffix import verify_processor_choice_suffix


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [10, 11] if text == "prompt" else [10, 11, ord(text[-1])]


class ProcessorChoiceSuffixTests(unittest.TestCase):
    def test_accepts_all_five_matching_processor_suffixes(self):
        def process(text):
            prefix = [99, *range(20, 26)]
            return prefix if text == "prompt" else prefix + [ord(text[-1])]

        result = verify_processor_choice_suffix(tokenizer=Tokenizer(),
            rendered_prompt="prompt", process_text=process)
        self.assertEqual(result["processed_suffix_matches"], 5)
        self.assertEqual(result["answer_token_ids"], [ord(x) for x in "ABCDE"])

    def test_rejects_processor_retokenization_or_wrong_suffix(self):
        def process(text):
            if text == "prompt":
                return [99, 20, 21]
            if text.endswith("A"):
                return [99, 20, 22, ord("A")]
            return [99, 20, 21, ord(text[-1])]

        with self.assertRaisesRegex(ValueError, "processor_choice_suffix_mismatch"):
            verify_processor_choice_suffix(tokenizer=Tokenizer(),
                rendered_prompt="prompt", process_text=process)

    def test_rejects_callback_output_that_is_not_a_token_sequence(self):
        with self.assertRaisesRegex(ValueError, "processor_choice_base_ids_invalid"):
            verify_processor_choice_suffix(tokenizer=Tokenizer(),
                rendered_prompt="prompt", process_text=lambda _text: "token ids")


if __name__ == "__main__":
    unittest.main()
