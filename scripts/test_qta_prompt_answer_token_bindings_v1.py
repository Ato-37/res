from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qta_prompt_answer_token_bindings_v1 import (
    build_qta_prompt_answer_token_bindings,
    validate_qta_prompt_answer_token_bindings,
)

VIDEO = "<|video_pad|>"


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids, cursor = [], 0
        while cursor < len(text):
            if text.startswith(VIDEO, cursor):
                ids.append(99)
                cursor += len(VIDEO)
            else:
                ids.append(ord(text[cursor]) + 100)
                cursor += 1
        return ids


class FakeProcessor:
    video_token_id = 99

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        text = next(part["text"] for part in messages[0]["content"]
                    if part["type"] == "text")
        return VIDEO + "\n" + text


def _prompt_inputs():
    frames = [Image.new("RGB", (2, 2), (i, 4, 5)) for i in range(32)]
    return [{"review_token": f"opaque-{i}", "question": f"Question {i}?",
             "choices": [f"choice {i}-{letter}" for letter in "ABCDE"],
             "frames": frames, "sample_fps": [1, 1]} for i in range(3)]


def _sampler_rows(prompt_inputs):
    return [{"sampled_frame_count": 32, "effective_sample_fps": [1, 1],
             "sampled_rgb_sha256": [hashlib.sha256(
                 frame.convert("RGB").tobytes()).hexdigest()
                 for frame in row["frames"]]} for row in prompt_inputs]


def _live_batch(tokenizer, processor, row):
    text = processor.apply_chat_template([{"role": "user", "content": [
        {"type": "video", "video": row["frames"], "sample_fps": 1.0},
        {"type": "text", "text": (f"{row['question']}\nOptions:\n" +
            "\n".join(f"{letter}. {choice}" for letter, choice in
                       zip("ABCDE", row["choices"], strict=True)) +
            "\nChoose the best answer. Respond with one letter only.\nAnswer:")},
    ]}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    at = ids.index(processor.video_token_id)
    return {"input_ids": [ids[:at] + [processor.video_token_id] * 32 + ids[at + 1:] ]}


def _fixture():
    candidates, expected, processor = [], [], []
    for index in range(3):
        token = f"opaque-{index}"
        h = lambda label: hashlib.sha256(f"{label}-{index}".encode()).hexdigest()
        row = {
            "review_token": token,
            "question_sha256": h("question"),
            "options_sha256": h("options"),
            "rendered_prompt_sha256": h("prompt"),
            "tokenizer_sha256": h("tokenizer"),
            "prompt_token_ids_sha256": h("prompt-ids"),
            "choice_token_binding_sha256": h("choice-binding"),
            "input_ids_sha256": h("input-ids"),
            "answer_token_ids": [index * 10 + n for n in range(5)],
        }
        candidates.append(row)
        expected.append({"review_token": token,
                         "question_sha256": row["question_sha256"],
                         "options_sha256": row["options_sha256"]})
        processor.append({"prompt_sha256": row["rendered_prompt_sha256"],
                          "tokenizer_sha256": row["tokenizer_sha256"],
                          "input_ids_sha256": row["input_ids_sha256"]})
    return candidates, expected, processor


class PromptAnswerTokenBindingsTests(unittest.TestCase):
    def test_build_is_non_authorizing_and_keeps_private_text_out(self):
        candidates, expected, processor = _fixture()
        raw = build_qta_prompt_answer_token_bindings(candidates)
        self.assertEqual(json.loads(raw)["execution_authorization"], "not_granted")
        for secret in ("opaque-0", "question-0", "options-0"):
            self.assertNotIn(secret.encode(), raw)

    def test_rejects_spliced_source_or_processor_pins(self):
        raw, expected, processor, live, samplers, prompts, proc, tokenizer = self._valid_material()
        expected[1] = {**expected[1], "options_sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "independent_recomputation_mismatch"):
            validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                processor_rows=processor, live_processor_rows=live, sampler_rows=samplers,
                prompt_reconstruction_inputs=prompts, processor=proc, tokenizer=tokenizer)
        raw, expected, processor, live, samplers, prompts, proc, tokenizer = self._valid_material()
        processor[2] = {**processor[2], "input_ids_sha256": "e" * 64}
        with self.assertRaisesRegex(ValueError, "independent_recomputation_mismatch"):
            validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                processor_rows=processor, live_processor_rows=live, sampler_rows=samplers,
                prompt_reconstruction_inputs=prompts, processor=proc, tokenizer=tokenizer)

    def test_rejects_noncanonical_duplicate_or_mutated_manifest(self):
        candidates, expected, processor = _fixture()
        raw = build_qta_prompt_answer_token_bindings(candidates)
        with self.assertRaisesRegex(ValueError, "canonical_invalid"):
            self._validate(raw + b" ", expected, processor)
        duplicate = raw[:-1] + b',"purpose":"other"}'
        with self.assertRaisesRegex(ValueError, "duplicate_key"):
            self._validate(duplicate, expected, processor)
        obj = json.loads(raw)
        obj["rows"][0]["answer_token_ids"][0] = True
        bad = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
        with self.assertRaisesRegex(ValueError, "token_ids_invalid"):
            self._validate(bad, expected, processor)

    def test_rejects_invalid_answer_ids_at_build(self):
        candidates, _, _ = _fixture()
        candidates[0]["answer_token_ids"] = [1, 1, 2, 3, 4]
        with self.assertRaisesRegex(ValueError, "token_ids_invalid"):
            build_qta_prompt_answer_token_bindings(candidates)

    def _valid_material(self):
        tokenizer = FakeTokenizer()
        processor = FakeProcessor(tokenizer)
        prompts = _prompt_inputs()
        samplers = _sampler_rows(prompts)
        expected, processor_rows, candidates, live = [], [], [], []
        for i, source in enumerate(prompts):
            source_sha = hashlib.sha256(f"video-{i}".encode()).hexdigest()
            samplers[i]["source_sha256"] = source_sha
            sampler_sha = hashlib.sha256(json.dumps(samplers[i], ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            qsha = hashlib.sha256(source["question"].encode()).hexdigest()
            osha = hashlib.sha256(json.dumps(source["choices"], ensure_ascii=False,
                sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "video", "video": source["frames"], "sample_fps": 1.0},
                {"type": "text", "text": (f"{source['question']}\nOptions:\n" +
                    "\n".join(f"{letter}. {choice}" for letter, choice in
                        zip("ABCDE", source["choices"], strict=True)) +
                    "\nChoose the best answer. Respond with one letter only.\nAnswer:")},
            ]}], tokenize=False, add_generation_prompt=True)
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            answer = [tokenizer.encode(prompt + letter, add_special_tokens=False)[len(prefix)]
                      for letter in "ABCDE"]
            tokenizer_sha = hashlib.sha256(b"test-tokenizer").hexdigest()
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            prefix_sha = hashlib.sha256(json.dumps(prefix, sort_keys=True,
                separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()
            choice_sha = hashlib.sha256(json.dumps({"labels": list("ABCDE"),
                "prompt_sha256": prompt_sha, "tokenizer_sha256": tokenizer_sha,
                "prompt_token_ids_sha256": prefix_sha,
                "answer_token_ids": answer}, sort_keys=True, separators=(",", ":"),
                ensure_ascii=True).encode("ascii")).hexdigest()
            batch = _live_batch(tokenizer, processor, source)
            input_sha = hashlib.sha256(json.dumps(batch["input_ids"][0], sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            token = source["review_token"]
            expected.append({"review_token": token, "source_video_sha256": source_sha,
                "question_sha256": qsha, "options_sha256": osha,
                "tokenizer_sha256": tokenizer_sha, "sampler_row_sha256": sampler_sha})
            processor_rows.append({"prompt_sha256": prompt_sha,
                "tokenizer_sha256": tokenizer_sha, "input_ids_sha256": input_sha})
            live.append({"review_token": token, "batch": batch})
            candidates.append({"review_token": token, "question_sha256": qsha,
                "options_sha256": osha, "rendered_prompt_sha256": prompt_sha,
                "tokenizer_sha256": tokenizer_sha,
                "prompt_token_ids_sha256": prefix_sha,
                "choice_token_binding_sha256": choice_sha,
                "input_ids_sha256": input_sha, "answer_token_ids": answer})
        raw = build_qta_prompt_answer_token_bindings(candidates)
        return raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer

    def _validate(self, raw, expected, processor_rows):
        material = self._valid_material()
        _canonical_raw, canonical_expected, canonical_processor, live, samplers, prompts, processor, tokenizer = material
        return validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
            processor_rows=processor_rows, live_processor_rows=live,
            sampler_rows=samplers, prompt_reconstruction_inputs=prompts,
            processor=processor, tokenizer=tokenizer)

    def test_independently_recomputes_all_three_rows_without_authorizing(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()
        report = validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
            processor_rows=processor_rows, live_processor_rows=live,
            sampler_rows=samplers, prompt_reconstruction_inputs=prompts,
            processor=processor, tokenizer=tokenizer)
        self.assertEqual(report["validated_rows"], 3)
        self.assertTrue(report["prompt_rendering_independently_reconstructed"])
        self.assertFalse(report["prompt_renderer_origin_authenticated"])
        self.assertFalse(report["tokenizer_authenticity_verified"])
        self.assertFalse(report["processor_authenticity_verified"])
        self.assertFalse(report["scoring_authorized"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_rejects_each_signed_declaration_field_mutation(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()
        original = json.loads(raw)
        for field in ("record_token", "question_sha256", "options_sha256", "prompt_sha256",
                      "tokenizer_sha256", "prompt_token_ids_sha256",
                      "choice_token_binding_sha256", "input_ids_sha256", "answer_token_ids"):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(original))
                if field == "answer_token_ids":
                    changed["rows"][1][field][0] += 1
                elif field == "record_token":
                    changed["rows"][1][field] = "f" * 64
                else:
                    changed["rows"][1][field] = ("e" * 64 if field != "answer_token_ids" else [1,2,3,4,5])
                bad = json.dumps(changed, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":")).encode()
                with self.assertRaises(ValueError):
                    validate_qta_prompt_answer_token_bindings(bad, expected_rows=expected,
                        processor_rows=processor_rows, live_processor_rows=live,
                        sampler_rows=samplers, prompt_reconstruction_inputs=prompts,
                        processor=processor, tokenizer=tokenizer)

    def test_rejects_order_source_frame_and_live_batch_mutations(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()
        for altered_prompts, altered_samplers, altered_live in (
                (list(reversed(prompts)), samplers, live),
                (prompts, [{**samplers[0], "sampled_rgb_sha256": ["f" * 64] * 32}, *samplers[1:]], live),
                (prompts, samplers, [{**live[0], "batch": {"input_ids": [[999]]}}, *live[1:]])):
            with self.subTest(altered=altered_prompts is not prompts or altered_live is not live):
                with self.assertRaises(ValueError):
                    validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                        processor_rows=processor_rows, live_processor_rows=altered_live,
                        sampler_rows=altered_samplers, prompt_reconstruction_inputs=altered_prompts,
                        processor=processor, tokenizer=tokenizer)

    def test_rejects_renderer_mutating_sampled_frames(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()

        class MutatingProcessor(FakeProcessor):
            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                frames = next(part["video"] for part in messages[0]["content"]
                              if part["type"] == "video")
                frames[0].putpixel((0, 0), (255, 0, 0))
                return super().apply_chat_template(messages, tokenize=tokenize,
                                                    add_generation_prompt=add_generation_prompt)

        mutating = MutatingProcessor(tokenizer)
        with self.assertRaisesRegex(ValueError, "sampler_frame_or_fps_mismatch"):
            validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                processor_rows=processor_rows, live_processor_rows=live,
                sampler_rows=samplers, prompt_reconstruction_inputs=prompts,
                processor=mutating, tokenizer=tokenizer)

    def test_missing_ephemeral_material_fails_closed(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()
        with self.assertRaises(ValueError):
            validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                processor_rows=processor_rows, live_processor_rows=live,
                sampler_rows=samplers, prompt_reconstruction_inputs=None,
                processor=processor, tokenizer=tokenizer)

    def test_renderer_without_tokenizer_attribute_fails_closed(self):
        raw, expected, processor_rows, live, samplers, prompts, processor, tokenizer = self._valid_material()

        class RendererWithoutTokenizer:
            video_token_id = 99

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                return processor.apply_chat_template(messages, tokenize=tokenize,
                    add_generation_prompt=add_generation_prompt)

        with self.assertRaisesRegex(ValueError, "processor_tokenizer_mismatch"):
            validate_qta_prompt_answer_token_bindings(raw, expected_rows=expected,
                processor_rows=processor_rows, live_processor_rows=live,
                sampler_rows=samplers, prompt_reconstruction_inputs=prompts,
                processor=RendererWithoutTokenizer(), tokenizer=tokenizer)


if __name__ == "__main__":
    unittest.main()
