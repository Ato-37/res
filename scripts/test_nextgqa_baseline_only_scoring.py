from __future__ import annotations

import hashlib
import json
import unittest

import torch

from nextgqa_atomic_batch_freeze import FrozenBatch, batch_sha256
from nextgqa_baseline_only_scoring import score_full_video_baseline_three_rows


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


class _Output:
    def __init__(self, logits):
        self.logits = logits


class _FakeAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.peak_index = None

    def forward(self, hidden):
        batch, seq, _width = hidden.shape
        attention = torch.full((batch, 2, seq, seq), 1.0 / seq,
                               dtype=torch.float32)
        if self.peak_index is not None:
            attention[..., self.peak_index] = 0.5
            attention /= attention.sum(dim=-1, keepdim=True)
        return hidden, attention


class _FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeAttention()


class _FakeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_FakeLayer() for _ in range(16)])


class _FakeOuter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()


class _FakeModel(torch.nn.Module):
    def __init__(self, *, fail_at=None, nan_at=None, attention_peak_index=None,
                 skip_attention_layer=None):
        super().__init__()
        self.config = type("Config", (), {
            "vocab_size": 16, "_attn_implementation": "eager"})()
        self.model = _FakeOuter()
        self.calls = 0
        self.fail_at = fail_at
        self.nan_at = nan_at
        self.attention_peak_index = attention_peak_index
        self.skip_attention_layer = skip_attention_layer

    def forward(self, input_ids, attention_mask, pixel_values_videos,
                video_grid_thw, second_per_grid_ts, use_cache, logits_to_keep):
        del attention_mask, pixel_values_videos, video_grid_thw, second_per_grid_ts
        if use_cache is not False or logits_to_keep != 1:
            raise AssertionError("unexpected_model_options")
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic_forward_abort")
        logits = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
        logits = logits + input_ids[0, -1].float()
        hidden = torch.ones((1, input_ids.shape[1], 2), dtype=torch.float32)
        for index, layer in enumerate(self.model.language_model.layers[:16]):
            layer.self_attn.peak_index = self.attention_peak_index
            if index != self.skip_attention_layer:
                layer.self_attn(hidden)
        if self.calls == self.nan_at:
            logits[0, 0, 0] = float("nan")
        return _Output(logits)


def _fixture(group_count=2):
    rows, pins = [], []
    run_digests = {field: hashlib.sha256(field.encode()).hexdigest()
                   for field in ("model_sha256", "processor_sha256",
                                 "tokenizer_sha256", "runtime_sha256",
                                 "protocol_sha256", "scorer_sha256")}
    for position in range(1, 4):
        record_id = f"record-{position}"
        source_sha = hashlib.sha256(f"video-{position}".encode()).hexdigest()
        prompt_sha = hashlib.sha256(f"prompt-{position}".encode()).hexdigest()
        batch = FrozenBatch({
            "input_ids": torch.tensor([[1] * group_count + [position]], dtype=torch.int64),
            "attention_mask": torch.ones((1, group_count + 1), dtype=torch.int64),
            "pixel_values_videos": torch.full((group_count, 3), position / 10,
                                               dtype=torch.float32),
            "video_grid_thw": torch.tensor([[group_count, 1, 1]], dtype=torch.int64),
            "second_per_grid_ts": torch.tensor([1.0], dtype=torch.float32),
        })
        input_ids = batch["input_ids"][0].tolist()
        ids_sha = hashlib.sha256(_canonical(input_ids)).hexdigest()
        batch_sha = batch_sha256(batch)
        binding = {
            "record_id": record_id,
            "source_video_sha256": source_sha,
            "prompt_sha256": prompt_sha,
            "input_ids_sha256": ids_sha,
            "batch_sha256": batch_sha,
            "question_positions": [group_count],
            "temporal_groups": [[i] for i in range(group_count)],
            "opaque_record_token": f"opaque_record_token_{position:04d}",
            "answer_token_ids": [5, 6, 7, 8, 9],
        }
        pins.append(binding)
        rows.append({**binding, "condition": "full", "answer_token_ids": [5, 6, 7, 8, 9],
                     "question_positions": [group_count],
                     "temporal_groups": [[i] for i in range(group_count)],
                     "opaque_record_token": binding["opaque_record_token"],
                     "batch": batch})
    return rows, pins, run_digests


class BaselineOnlyScoringTests(unittest.TestCase):
    def score(self, model=None, rows=None, pins=None, digests=None):
        defaults = _fixture()
        return score_full_video_baseline_three_rows(
            model=model or _FakeModel().eval(),
            rows=defaults[0] if rows is None else rows,
            expected_record_bindings=defaults[1] if pins is None else pins,
            run_digests=defaults[2] if digests is None else digests,
            device="cpu",
        )

    def test_exactly_three_full_video_forwards_and_provenance(self):
        rows, pins, digests = _fixture()
        model = _FakeModel().eval()
        report = self.score(model, rows, pins, digests)
        self.assertEqual(report["status"], "completed_non_authorizing")
        self.assertEqual(report["requested_forwards"], 3)
        self.assertEqual(report["attempted_forwards"], 3)
        self.assertEqual(report["completed_forwards"], 3)
        self.assertEqual(report["root_call_entries"], 3)
        self.assertEqual(report["verified_root_calls"], 3)
        self.assertEqual(report["completed_boundary_calls"], 3)
        self.assertEqual([r["selector_attention_capture_status"] for r in report["records"]],
                         ["one_forward_attention_rows_captured_non_authorizing"] * 3)
        self.assertEqual(report["records"][0]["selector_attention_layers_1_based"],
                         list(range(5, 17)))
        self.assertEqual(report["records"][0]["selector_top_k"], 1)
        self.assertEqual(report["records"][0]["sham_seed"],
                         "qta-sham-seed-20261018-v1")
        self.assertEqual(len(report["records"][0]["target_token_positions"]), 1)
        self.assertEqual(len(report["records"][0]["sham_token_positions"]), 1)
        self.assertEqual(model.calls, 3)
        self.assertEqual([row["record_id"] for row in report["records"]],
                         ["record-1", "record-2", "record-3"])
        for result, source, batch in zip(report["records"], pins, rows, strict=True):
            self.assertEqual(result["source_video_sha256"], source["source_video_sha256"])
            self.assertEqual(result["processor_batch_sha256"], batch["batch_sha256"])
            self.assertEqual(set(result["answer_label_log_probabilities_nats"]),
                             {"A", "B", "C", "D", "E"})
            self.assertEqual((result["p_label"], result["q_label"]), ("E", "D"))
            self.assertEqual(result["score_position"], 2)
        self.assertEqual(report["run_digests"], digests)
        self.assertIs(report["scoring_authorized"], False)
        self.assertIs(report["model_load_permitted"], False)
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertNotIn("treatment", report)
        self.assertNotIn("branches", report)

    def test_bad_row_count_rejects_before_any_forward(self):
        rows, pins, _ = _fixture()
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_three_row_inventory_required"):
            self.score(model, rows[:2], pins[:2])
        self.assertEqual(model.calls, 0)

    def test_row_order_and_source_tampering_reject_before_forward(self):
        rows, pins, _ = _fixture()
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_record_order_or_identity_mismatch"):
            self.score(model, list(reversed(rows)), pins)
        self.assertEqual(model.calls, 0)

        rows, pins, _ = _fixture()
        rows[1]["source_video_sha256"] = "0" * 64
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_source_video_sha256_binding_mismatch"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_batch_digest_and_treatment_inventory_tampering_reject_pre_call(self):
        rows, pins, _ = _fixture()
        rows[0]["batch"]["input_ids"][0, 0] = 13
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_input_ids_digest_mismatch"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

        rows, pins, _ = _fixture()
        rows[0]["batch"]["pixel_values_videos"][0, 0] = 0.99
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_batch_digest_mismatch"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

        rows, pins, _ = _fixture()
        rows[0]["condition"] = "claim_evidence_black"
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_only_full_condition_required"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_duplicate_or_out_of_range_answer_token_rejects_pre_call(self):
        for token_ids in ([5, 6, 7, 8, 8], [5, 6, 7, 8, 16]):
            with self.subTest(token_ids=token_ids):
                rows, pins, _ = _fixture()
                rows[0]["answer_token_ids"] = token_ids
                model = _FakeModel().eval()
                with self.assertRaisesRegex(ValueError,
                                            "baseline_answer_token_contract_invalid"):
                    self.score(model, rows, pins)
                self.assertEqual(model.calls, 0)

    def test_answer_token_pin_and_distinct_sources_reject_pre_call(self):
        rows, pins, _ = _fixture()
        rows[0]["answer_token_ids"] = [4, 6, 7, 8, 9]
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_answer_token_binding_mismatch"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

        rows, pins, _ = _fixture()
        rows[1]["source_video_sha256"] = rows[0]["source_video_sha256"]
        pins[1]["source_video_sha256"] = pins[0]["source_video_sha256"]
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_duplicate_source_video_rejected"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_ranked_selection_is_preserved_and_plan_positions_are_sorted(self):
        rows, pins, _ = _fixture(group_count=20)
        report = self.score(_FakeModel(attention_peak_index=19).eval(), rows, pins)
        record = report["records"][0]
        self.assertEqual(record["selector_top_k"], 2)
        self.assertEqual(record["target_ranked_group_indices"], [19, 0])
        self.assertEqual(record["target_group_indices"], [0, 19])
        self.assertEqual(record["target_token_positions"], [0, 19])

    def test_missing_or_ambiguous_video_fields_reject_before_forward(self):
        rows, pins, _ = _fixture()
        del rows[0]["batch"]["pixel_values_videos"]
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_full_video_batch_required"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_video_mask_grid_and_timing_layout_reject_before_forward(self):
        mutations = (
            ("mask", lambda batch: batch["attention_mask"].__setitem__((0, 1), 0),
             "baseline_unpadded_attention_mask_required"),
            ("pixel_rows", lambda batch: batch.__setitem__(
                "pixel_values_videos", torch.ones((1, 3))),
             "baseline_video_pixel_grid_count_mismatch"),
            ("pixel_rank", lambda batch: batch.__setitem__(
                "pixel_values_videos", torch.ones((2, 1, 3))),
             "baseline_video_pixel_payload_invalid"),
            ("pixel_dtype", lambda batch: batch.__setitem__(
                "pixel_values_videos", torch.ones((2, 3), dtype=torch.int64)),
             "baseline_video_pixel_payload_invalid"),
            ("grid_dtype", lambda batch: batch.__setitem__(
                "video_grid_thw", torch.tensor([[2.0, 1.0, 1.0]])),
             "baseline_video_grid_invalid"),
            ("timing", lambda batch: batch.__setitem__(
                "second_per_grid_ts", torch.tensor([1.0, 2.0])),
             "baseline_video_timing_invalid"),
        )
        for label, mutate, expected_error in mutations:
            with self.subTest(label=label):
                rows, pins, _ = _fixture()
                mutate(rows[0]["batch"])
                model = _FakeModel().eval()
                with self.assertRaisesRegex(ValueError, expected_error):
                    self.score(model, rows, pins)
                self.assertEqual(model.calls, 0)

        rows, pins, _ = _fixture()
        rows[0]["batch"]["pixel_values"] = torch.ones((1, 2))
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_full_video_batch_required"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_training_mode_and_non_eager_attention_reject_before_forward(self):
        rows, pins, _ = _fixture()
        model = _FakeModel()
        with self.assertRaisesRegex(ValueError, "baseline_model_must_be_in_eval_mode"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

        model = _FakeModel().eval()
        model.model.language_model.layers[5].self_attn.train()
        with self.assertRaisesRegex(ValueError,
                                   "baseline_attention_modules_must_be_in_eval_mode"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

        model = _FakeModel().eval()
        model.config._attn_implementation = "sdpa"
        with self.assertRaisesRegex(ValueError, "baseline_eager_attention_required"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_question_position_and_temporal_groups_are_pinned(self):
        rows, pins, _ = _fixture()
        pins[0]["question_positions"] = [1]
        model = _FakeModel().eval()
        with self.assertRaisesRegex(ValueError, "baseline_selector_position_binding_mismatch"):
            self.score(model, rows, pins)
        self.assertEqual(model.calls, 0)

    def test_nonfinite_logits_fail_closed_without_partial_scores(self):
        model = _FakeModel(nan_at=2).eval()
        report = self.score(model)
        self.assertEqual(report["status"], "failed_non_authorizing")
        self.assertEqual(report["failure_code"], "baseline_nonfinite_logits")
        self.assertEqual(report["attempted_forwards"], 2)
        self.assertEqual(report["completed_forwards"], 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(report["records"], [])
        self.assertIsNone(report["run_digests"])
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_aborted_forward_fails_closed_and_omits_partial_scores(self):
        model = _FakeModel(fail_at=2).eval()
        report = self.score(model)
        self.assertEqual(report["status"], "failed_non_authorizing")
        self.assertEqual(report["attempted_forwards"], 2)
        self.assertEqual(report["completed_forwards"], 1)
        self.assertEqual(model.calls, 2)
        self.assertEqual(report["records"], [])
        self.assertEqual(report["failure_code"],
                         "baseline_forward_or_score_failed_RuntimeError")

    def test_missing_attention_hook_layer_counts_returned_forward_but_discards_rows(self):
        model = _FakeModel(skip_attention_layer=7).eval()
        report = self.score(model)
        self.assertEqual(report["status"], "failed_non_authorizing")
        self.assertEqual(report["attempted_forwards"], 1)
        self.assertEqual(report["completed_forwards"], 1)
        self.assertEqual(report["completed_boundary_calls"], 1)
        self.assertEqual(model.calls, 1)
        self.assertEqual(report["records"], [])
        self.assertEqual(report["failure_code"],
                         "baseline_forward_or_score_failed_RuntimeError")

    def test_extra_root_call_is_rejected(self):
        rows, _, _ = _fixture()
        model = _FakeModel().eval()
        reentered = {"done": False}

        def extra_call(module, _args, kwargs):
            if not reentered["done"]:
                reentered["done"] = True
                module(**kwargs)

        model.register_forward_pre_hook(extra_call, with_kwargs=True)
        report = self.score(model, rows)
        self.assertEqual(report["status"], "failed_non_authorizing")
        self.assertEqual(report["failure_code"],
                         "baseline_forward_or_score_failed_RuntimeError")
        self.assertEqual(report["attempted_forwards"], 1)
        self.assertEqual(report["completed_forwards"], 0)
        self.assertEqual(report["root_call_entries"], 2)
        self.assertEqual(model.calls, 2)
        self.assertEqual(report["records"], [])


if __name__ == "__main__":
    unittest.main()
