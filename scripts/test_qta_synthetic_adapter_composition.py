from __future__ import annotations

import copy
import hashlib
import json
import unittest
from fractions import Fraction

from necessity_choice_margin_utils import uniform_sample_indices
from qta_prescore_clock_row_adapter import adapt_calibration_to_clock_row
from qta_prescore_processor_row_adapter import processor_binding_row
from qta_prescore_mapping_binding import interval_list_sha256
from nextgqa_atomic_batch_freeze import (
    CONDITIONS, PreflightError, batch_sha256, freeze_all_batches,
)
from nextgqa_condition_batch_builder import build_exact_triplet_batches
from qwen_processor_batch_review_mapping import map_review_interval_from_processor_batch
from qta_synthetic_cohort_preflight import preflight_synthetic_qta_cohort
import test_qta_prescore_clock_row_adapter as clock_adapter_test
import test_qta_prescore_processor_row_adapter as processor_adapter_test
import test_nextgqa_condition_batch_builder as batch_builder_test
from test_qta_prescore_clock_row_adapter import seal
from test_qta_synthetic_cohort_preflight import synthetic_mapping_inputs


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _replace_tokens(inputs):
    old_tokens = list(inputs["expected_cohort_tokens"])
    new_tokens = [f"{index + 1:024x}" for index in range(3)]
    mapping = dict(zip(old_tokens, new_tokens, strict=True))
    inputs["expected_cohort_tokens"] = new_tokens
    for name in ("review_rows", "clock_rows", "processor_rows", "baseline_rows"):
        for row in inputs[name]:
            row["review_token"] = mapping[row["review_token"]]
    for name in ("expected_video_sha256_by_token",
                 "expected_processor_batch_sha256_by_token",
                 "expected_sampling_metadata_sha256_by_token",
                 "expected_processor_groups_by_token",
                 "expected_calibration_sha256_by_token",
                 "expected_baseline_source_sha256_by_token"):
        inputs[name] = {mapping[token]: value for token, value in inputs[name].items()}
    return inputs


def _actual_adapter_rows(inputs, *, full_triplets=None, source_rows=None):
    clock_fixture = clock_adapter_test.QtaPrescoreClockRowAdapterTests()
    clock_fixture.setUp()
    processor_fixture = processor_adapter_test.QtaPrescoreProcessorRowAdapterTests()
    processor_fixture.setUp()
    clock_rows, processor_rows = [], []
    try:
        for index, token in enumerate(inputs["expected_cohort_tokens"]):
            video_sha = inputs["expected_video_sha256_by_token"][token]
            capture_obj = json.loads(clock_fixture.capture_raw)
            capture_obj["source_media_sha256"] = video_sha
            capture_raw = json.dumps(capture_obj, sort_keys=True,
                                     separators=(",", ":")).encode()
            calibration_body = {key: value for key, value in clock_fixture.calibration.items()
                                if key not in {"artifact_raw", "artifact_sha256"}}
            calibration_body.update({
                "review_token": token,
                "source_media_sha256": video_sha,
                "capture_sha256": sha(capture_raw),
            })
            calibration = seal(calibration_body)
            clock_row = adapt_calibration_to_clock_row(
                review_token=token,
                source_video_sha256=video_sha,
                video_claim_review_receipt_sha256=(
                    inputs["expected_video_review_receipt_sha256"]),
                calibration=calibration,
                capture_raw=capture_raw,
            )
            clock_rows.append(clock_row)
            inputs["expected_calibration_sha256_by_token"][token] = (
                clock_row["calibration_sha256"])

            if full_triplets is None:
                mapping = copy.deepcopy(processor_fixture.mapping)
                # Existing row-adapter-only coverage uses a small synthetic
                # snapshot. The integrated caller test below supplies actual
                # first-pass builder batches instead.
                mapping["_processor_batch_snapshot"] = {
                    "input_ids": [[index + 1, index + 2]],
                    "pixel_values_videos": [[float(index + 1), float(index + 2)]],
                    "video_grid_thw": [[5, 2, 2]],
                }
                mapping["typed_batch_sha256"] = batch_sha256(
                    mapping["_processor_batch_snapshot"])
            else:
                import torch

                source = source_rows[index]
                mapping = map_review_interval_from_processor_batch(
                    batch=full_triplets[index]["batches"]["full"],
                    torch_module=torch,
                    video_token_id=9999,
                    spatial_merge_size=1,
                    temporal_patch_size=4,
                    sampled_frame_indices=source["sampled_indices"],
                    sampled_pts_seconds=source["presentation_times"],
                    source_frame_count=len(source["frames"]),
                    video_duration_seconds=(source["presentation_times"][-1]
                                             + 1 / source["sample_fps"]),
                    sample_fps=float(source["sample_fps"]),
                    processor_revision="synthetic-processor-v1",
                    processor_fingerprint_sha256=sha(b"synthetic-processor-v1"),
                    reviewed_intervals_seconds=source["intervals"],
                    interval_boundary_guard_seconds=(
                        clock_row["calibration_binding_payload"]["guard_pts_seconds"]),
                )
                # The mapper returns geometry and a digest; the row adapter's
                # projection also requires the exact grid field retained by
                # that same batch. Reattach that value from the mapped snapshot.
                mapping["video_grid_thw"] = [
                    int(value) for value in
                    full_triplets[index]["batches"]["full"]["video_grid_thw"][0].tolist()]
                mapping["sampled_frame_indices"] = list(source["sampled_indices"])
                mapping["temporal_group_token_positions"] = [
                    group["token_positions"] for group in mapping["groups"]]
                groups = inputs["expected_processor_groups_by_token"][token]
                groups["video_grid_thw"] = mapping["video_grid_thw"]
                groups["evidence_group_indices"] = mapping["evidence_group_indices"]
                groups["control_group_indices"] = mapping["control_group_indices"]
                groups["temporal_group_token_positions"] = (
                    mapping["temporal_group_token_positions"])
                inputs["expected_sampling_metadata_sha256_by_token"][token] = (
                    mapping["sampling_metadata_sha256"])
            inputs["expected_processor_batch_sha256_by_token"][token] = (
                mapping["typed_batch_sha256"])
            if full_triplets is None:
                mapping["sampling_metadata_sha256"] = (
                    inputs["expected_sampling_metadata_sha256_by_token"][token])
                mapping["video_grid_thw"] = inputs[
                    "expected_processor_groups_by_token"][token]["video_grid_thw"]
            groups = inputs["expected_processor_groups_by_token"][token]
            if full_triplets is None:
                mapping["interval_boundary_guard_rational_seconds"] = (
                    clock_row["calibration_binding_payload"]["guard_pts_seconds"])
                mapping["evidence_group_indices"] = groups["evidence_group_indices"]
                mapping["control_group_indices"] = groups["control_group_indices"]
                mapping["groups"] = [{"token_positions": positions}
                                      for positions in groups["temporal_group_token_positions"]]
                sample_indices = uniform_sample_indices(64, 32)
                sample_pts = [[index, 1] for index in range(32)]
                mapping["sampled_frame_indices"] = sample_indices
                mapping["sampled_pts_rational_seconds"] = [
                    str(Fraction(*pair)) for pair in sample_pts]
            else:
                sample_indices = source_rows[index]["sampled_indices"]
                sample_pts = [[value.numerator, value.denominator]
                              for value in source_rows[index]["presentation_times"]]
            sampler_row = {
                "source_sha256": video_sha,
                "decoded_frame_count": (64 if full_triplets is None
                                         else len(sample_indices)),
                "sampled_frame_count": 32,
                "sampled_indices": sample_indices,
                "sampled_pts": sample_pts,
            }
            processor_rows.append(processor_binding_row(
                review_token=token, source_video_sha256=video_sha,
                mapping=mapping, sampler_row=sampler_row))
    finally:
        clock_fixture.tearDown()
    return clock_rows, processor_rows


class SyntheticProcessorBatch(dict):
    def to(self, device):
        if device != "cpu":
            raise AssertionError("synthetic freezer test permits CPU snapshots only")
        return self


class VideoTokenFakeProcessor(batch_builder_test.FakeProcessor):
    """Give the builder fixture the video-token run expected by the mapper."""

    def __init__(self, *, drift_first_row=False):
        super().__init__()
        self.drift_first_row = drift_first_row

    def __call__(self, **kwargs):
        import torch

        result = super().__call__(**kwargs)
        # Model the review mapper's documented 2-D patch-row processor
        # contract while keeping this fixture tiny and deterministic.
        video = kwargs["videos"][0]
        result["pixel_values_videos"] = video.reshape(8, 4, 3, -1).mean(dim=(1, 3))
        result["second_per_grid_ts"] = torch.tensor([0.4], dtype=torch.float32)
        if self.drift_first_row and self.calls <= 3:
            result["input_ids"] = result["input_ids"].clone()
            result["input_ids"][0, 0] += 1
        marker = torch.full((1, 8), 9999, dtype=torch.int64)
        result["input_ids"] = torch.cat((result["input_ids"], marker), dim=1)
        result["attention_mask"] = torch.ones_like(result["input_ids"])
        result["mm_token_type_ids"] = torch.zeros_like(result["input_ids"])
        return SyntheticProcessorBatch(result)


def _build_snapshot_bound_inputs(records):
    """Run the real CPU mapper/row adapter over builder-produced full batches."""
    first = build_exact_triplet_batches(
        records=records, processor=VideoTokenFakeProcessor(),
        vision_info_fn=batch_builder_test.FakeVisionInfo())
    inputs = _replace_tokens(synthetic_mapping_inputs())
    for row in inputs["review_rows"]:
        row["evidence_intervals_seconds"] = [[0.5, 0.7]]
    for row in inputs["baseline_rows"]:
        row["reviewed_intervals_sha256"] = interval_list_sha256([[0.5, 0.7]])
    clocks, processors = _actual_adapter_rows(
        inputs, full_triplets=first["triplets"], source_rows=records)
    inputs["clock_rows"] = clocks
    inputs["processor_rows"] = processors
    return first, inputs


def composed_inputs():
    inputs = _replace_tokens(synthetic_mapping_inputs())
    clocks, processors = _actual_adapter_rows(inputs)
    inputs["clock_rows"] = clocks
    inputs["processor_rows"] = processors
    return inputs


class QtaSyntheticAdapterCompositionTests(unittest.TestCase):
    def denied(self, report):
        self.assertFalse(report["model_load_permitted"])
        self.assertEqual(report["branch_forwards"], 0)
        self.assertEqual(report["scoring_authorization"], "not_granted")
        self.assertEqual(report["execution_authorization"], "not_granted")

    def test_real_clock_and_processor_row_adapters_compose_for_three_synthetic_rows(self):
        inputs = composed_inputs()
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "structural_row_binding_passed_unvalidated")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 3)
        self.assertEqual(len(report["row_binding_sha256"]), 3)
        self.assertEqual(len(report["bundle_binding_sha256"]), 64)
        self.assertFalse(report["clock_joint_freeze_authenticated"])
        self.assertEqual(report["prediction_link_authentication"],
                         "self_asserted_unauthenticated")
        self.denied(report)
        public = json.dumps(report, sort_keys=True)
        for token in inputs["expected_cohort_tokens"]:
            self.assertNotIn(token, public)
        for video_sha in inputs["expected_video_sha256_by_token"].values():
            self.assertNotIn(video_sha, public)

    def test_one_processor_batch_mismatch_blocks_whole_three_row_join(self):
        inputs = composed_inputs()
        inputs["processor_rows"][1]["typed_batch_sha256"] = "f" * 64
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self.denied(report)

    def test_clock_video_mismatch_blocks_whole_three_row_join(self):
        inputs = composed_inputs()
        inputs["clock_rows"][2]["source_video_sha256"] = "e" * 64
        report = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(report["status"], "blocked_structural_row_binding_integration")
        self.assertEqual(report["verified_structural_rows"], 0)
        self.assertEqual(report["row_binding_sha256"], [])
        self.denied(report)

    @unittest.skipIf(batch_builder_test.torch is None or batch_builder_test.Image is None
                     or batch_builder_test.np is None,
                     "torch/Pillow/NumPy unavailable")
    def test_builder_full_snapshot_rebinds_on_second_build_and_freezes_non_authorizing(self):
        records = batch_builder_test.source_records()
        # Match the independent synthetic receipt fixture's interval while
        # retaining the builder's 32-frame source and exact PTS grid. The
        # compressed 10 FPS fixture keeps the evidence inside the pinned
        # synthetic clock anchor span [0, 1] seconds.
        for record in records:
            record["presentation_times"] = [Fraction(index, 10)
                                             for index in range(32)]
            record["sample_fps"] = Fraction(10, 1)
            record["intervals"] = [[Fraction(1, 2), Fraction(7, 10)]]

        first, inputs = _build_snapshot_bound_inputs(records)
        joined = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(joined["status"], "structural_row_binding_passed_unvalidated")
        self.denied(joined)

        tokens = inputs["expected_cohort_tokens"]
        expected_hashes = {
            token: {condition: batch_sha256(first["triplets"][index]["batches"][condition])
                    for condition in CONDITIONS}
            for index, token in enumerate(tokens)
        }
        for index, (token, processor_row) in enumerate(
                zip(tokens, inputs["processor_rows"], strict=True)):
            self.assertEqual(processor_row["typed_batch_sha256"],
                             expected_hashes[token]["full"])

        # A second independent builder invocation represents scoring-time
        # reprocessing. Compare typed bytes, not a copied digest field.
        second = build_exact_triplet_batches(
            records=records, processor=VideoTokenFakeProcessor(),
            vision_info_fn=batch_builder_test.FakeVisionInfo())
        self.assertEqual(len(tokens), 3)
        self.assertEqual(len(first["triplets"]), len(second["triplets"]))
        for index, (token, processor_row, scorer_row) in enumerate(zip(
                tokens, inputs["processor_rows"], second["triplets"], strict=True)):
            self.assertEqual(processor_row["review_token"], token)
            self.assertEqual(set(scorer_row["batches"]), set(CONDITIONS))
            for condition in CONDITIONS:
                self.assertEqual(expected_hashes[token][condition],
                                 batch_sha256(scorer_row["batches"][condition]),
                                 f"row {index + 1} {condition} processor batch changed")

        scorer_triplets = []
        for token, row in zip(tokens, second["triplets"], strict=True):
            scorer_triplets.append({
                "record_id": token,
                "gold_index": row["gold_index"],
                "batches": row["batches"],
            })
        callbacks = []
        frozen = freeze_all_batches(
            triplets=scorer_triplets,
            token_ids=[11, 12, 13, 14, 15],
            expected_record_ids=tokens,
            expected_batch_hashes=expected_hashes,
            upstream_precheck=lambda: callbacks.append("precheck") or True,
            authenticated_gate_check=lambda: callbacks.append("final") or True,
        )
        self.assertEqual(callbacks, ["precheck", "final"])
        self.assertEqual(frozen["status"], "all_nine_batches_frozen")
        self.assertEqual(frozen["requested_forwards"], 9)
        self.assertEqual(frozen["execution_authorization"], "not_granted")
        # The test never constructs a model and the freezer has no scoring API.
        self.assertNotIn("model_load_permitted", frozen)
        self.assertNotIn("completed_forwards", frozen)

        callbacks = []
        with self.assertRaisesRegex(PreflightError, "record_order_or_identity_mismatch"):
            freeze_all_batches(
                triplets=list(reversed(scorer_triplets)),
                token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=tokens,
                expected_batch_hashes=expected_hashes,
                upstream_precheck=lambda: callbacks.append("precheck") or True,
                authenticated_gate_check=lambda: callbacks.append("final") or True,
            )
        self.assertEqual(callbacks, [])

    @unittest.skipIf(batch_builder_test.torch is None or batch_builder_test.Image is None
                     or batch_builder_test.np is None,
                     "torch/Pillow/NumPy unavailable")
    def test_second_build_mutation_and_row_permutation_fail_the_join(self):
        records = batch_builder_test.source_records()
        for record in records:
            record["presentation_times"] = [Fraction(index, 10)
                                             for index in range(32)]
            record["sample_fps"] = Fraction(10, 1)
            record["intervals"] = [[Fraction(1, 2), Fraction(7, 10)]]
        first, inputs = _build_snapshot_bound_inputs(records)
        baseline_hash = inputs["processor_rows"][0]["typed_batch_sha256"]

        mutated = build_exact_triplet_batches(
            records=records,
            processor=VideoTokenFakeProcessor(drift_first_row=True),
            vision_info_fn=batch_builder_test.FakeVisionInfo())
        changed_hash = batch_sha256(mutated["triplets"][0]["batches"]["full"])
        self.assertNotEqual(changed_hash, baseline_hash)

        expected_hashes = {
            token: {condition: batch_sha256(row["batches"][condition])
                    for condition in CONDITIONS}
            for token, row in zip(inputs["expected_cohort_tokens"],
                                  first["triplets"], strict=True)
        }
        changed_triplets = [
            {"record_id": token, **row}
            for token, row in zip(inputs["expected_cohort_tokens"],
                                  mutated["triplets"], strict=True)
        ]
        callbacks = []
        with self.assertRaisesRegex(PreflightError,
                                    "exact_condition_batch_hash_mismatch"):
            freeze_all_batches(
                triplets=changed_triplets,
                token_ids=[11, 12, 13, 14, 15],
                expected_record_ids=inputs["expected_cohort_tokens"],
                expected_batch_hashes=expected_hashes,
                upstream_precheck=lambda: callbacks.append("precheck") or True,
                authenticated_gate_check=lambda: callbacks.append("final") or True,
            )
        self.assertEqual(callbacks, ["precheck"])

        inputs["processor_rows"] = list(reversed(inputs["processor_rows"]))
        permuted = preflight_synthetic_qta_cohort(**inputs)
        self.assertEqual(permuted["status"],
                         "blocked_structural_row_binding_integration")
        self.assertEqual(permuted["verified_structural_rows"], 0)
        self.assertEqual(permuted["row_binding_sha256"], [])
        self.denied(permuted)


if __name__ == "__main__":
    unittest.main()
