from fractions import Fraction
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_qwen_pts_to_processor_cpu import encode_fixture
from claim_mask_timing_adapter import planner_timing_record
from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels


def image_sha(image):
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


class ClaimMaskTimingAdapterTests(unittest.TestCase):
    def fixture(self, directory, *, include_timing_map=True):
        path = Path(directory) / "synthetic_offset.mkv"
        encode_fixture(path, pts=(100, 102, 104, 106, 108),
                       time_base=Fraction(1, 10))
        frames, metadata = decode_sampled_video(
            path, num_frames=3, uniform_tolerance_seconds=Fraction(0),
            include_timing_map=include_timing_map)
        verify_sampled_frame_pixels(frames, metadata)
        return frames, metadata

    def processor_hashes(self, frames):
        return [image_sha(frame) for frame in frames]

    def test_real_synthetic_pyav_offset_pts_builds_planner_timing_row(self):
        with tempfile.TemporaryDirectory() as temp:
            frames, metadata = self.fixture(temp)
            record = planner_timing_record(
                metadata, video_id="synthetic-video",
                processor_pixel_sha256=self.processor_hashes(frames))
        timing_map = metadata["full_timing_map"]
        self.assertEqual(timing_map["pts"][0], 10000)
        self.assertNotEqual(timing_map["pts"][0], 0)
        self.assertEqual(record["pts"], timing_map["pts"])
        self.assertEqual(record["num_frames"], 5)
        self.assertEqual(len(record["sampled_pixel_sha256"]), 3)
        self.assertEqual(record["sampled_pixel_sha256"], record["processor_pixel_sha256"])
        self.assertEqual(record["annotation_clock_provenance"],
                         "unauthenticated_not_calibrated_by_decoder")
        self.assertEqual(record["review_provenance"],
                         "unauthenticated_not_provided_by_decoder")
        self.assertEqual(record["execution_authorization"], "not_granted")
        self.assertNotIn("score", record)
        json.dumps(record)

    def test_default_decoder_output_is_unchanged_and_has_no_full_map(self):
        with tempfile.TemporaryDirectory() as temp:
            _, metadata = self.fixture(temp, include_timing_map=False)
        self.assertNotIn("full_timing_map", metadata)

    def test_missing_tampered_digest_sample_mapping_and_pixel_hash_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            frames, original = self.fixture(temp)
            processor_hashes = self.processor_hashes(frames)
        variants = []
        missing = dict(original); missing.pop("full_timing_map")
        variants.append((missing, processor_hashes, "full_timing_map_missing_or_invalid"))
        bad_digest = copy.deepcopy(original)
        bad_digest["full_timing_map"]["pts"][1] += 1
        variants.append((bad_digest, processor_hashes, "full_timing_map_digest_mismatch"))
        bad_index = copy.deepcopy(original)
        bad_index["sampled_frames"][0]["decoded_frame_index"] += 1
        variants.append((bad_index, processor_hashes, "sampled_timing_binding_mismatch"))
        bad_pts = copy.deepcopy(original)
        bad_pts["sampled_frames"][0]["pts"] += 1
        variants.append((bad_pts, processor_hashes, "sampled_timing_binding_mismatch"))
        bad_relative_time = copy.deepcopy(original)
        bad_relative_time["sampled_frames"][0]["relative_presentation_time"] = [9, 1]
        variants.append((bad_relative_time, processor_hashes,
                         "sampled_timing_binding_mismatch"))
        bad_sample_hash = copy.deepcopy(original)
        bad_sample_hash["sampled_frames"][0]["rgb_pixel_sha256"] = "0" * 64
        variants.append((bad_sample_hash, processor_hashes,
                         "processor_pixel_hash_binding_mismatch"))
        variants.append((original, ["0" * 64] * len(processor_hashes),
                         "processor_pixel_hash_binding_mismatch"))
        for metadata, hashes, error in variants:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                planner_timing_record(metadata, video_id="synthetic-video",
                                      processor_pixel_sha256=hashes)

    def test_nonmonotonic_full_pts_map_fails_even_with_matching_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            frames, metadata = self.fixture(temp)
            hashes = self.processor_hashes(frames)
        timing = metadata["full_timing_map"]
        timing["pts"][2] = timing["pts"][1]
        raw = json.dumps([[value, *timing["time_base"]] for value in timing["pts"]],
                         separators=(",", ":")).encode("ascii")
        digest = hashlib.sha256(raw).hexdigest()
        timing["decoded_pts_sequence_sha256"] = digest
        metadata["decoded_pts_sequence_sha256"] = digest
        with self.assertRaisesRegex(ValueError, "full_timing_map_pts_invalid"):
            planner_timing_record(metadata, video_id="synthetic-video",
                                  processor_pixel_sha256=hashes)

    def test_bool_numeric_fields_and_inconsistent_decoder_contract_fail(self):
        with tempfile.TemporaryDirectory() as temp:
            frames, original = self.fixture(temp)
            hashes = self.processor_hashes(frames)
        cases = (
            ("full_timing_map.schema_version", "full_timing_map_missing_or_invalid"),
            ("decoded_frame_count", "full_timing_map_metadata_mismatch"),
            ("decoded_time_base", "full_timing_map_metadata_mismatch"),
            ("requested_frame_count", "sampled_frame_count_invalid"),
            ("sampled_frames.0.decoded_frame_index", "sampled_timing_binding_mismatch"),
            ("sampled_frames.0.pts", "sampled_timing_binding_mismatch"),
            ("sampled_frames.0.relative_presentation_time.0",
             "sampled_timing_binding_mismatch"),
            ("sampler", "decoder_sampling_contract_invalid"),
            ("timestamp_origin", "decoder_sampling_contract_invalid"),
            ("execution_authorization", "decoder_execution_authorization_invalid"),
        )
        for field, error in cases:
            with self.subTest(field=field):
                metadata = copy.deepcopy(original)
                if field == "full_timing_map.schema_version":
                    metadata["full_timing_map"]["schema_version"] = True
                elif field == "decoded_frame_count":
                    metadata[field] = True
                elif field == "decoded_time_base":
                    metadata[field] = [True, 1000]
                elif field == "requested_frame_count":
                    metadata[field] = True
                elif field == "sampled_frames.0.decoded_frame_index":
                    metadata["sampled_frames"][0]["decoded_frame_index"] = False
                elif field == "sampled_frames.0.pts":
                    metadata["sampled_frames"][0]["pts"] = True
                elif field == "sampled_frames.0.relative_presentation_time.0":
                    metadata["sampled_frames"][0]["relative_presentation_time"][0] = False
                elif field == "sampler":
                    metadata[field] = "unrecognized_sampler"
                elif field == "timestamp_origin":
                    metadata[field] = "unknown_origin"
                else:
                    metadata[field] = "granted"
                with self.assertRaisesRegex(ValueError, error):
                    planner_timing_record(metadata, video_id="synthetic-video",
                                          processor_pixel_sha256=hashes)

    def test_requested_frame_count_below_two_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            frames, metadata = self.fixture(temp)
            hashes = self.processor_hashes(frames)
        metadata["requested_frame_count"] = 1
        with self.assertRaisesRegex(ValueError, "sampled_frame_count_invalid"):
            planner_timing_record(metadata, video_id="synthetic-video",
                                  processor_pixel_sha256=hashes)


if __name__ == "__main__":
    unittest.main()
