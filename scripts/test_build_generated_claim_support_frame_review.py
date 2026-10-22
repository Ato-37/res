from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from build_generated_claim_support_frame_review import (
    PRIVATE, _private_output, _validate_lock_pair_binding, _validate_structured_rows,
)


class GeneratedClaimReviewOutputPathTests(unittest.TestCase):
    def test_rejects_traversal_out_of_private_directory(self):
        with self.assertRaisesRegex(ValueError, "review_output_or_video_root_invalid"):
            _private_output(Path("data/private_nextgqa/../../experiments/out.html"))

    def test_rejects_absolute_output(self):
        with self.assertRaisesRegex(ValueError, "review_output_or_video_root_invalid"):
            _private_output(Path("/tmp/out.html"))

    def test_rejects_symlinked_output_component(self):
        with tempfile.TemporaryDirectory(dir=PRIVATE) as temp_dir:
            temp = Path(temp_dir)
            (temp / "escape").symlink_to("/tmp", target_is_directory=True)
            rel = Path("data/private_nextgqa") / temp.name / "escape" / "out.html"
            with self.assertRaisesRegex(ValueError, "review_output_or_video_root_invalid"):
                _private_output(rel)

    def test_accepts_new_file_directly_in_private_directory(self):
        output = _private_output(Path("data/private_nextgqa/test-safe-output.html"))
        self.assertTrue(output.parent.is_relative_to(output.parents[1]))

    def test_lock_must_bind_selected_pair_filter_and_strict_integer_overlap(self):
        lock = {"schema_version": 1,
                "status": "frozen_before_claim_generation_and_scoring",
                "selected_pair_filter_sha256": "a" * 64,
                "local_scored_claim_overlap_count": 0}
        _validate_lock_pair_binding(lock, "a" * 64)
        lock["selected_pair_filter_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "packet_lock_input_binding_invalid"):
            _validate_lock_pair_binding(lock, "a" * 64)
        lock["selected_pair_filter_sha256"] = "a" * 64
        lock["local_scored_claim_overlap_count"] = False
        with self.assertRaisesRegex(ValueError, "packet_lock_input_binding_invalid"):
            _validate_lock_pair_binding(lock, "a" * 64)

    def test_structured_count_must_match_actual_records_and_be_strict_integer(self):
        records = [{"generated_parsed": {"valid": True}},
                   {"generated_parsed": {"valid": False}}]
        packet = {"fully_structured_claim_records": 1, "status": "partial"}
        self.assertEqual(_validate_structured_rows(packet, records), 1)
        packet["status"] = "passed"
        self.assertEqual(_validate_structured_rows(packet, records[:1]), 1)
        packet["fully_structured_claim_records"] = 2
        with self.assertRaisesRegex(ValueError, "packet_structured_count_invalid"):
            _validate_structured_rows(packet, records)
        packet["fully_structured_claim_records"] = True
        packet["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "packet_structured_count_invalid"):
            _validate_structured_rows(packet, records)
        packet["fully_structured_claim_records"] = 2
        with self.assertRaisesRegex(ValueError, "packet_structured_count_invalid"):
            _validate_structured_rows(packet, records)


if __name__ == "__main__":
    unittest.main()
