from pathlib import Path
import hashlib
import json
import os
import stat
import tempfile
import unittest
from unittest import mock
import uuid

import run_qta_real_video_sampler_preflight as runner
from run_qta_real_video_sampler_preflight import (
    DEFAULT_MEDIA_DIR,
    PRIVATE_ROOT,
    build_report,
    load_candidate_inventory,
    write_private_report,
)


class QtaRealVideoSamplerPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PRIVATE_ROOT)
        self.directory = Path(self.temp.name)
        os.chmod(self.directory, 0o700)

    def tearDown(self):
        self.temp.cleanup()

    def _files(self, count=3):
        paths = []
        for index in range(count):
            path = self.directory / (f"{index:024d}.mp4")
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            paths.append(path)
        return paths

    def test_requires_exactly_three_media_and_keeps_fixed_denominator(self):
        self._files(2)
        with mock.patch("run_qta_real_video_sampler_preflight.DEFAULT_MEDIA_DIR",
                        self.directory):
            with mock.patch("run_qta_real_video_sampler_preflight.load_candidate_inventory",
                            return_value=({f"staged_media_{i:02d}": "a" * 64
                                           for i in range(1, 4)}, "b" * 64)):
                with self.assertRaisesRegex(
                        ValueError, "expected_exactly_three_private_mp4_files"):
                    build_report(self.directory, self.directory / "inventory.json")

    def test_builds_three_row_report_without_model_authorization(self):
        self._files(3)
        with mock.patch("run_qta_real_video_sampler_preflight.DEFAULT_MEDIA_DIR",
                        self.directory):
            with mock.patch("run_qta_real_video_sampler_preflight.load_candidate_inventory",
                            return_value=({f"staged_media_{i:02d}": "a" * 64
                                           for i in range(1, 4)}, "b" * 64)):
                with mock.patch("run_qta_real_video_sampler_preflight._decode_row") as decode:
                    decode.side_effect = lambda index, _, expected: {
                        "record_index": index, "expected_sha256": expected}
                    result = build_report(self.directory, self.directory / "inventory.json")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["requested_records"], 3)
        self.assertEqual(result["effective_records"], 3)
        self.assertFalse(result["gpu_used"])
        self.assertFalse(result["model_loaded"])
        self.assertFalse(result["faithfulness_scored"])
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(decode.call_count, 3)

    def test_writer_is_private_exclusive_and_never_overwrites(self):
        output = PRIVATE_ROOT / f"test_qta_sampler_{uuid.uuid4().hex}.json"
        report = {"purpose": "test", "fraction": [1, 2]}
        try:
            write_private_report(report, output)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaisesRegex(ValueError, "output_already_exists"):
                write_private_report(report, output)
        finally:
            output.unlink(missing_ok=True)

    def test_writer_refuses_non_private_output_parent(self):
        with tempfile.TemporaryDirectory() as outside:
            with self.assertRaisesRegex(
                    ValueError, "output_must_be_direct_child_of_private_root"):
                write_private_report({"purpose": "test"}, Path(outside) / "out.json")

    def test_inventory_binds_candidate_media_and_current_sources(self):
        role_files = {
            "qta_protocol_draft": runner.REPO_ROOT / "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
            "review_runbook": runner.REPO_ROOT / "docs/nextgqa_v6_human_review_runbook.md",
            "video_decoder": runner.REPO_ROOT / "scripts/claim_mask_video_decoder.py",
            "necessity_choice_margin_utils": runner.REPO_ROOT / "scripts/necessity_choice_margin_utils.py",
            "presentation_time_utils": runner.REPO_ROOT / "scripts/presentation_time_utils.py",
            "qta_real_video_sampler_preflight": Path(runner.__file__).resolve(),
        }
        artifacts = []
        for role, path in role_files.items():
            artifacts.append({"role": role,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                              "size_bytes": path.stat().st_size})
        for index in range(1, 4):
            artifacts.append({"role": f"staged_media_{index:02d}",
                              "sha256": str(index) * 64, "size_bytes": 10})
        candidate = {
            "purpose": "pre_review_candidate_byte_inventory",
            "status": "candidate_only", "execution_authorization": "not_granted",
            "joint_freeze": False, "score_ready": False,
            "rights_status": "unverified_local_only",
            "missing_required_slots": [], "artifacts": artifacts,
        }
        path = self.directory / "candidate.json"
        path.write_text(json.dumps(candidate), encoding="utf-8")
        os.chmod(path, 0o600)
        hashes, digest = load_candidate_inventory(path)
        self.assertEqual(hashes, {f"staged_media_{i:02d}": str(i) * 64
                                  for i in range(1, 4)})
        self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
        candidate["artifacts"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(candidate), encoding="utf-8")
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ValueError, "candidate_inventory_source_binding_stale"):
            load_candidate_inventory(path)

    def test_sorted_media_rows_bind_to_inventory_roles_and_hash_mismatch_fails(self):
        paths = self._files(3)
        expected = {"staged_media_01": "1" * 64,
                    "staged_media_02": "2" * 64,
                    "staged_media_03": "3" * 64}
        actual = ["1" * 64, "2" * 64, "3" * 64]
        calls = []

        def decode(index, path, digest):
            calls.append((index, path.name, digest))
            if digest != actual[index - 1]:
                raise ValueError("source_inventory_hash_mismatch")
            return {"record_index": index}

        with mock.patch("run_qta_real_video_sampler_preflight.DEFAULT_MEDIA_DIR",
                        self.directory):
            with mock.patch("run_qta_real_video_sampler_preflight.load_candidate_inventory",
                            return_value=(expected, "b" * 64)):
                with mock.patch("run_qta_real_video_sampler_preflight._decode_row",
                                side_effect=decode):
                    report = build_report(self.directory, self.directory / "inventory.json")
                self.assertEqual(report["effective_records"], 3)
                self.assertEqual([call[0] for call in calls], [1, 2, 3])
                self.assertEqual([call[1] for call in calls],
                                 [path.name for path in sorted(paths)])
                self.assertEqual([call[2] for call in calls],
                                 [expected[f"staged_media_{i:02d}"] for i in range(1, 4)])
                expected["staged_media_02"] = "f" * 64
                with mock.patch("run_qta_real_video_sampler_preflight._decode_row",
                                side_effect=decode):
                    with self.assertRaisesRegex(
                            ValueError, "source_inventory_hash_mismatch"):
                        build_report(self.directory, self.directory / "inventory.json")


if __name__ == "__main__":
    unittest.main()
