import json
import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.build_samples import (
    selection_audit_record,
    write_private_json_exclusive,
    write_sample_and_audit_exclusive,
    main,
)


class BuildSamplesProvenanceTests(unittest.TestCase):
    def test_sidecar_binds_inputs_config_counts_and_output_without_private_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_csv = root / "test.csv"
            gsub = root / "grounding.json"
            video_map = root / "video_map.json"
            manifest = root / "samples.json"
            test_csv.write_text("video_id,question\nprivate-id,private question\n", encoding="utf-8")
            gsub.write_text('{"private-id": {}}', encoding="utf-8")
            video_map.write_text('{"private-id": "private/path"}', encoding="utf-8")
            write_private_json_exclusive(manifest, [{"video_id": "private-id",
                                                     "question": "private question",
                                                     "video": "/private/path.mp4"}])
            args = SimpleNamespace(test_csv=str(test_csv), gsub_json=str(gsub),
                                   map_json=str(video_map), exclude=[], zip_path=None,
                                   seed=17, n=2, unique_videos=True)

            with patch.dict(os.environ, {"OPENCV_FFMPEG_CAPTURE_OPTIONS": "private-option"}):
                record = selection_audit_record(
                    args=args, effective_zip_path=None, candidates_before_exclusion=5,
                    preexcluded=1, candidates_after_exclusion=4,
                    exclusion_counts={"missing_video_map": 2},
                    built_exclusions={"video_unreadable": 1}, results=[{}, {}],
                    manifest_path=manifest,
                )
            serialized = json.dumps(record)
            self.assertEqual(record["status"], "complete")
            self.assertEqual(record["configuration"]["seed"], 17)
            self.assertEqual(record["selection_counts"]["candidate_rows_excluded_as_previously_used"], 1)
            self.assertEqual(record["selection_counts"]["exclusions_by_reason"],
                             {"missing_video_map": 2, "video_unreadable": 1})
            self.assertIn("not an all-eligible-video enumerator", record["selection_semantics"])
            self.assertNotIn("private-id", serialized)
            self.assertNotIn("private question", serialized)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn("private-option", serialized)
            self.assertEqual(record["configuration"]["opencv_decoder_environment_sha256"]
                             ["OPENCV_FFMPEG_CAPTURE_OPTIONS"],
                             hashlib.sha256(b"private-option").hexdigest())

    def test_nonpositive_sample_count_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "must_not_exist.json"
            argv = ["build_samples.py", "--test-csv", "missing.csv", "--gsub-json",
                    "missing.json", "--map-json", "missing-map.json", "--video-root",
                    tmp, "--n", "0", "--out", str(output)]
            with patch("sys.argv", argv), self.assertRaises(SystemExit) as raised:
                main()
            self.assertEqual(raised.exception.code, 2)
            self.assertFalse(output.exists())

    def test_underfilled_status_and_private_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "private.json"
            write_private_json_exclusive(target, {"count": 1})
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_private_json_exclusive(target, {"count": 2})
            self.assertEqual(json.loads(target.read_text()), {"count": 1})

    def test_pair_writer_binds_manifest_and_keeps_both_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples.json"
            audit = Path(tmp) / "audit.json"
            write_sample_and_audit_exclusive(
                sample, [{"count": 1}], audit,
                lambda staged: {"sample_manifest_sha256": hashlib.sha256(
                    Path(staged).read_bytes()).hexdigest()},
            )
            self.assertEqual(os.stat(sample).st_mode & 0o777, 0o600)
            self.assertEqual(os.stat(audit).st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(audit.read_text())["sample_manifest_sha256"],
                             hashlib.sha256(sample.read_bytes()).hexdigest())
            self.assertFalse(list(Path(tmp).glob(".selection_tmp_*")))

    def test_pair_writer_rolls_back_partial_publish_and_preserves_collision(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples.json"
            audit = Path(tmp) / "audit.json"
            audit.write_text('{"preserve": true}\n', encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_sample_and_audit_exclusive(
                    sample, [{"count": 1}], audit, lambda _staged: {"ok": True},
                )
            self.assertFalse(sample.exists())
            self.assertEqual(audit.read_text(encoding="utf-8"), '{"preserve": true}\n')
            self.assertFalse(list(Path(tmp).glob(".selection_tmp_*")))

    def test_pair_writer_cleans_up_when_audit_construction_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples.json"
            audit = Path(tmp) / "audit.json"

            def fail(_staged):
                raise RuntimeError("synthetic audit serialization failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                write_sample_and_audit_exclusive(sample, [{"count": 1}], audit, fail)
            self.assertFalse(sample.exists())
            self.assertFalse(audit.exists())
            self.assertFalse(list(Path(tmp).glob(".selection_tmp_*")))

    def test_pair_writer_rolls_back_on_noncollision_second_link_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "samples.json"
            audit = Path(tmp) / "audit.json"
            real_link = os.link
            calls = 0

            def fail_second_link(source, target):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise PermissionError("synthetic second-link failure")
                return real_link(source, target)

            with patch("scripts.build_samples.os.link", side_effect=fail_second_link):
                with self.assertRaisesRegex(PermissionError, "synthetic"):
                    write_sample_and_audit_exclusive(
                        sample, [{"count": 1}], audit, lambda _staged: {"ok": True},
                    )
            self.assertFalse(sample.exists())
            self.assertFalse(audit.exists())
            self.assertFalse(list(Path(tmp).glob(".selection_tmp_*")))


if __name__ == "__main__":
    unittest.main()
