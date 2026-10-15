from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_nextgqa_claim_packet_lineage import DATASET_KEYS
from audit_nextgqa_candidate_clock_extent import audit, _read_candidate_durations
from audit_qwen_pts_to_processor_cpu import encode_fixture


class CandidateClockExtentTests(unittest.TestCase):
    def test_duration_reader_skips_annotation_span_payloads(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsub.json"
            path.write_text('{"keep":{"duration":2,"location":{"hidden":[1,{"x":"}\\\"{"}]},"other":[true,false]},"skip":{"location":[1,2]}}')
            self.assertEqual(_read_candidate_durations(path, {"keep"}), {"keep": 2})

    def test_duration_reader_rejects_malformed_or_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "gsub.json"
            for raw in ('{"v":{"duration":2,}}', '{"v" {"duration":2}}',
                        '{"v":{"duration":2},}', '{"v":{"duration":NaN}}',
                        '{"v":{"duration":2,"duration":3}}',
                        '{"v":{"duration":2}} trailing'):
                path.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    _read_candidate_durations(path, {"v"})

    def fixture(self, root):
        repo = root / "repo"
        dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
        dataset.mkdir(parents=True)
        video_root = root / "videos"
        video_root.mkdir()
        mapping, gsub, rows = {}, {}, []
        for index, pts in enumerate(((100, 102, 104, 106, 108),
                                     (0, 3, 6, 9, 12), (0, 2, 4, 6, 8))):
            vid = f"synthetic-video-id-{index}"
            relative = f"group/synthetic-video-{index}.mkv"
            path = video_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            encode_fixture(path, pts=pts, time_base=Fraction(1, 10))
            media_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            question_sha = hashlib.sha256(f"question-{index}".encode()).hexdigest()
            claim_sha = hashlib.sha256(f"claim-{index}".encode()).hexdigest()
            claim_id = "claim-" + hashlib.sha256(
                (media_sha + "\0" + f"qid-{index}" + "\0" + claim_sha).encode()
            ).hexdigest()
            mapping[vid] = relative
            gsub[vid] = {"duration": 2, "location": {"hidden": "not exported"}}
            rows.append({
                "video_id": "video-" + media_sha, "claim_id": claim_id,
                "dataset_question_id": f"qid-{index}", "dataset_video_id": vid,
                "video_key": f"group/synthetic-video-{index}",
                "relative_video_path": relative, "video_sha256": media_sha,
                "question_sha256": question_sha, "claim_text_sha256": claim_sha,
            })
        (dataset / "test.csv").write_text("synthetic fixture only\n")
        (dataset / "gsub_test.json").write_text(json.dumps(gsub))
        (dataset / "map_vid_vidorID.json").write_text(json.dumps(mapping))
        (dataset / "frame2time_test.json").write_text("{}")
        dataset_hashes = {key: hashlib.sha256((dataset / key).read_bytes()).hexdigest()
                          for key in DATASET_KEYS}
        manifest = {
            "schema_version": 1,
            "purpose": "private_source_lineage_crosscheck_for_claim_review",
            "created_by": "synthetic-test",
            "source_packet_sha256": "a" * 64,
            "cohort_sha256": "b" * 64,
            "dataset_inputs_sha256": dataset_hashes,
            "requested_records": 3, "effective_records": 3,
            "verification_scope": "synthetic only", "status": "metadata_lineage_crosschecked",
            "records": rows,
        }
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        return manifest, raw, repo, video_root

    def test_exact_pts_summary_is_aggregate_only_and_never_calibrates_clock(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, raw, repo, video_root = self.fixture(Path(td))
            report = audit(manifest, raw, hashlib.sha256(raw).hexdigest(), repo, video_root)
        self.assertEqual(report["status"], "metadata_decoder_extent_compared_not_calibrated")
        self.assertEqual(report["requested_records"], 3)
        self.assertEqual(report["effective_records"], 3)
        self.assertEqual(report["decode_failures"], 0)
        self.assertEqual(report["decoded_frame_counts"], [5, 5, 5])
        self.assertEqual(report["pts_strictly_increasing_count"], 3)
        self.assertEqual(report["codec_counts"], {"ffv1": 3})
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertFalse(report["annotation_clock_calibrated"])
        self.assertIn("preceding PTS interval", report["interpretation_limit"])
        serialized = json.dumps(report)
        for forbidden in ("synthetic-video-id", "synthetic-video-0.mkv", "question-0",
                          "claim-0", "location", "hidden", '"records":'):
            self.assertNotIn(forbidden, serialized)

    def test_manifest_hash_and_media_hash_mismatches_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            manifest, raw, repo, video_root = self.fixture(Path(td))
            with self.assertRaisesRegex(ValueError, "lineage_manifest_invalid_or_hash_mismatch"):
                audit(manifest, raw, "0" * 64, repo, video_root)
            target = video_root / manifest["records"][0]["relative_video_path"]
            target.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source_media_hash_mismatch_or_duplicate"):
                audit(manifest, raw, hashlib.sha256(raw).hexdigest(), repo, video_root)


if __name__ == "__main__":
    unittest.main()
