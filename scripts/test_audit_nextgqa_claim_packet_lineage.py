import csv
import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from audit_nextgqa_claim_packet_lineage import DATASET_KEYS, audit, main
from test_build_nextgqa_claim_mapping_packet import packet


class AuditNextgqaClaimPacketLineageTests(unittest.TestCase):
    def fixture(self, root):
        repo = root / "repo"
        ds = repo / "NExT-GQA" / "datasets" / "nextgqa"
        ds.mkdir(parents=True)
        video_root = root / "videos"
        records = []
        mapping = {}
        csv_path = ds / "test.csv"
        for i in range(3):
            relative = f"group/video-{i}.mp4"
            path = video_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            media = f"synthetic media {i}".encode()
            path.write_bytes(media)
            mapping[f"vid-{i}"] = relative
            records.append({"qid": f"qid-{i}", "video_id": f"vid-{i}",
                            "question": f"Synthetic question {i}?"})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "qid", "question"])
            writer.writeheader(); writer.writerows(records)
        for name, content in (("gsub_test.json", b"{}"),
                              ("map_vid_vidorID.json", json.dumps(mapping).encode()),
                              ("frame2time_test.json", b"{}")):
            (ds / name).write_bytes(content)
        value, _ = packet(3)
        for i, row in enumerate(value["records"]):
            path = video_root / f"group/video-{i}.mp4"
            row["local_video_path"] = str(path)
            row["video_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["dataset_question_id"] = f"qid-{i}"
            row["question"] = f"Synthetic question {i}?"
        value["dataset_inputs_sha256"] = {
            name: hashlib.sha256((ds / name).read_bytes()).hexdigest() for name in DATASET_KEYS
        }
        raw = (json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n").encode()
        return value, raw, repo, video_root

    def test_crosschecks_video_question_and_media_without_copying_gold(self):
        with tempfile.TemporaryDirectory() as td:
            value, raw, repo, video_root = self.fixture(Path(td))
            manifest, encoded = audit(value, raw, repo, video_root)
            self.assertEqual(manifest["status"], "metadata_lineage_crosschecked")
            self.assertEqual(manifest["effective_records"], 3)
            for forbidden in ("gold_answer", "gold_interval_seconds", "choices", "claim text"):
                self.assertNotIn(forbidden, encoded.decode())
            self.assertTrue(all(r["relative_video_path"].endswith(".mp4")
                                for r in manifest["records"]))

    def test_question_video_or_media_mismatch_fails_without_details(self):
        with tempfile.TemporaryDirectory() as td:
            value, raw, repo, video_root = self.fixture(Path(td))
            value["records"][0]["question"] = "different question"
            with self.assertRaisesRegex(ValueError, "dataset_video_question_binding_mismatch"):
                audit(value, json.dumps(value).encode(), repo, video_root)
            value, raw, repo, video_root = self.fixture(Path(td) / "second")
            Path(value["records"][0]["local_video_path"]).write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "source_video_hash_mismatch_or_duplicate"):
                audit(value, raw, repo, video_root)

    def test_cli_requires_private_packet_and_writes_exclusive_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            value, raw, repo, video_root = self.fixture(Path(td))
            packet_path = Path(td) / "packet.json"
            output = Path(td) / "lineage.json"
            packet_path.write_bytes(raw); os.chmod(packet_path, 0o600)
            args = ["--packet", str(packet_path), "--repo", str(repo),
                    "--video-root", str(video_root), "--output", str(output),
                    "--user-authorized-local-only"]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            manifest = json.loads(output.read_bytes())
            self.assertEqual(manifest["effective_records"], 3)
            self.assertEqual(manifest["status"], "metadata_lineage_crosschecked")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            output.unlink()
            os.chmod(packet_path, 0o644)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
