"""CPU-only tests for private NOAA human review packet generation."""

import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import subprocess
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_noaa_qa_gate as gate  # noqa: E402
from build_noaa_human_review_packet import (  # noqa: E402
    PacketError, _prepare_silent_video, build_review_packet,
)


def _raw_json(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def make_fixture(root, *, malicious_question=False, bad_join=False, bad_video_path=False):
    root.mkdir(parents=True, exist_ok=True)
    (root / "private").mkdir(exist_ok=True)
    groups = []
    for group_index, group_id in enumerate(("dive03", "dive04", "dive09")):
        directory = "." if group_index == 0 else group_id
        group_root = root / directory
        media_path = group_root / "media" / "source.mp4"
        private = group_root / "private"
        media_path.parent.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        media = f"synthetic private clip {group_index}".encode()
        media_path.write_bytes(media)
        media_hash = _sha(media)
        video_id = f"fixture-video-{group_index}"
        samples = []
        gold = []
        audit_items = []
        for item_index in range(2):
            qid = f"fixture-{group_index}-{item_index}"
            question = ("</script><img src=x onerror=alert(1)>" if malicious_question and item_index == 0
                       else f"Fixture question {qid}?")
            sample_video = "/different/path.mp4" if bad_video_path and group_index == 0 else str(media_path)
            samples.append({
                "qid": qid,
                "video_id": video_id,
                "video": sample_video,
                "video_sha256": media_hash,
                "question": question,
                "choices": [f"Option {letter}" for letter in "ABCDE"],
                "window_start_seconds": float(item_index + 1),
                "window_end_seconds": float(item_index + 2),
            })
            gold.append({
                "qid": "wrong-qid" if bad_join and group_index == 0 and item_index == 1 else qid,
                "video_id": video_id,
                "answer_choice_index": item_index,
            })
            audit_items.append({
                "qid": qid,
                "independent_review": {"reviewer_role": "AI blind reviewer"},
            })
        sample_raw = _raw_json(samples)
        gold_raw = _raw_json(gold)
        audit_raw = _raw_json({"method": "AI blind review", "items": audit_items})
        rights_raw = _raw_json({
            "media_sha256": media_hash,
            "video_id": video_id,
            "audio_used": False,
            "human_verified_manifest_status": "pending: manual review required",
        })
        (private / "samples.json").write_bytes(sample_raw)
        (private / "gold.json").write_bytes(gold_raw)
        (private / "annotation_audit_v2.json").write_bytes(audit_raw)
        (private / "rights_review_v2.json").write_bytes(rights_raw)
        groups.append({
            "group_id": group_id,
            "directory": directory,
            "audit_file": "annotation_audit_v2.json",
            "media_sha256": media_hash,
            "rights_sha256": _sha(rights_raw),
        })
    return tuple(groups)


def embedded_json(page, element_id):
    match = re.search(
        rf'<script id="{re.escape(element_id)}" type="application/json">(.*?)</script>',
        page,
        flags=re.DOTALL,
    )
    if not match:
        raise AssertionError(f"missing embedded JSON element {element_id}")
    return json.loads(match.group(1))


class BuildNoaaHumanReviewPacketTests(unittest.TestCase):
    def _build_fixture(self, root, **kwargs):
        groups = make_fixture(root, **kwargs)
        return mock.patch.object(gate, "GROUPS", groups)

    def _mock_media_preparer(self, source, output):
        raw = Path(source).read_bytes()
        Path(output).write_bytes(raw)
        digest = _sha(raw)
        return {
            "original_media_sha256": digest,
            "silent_copy_sha256": _sha(raw),
            "source_video_packet_sha256": digest,
            "copy_video_packet_sha256": digest,
            "source_video_start_time": 0.0,
            "copy_video_start_time": 0.0,
            "source_video_duration": 2.0,
            "copy_video_duration": 2.0,
            "source_video_nb_frames": 2,
            "copy_video_nb_frames": 2,
            "video_pts_count": 2,
            "source_audio_streams_removed": 4,
            "copy_audio_streams": 0,
        }

    def _build(self, root, packet_name=None):
        return build_review_packet(root, packet_name, media_preparer=self._mock_media_preparer)

    def test_unsigned_blank_template_hashes_qids_and_private_local_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "noaa"
            group_patch = self._build_fixture(root)
            with group_patch:
                index, counts = self._build(root)
            page = index.read_text(encoding="utf-8")
            gold_page = (index.parent / "gold_reference.html").read_text(encoding="utf-8")
            template = embedded_json(page, "attestation-template")
            review_data = embedded_json(page, "review-data")

        self.assertEqual(counts, {
            "groups": 3, "review_items": 6, "attestation_status": "missing",
            "silent_copy_audio_streams": 0})
        self.assertEqual(template["signed"], False)
        self.assertIsNone(template["signed_at_utc"])
        self.assertEqual(template["attestor"], {"name": "", "role": "", "human": False})
        self.assertIsNone(template["first_pass"])
        self.assertEqual(sum(x["copy_audio_streams"] for x in template["review_media"]), 0)
        self.assertEqual(len(template["groups"]), 3)
        self.assertEqual(len(review_data["groups"]), 3)
        for group in template["groups"]:
            self.assertEqual(group["reviewed_qids"], [])
            self.assertEqual(set(group["checks"]), set(gate.REQUIRED_CHECKS))
            self.assertTrue(all(value is False for value in group["checks"].values()))
            self.assertEqual(set(group["rights_scope"]), {
                "video_only_model_inference", "audio_use", "external_media_publication",
                "required_credit"})
            for field in ("media_sha256", "samples_sha256", "gold_sha256",
                          "audit_sha256", "rights_sha256"):
                self.assertRegex(group[field], r"^[a-f0-9]{64}$")
        self.assertTrue(all(len(group["items"]) == 2 for group in review_data["groups"]))
        self.assertIn('value=""', page)
        self.assertIn('id="human-claim" type="checkbox"', page)
        self.assertNotRegex(page, r'<input[^>]+\bchecked(?:\s|>)')
        self.assertRegex(page, r'<button id="download" type="button" disabled>')
        self.assertTrue(index.parent.resolve().is_relative_to((root / "private").resolve()))
        self.assertIn('src="assets/dive03_silent.mp4"', page)
        self.assertNotIn('src="../../media/source.mp4"', page)
        self.assertNotIn(str(root.resolve()), page)
        self.assertNotIn("gold_choice_index", page)
        self.assertNotIn("gold_choice_index", json.dumps(review_data))
        self.assertIn("not secure blinding", page)
        self.assertIn("source_video_packet_sha256", page)
        self.assertIn("gold_reference.html", page)
        self.assertIn("first-pass JSON", gold_page)
        self.assertIn("not secure blinding", gold_page.lower())
        self.assertFalse(list(index.parent.glob("*attestation*.json")))
        self.assertFalse((root / "results").exists())

    def test_escapes_question_markup_and_json_script_terminator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "noaa"
            group_patch = self._build_fixture(root, malicious_question=True)
            with group_patch:
                index, _ = self._build(root)
            page = index.read_text(encoding="utf-8")
            review_data = embedded_json(page, "review-data")

        question = review_data["groups"][0]["items"][0]["question"]
        self.assertEqual(question, "</script><img src=x onerror=alert(1)>")
        self.assertNotIn("</script><img", page)
        self.assertNotIn("<img src=x", page)
        self.assertIn("&lt;/script&gt;&lt;img", page)
        self.assertIn(r"\u003c/script\u003e", page)

    def test_refuses_data_join_failure_and_video_path_mismatch(self):
        for fixture_options in ({"bad_join": True}, {"bad_video_path": True}):
            with self.subTest(fixture_options=fixture_options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "noaa"
                group_patch = self._build_fixture(root, **fixture_options)
                with group_patch:
                    with self.assertRaises(PacketError):
                        self._build(root)
                self.assertEqual(list((root / "private").glob("review_packet_*")), [])

    def test_existing_packet_directory_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "noaa"
            group_patch = self._build_fixture(root)
            existing = root / "private" / "review_packet_taken"
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("do not overwrite")
            with group_patch:
                with self.assertRaises(FileExistsError):
                    self._build(root, packet_name="review_packet_taken")
            self.assertEqual(marker.read_text(), "do not overwrite")

    def test_refuses_media_symlink_outside_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "noaa"
            group_patch = self._build_fixture(root)
            media = root / "media" / "source.mp4"
            outside = base / "outside.mp4"
            outside.write_bytes(media.read_bytes())
            media.unlink()
            media.symlink_to(outside)
            with group_patch:
                with self.assertRaisesRegex(PacketError, "outside the private dataset root"):
                    self._build(root)
            self.assertEqual(list((root / "private").glob("review_packet_*")), [])

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"),
                         "ffmpeg and ffprobe are needed for this CPU media test")
    def test_silent_copy_removes_audio_and_preserves_video_packets_and_timestamps(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            output = root / "silent.mp4"
            command = [
                shutil.which("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "lavfi", "-i", "color=c=black:s=64x64:r=2:d=1",
                "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=8000",
                "-map", "0:v:0", "-map", "1:a:0", "-map", "1:a:0", "-map", "1:a:0", "-map", "1:a:0",
                "-t", "1", "-c:v", "mpeg4", "-q:v", "5", "-c:a", "aac", str(source),
            ]
            subprocess.run(command, check=True, capture_output=True)
            provenance = _prepare_silent_video(source, output)
            self.assertEqual(provenance["source_audio_streams_removed"], 4)
            self.assertEqual(provenance["copy_audio_streams"], 0)
            self.assertEqual(provenance["source_video_packet_sha256"],
                             provenance["copy_video_packet_sha256"])
            self.assertEqual(provenance["source_video_nb_frames"],
                             provenance["copy_video_nb_frames"])
            self.assertEqual(provenance["video_pts_count"],
                             provenance["source_video_nb_frames"])
            self.assertEqual(provenance["original_media_sha256"], _sha(source.read_bytes()))
            self.assertEqual(provenance["silent_copy_sha256"], _sha(output.read_bytes()))


if __name__ == "__main__":
    unittest.main()
