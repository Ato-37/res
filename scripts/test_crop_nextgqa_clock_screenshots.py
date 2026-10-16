from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import tempfile
import unittest
from unittest.mock import patch

import av
from PIL import Image

from crop_nextgqa_clock_screenshots import SOURCE_FILES, crop_screenshots


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def enc(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


class ScreenshotCropTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.meta = self.root / "meta"
        self.shots = self.root / "shots"
        self.out_parent = self.root / "out"
        for directory in (self.meta, self.shots, self.out_parent):
            directory.mkdir(mode=0o700)
        self.tokens = [f"tok_{i:020d}"[:24] for i in range(3)]
        anchor_order = ("fit_begin", "heldout_quarter", "fit_midpoint",
                        "heldout_three_quarter", "fit_end")
        fit = {"fit_begin", "fit_midpoint", "fit_end"}
        self.assignments = {(token, anchor) for token in self.tokens for anchor in anchor_order}
        self.plan = {"schema_version": 1, "purpose": "predeclared_html5_clock_pts_anchor_plan",
                     "status": "predeclared_frozen", "execution_authorization": "not_granted",
                     "records": [{"review_token": token, "source_media_sha256": "a" * 64,
                                  "fit_anchors": [{"anchor_id": a, "decoder_pts_seconds": "1"}
                                                  for a in anchor_order if a in fit],
                                  "heldout_anchors": [{"anchor_id": a, "decoder_pts_seconds": "1"}
                                                      for a in anchor_order if a not in fit]}
                                 for token in self.tokens]}
        self.plan_raw = enc(self.plan)
        self.plan_path = self.meta / "plan.json"
        self.write_private(self.plan_path, self.plan_raw)
        self.plan_sha = sha(self.plan_raw)
        self.provenance = {"schema_version": 1,
            "purpose": "private_clock_anchor_plan_candidate_provenance",
            "status": "candidate_not_jointly_frozen", "execution_authorization": "not_granted",
            "input_sha256": {"worksheet": "1" * 64, "lineage_manifest": "2" * 64,
                             "custody_manifest": "3" * 64},
            "source_sha256": {key: sha((Path(__file__).resolve().parent / filename).read_bytes())
                               for key, filename in SOURCE_FILES.items()},
            "plan_sha256": self.plan_sha,
            "created_at_utc": "2026-10-17T00:00:00Z", "python_version": platform.python_version(),
            "pyav_version": av.__version__,
            "ffmpeg_library_versions": {key: list(value) for key, value in av.library_versions.items()},
            "joint_freeze_receipt_sha256": None,
            "records": [{"review_token": token, "staged_media_sha256": "a" * 64,
                         "decoded_frame_count": 10, "decoded_pts_sequence_sha256": "c" * 64,
                         "time_base": [1, 10], "pixel_spec": [100, 80, "rgb24"],
                         "decoder_stream_info": {}} for token in self.tokens]}
        self.prov_raw = enc(self.provenance)
        self.prov_path = self.meta / "provenance.json"
        self.write_private(self.prov_path, self.prov_raw)
        self.prov_sha = sha(self.prov_raw)
        self.receipt = {"schema_version": 1,
            "purpose": "private_windows_compositor_screenshot_capture_receipt_candidate",
            "status": "self_reported_capture_receipt_not_independently_authenticated",
            "execution_authorization": "not_granted", "plan_sha256": self.plan_sha,
            "plan_provenance_sha256": self.prov_sha,
            "capture_method": "windows_gdi_copyfromscreen_foreground_window_v1",
            "capture_helper_file_sha256": "b" * 64,
            "created_at_utc": "2026-10-17T00:00:00.000Z", "records": []}
        self.crop_map = {"schema_version": 1,
            "purpose": "private_candidate_screenshot_video_crop_map",
            "plan_sha256": self.plan_sha, "plan_provenance_sha256": self.prov_sha,
            "records": []}
        for token, anchor in sorted(self.assignments):
            filename = token + "_" + anchor + ".png"
            path = self.shots / filename
            image = Image.new("RGB", (100, 80), (20, 40, 60))
            image.save(path, format="PNG")
            os.chmod(path, 0o600)
            raw = path.read_bytes()
            self.receipt["records"].append({
                "review_token": token, "anchor_id": anchor, "screenshot_file": filename,
                "screenshot_sha256": sha(raw), "screenshot_captured_at_utc": "2026-10-17T00:00:00.000Z",
                "capture_method": self.receipt["capture_method"], "foreground_process": "msedge",
                "foreground_pid": 1, "foreground_window_title": "Private offline video review",
                "foreground_window_rect_xywh": [0, 0, 100, 80],
                "virtual_screen_rect_xywh": [0, 0, 100, 80],
                "hotkey_message_tick_ms": 1, "capture_complete_tick_ms": 2,
                "hotkey_to_capture_ms": 1})
            self.crop_map["records"].append({"review_token": token, "anchor_id": anchor,
                                               "bbox_xywh": [10, 10, 30, 25]})
        self.receipt_raw = enc(self.receipt)
        self.receipt_path = self.meta / "receipt.json"
        self.write_private(self.receipt_path, self.receipt_raw)
        self.receipt_sha = sha(self.receipt_raw)
        self.crop_map["capture_receipt_sha256"] = self.receipt_sha
        self.crop_raw = enc(self.crop_map)
        self.crop_path = self.meta / "crop-map.json"
        self.write_private(self.crop_path, self.crop_raw)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def write_private(path: Path, raw: bytes):
        path.write_bytes(raw)
        os.chmod(path, 0o600)

    def run_crop(self, name="result", **kwargs):
        values = {"plan_path": self.plan_path, "expected_plan_sha": self.plan_sha,
                  "provenance_path": self.prov_path, "expected_provenance_sha": self.prov_sha,
                  "receipt_path": self.receipt_path, "expected_receipt_sha": self.receipt_sha,
                  "screenshot_dir": self.shots, "crop_map_path": self.crop_path,
                  "output_dir": self.out_parent / name}
        values.update(kwargs)
        return crop_screenshots(**values)

    def refresh_provenance_bindings(self):
        self.prov_raw = enc(self.provenance)
        self.prov_sha = sha(self.prov_raw)
        self.write_private(self.prov_path, self.prov_raw)
        self.receipt["plan_provenance_sha256"] = self.prov_sha
        self.receipt_raw = enc(self.receipt)
        self.receipt_sha = sha(self.receipt_raw)
        self.write_private(self.receipt_path, self.receipt_raw)
        self.crop_map["plan_provenance_sha256"] = self.prov_sha
        self.crop_map["capture_receipt_sha256"] = self.receipt_sha
        self.write_private(self.crop_path, enc(self.crop_map))

    def test_outputs_exact_private_rgb_crops_and_crosswalk(self):
        result = self.run_crop()
        output = self.out_parent / "result"
        self.assertEqual(result["records"], 15)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        pngs = list(output.glob("*.png"))
        self.assertEqual(len(pngs), 15)
        expected_names = {token + "_" + anchor + ".png" for token, anchor in self.assignments}
        self.assertEqual({path.name for path in pngs}, expected_names)
        for path in pngs:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (30, 25))
        crosswalk = json.loads((output / "crop_crosswalk.json").read_bytes())
        self.assertEqual(crosswalk["bbox_video_only_purity"], "not_established_human_review_required")
        self.assertEqual(len(crosswalk["records"]), 15)
        self.assertEqual({row["crop_file"] for row in crosswalk["records"]}, expected_names)
        self.assertEqual({(row["review_token"], row["anchor_id"])
                          for row in crosswalk["records"]}, self.assignments)
        for row in crosswalk["records"]:
            self.assertIn((row["review_token"], row["anchor_id"]), self.assignments)
            self.assertEqual(row["crop_file"], row["review_token"] + "_" + row["anchor_id"] + ".png")
            self.assertEqual(sha((output / row["crop_file"]).read_bytes()), row["crop_sha256"])
        self.assertEqual(stat.S_IMODE((output / "crop_crosswalk.json").stat().st_mode), 0o600)

    def test_rejects_plan_provenance_and_receipt_binding_mismatch(self):
        with self.assertRaisesRegex(ValueError, "plan_sha256_mismatch"):
            self.run_crop("bad-plan", expected_plan_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "provenance_sha256_mismatch"):
            self.run_crop("bad-prov", expected_provenance_sha="0" * 64)
        with self.assertRaisesRegex(ValueError, "capture_receipt_sha256_mismatch"):
            self.run_crop("bad-receipt", expected_receipt_sha="0" * 64)
        self.assertFalse(any(self.out_parent.iterdir()))

    def test_rejects_provenance_source_and_media_alignment_drift(self):
        original = self.provenance["source_sha256"]["decoder"]
        self.provenance["source_sha256"]["decoder"] = "0" * 64
        self.refresh_provenance_bindings()
        with self.assertRaisesRegex(ValueError, "provenance_source_hash_mismatch"):
            self.run_crop()
        self.provenance["source_sha256"]["decoder"] = original
        self.provenance["records"][0]["staged_media_sha256"] = "0" * 64
        self.refresh_provenance_bindings()
        with self.assertRaisesRegex(ValueError, "provenance_records_invalid"):
            self.run_crop()
        self.provenance["records"][0]["staged_media_sha256"] = "a" * 64
        self.provenance["input_sha256"]["lineage_manifest"] = "invalid"
        self.refresh_provenance_bindings()
        with self.assertRaisesRegex(ValueError, "provenance_input_hashes_invalid"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())

    def test_rejects_provenance_python_version_drift(self):
        self.provenance["python_version"] = "0.0.0"
        self.refresh_provenance_bindings()
        with self.assertRaisesRegex(ValueError, "provenance_python_version_mismatch"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())

    def test_malformed_identity_types_fail_as_value_errors(self):
        self.crop_map["records"][0]["review_token"] = {"not": "hashable"}
        self.write_private(self.crop_path, enc(self.crop_map))
        with self.assertRaisesRegex(ValueError, "crop_map_record_invalid"):
            self.run_crop()
        self.crop_map["records"][0]["review_token"] = self.tokens[0]
        self.write_private(self.crop_path, enc(self.crop_map))
        self.receipt["records"][0]["anchor_id"] = ["not", "a", "string"]
        self.write_private(self.receipt_path, enc(self.receipt))
        with self.assertRaisesRegex(ValueError, "capture_receipt_sha256_mismatch"):
            self.run_crop()
        # Re-pin the deliberately malformed receipt and map to exercise identity validation.
        self.receipt_raw = enc(self.receipt)
        self.receipt_sha = sha(self.receipt_raw)
        self.write_private(self.receipt_path, self.receipt_raw)
        self.crop_map["capture_receipt_sha256"] = self.receipt_sha
        self.write_private(self.crop_path, enc(self.crop_map))
        with self.assertRaisesRegex(ValueError, "capture_receipt_record_invalid"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())

    def test_preexisting_or_racing_output_is_never_removed(self):
        output = self.out_parent / "occupied"
        output.mkdir(mode=0o700)
        sentinel = output / "keep.txt"
        sentinel.write_text("retain", encoding="utf-8")
        os.chmod(sentinel, 0o600)
        with self.assertRaisesRegex(ValueError, "output_directory_exists"):
            self.run_crop("occupied")
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "retain")

        racing = self.out_parent / "racing"
        original_mkdir = Path.mkdir
        def raced_mkdir(path, *args, **kwargs):
            if path == racing:
                original_mkdir(path, *args, **kwargs)
                marker = path / "concurrent.txt"
                marker.write_text("retain", encoding="utf-8")
                os.chmod(marker, 0o600)
                raise FileExistsError("simulated concurrent creation")
            return original_mkdir(path, *args, **kwargs)
        with patch.object(Path, "mkdir", raced_mkdir):
            with self.assertRaises(FileExistsError):
                self.run_crop("racing")
        self.assertEqual((racing / "concurrent.txt").read_text(encoding="utf-8"), "retain")

    def test_rejects_screenshot_tampering_and_cleans_partial_output(self):
        row = self.receipt["records"][0]
        (self.shots / row["screenshot_file"]).write_bytes(b"changed")
        os.chmod(self.shots / row["screenshot_file"], 0o600)
        with self.assertRaisesRegex(ValueError, "screenshot_sha256_mismatch"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())

    def test_rejects_out_of_bounds_bbox_and_incomplete_coverage(self):
        self.crop_map["records"][0]["bbox_xywh"] = [90, 70, 30, 25]
        self.crop_raw = enc(self.crop_map)
        self.write_private(self.crop_path, self.crop_raw)
        with self.assertRaisesRegex(ValueError, "crop_bbox_out_of_bounds"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())
        self.crop_map["records"].pop()
        self.crop_raw = enc(self.crop_map)
        self.write_private(self.crop_path, self.crop_raw)
        with self.assertRaisesRegex(ValueError, "crop_map_binding_or_count_invalid"):
            self.run_crop("missing")
        self.assertFalse((self.out_parent / "missing").exists())

    def test_rejects_relaxed_file_permissions(self):
        os.chmod(self.plan_path, 0o640)
        with self.assertRaisesRegex(ValueError, "plan_file_invalid"):
            self.run_crop()
        self.assertFalse((self.out_parent / "result").exists())


if __name__ == "__main__":
    unittest.main()
