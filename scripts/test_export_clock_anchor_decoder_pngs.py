from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

import av
from PIL import Image

from export_clock_anchor_decoder_pngs import SOURCE_FILES, export_anchor_pngs


class ExportClockAnchorDecoderPngTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        os.chmod(self.root, 0o700)
        self.inputs = self.root / "inputs"
        self.media = self.root / "media"
        self.inputs.mkdir(mode=0o700)
        self.media.mkdir(mode=0o700)
        self.tokens = [f"tok_{i:020d}"[:24] for i in range(3)]
        self.plan_rows = []
        self.provenance_rows = []
        for i, token in enumerate(self.tokens):
            media_path = self.media / (token + ".mp4")
            self.make_video(media_path, i)
            raw = media_path.read_bytes()
            os.chmod(media_path, 0o600)
            pts, tb, pixel_spec = self.inspect_video(media_path)
            self.plan_rows.append({"review_token": token,
                                   "source_media_sha256": hashlib.sha256(raw).hexdigest(),
                                   "fit_anchors": self.anchors(pts, tb, ("fit_begin", "fit_midpoint", "fit_end")),
                                   "heldout_anchors": self.anchors(pts, tb,
                                       ("heldout_quarter", "heldout_three_quarter"))})
            pts_hash = hashlib.sha256(json.dumps(pts, separators=(",", ":")).encode("ascii")).hexdigest()
            self.provenance_rows.append({"review_token": token,
                "staged_media_sha256": hashlib.sha256(raw).hexdigest(),
                "decoded_frame_count": len(pts), "decoded_pts_sequence_sha256": pts_hash,
                "time_base": [tb.numerator, tb.denominator], "pixel_spec": list(pixel_spec),
                "decoder_stream_info": {"codec_name": "ffv1"}})
        self.plan = {"schema_version": 1, "purpose": "predeclared_html5_clock_pts_anchor_plan",
                     "status": "predeclared_frozen", "execution_authorization": "not_granted",
                     "records": self.plan_rows}
        self.plan_raw = self.encode(self.plan)
        self.plan_sha = hashlib.sha256(self.plan_raw).hexdigest()
        scripts_dir = Path(__file__).resolve().parent
        source_sha256 = {key: hashlib.sha256((scripts_dir / filename).read_bytes()).hexdigest()
                         for key, filename in SOURCE_FILES.items()}
        self.provenance = {"schema_version": 1,
            "purpose": "private_clock_anchor_plan_candidate_provenance",
            "status": "candidate_not_jointly_frozen", "execution_authorization": "not_granted",
            "input_sha256": {}, "source_sha256": source_sha256, "plan_sha256": self.plan_sha,
            "created_at_utc": "2026-10-17T00:00:00Z", "python_version": "test",
            "pyav_version": av.__version__,
            "ffmpeg_library_versions": {key: list(value) for key, value in av.library_versions.items()},
            "joint_freeze_receipt_sha256": None, "records": self.provenance_rows}
        self.prov_raw = self.encode(self.provenance)
        self.prov_sha = hashlib.sha256(self.prov_raw).hexdigest()
        self.plan_path, self.prov_path = self.inputs / "plan.json", self.inputs / "provenance.json"
        self.plan_path.write_bytes(self.plan_raw)
        self.prov_path.write_bytes(self.prov_raw)
        os.chmod(self.plan_path, 0o600)
        os.chmod(self.prov_path, 0o600)

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    @staticmethod
    def make_video(path, variant):
        container = av.open(str(path), mode="w", format="matroska")
        stream = container.add_stream("ffv1", rate=10)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr0"
        for index in range(20):
            color = ((index * 11 + variant * 3) % 256, (index * 7) % 256, (index * 13) % 256)
            frame = Image.new("RGB", (32, 24), color)
            av_frame = av.VideoFrame.from_image(frame)
            av_frame.pts, av_frame.time_base = index, Fraction(1, 10)
            for packet in stream.encode(av_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()

    @staticmethod
    def inspect_video(path):
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = list(container.decode(stream))
        return ([int(frame.pts) for frame in frames], Fraction(frames[0].time_base),
                (frames[0].width, frames[0].height, frames[0].format.name))

    @staticmethod
    def anchors(pts, tb, ids):
        duration = (pts[-1] - pts[0]) * tb
        fractions = {"fit_begin": Fraction(1, 20), "heldout_quarter": Fraction(1, 4),
                     "fit_midpoint": Fraction(1, 2), "heldout_three_quarter": Fraction(3, 4),
                     "fit_end": Fraction(19, 20)}
        targets = []
        for anchor_id in ids:
            target = duration * fractions[anchor_id]
            nearest = min(((abs((point - pts[0]) * tb - target), point)
                           for point in pts), key=lambda row: (row[0], row[1]))[1]
            targets.append({"anchor_id": anchor_id,
                            "decoder_pts_seconds": str((nearest - pts[0]) * tb)})
        return targets

    def run_export(self, name="output"):
        return export_anchor_pngs(self.plan_path, self.prov_path, self.media,
                                  self.root / name, self.plan_sha, self.prov_sha)

    def test_exports_exact_fifteen_private_rgb_pngs_and_crosswalk(self):
        with patch("export_clock_anchor_decoder_pngs.av.open", wraps=av.open) as open_video:
            result = self.run_export()
        self.assertEqual(open_video.call_count, 3)
        output = self.root / "output"
        self.assertEqual(result["records"], 15)
        self.assertEqual(result["execution_authorization"], "not_granted")
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
        pngs = list(output.glob("*.png"))
        self.assertEqual(len(pngs), 15)
        for path in pngs:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            with Image.open(path) as image:
                self.assertEqual(image.mode, "RGB")
        crosswalk = json.loads((output / "crosswalk.json").read_bytes())
        self.assertEqual(crosswalk["effective_anchors"], 15)
        self.assertEqual(len(crosswalk["records"]), 15)
        self.assertEqual(crosswalk["plan_sha256"], self.plan_sha)
        for row in crosswalk["records"]:
            raw = (output / row["png_file"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["png_sha256"])

    def test_rejects_wrong_provenance_hash_and_removes_partial_output(self):
        with self.assertRaisesRegex(ValueError, "provenance_sha256_mismatch"):
            export_anchor_pngs(self.plan_path, self.prov_path, self.media,
                               self.root / "bad", self.plan_sha, "0" * 64)
        self.assertFalse((self.root / "bad").exists())

    def test_rejects_runtime_and_source_provenance_drift(self):
        mutations = (
            ("pyav_version", "0.0.0", "provenance_pyav_version_mismatch"),
            ("ffmpeg_library_versions", {"avcodec": [0, 0, 0]},
             "provenance_ffmpeg_versions_mismatch"),
            ("source_sha256", {**self.provenance["source_sha256"], "decoder": "0" * 64},
             "provenance_source_hash_mismatch"),
        )
        for index, (field, value, expected_error) in enumerate(mutations):
            with self.subTest(field=field):
                self.provenance[field] = value
                self.prov_raw = self.encode(self.provenance)
                self.prov_path.write_bytes(self.prov_raw)
                os.chmod(self.prov_path, 0o600)
                self.prov_sha = hashlib.sha256(self.prov_raw).hexdigest()
                output_name = f"runtime-drift-{index}"
                with self.assertRaisesRegex(ValueError, expected_error):
                    self.run_export(output_name)
                self.assertFalse((self.root / output_name).exists())
                # Restore the valid fixture before testing the next independent mutation.
                self.provenance[field] = {
                    "pyav_version": av.__version__,
                    "ffmpeg_library_versions": {key: list(item) for key, item in av.library_versions.items()},
                    "source_sha256": {key: hashlib.sha256(
                        (Path(__file__).resolve().parent / filename).read_bytes()).hexdigest()
                                      for key, filename in SOURCE_FILES.items()},
                }[field]

    def test_rejects_wrong_plan_hash_and_changed_staged_media(self):
        with self.assertRaisesRegex(ValueError, "plan_sha256_mismatch"):
            export_anchor_pngs(self.plan_path, self.prov_path, self.media,
                               self.root / "bad-plan", "0" * 64, self.prov_sha)
        media_path = self.media / (self.tokens[0] + ".mp4")
        raw = bytearray(media_path.read_bytes())
        raw[-1] ^= 1
        media_path.write_bytes(raw)
        os.chmod(media_path, 0o600)
        with self.assertRaisesRegex(ValueError, "staged_media_sha256_mismatch"):
            self.run_export("bad-media")
        self.assertFalse((self.root / "bad-media").exists())

    def test_rejects_non_candidate_authorization_and_pts_sequence_change(self):
        self.provenance["execution_authorization"] = "granted"
        self.prov_raw = self.encode(self.provenance)
        self.prov_path.write_bytes(self.prov_raw)
        os.chmod(self.prov_path, 0o600)
        self.prov_sha = hashlib.sha256(self.prov_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "provenance_schema_or_plan_binding_invalid"):
            self.run_export("unauthorized")
        self.provenance["execution_authorization"] = "not_granted"
        self.provenance_rows[0]["decoded_pts_sequence_sha256"] = "0" * 64
        self.provenance["records"] = self.provenance_rows
        self.prov_raw = self.encode(self.provenance)
        self.prov_path.write_bytes(self.prov_raw)
        os.chmod(self.prov_path, 0o600)
        self.prov_sha = hashlib.sha256(self.prov_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "decoded_sequence_or_anchor_mismatch"):
            self.run_export("pts-mismatch")
        self.assertFalse((self.root / "pts-mismatch").exists())


if __name__ == "__main__":
    unittest.main()
