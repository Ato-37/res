import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_nextgqa_review_cohort_gpu_capacity_smoke import validate_cohort


def put_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path


class CohortValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.media = self.root / "media"
        self.media.mkdir(mode=0o700)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir(mode=0o700)

    def fixture(self, n=3):
        cr, pr, vr = [], [], []
        for i in range(n):
            token, name, data = f"opaque-{i}", f"{i}.mp4", f"video-{i}".encode()
            (self.media / name).write_bytes(data)
            (self.media / name).chmod(0o600)
            sha = hashlib.sha256(data).hexdigest()
            source_sha = hashlib.sha256(f"source-{i}".encode()).hexdigest()
            cr.append({"review_token": token, "media_file": name,
                "source_media_sha256": source_sha, "staged_media_sha256": sha,
                "size_bytes": len(data)})
            pr.append({"review_token": token, "source_media_sha256": source_sha})
            vr.append({"review_token": token, "staged_media_sha256": sha})
        c = put_json(self.root / "c.json", {"schema_version": 1, "record_count": n,
            "records": cr})
        p = put_json(self.root / "p.json", {"schema_version": 1,
            "status": "predeclared_frozen", "records": pr})
        v = put_json(self.root / "v.json", {"schema_version": 1,
            "status": "candidate_not_jointly_frozen", "plan_sha256":
            hashlib.sha256(p.read_bytes()).hexdigest(), "records": vr})
        pins = tuple(hashlib.sha256(x.read_bytes()).hexdigest() for x in (c, p, v))
        return c, p, v, pins

    def test_exact_three_cohort_is_snapshotted(self):
        c, p, v, pins = self.fixture()
        files = validate_cohort(c, p, v, self.media, pins, self.scratch)
        self.assertEqual(len(files), 3)
        self.assertEqual([x.read_bytes() for x in files], [b"video-0", b"video-1", b"video-2"])
        (self.media / "0.mp4").write_bytes(b"changed-after-validation")
        self.assertEqual(files[0].read_bytes(), b"video-0")

    def test_refuses_incomplete_cohort(self):
        c, p, v, pins = self.fixture(n=1)
        with self.assertRaisesRegex(ValueError, "exact_three_record_cohort_required"):
            validate_cohort(c, p, v, self.media, pins, self.scratch)

    def test_refuses_wrong_manifest_pin(self):
        c, p, v, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, "manifest_sha256_pin_mismatch"):
            validate_cohort(c, p, v, self.media, ("0" * 64,) * 3, self.scratch)

    def test_refuses_media_hash_mismatch(self):
        c, p, v, pins = self.fixture()
        (self.media / "0.mp4").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "staged_media_hash_mismatch"):
            validate_cohort(c, p, v, self.media, pins, self.scratch)

    def test_refuses_duplicate_video_identity(self):
        c, p, v, _ = self.fixture()
        custody = json.loads(c.read_text())
        plan = json.loads(p.read_text())
        prov = json.loads(v.read_text())
        custody["records"][1]["source_media_sha256"] = custody["records"][0]["source_media_sha256"]
        custody["records"][1]["staged_media_sha256"] = custody["records"][0]["staged_media_sha256"]
        plan["records"][1]["source_media_sha256"] = plan["records"][0]["source_media_sha256"]
        prov["records"][1]["staged_media_sha256"] = prov["records"][0]["staged_media_sha256"]
        put_json(c, custody); put_json(p, plan)
        prov["plan_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        put_json(v, prov)
        pins = tuple(hashlib.sha256(x.read_bytes()).hexdigest() for x in (c, p, v))
        with self.assertRaisesRegex(ValueError, "duplicate_staged_video_identity"):
            validate_cohort(c, p, v, self.media, pins, self.scratch)

    def test_refuses_symlink_media(self):
        c, p, v, pins = self.fixture()
        target = self.root / "outside.mp4"
        target.write_bytes(b"video-0")
        (self.media / "0.mp4").unlink()
        (self.media / "0.mp4").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "staged_media_missing_or_not_private"):
            validate_cohort(c, p, v, self.media, pins, self.scratch)

    def test_refuses_bad_manifest_mode(self):
        c, p, v, pins = self.fixture()
        c.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private_manifest_permissions_or_type_invalid"):
            validate_cohort(c, p, v, self.media, pins, self.scratch)


if __name__ == "__main__":
    unittest.main()
