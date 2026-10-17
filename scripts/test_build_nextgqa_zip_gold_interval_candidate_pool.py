import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile

from build_nextgqa_zip_gold_interval_candidate_pool import (
    _extract_member, _sha_member, _token, build, candidate_pair_sha256,
)


KEY = b"k" * 32


class ZipPoolTests(unittest.TestCase):
    def fixture(self, root, *, bad_zip=None, prior=(), rows=None, key=KEY):
        root.mkdir(parents=True, exist_ok=True)
        zp = root / "videos.zip"
        if bad_zip is None:
            with zipfile.ZipFile(zp, "w") as z:
                z.writestr("folder/", b"")
                z.writestr("folder/clip.mp4", b"synthetic media bytes")
        else:
            with zipfile.ZipFile(zp, "w") as z:
                if bad_zip == "traversal": z.writestr("../clip.mp4", b"x")
                elif bad_zip == "symlink":
                    info = zipfile.ZipInfo("folder/clip.mp4")
                    info.create_system = 3
                    info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    z.writestr(info, "target")
                elif bad_zip == "duplicate-map": z.writestr("folder/clip.mp4", b"x")
        cp = root / "test.csv"
        fields = ["video_id", "qid", "question", "answer", "a0", "a1", "a2", "a3", "a4"]
        with cp.open("w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for row in rows or [{"video_id":"v1","qid":"q1","question":"synthetic?","answer":"yes",
                                 "a0":"yes","a1":"no","a2":"maybe","a3":"other","a4":"none"}]:
                w.writerow(row)
        qloc = {row["qid"]:[[1,2]] for row in (rows or [{"qid":"q1"}])}
        gp = root / "gsub.json"; gp.write_text(json.dumps({"v1":{"duration":10,"location":qloc}}))
        mp = root / "map.json"
        map_obj = {"v1":"folder/clip.mp4"}
        if bad_zip == "duplicate-map": map_obj["v2"] = "other/folder/clip.mp4"
        mp.write_text(json.dumps(map_obj))
        sd = root / "samples"; sd.mkdir(exist_ok=True)
        (sd / "samples.json").write_text("[]")
        secret = root / "secret"; secret.write_bytes(key); secret.chmod(0o600)
        index = root / "index.json"
        ids = {"dataset_video_hmac_sha256": [], "media_sha256_hmac_sha256": [],
               "video_question_pair_hmac_sha256": []}
        for kind, identity in prior:
            ids[{"video":"dataset_video_hmac_sha256","media":"media_sha256_hmac_sha256",
                 "pair":"video_question_pair_hmac_sha256"}[kind]].append(identity)
        index.write_text(json.dumps({"schema_version":1,
            "purpose":"outcome_blind_prior_exposure_index",
            "canonicalization_version":"explicit-identifiers-v1-no-path-stems",
            "sources":[{"identity_namespace":"nextgqa"}],"identities":ids}))
        return zp, cp, gp, mp, sd, index, secret

    def run_build(self, root, paths, out):
        zp, cp, gp, mp, sd, ix, sec = paths
        return build(zp, cp, gp, mp, sd, "*.json", ix, sec, out,
                     expected_secret_sha256=hashlib.sha256(KEY).hexdigest())

    def test_eligible_extraction_hash_permissions_and_pair_schema(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root); out=root/"out"
            summary=self.run_build(root,paths,out)
            self.assertEqual(summary["counts"]["eligible"],1)
            media=out/"folder/clip.mp4"
            self.assertEqual(hashlib.sha256(media.read_bytes()).hexdigest(),hashlib.sha256(b"synthetic media bytes").hexdigest())
            self.assertEqual(stat.S_IMODE(out.stat().st_mode),0o700)
            self.assertEqual(stat.S_IMODE(media.stat().st_mode),0o600)
            pair=json.loads((out/"eligible_pairs.json").read_text())["pair_sha256"]
            self.assertEqual(pair,[candidate_pair_sha256("v1","q1")])
            for name in ("eligible_pairs.json","summary.json","prior_media_sha256_exclusions.json"):
                self.assertEqual(stat.S_IMODE((out/name).stat().st_mode),0o600)

    def test_zip_traversal_symlink_and_duplicate_mapped_keys_fail(self):
        for kind in ("traversal","symlink","duplicate-map"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as d:
                root=Path(d); paths=self.fixture(root,bad_zip=kind)
                with self.assertRaises(ValueError): self.run_build(root,paths,root/"out")
                self.assertFalse((root/"out").exists())

    def test_prior_video_pair_media_hmac_exclusions(self):
        media_sha=hashlib.sha256(b"synthetic media bytes").hexdigest()
        identities=[
            ("video",_token(KEY,"dataset-video-id",("nextgqa","v1"))),
            ("pair",_token(KEY,"video-question-pair",("nextgqa","dataset-video-id","nextgqa","v1","q1"))),
            ("media",_token(KEY,"media-sha256",(media_sha,))),
        ]
        for expected_field, prior in (("prior_video_excluded", identities[0]),
                                      ("prior_pair_excluded", identities[1]),
                                      ("prior_media_excluded", identities[2])):
            with self.subTest(field=expected_field), tempfile.TemporaryDirectory() as d:
                root=Path(d); paths=self.fixture(root,prior=[prior]); out=root/"out"
                result=self.run_build(root,paths,out)
                self.assertEqual(result["counts"][expected_field],1)
                if expected_field == "prior_media_excluded":
                    self.assertEqual(json.loads((out/"prior_media_sha256_exclusions.json").read_text()),
                                     {"schema_version":1,"media_sha256":[media_sha]})
                else:
                    self.assertEqual(result["counts"]["eligible"],0)

    def test_invalid_hmac_key_and_index_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root,key=b"short")
            with self.assertRaisesRegex(ValueError,"secret_too_short"):
                build(*paths[:4],paths[4],"*.json",paths[5],paths[6],root/"out",
                      expected_secret_sha256=hashlib.sha256(b"short").hexdigest())
            paths=self.fixture(root); paths[5].write_text('{"schema_version":true,"identities":{}}')
            with self.assertRaisesRegex(ValueError,"prior_index_schema_invalid"):
                self.run_build(root,paths,root/"out")

    def test_wrong_pinned_secret_fails_before_extraction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root)
            with self.assertRaisesRegex(ValueError,"secret_sha256_mismatch"):
                build(*paths[:4],paths[4],"*.json",paths[5],paths[6],root/"out",
                      expected_secret_sha256="0"*64)
            self.assertFalse((root/"out").exists())

    def test_output_parent_mode_and_symlink_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root)
            root.chmod(0o755)
            with self.assertRaisesRegex(ValueError,"output_parent_must_be_owned_mode_0700"):
                self.run_build(root,paths,root/"out")
            root.chmod(0o700)
            original=paths[1]; alias=root/"csv-link"; alias.symlink_to(original)
            paths=list(paths); paths[1]=alias
            with self.assertRaisesRegex(ValueError,"input_unavailable_or_symlink"):
                self.run_build(root,paths,root/"out")

    def test_actual_decompressed_member_cap(self):
        class FakeZip:
            def open(self, _info):
                return io.BytesIO(b"four")
        class Info:
            header_offset=1
        with self.assertRaisesRegex(ValueError,"zip_actual_member_size_limit"):
            _sha_member(FakeZip(),Info(),3)
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"media.mp4"
            with self.assertRaisesRegex(ValueError,"zip_actual_extraction_size_limit"):
                _extract_member(FakeZip(),Info(),dest,hashlib.sha256(b"four").hexdigest(),3,100)
            self.assertFalse(dest.exists())

    def test_two_eligible_questions_extract_media_once_and_keep_both_pairs(self):
        rows=[]
        for qid in ("q1","q2"):
            rows.append({"video_id":"v1","qid":qid,"question":"synthetic?","answer":"yes",
                         "a0":"yes","a1":"no","a2":"maybe","a3":"other","a4":"none"})
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root,rows=rows); out=root/"out"
            result=self.run_build(root,paths,out)
            self.assertEqual(result["counts"]["eligible"],2)
            self.assertEqual(result["extracted_unique_media_count"],1)
            self.assertEqual((out/"folder/clip.mp4").read_bytes(),b"synthetic media bytes")
            expected={candidate_pair_sha256("v1","q1"),candidate_pair_sha256("v1","q2")}
            self.assertEqual(set(json.loads((out/"eligible_pairs.json").read_text())["pair_sha256"]),expected)

    def test_ineligible_interval_choices_sample_path_and_exclusive_output(self):
        bad={"video_id":"v1","qid":"q1","question":"q?","answer":"yes",
             "a0":"yes","a1":"yes","a2":"maybe","a3":"other","a4":"none"}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root,rows=[bad]); out=root/"out"
            result=self.run_build(root,paths,out)
            self.assertEqual(result["counts"]["choice_ineligible"],1)
            with self.assertRaisesRegex(ValueError,"output_directory_already_exists"):
                self.run_build(root,paths,out)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); paths=self.fixture(root)
            (paths[4]/"samples.json").write_text(json.dumps([{"video":"folder/clip.mp4"}]))
            result=self.run_build(root,paths,root/"out")
            self.assertEqual(result["counts"]["sample_path_excluded"],1)


if __name__ == "__main__":
    unittest.main()
