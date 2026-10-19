import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from stage_nextgqa_review_exports import StageError, main, stage_exports


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def fixture(root: Path):
    source = root / "downloads"
    source.mkdir()
    repo = root / "repo"
    (repo / "data").mkdir(parents=True)
    (repo / "data" / "private_nextgqa").mkdir(mode=0o700)
    review_raw = encode({
        "schema_version": 3,
        "purpose": "anonymous_local_claim_video_human_review",
        "status": "human_review_recorded_unvalidated",
        "execution_authorization": "not_granted",
        "private_payload_marker": "SECRET REVIEW TEXT",
    })
    review_index_raw = encode({
        "schema_version": 1,
        "purpose": "anonymous_unvalidated_human_review_handoff",
        "review_payload_sha256": hashlib.sha256(review_raw).hexdigest(),
        "review_status": "unvalidated_requires_custodian_audit",
        "execution_authorization": "not_granted",
    })
    clock_raw = encode({
        "schema_version": 2,
        "purpose": "local_unvalidated_html5_player_clock_observations",
        "status": "raw_player_clock_observations_only",
        "execution_authorization": "not_granted",
        "private_payload_marker": "SECRET CLOCK TEXT",
    })
    clock_index_raw = encode({
        "schema_version": 1,
        "purpose": "unvalidated_html5_clock_observation_integrity_index",
        "payload_sha256": hashlib.sha256(clock_raw).hexdigest(),
        "status": "raw_observations_only",
        "execution_authorization": "not_granted",
    })
    files = {}
    for name, raw in (("review.json", review_raw), ("review_index.json", review_index_raw),
                      ("clock.json", clock_raw), ("clock_index.json", clock_index_raw)):
        path = source / name
        path.write_bytes(raw)
        path.chmod(0o644)
        files[name] = path
    return repo, files, {
        "review.json": review_raw,
        "review_index.json": review_index_raw,
        "clock.json": clock_raw,
        "clock_index.json": clock_index_raw,
    }


class StageNextgqaReviewExportsTests(unittest.TestCase):
    def test_review_and_optional_clock_are_staged_byte_identically_as_0600(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, expected = fixture(Path(temp))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main([
                    "--review-payload", str(files["review.json"]),
                    "--review-index", str(files["review_index.json"]),
                    "--clock-observations", str(files["clock.json"]),
                    "--clock-index", str(files["clock_index.json"]),
                    "--output-prefix", "browser_run_01",
                ], repo_root=repo)
            self.assertEqual(result, 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["artifact_count"], 4)
            self.assertEqual(summary["execution_authorization"], "not_granted")
            self.assertEqual(summary["provenance_status"],
                             "bytes_and_declared_fields_only_unauthenticated")
            private = repo / "data" / "private_nextgqa"
            manifest_path = private / "browser_run_01_manifest.json"
            manifest_raw = manifest_path.read_bytes()
            manifest = json.loads(manifest_raw)
            self.assertEqual(manifest["execution_authorization"], "not_granted")
            self.assertEqual(manifest["provenance_status"],
                             "bytes_and_declared_fields_only_unauthenticated")
            for artifact in manifest["artifacts"]:
                destination = private / artifact["basename"]
                source_role = artifact["role"]
                source_key = {
                    "review_payload": "review.json",
                    "review_index": "review_index.json",
                    "clock_observations": "clock.json",
                    "clock_index": "clock_index.json",
                }[source_role]
                self.assertEqual(destination.read_bytes(), expected[source_key])
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
                self.assertEqual(artifact["sha256"], hashlib.sha256(expected[source_key]).hexdigest())
                self.assertEqual(set(artifact), {"role", "basename", "sha256", "status"})
            self.assertEqual(stat.S_IMODE(manifest_path.stat().st_mode), 0o600)
            self.assertNotIn("SECRET", manifest_raw.decode())
            self.assertNotIn(str(files["review.json"]), manifest_raw.decode())
            self.assertNotIn("SECRET", stdout.getvalue())
            self.assertNotIn(str(repo), stdout.getvalue())

    def test_review_pair_only_is_valid(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            result = stage_exports(
                review_payload_path=str(files["review.json"]),
                review_index_path=str(files["review_index.json"]),
                output_prefix="review_only", repo_root=repo)
            self.assertEqual(result["artifact_count"], 2)
            self.assertTrue((repo / "data/private_nextgqa/review_only_review_payload.json").exists())
            self.assertFalse((repo / "data/private_nextgqa/review_only_clock_observations.json").exists())

    def test_current_renderer_clock_schema_v3_is_staged_byte_identically(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            clock = json.loads(files["clock.json"].read_bytes())
            clock["schema_version"] = 3
            raw = encode(clock)
            files["clock.json"].write_bytes(raw)
            index = json.loads(files["clock_index.json"].read_bytes())
            index["payload_sha256"] = hashlib.sha256(raw).hexdigest()
            files["clock_index.json"].write_bytes(encode(index))
            result = stage_exports(
                review_payload_path=str(files["review.json"]),
                review_index_path=str(files["review_index.json"]),
                clock_observations_path=str(files["clock.json"]),
                clock_index_path=str(files["clock_index.json"]),
                output_prefix="renderer_clock_v3", repo_root=repo)
            self.assertEqual(result["artifact_count"], 4)
            staged = repo / "data/private_nextgqa/renderer_clock_v3_clock_observations.json"
            self.assertEqual(staged.read_bytes(), raw)
            self.assertEqual(json.loads(staged.read_bytes())["schema_version"], 3)

    def test_unsupported_clock_schema_is_rejected_without_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            clock = json.loads(files["clock.json"].read_bytes())
            clock["schema_version"] = 4
            raw = encode(clock)
            files["clock.json"].write_bytes(raw)
            index = json.loads(files["clock_index.json"].read_bytes())
            index["payload_sha256"] = hashlib.sha256(raw).hexdigest()
            files["clock_index.json"].write_bytes(encode(index))
            with self.assertRaisesRegex(StageError, "clock_observations_declaration_invalid"):
                stage_exports(
                    review_payload_path=str(files["review.json"]),
                    review_index_path=str(files["review_index.json"]),
                    clock_observations_path=str(files["clock.json"]),
                    clock_index_path=str(files["clock_index.json"]),
                    output_prefix="renderer_clock_v4", repo_root=repo)
            self.assertEqual(list((repo / "data/private_nextgqa").iterdir()), [])

    def test_optional_clock_pair_is_all_or_none(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            with self.assertRaisesRegex(StageError, "clock_export_pair_incomplete"):
                stage_exports(review_payload_path=str(files["review.json"]),
                              review_index_path=str(files["review_index.json"]),
                              clock_observations_path=str(files["clock.json"]),
                              output_prefix="partial", repo_root=repo)
            self.assertEqual(list((repo / "data/private_nextgqa").iterdir()), [])

    def test_review_and_clock_digest_mismatches_fail_before_writing(self):
        for which, expected_code in (("review", "review_payload_hash_mismatch"),
                                     ("clock", "clock_observations_hash_mismatch")):
            with self.subTest(which=which), tempfile.TemporaryDirectory() as temp:
                repo, files, _ = fixture(Path(temp))
                index_path = files["review_index.json"] if which == "review" else files["clock_index.json"]
                index = json.loads(index_path.read_bytes())
                key = "review_payload_sha256" if which == "review" else "payload_sha256"
                index[key] = "0" * 64
                index_path.write_bytes(encode(index))
                with self.assertRaisesRegex(StageError, expected_code):
                    stage_exports(
                        review_payload_path=str(files["review.json"]),
                        review_index_path=str(files["review_index.json"]),
                        clock_observations_path=(str(files["clock.json"]) if which == "clock" else None),
                        clock_index_path=(str(files["clock_index.json"]) if which == "clock" else None),
                        output_prefix="mismatch", repo_root=repo)
                self.assertEqual(list((repo / "data/private_nextgqa").iterdir()), [])

    def test_duplicate_index_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            files["review_index.json"].write_bytes(
                b'{"schema_version":1,"purpose":"anonymous_unvalidated_human_review_handoff",'
                b'"purpose":"duplicate","review_payload_sha256":"' +
                hashlib.sha256(files["review.json"].read_bytes()).hexdigest().encode() +
                b'","review_status":"unvalidated_requires_custodian_audit",'
                b'"execution_authorization":"not_granted"}')
            with self.assertRaisesRegex(StageError, "json_duplicate_key"):
                stage_exports(review_payload_path=str(files["review.json"]),
                              review_index_path=str(files["review_index.json"]),
                              output_prefix="duplicate", repo_root=repo)

    def test_symlink_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            link = files["review.json"].parent / "link.json"
            link.symlink_to(files["review.json"])
            with self.assertRaisesRegex(StageError, "input_file_invalid"):
                stage_exports(review_payload_path=str(link),
                              review_index_path=str(files["review_index.json"]),
                              output_prefix="symlink", repo_root=repo)

    def test_output_collision_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            private = repo / "data" / "private_nextgqa"
            collision = private / "collision_review_payload.json"
            collision.write_bytes(b"keep")
            collision.chmod(0o600)
            with self.assertRaisesRegex(StageError, "output_already_exists"):
                stage_exports(review_payload_path=str(files["review.json"]),
                              review_index_path=str(files["review_index.json"]),
                              output_prefix="collision", repo_root=repo)
            self.assertEqual(collision.read_bytes(), b"keep")
            self.assertEqual({p.name for p in private.iterdir()}, {collision.name})

    def test_write_failure_cleans_all_created_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            original_write = os.write
            calls = 0

            def fail_after_partial(fd, data):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_write(fd, data[:2])
                raise OSError("synthetic write failure")

            with mock.patch("stage_nextgqa_review_exports.os.write", side_effect=fail_after_partial):
                with self.assertRaisesRegex(StageError, "output_write_failed"):
                    stage_exports(review_payload_path=str(files["review.json"]),
                                  review_index_path=str(files["review_index.json"]),
                                  output_prefix="write_failure", repo_root=repo)
            self.assertEqual(list((repo / "data/private_nextgqa").iterdir()), [])

    def test_keyboard_interrupt_cleans_partial_and_completed_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            repo, files, _ = fixture(Path(temp))
            original_write = os.write
            calls = 0

            def interrupt_after_first_artifact(fd, data):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return original_write(fd, data)
                raise KeyboardInterrupt

            with mock.patch("stage_nextgqa_review_exports.os.write",
                            side_effect=interrupt_after_first_artifact):
                with self.assertRaises(KeyboardInterrupt):
                    stage_exports(review_payload_path=str(files["review.json"]),
                                  review_index_path=str(files["review_index.json"]),
                                  output_prefix="keyboard_interrupt", repo_root=repo)
            self.assertEqual(list((repo / "data/private_nextgqa").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
