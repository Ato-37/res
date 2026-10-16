import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.build_nextgqa_prior_exposure_manifest import build, main


KEY = b"synthetic test key with enough entropy 0123456789abcdef"


def raw(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")


class PriorExposureManifestTests(unittest.TestCase):
    def test_explicit_video_pair_and_media_hashes_are_hmac_only(self):
        secret = KEY
        media_sha = "a" * 64
        manifest, encoded = build([
            ("sample-json", "nextgqa", "synthetic-one", raw([
                {"video_id": "synthetic-video", "qid": "synthetic-question",
                 "question": "private question", "video": "/private/path.mp4"}
            ])),
            ("lineage-json", "nextgqa", "synthetic-two", raw({"records": [{
                "dataset_video_id": "synthetic-video", "dataset_question_id": "synthetic-question",
                "video_sha256": media_sha, "relative_video_path": "private/path.mp4"
            }]})),
        ], secret)
        self.assertEqual(manifest["sources"][0]["rows_with_any_identity"], 1)
        self.assertEqual(manifest["sources"][0]["rows_with_video_question_pair"], 1)
        self.assertEqual(len(manifest["identities"]["dataset_video_hmac_sha256"]), 1)
        self.assertEqual(len(manifest["identities"]["video_question_pair_hmac_sha256"]), 1)
        text = encoded.decode("utf-8")
        for private in ("synthetic-video", "synthetic-question", "private question", "/private/path.mp4",
                        "private/path.mp4", media_sha):
            self.assertNotIn(private, text)
        self.assertEqual(manifest["coverage"]["historical_completeness_authenticated"], False)

    def test_lock_qid_hash_is_not_misrepresented_as_pair_identity(self):
        manifest, _ = build([("lock-json", "nextgqa", "synthetic-lock", raw({
            "schema_version": 1,
            "assignments": [{"media_sha256": "b" * 64, "qid_sha256": "c" * 64}],
        }))], KEY)
        self.assertEqual(len(manifest["identities"]["media_sha256_hmac_sha256"]), 1)
        self.assertEqual(manifest["identities"]["video_question_pair_hmac_sha256"], [])
        self.assertEqual(manifest["sources"][0]["rows_with_any_identity"], 1)
        self.assertEqual(manifest["sources"][0]["rows_with_media_sha256"], 1)
        self.assertEqual(manifest["sources"][0]["rows_with_video_question_pair"], 0)
        self.assertEqual(manifest["sources"][0]["video_question_pair_count"], 0)
        self.assertEqual(manifest["sources"][0]["unresolved_reason_counts"][
            "qid_hash_is_opaque_pair_unresolved"], 1)

    def test_path_only_and_aggregate_rows_are_unresolved(self):
        manifest, _ = build([
            ("sample-json", "nextgqa", "synthetic-path-only", raw([{"video": "/private/a.mp4", "qid": "q"}])),
            ("result-json", "nextgqa", "synthetic-aggregate", raw({"n": 9, "effect": 0.2})),
        ], KEY)
        self.assertEqual(manifest["sources"][0]["rows_without_any_identity"], 1)
        self.assertEqual(manifest["sources"][0]["unresolved_reason_counts"]["path_only_video_unresolved"], 1)
        self.assertIsNone(manifest["sources"][1]["rows_total"])

    def test_ambiguous_result_json_and_bad_csv_width_fail_closed(self):
        manifest, _ = build([
            ("result-json", "nextgqa", "synthetic-ambiguous", raw({"records": [], "rows": []})),
            ("result-csv", "nextgqa", "synthetic-bad-csv", b"video_id,qid\nv1\n"),
        ], KEY)
        for source in manifest["sources"]:
            self.assertEqual(source["status"], "unresolved_invalid_source")
            self.assertIsNone(source["rows_total"])
            self.assertIsNone(source["rows_without_any_identity"])
        self.assertEqual(
            manifest["sources"][0]["unresolved_reason_counts"],
            {"result_json_row_container_missing_or_ambiguous": 1})
        self.assertEqual(
            manifest["sources"][1]["unresolved_reason_counts"],
            {"csv_row_width_mismatch": 1})

    def test_source_fingerprint_is_keyed_and_duplicate_content_is_rejected(self):
        source = raw([{"video_id": "synthetic-id"}])
        manifest, encoded = build([("sample-json", "nextgqa", "synthetic-one", source)], KEY)
        digest = hashlib.sha256(source).hexdigest()
        self.assertNotIn(digest, encoded.decode())
        self.assertIn("source_fingerprint_hmac_sha256", manifest["sources"][0])
        with self.assertRaisesRegex(ValueError, "duplicate_source_content"):
            build([("sample-json", "nextgqa", "one", source),
                   ("sample-json", "nextgqa", "two", source)], KEY)

    def test_video_id_hmac_is_namespaced(self):
        manifest, _ = build([
            ("sample-json", "dataset_a", "source-a", raw([{"video_id": "same-id", "qid": "q1"}])),
            ("sample-json", "dataset_b", "source-b", raw([{"video_id": "same-id", "qid": "q2"}])),
        ], KEY)
        self.assertEqual(len(manifest["identities"]["dataset_video_hmac_sha256"]), 2)
        self.assertEqual(manifest["sources"][0]["identity_namespace"], "dataset_a")

    def test_opaque_legacy_counts_rows_without_interpreting_values(self):
        private_hash = "d" * 64
        manifest, encoded = build([("opaque-legacy", "nextgqa", "synthetic-legacy", raw([private_hash, "synthetic-id"]))], KEY)
        source = manifest["sources"][0]
        self.assertEqual(source["rows_total"], 2)
        self.assertIsNone(source["rows_with_any_identity"])
        self.assertIsNone(source["rows_without_any_identity"])
        self.assertNotIn(private_hash, encoded.decode())
        self.assertEqual(manifest["identities"]["media_sha256_hmac_sha256"], [])

    def test_media_set_parses_hashes_but_marks_provenance_unverified(self):
        media_sha = "e" * 64
        manifest, encoded = build([("media-set-json", "nextgqa-media", "synthetic-media-set",
                                    raw({"media_sha256": [media_sha]}))], KEY)
        source = manifest["sources"][0]
        self.assertEqual(source["status"], "parsed_provenance_unverified")
        self.assertEqual(source["rows_total"], 1)
        self.assertEqual(source["rows_with_any_identity"], 1)
        self.assertEqual(len(manifest["identities"]["media_sha256_hmac_sha256"]), 1)
        self.assertNotIn(media_sha, encoded.decode("utf-8"))
        self.assertFalse(manifest["coverage"]["historical_completeness_authenticated"])

    def test_media_set_allows_optional_metadata_without_emitting_it(self):
        synthetic_sha = "f" * 64
        for metadata in ({"schema_version": 1}, {"purpose": "synthetic fixture only"}):
            with self.subTest(metadata_key=next(iter(metadata))):
                manifest, encoded = build([("media-set-json", "nextgqa-media", "synthetic-metadata",
                                            raw({"media_sha256": [synthetic_sha], **metadata}))], KEY)
                self.assertEqual(manifest["sources"][0]["status"], "parsed_provenance_unverified")
                self.assertNotIn("synthetic fixture only", encoded.decode("utf-8"))
                self.assertNotIn("schema_version", manifest["sources"][0])

    def test_media_set_rejects_ambiguous_or_malformed_shapes(self):
        invalid = (
            {"media_sha256": ["F" * 64]},
            {"media_sha256": ["a" * 63]},
            {"media_sha256": [True]},
            {"media_sha256": [1]},
            {"media_sha256": "a" * 64},
            {"media_sha256": [], "schema_version": True},
            {"media_sha256": [], "schema_version": 2},
            {"media_sha256": [], "purpose": "  "},
            {"media_sha256": [], "purpose": "synthetic", "schema_version": 1},
            [],
        )
        for value in invalid:
            with self.subTest(shape=type(value).__name__):
                manifest, _ = build([("media-set-json", "nextgqa-media", "synthetic-invalid", raw(value))], KEY)
                self.assertEqual(manifest["sources"][0]["status"], "unresolved_invalid_source")
                self.assertEqual(manifest["sources"][0]["media_identity_count"], 0)
        duplicate_key = b'{"media_sha256":[],"media_sha256":[]}'
        manifest, _ = build([("media-set-json", "nextgqa-media", "synthetic-duplicate", duplicate_key)], KEY)
        self.assertEqual(manifest["sources"][0]["status"], "unresolved_invalid_source")

    def test_review_custody_extracts_only_media_hash_and_marks_unverified(self):
        synthetic_sha = "9" * 64
        payload = {"records": [{"source_media_sha256": synthetic_sha,
                                "review_token": "synthetic-private-token",
                                "unrelated": "synthetic private text"}],
                   "purpose": "synthetic custody fixture"}
        manifest, encoded = build([("review-custody-json", "review-media", "synthetic-custody",
                                    raw(payload))], KEY)
        source = manifest["sources"][0]
        self.assertEqual(source["status"], "parsed_provenance_unverified")
        self.assertEqual(source["rows_total"], 1)
        self.assertEqual(source["rows_with_media_sha256"], 1)
        serialized = encoded.decode("utf-8")
        for private in (synthetic_sha, "synthetic-private-token", "synthetic private text",
                        "synthetic custody fixture"):
            self.assertNotIn(private, serialized)

    def test_review_custody_requires_valid_record_hashes_and_rejects_duplicate_keys(self):
        invalid_payloads = (
            {"records": [{"review_token": "opaque"}]},
            {"records": [{"source_media_sha256": "A" * 64}]},
            {"records": [{"source_media_sha256": "a" * 63}]},
            {"records": [True]},
            {"other": []},
        )
        for payload in invalid_payloads:
            with self.subTest(shape=type(payload.get("records", "missing")).__name__):
                manifest, _ = build([("review-custody-json", "review-media", "synthetic-invalid",
                                      raw(payload))], KEY)
                self.assertEqual(manifest["sources"][0]["status"], "unresolved_invalid_source")
                self.assertEqual(manifest["sources"][0]["media_identity_count"], 0)
        duplicate_key = b'{"records":[{"source_media_sha256":"' + b"a" * 64 + b'","source_media_sha256":"' + b"b" * 64 + b'"}]}'
        manifest, _ = build([("review-custody-json", "review-media", "synthetic-duplicate",
                              duplicate_key)], KEY)
        self.assertEqual(manifest["sources"][0]["status"], "unresolved_invalid_source")

    def test_opaque_legacy_non_array_row_count_is_unknown(self):
        manifest, _ = build([
            ("opaque-legacy", "nextgqa", "synthetic-dict", raw({"a": [1, 2], "b": [3]})),
            ("opaque-legacy", "nextgqa", "synthetic-scalar", raw(42)),
        ], KEY)
        for source in manifest["sources"]:
            self.assertIsNone(source["rows_total"])
            self.assertIsNone(source["rows_without_any_identity"])
            self.assertIn("opaque_legacy_row_count_unknown",
                          source["unresolved_reason_counts"])

    def test_lock_schema_version_boolean_is_rejected(self):
        manifest, _ = build([("lock-json", "nextgqa", "synthetic-bool-lock", raw({
            "schema_version": True, "assignments": []}))], KEY)
        source = manifest["sources"][0]
        self.assertEqual(source["status"], "unresolved_invalid_source")
        self.assertIsNone(source["rows_total"])

    def test_duplicate_json_keys_fail_source_closed_without_values(self):
        manifest, _ = build([("sample-json", "nextgqa", "synthetic-bad", b'[{"video_id":"x","video_id":"y"}]')], KEY)
        source = manifest["sources"][0]
        self.assertEqual(source["status"], "unresolved_invalid_source")
        self.assertIsNone(source["rows_with_any_identity"])
        self.assertEqual(source["unresolved_reason_counts"], {"source_parse_failed": 1})

    def test_cli_requires_private_secret_and_writes_exclusive_0600_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "synthetic.json"
            secret = root / "secret"
            output = root / "manifest.json"
            source.write_bytes(raw([{"video_id": "synthetic-id"}]))
            secret.write_bytes(KEY)
            secret.chmod(0o600)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", f"sample-json@nextgqa={source}", "--secret-file", str(secret),
                                       "--output", str(output)]), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertNotIn("synthetic-id", output.read_text())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--input", f"sample-json@nextgqa={source}", "--secret-file", str(secret),
                      "--output", str(output)])

    def test_cli_rejects_insecure_key_and_path_alias_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            secret = root / "secret"
            output = root / "out.json"
            source.write_bytes(raw([]))
            secret.write_bytes(KEY)
            secret.chmod(0o644)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--input", f"sample-json@nextgqa={source}", "--secret-file", str(secret), "--output", str(output)])
            secret.chmod(0o600)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--input", f"unknown@nextgqa={source}", "--secret-file", str(secret), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
