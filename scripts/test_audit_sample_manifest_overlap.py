import json
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from scripts.audit_sample_manifest_overlap import audit, main


def raw(rows):
    return json.dumps(rows).encode("utf-8")


class SampleManifestOverlapTests(unittest.TestCase):
    def test_counts_identity_overlap_without_emitting_ids(self):
        candidate = raw([{"video_id": f"v{i}"} for i in range(4)])
        historical = [raw([{"video": "/private/v0.mp4"}, {"video_id": "v1"}]),
                      raw([{"video_id": "v1"}, {"video_id": "v2"}])]
        result = audit(candidate, historical)
        self.assertEqual(result["candidate_identities_seen_historically"], 3)
        self.assertEqual(result["candidate_identities_unseen_historically"], 1)
        self.assertEqual([row["candidate_identity_overlap"] for row in result["per_historical_manifest"]], [2, 2])
        self.assertEqual([row["manifest_label"] for row in result["per_historical_manifest"]], ["historical_01", "historical_02"])
        self.assertEqual(result["historical_identities_shared_across_manifests"], 1)
        self.assertNotIn("v0", json.dumps(result))

    def test_detects_duplicate_candidate_and_historical_rows(self):
        result = audit(raw([{"video_id": "v"}, {"video_id": "v"}]),
                       [raw([{"video_id": "v"}, {"video_id": "v"}])])
        self.assertEqual(result["status"], "audit_complete")
        self.assertEqual(result["candidate_identity_status"], "duplicates_present")
        self.assertEqual(result["candidate_duplicate_identity_rows"], 1)
        self.assertEqual(result["historical_duplicate_identity_rows_within_manifests"], 1)

    def test_rejects_malformed_identity_fields_even_when_other_field_is_valid(self):
        for row in ({"video_id": [], "video": "/private/alpha.mp4"},
                    {"video_id": "alpha", "video": {}},
                    {"video_id": None}, {"video": "  "}):
            with self.subTest(row_type=type(row.get("video_id")).__name__):
                with self.assertRaises(ValueError):
                    audit(raw([row]), [raw([{"video_id": "old"}])])

    def test_rejects_mismatched_video_id_and_video_stem(self):
        with self.assertRaisesRegex(ValueError, "disagree"):
            audit(raw([{"video_id": "one", "video": "/private/two.mp4"}]),
                  [raw([{"video_id": "old"}])])

    def test_rejects_duplicate_keys_and_nonfinite_json(self):
        with self.assertRaises(ValueError):
            audit(b'[{"video_id":"a","video_id":"b"}]', [raw([{"video_id": "x"}])])
        with self.assertRaises(ValueError):
            audit(b'[{"video_id":"a","value":NaN}]', [raw([{"video_id": "x"}])])
        with self.assertRaises(ValueError):
            audit(b'[{"video_id":"a","value":1e9999}]', [raw([{"video_id": "x"}])])

    def test_cli_writes_private_aggregate_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "candidate.json"
            historical = root / "old.json"
            output = root / "summary.json"
            candidate.write_bytes(raw([{"video_id": "private-video-id"}]))
            historical.write_bytes(raw([{"video_id": "different-private-id"}]))
            self.assertEqual(main(["--candidate", str(candidate), "--historical", str(historical),
                                   "--output", str(output)]), 0)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)
            serialized = output.read_text()
            self.assertNotIn("private-video-id", serialized)
            self.assertNotIn("different-private-id", serialized)
            self.assertNotIn("old.json", serialized)
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(["--candidate", str(candidate), "--historical", str(historical),
                      "--output", str(output)])

    def test_dotted_video_id_remains_opaque(self):
        result = audit(raw([{"video_id": "clip.v1"}]), [raw([{"video_id": "clip.v2"}])])
        self.assertEqual(result["candidate_identities_seen_historically"], 0)

    def test_opaque_ids_reject_surrounding_whitespace(self):
        with self.assertRaisesRegex(ValueError, "surrounding whitespace"):
            audit(raw([{"video_id": "v"}]), [raw([{"video_id": " v "}])])


if __name__ == "__main__":
    unittest.main()
