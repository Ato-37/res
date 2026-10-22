from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.build_claim_exposure_union_manifest import (
    _read_private_source,
    _safe_private_root,
    build_manifest,
    main,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ClaimExposureUnionManifestTests(unittest.TestCase):
    def test_temporal_order_generation_and_scoring_markers_union_deterministically(self):
        a, b, c, d = (letter * 64 for letter in "abcd")
        generation = encoded({
            "schema_version": 1,
            "purpose": "temporal_order_claim_generation",
            "records": [{"video_sha256": a, "nested": {"media_sha256": [b]}}],
            "private_claim": "synthetic claim text",
        })
        score = encoded({
            "status": "completed",
            "records": [{"source_video_sha256": c}],
            "audit": {"completed_forwards": 2},
            "media_sha256": d,
        })
        inventory = encoded({
            "purpose": "candidate_inventory",
            "records": [{"video_sha256": "e" * 64, "media_sha256": "f" * 64}],
        })
        sources = [
            ("data/private_nextgqa/z_generation.json", generation),
            ("data/private_nextgqa/a_score.json", score),
            ("data/private_nextgqa/m_inventory.json", inventory),
        ]

        manifest, reversed_manifest = build_manifest(sources), build_manifest(list(reversed(sources)))

        self.assertEqual(manifest, reversed_manifest)
        self.assertEqual(manifest["source_file_count"], 2)
        self.assertEqual(manifest["unique_media_count"], 4)
        self.assertEqual(manifest["media_sha256"], sorted((a, b, c, d)))
        self.assertEqual([row["role"] for row in manifest["sources"]],
                         ["scoring_receipt", "claim_generation"])
        self.assertEqual(manifest["source_inventory_sha256"], hashlib.sha256(
            json.dumps([{"relative_filename": row["relative_filename"],
                         "source_sha256": row["source_sha256"], "role": row["role"]}
                        for row in manifest["sources"]], sort_keys=True,
            separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        self.assertNotIn("private_claim", json.dumps(manifest))
        self.assertNotIn("e" * 64, manifest["media_sha256"])

    def test_generation_schema_and_each_explicit_scoring_marker_are_recognized(self):
        digests = [str(i) * 64 for i in range(1, 10)]
        payloads = [
            {"schema": "saved_claim_generation_packet_v1", "schema_version": 1,
             "records": [{"video_sha256": digests[0]}]},
            {"records": [{"source_video_sha256": digests[1]}],
             "scoring_performed": True},
            {"records": [{"media_sha256": digests[2]}],
             "target_scoring_performed": True},
            {"records": [{"video_sha256": digests[3]}],
             "faithfulness_scored": True},
            {"records": [{"source_video_sha256": digests[4]}],
             "completed_forwards": 1},
        ]
        manifest = build_manifest([
            (f"data/private_nextgqa/source_{index}.json", encoded(value))
            for index, value in enumerate(payloads)
        ])
        self.assertEqual(manifest["source_file_count"], len(payloads))
        self.assertEqual(manifest["unique_media_count"], 5)
        self.assertEqual(manifest["media_sha256"], sorted(digests[:5]))

    def test_candidate_inventory_and_nonpositive_or_boolean_forward_counts_are_not_scoring(self):
        inventory = {"purpose": "candidate_inventory", "video_sha256": "a" * 64}
        bool_forward = {"nested": {"completed_forwards": True}, "source_video_sha256": "b" * 64}
        zero_forward = {"completed_forwards": 0, "media_sha256": "c" * 64}
        false_markers = {"scoring_performed": False, "target_scoring_performed": 1,
                         "faithfulness_scored": "true", "video_sha256": "d" * 64}
        manifest = build_manifest([
            (f"data/private_nextgqa/excluded_{i}.json", encoded(item))
            for i, item in enumerate((inventory, bool_forward, zero_forward, false_markers))
        ])
        self.assertEqual(manifest["source_file_count"], 0)
        self.assertEqual(manifest["unique_media_count"], 0)
        self.assertEqual(manifest["media_sha256"], [])

    def test_cli_writes_owner_only_exclusive_manifest_and_stdout_has_only_counts_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            private = repo / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            os.chmod(private, 0o700)
            source = private / "synthetic_generation.json"
            source.write_bytes(encoded({"purpose": "temporal_order_claim_generation",
                                        "video_sha256": "a" * 64}))
            os.chmod(source, 0o600)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(["--repo-root", str(repo), "--input",
                               "data/private_nextgqa/synthetic_generation.json",
                               "--output-name", "union.json"])
            self.assertEqual(result, 0)
            output = private / "union.json"
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            payload = json.loads(output.read_text())
            self.assertEqual(payload["source_file_count"], 1)
            self.assertEqual(payload["media_sha256"], ["a" * 64])
            printed = json.loads(stdout.getvalue())
            self.assertEqual(set(printed), {"status", "source_file_count", "unique_media_count",
                                            "source_inventory_sha256", "manifest_sha256"})
            self.assertNotIn("synthetic_generation", stdout.getvalue())
            with contextlib.redirect_stdout(io.StringIO()) as second_stdout:
                result = main(["--repo-root", str(repo), "--input",
                               "data/private_nextgqa/synthetic_generation.json",
                               "--output-name", "union.json"])
            self.assertEqual(result, 2)
            self.assertEqual(json.loads(second_stdout.getvalue())["failure_code"],
                             "manifest_build_failed")

    def test_symlink_unowned_and_wrong_mode_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            private = repo / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            os.chmod(private, 0o700)
            source = private / "source.json"
            source.write_bytes(encoded({"completed_forwards": 1}))
            os.chmod(source, 0o600)
            link = private / "link.json"
            link.symlink_to(source)
            with self.assertRaisesRegex(ValueError, "source_open_failed"):
                _read_private_source(repo, _safe_private_root(repo), Path(
                    "data/private_nextgqa/link.json"))
            os.chmod(source, 0o644)
            with self.assertRaisesRegex(ValueError, "source_permissions_or_size_invalid"):
                _read_private_source(repo, _safe_private_root(repo), Path(
                    "data/private_nextgqa/source.json"))
            os.chmod(source, 0o600)
            safe_root = _safe_private_root(repo)
            with patch("scripts.build_claim_exposure_union_manifest.os.getuid",
                       return_value=os.getuid() + 1):
                with self.assertRaisesRegex(ValueError, "source_permissions_or_size_invalid"):
                    _read_private_source(repo, safe_root, Path(
                        "data/private_nextgqa/source.json"))


if __name__ == "__main__":
    unittest.main()
