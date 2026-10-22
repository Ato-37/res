from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from freeze_temporal_order_claim_cohort import (
    _private_json_sha256, _read_exposure_manifest, _scored_video_digests,
    _selection_exclusions,
)
from build_claim_exposure_union_manifest import build_manifest


class ScoredVideoDigestAuditTests(unittest.TestCase):
    def test_selected_filter_hash_matches_the_private_file_bytes(self):
        payload = {"schema_version": 1, "pair_sha256": ["a" * 64]}
        expected = hashlib.sha256(
            (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        ).hexdigest()
        self.assertEqual(_private_json_sha256(payload), expected)

    def test_selection_exclusions_union_source_and_prior_scored_media(self):
        source = {"a" * 64, "b" * 64}
        scored = {"b" * 64, "c" * 64}
        self.assertEqual(_selection_exclusions(source, scored),
                         {"a" * 64, "b" * 64, "c" * 64})

    def test_reads_only_scoring_output_families_and_video_digest_fields(self):
        digest = hashlib.sha256(b"video").hexdigest()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, payload in (
                ("claim_generation_x.json", {"records": [{"video_sha256": digest}]}),
                ("saved_claim_result.json", {"source_video_sha256": digest}),
                ("candidate_pool.json", {"video_sha256": digest}),
                ("saved_claim_noise.json", {"media_sha256": digest}),
            ):
                p = root / name
                p.write_text(json.dumps(payload), encoding="utf-8")
                os.chmod(p, 0o600)
            self.assertEqual(_scored_video_digests(root), {digest})

    def test_nonstandard_result_filename_is_included_only_with_score_marker(self):
        scored = hashlib.sha256(b"scored-video").hexdigest()
        unscored = hashlib.sha256(b"unscored-video").hexdigest()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, payload in (
                ("opaque_result.json", {"rows": [{"video_sha256": scored}],
                 "audit": {"completed_forwards": 1}}),
                ("candidate_metadata.json", {"video_sha256": unscored}),
            ):
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                os.chmod(path, 0o600)
            self.assertEqual(_scored_video_digests(root), {scored})

    def test_nonstandard_claim_generation_purpose_is_prior_exposure(self):
        generated = hashlib.sha256(b"temporal-generation-video").hexdigest()
        inventory_only = hashlib.sha256(b"inventory-only-video").hexdigest()
        with TemporaryDirectory() as temp:
            root = Path(temp)
            payloads = (
                ("temporal_order_claim_generation_run.json", {
                    "purpose": "temporal_order_claim_generation",
                    "status": "passed",
                    "records": [{"video_sha256": generated}],
                }),
                ("candidate_inventory.json", {
                    "purpose": "outcome_blind_local_candidate_pool",
                    "records": [{"video_sha256": inventory_only}],
                }),
            )
            for name, payload in payloads:
                path = root / name
                path.write_text(json.dumps(payload), encoding="utf-8")
                os.chmod(path, 0o600)
            self.assertEqual(_scored_video_digests(root), {generated})

    def test_corrupt_candidate_receipt_is_ignored_without_exposing_contents(self):
        with TemporaryDirectory() as temp:
            p = Path(temp) / "claim_generation_bad.json"
            p.write_text("not-json", encoding="utf-8")
            self.assertEqual(_scored_video_digests(p.parent), set())

    def test_freezer_revalidates_exposure_manifest_and_source_bytes(self):
        with TemporaryDirectory() as temp:
            repo = Path(temp)
            private = repo / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            os.chmod(private, 0o700)
            source = private / "generation.json"
            source_raw = json.dumps({
                "purpose": "temporal_order_claim_generation",
                "video_sha256": hashlib.sha256(b"prior-video").hexdigest(),
            }, sort_keys=True, separators=(",", ":")).encode()
            source.write_bytes(source_raw)
            os.chmod(source, 0o600)
            manifest = build_manifest([("data/private_nextgqa/generation.json", source_raw)])
            output = private / "exposure.json"
            output.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
            os.chmod(output, 0o600)
            media, manifest_sha, verified = _read_exposure_manifest(
                repo, Path("data/private_nextgqa/exposure.json"))
            self.assertEqual(media, set(manifest["media_sha256"]))
            self.assertEqual(manifest_sha, hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(verified["source_inventory_sha256"],
                             manifest["source_inventory_sha256"])
            source.write_bytes(source_raw + b" ")
            with self.assertRaisesRegex(ValueError, "source_hash_mismatch"):
                _read_exposure_manifest(repo, Path("data/private_nextgqa/exposure.json"))


if __name__ == "__main__":
    unittest.main()
