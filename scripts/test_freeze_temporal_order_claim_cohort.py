from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from freeze_temporal_order_claim_cohort import (
    _private_json_sha256, _scored_video_digests, _selection_exclusions,
)


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

    def test_corrupt_candidate_receipt_is_ignored_without_exposing_contents(self):
        with TemporaryDirectory() as temp:
            p = Path(temp) / "claim_generation_bad.json"
            p.write_text("not-json", encoding="utf-8")
            self.assertEqual(_scored_video_digests(p.parent), set())


if __name__ == "__main__":
    unittest.main()
