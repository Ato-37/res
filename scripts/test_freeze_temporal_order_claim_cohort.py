from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from freeze_temporal_order_claim_cohort import _scored_video_digests


class ScoredVideoDigestAuditTests(unittest.TestCase):
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

    def test_corrupt_candidate_receipt_is_ignored_without_exposing_contents(self):
        with TemporaryDirectory() as temp:
            p = Path(temp) / "claim_generation_bad.json"
            p.write_text("not-json", encoding="utf-8")
            self.assertEqual(_scored_video_digests(p.parent), set())


if __name__ == "__main__":
    unittest.main()
