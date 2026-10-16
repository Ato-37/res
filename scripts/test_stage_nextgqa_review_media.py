import hashlib
import json
import os
import stat
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from stage_nextgqa_review_media import stage
from test_audit_nextgqa_review_handoff import frozen_worksheet
from test_build_nextgqa_claim_mapping_packet import encode


class StageNextgqaReviewMediaTests(unittest.TestCase):
    def test_staging_uses_token_names_and_private_verified_manifest(self):
        worksheet, _ = frozen_worksheet()
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp); private = repo / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700); os.chmod(private, 0o700)
            root = repo / "videos"; root.mkdir()
            lineage_rows = []
            for ordinal, row in enumerate(worksheet["records"], 1):
                relative = f"group/item-{ordinal}.mp4"
                source = root / relative
                source.parent.mkdir(parents=True, exist_ok=True)
                data = ("synthetic bytes " + str(ordinal)).encode()
                source.write_bytes(data)
                digest = hashlib.sha256(data).hexdigest()
                row["video_sha256"] = digest
                lineage_rows.append({"claim_id": row["claim_id"], "relative_video_path": relative})
            lineage = {"records": lineage_rows}
            worksheet_raw, lineage_raw = encode(worksheet), encode(lineage)
            media_dir, manifest = private / "media_v5", private / "custody_v5.json"
            with patch("stage_nextgqa_review_media.render"):
                result = stage(worksheet, worksheet_raw, lineage, lineage_raw,
                               hashlib.sha256(lineage_raw).hexdigest(), root, media_dir, manifest, repo)
            self.assertEqual(result["staged_records"], 3)
            self.assertEqual(stat.S_IMODE(media_dir.stat().st_mode), 0o700)
            custody = json.loads(manifest.read_bytes())
            self.assertEqual(custody["record_count"], 3)
            self.assertEqual({p.name for p in media_dir.iterdir()},
                             {row["review_token"] + ".mp4" for row in worksheet["records"]})
            self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in media_dir.iterdir()))
            for entry in custody["records"]:
                data = (media_dir / entry["media_file"]).read_bytes()
                self.assertEqual(hashlib.sha256(bytes.fromhex(entry["media_check_salt"]) + data).hexdigest(),
                                 entry["media_check_sha256"])


if __name__ == "__main__":
    unittest.main()
