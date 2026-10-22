from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from analyze_test_split_internal_bias_frame_mean_interaction import _source_matches


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FrozenSourceProvenanceTests(unittest.TestCase):
    def _repo(self, live: bytes, frozen: bytes | None, digest: str):
        root = Path(tempfile.mkdtemp())
        (root / "scripts" / "frozen_sources").mkdir(parents=True)
        (root / "scripts" / "tool.py").write_bytes(live)
        if frozen is not None:
            (root / "scripts" / "frozen_sources" / f"tool__sha256_{digest[:8]}.py").write_bytes(frozen)
        return root

    def test_live_file_with_matching_digest_passes(self):
        digest = _sha(b"v1")
        self.assertTrue(_source_matches(self._repo(b"v1", None, digest), "scripts/tool.py", digest))

    def test_drifted_live_file_without_frozen_copy_fails(self):
        digest = _sha(b"v1")
        self.assertFalse(_source_matches(self._repo(b"v2", None, digest), "scripts/tool.py", digest))

    def test_drifted_live_file_with_exact_frozen_copy_passes(self):
        digest = _sha(b"v1")
        self.assertTrue(_source_matches(self._repo(b"v2", b"v1", digest), "scripts/tool.py", digest))

    def test_frozen_copy_with_wrong_content_fails(self):
        digest = _sha(b"v1")
        self.assertFalse(_source_matches(self._repo(b"v2", b"tampered", digest), "scripts/tool.py", digest))


if __name__ == "__main__":
    unittest.main()
