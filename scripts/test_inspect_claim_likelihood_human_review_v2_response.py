from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from inspect_claim_likelihood_human_review_v2_response import inspect


class ReviewV2ResponseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.artifact = root / "artifact.html"
        self.response = root / "response.json"
        self.items = [{"item_id": f"item-{i}"} for i in range(1, 4)]
        self.binding = {"schema_version": 1, "packet_sha256": "a" * 64,
                        "result_sha256": "b" * 64,
                        "items": [{"item_id": x["item_id"], "video_sha256": "c" * 64,
                                   "claim_sha256": "d" * 64, "pts_sha256": "e" * 64}
                                  for x in self.items]}
        self.fingerprint = self._write_artifact()
        self.response.write_text(json.dumps({
            "schema_version": 1, "purpose": "exploratory_local_human_review",
            "status": "informal_observations_only",
            "artifact_fingerprint": self.fingerprint,
            "source_binding": self.binding,
            "respondents": [{"nickname": "reviewer-secret", "answers": [
                {"item_id": x["item_id"], "item_number": i,
                 "judgment": "roughly_matches", "note": "private-note"}
                for i, x in enumerate(self.items, 1)]}]}, ensure_ascii=False))

    def tearDown(self):
        self.tmp.cleanup()

    def _write_artifact(self):
        body = ('<meta name="review-artifact-sha256" content="__FP__">'
                '<script>const items=' + json.dumps(self.items) + ',binding=' +
                json.dumps(self.binding) + ';let idx=0;</script>')
        fp = hashlib.sha256(body.encode()).hexdigest()
        self.artifact.write_text(body.replace("__FP__", fp))
        return fp

    def test_valid_response_aggregates_without_names_or_notes(self):
        result = inspect(self.response, self.artifact)
        rendered = json.dumps(result)
        self.assertEqual(result["complete_reviewer_sessions"], 1)
        self.assertEqual(result["judgment_counts"]["roughly_matches"], 3)
        self.assertFalse(result["formal_faithfulness_evidence"])
        self.assertNotIn("reviewer-secret", rendered)
        self.assertNotIn("private-note", rendered)

    def test_rejects_artifact_substitution(self):
        self.artifact.write_text(self.artifact.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "fingerprint_invalid"):
            inspect(self.response, self.artifact)

    def test_rejects_incomplete_or_misbound_answers(self):
        payload = json.loads(self.response.read_text())
        payload["respondents"][0]["answers"].pop()
        self.response.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "session_incomplete"):
            inspect(self.response, self.artifact)

    def test_rejects_artifact_binding_mismatch(self):
        payload = json.loads(self.response.read_text())
        payload["source_binding"]["result_sha256"] = "0" * 64
        self.response.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError, "source_binding_mismatch"):
            inspect(self.response, self.artifact)

    def test_rejects_duplicate_json_keys(self):
        raw = self.response.read_text()
        self.response.write_text(raw.replace('"schema_version": 1',
                                              '"schema_version": 1, "schema_version": 1',
                                              1))
        with self.assertRaisesRegex(ValueError, "duplicate_json_key"):
            inspect(self.response, self.artifact)


if __name__ == "__main__":
    unittest.main()
