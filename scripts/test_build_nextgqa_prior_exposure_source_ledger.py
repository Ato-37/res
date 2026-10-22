import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.build_nextgqa_prior_exposure_source_ledger import (
    ROSTER_PURPOSE, build_source_ledger, main,
)


KEY = b"synthetic source ledger key with entropy 0123456789abcdef"


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


def roster(entries):
    return encoded({"schema_version": 1, "purpose": ROSTER_PURPOSE,
                    "inputs": entries, "omitted_duplicate_content": []})


class PriorExposureSourceLedgerTests(unittest.TestCase):
    def test_source_attribution_preserves_separate_identity_domains_and_redacts(self):
        rows = [
            [{"video_id": "PRIVATE VIDEO ID", "qid": "PRIVATE QID",
              "question": "PRIVATE QUESTION", "answer": "PRIVATE ANSWER"}],
            {"media_sha256": ["a" * 64], "purpose": "PRIVATE PURPOSE"},
        ]
        entries = [
            {"family": "sample-json", "namespace": "nextgqa",
             "relative_path": "private/source_one.json"},
            {"family": "media-set-json", "namespace": "nextgqa-media",
             "relative_path": "private/source_two.json"},
        ]
        manifest, raw = build_source_ledger(
            roster_raw=roster(entries), source_raws=[encoded(rows[0]), encoded(rows[1])],
            secret=KEY)
        self.assertEqual(len(manifest["sources"]), 2)
        self.assertEqual(len(manifest["sources"][0]["video_identity_hmac_sha256"]), 1)
        self.assertEqual(manifest["sources"][0]["media_identity_hmac_sha256"], [])
        self.assertEqual(len(manifest["sources"][1]["media_identity_hmac_sha256"]), 1)
        self.assertEqual(manifest["sources"][1]["video_identity_hmac_sha256"], [])
        self.assertEqual(manifest["coverage"]["freshness_conclusion"], "not_made")
        self.assertFalse(manifest["coverage"]["historical_completeness_authenticated"])
        text = raw.decode("utf-8")
        for private in ("PRIVATE VIDEO ID", "PRIVATE QID", "PRIVATE QUESTION",
                        "PRIVATE ANSWER", "PRIVATE PURPOSE", "private/source_one.json",
                        "private/source_two.json", "a" * 64):
            self.assertNotIn(private, text)
        self.assertRegex(manifest["sources"][0]["source_id_hmac_sha256"], r"^[0-9a-f]{64}$")

    def test_invalid_roster_path_duplicate_source_and_count_fail_closed(self):
        bad = [{"family": "sample-json", "namespace": "nextgqa",
                "relative_path": "../outside.json"}]
        with self.assertRaisesRegex(ValueError, "roster_path_invalid"):
            build_source_ledger(roster_raw=roster(bad), source_raws=[encoded([])], secret=KEY)
        entry = {"family": "sample-json", "namespace": "nextgqa",
                 "relative_path": "private/a.json"}
        with self.assertRaisesRegex(ValueError, "roster_duplicate_path"):
            build_source_ledger(roster_raw=roster([entry, dict(entry)]),
                                source_raws=[encoded([]), encoded([])], secret=KEY)
        with self.assertRaisesRegex(ValueError, "duplicate_source_content"):
            build_source_ledger(roster_raw=roster([
                dict(entry, relative_path="private/a.json"),
                dict(entry, relative_path="private/b.json")]),
                source_raws=[encoded([]), encoded([])], secret=KEY)

    def test_cli_writes_owner_only_output_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            private = root / "data" / "private_nextgqa"
            private.mkdir(parents=True, mode=0o700)
            private.chmod(0o700)
            source = private / "synthetic.json"
            source.write_bytes(encoded([{"video_id": "SECRET VIDEO"}]))
            source.chmod(0o600)
            roster_path = private / "roster.json"
            roster_path.write_bytes(roster([{"family": "sample-json", "namespace": "nextgqa",
                                             "relative_path": "data/private_nextgqa/synthetic.json"}]))
            roster_path.chmod(0o600)
            secret_path = private / "key.bin"
            secret_path.write_bytes(KEY)
            secret_path.chmod(0o600)
            output = private / "ledger.json"
            rel_roster = "data/private_nextgqa/roster.json"
            rel_secret = "data/private_nextgqa/key.bin"
            rel_output = "data/private_nextgqa/ledger.json"
            self.assertEqual(main(["--repo-root", str(root), "--roster", rel_roster,
                                   "--secret-file", rel_secret, "--output", rel_output]), 0)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            self.assertNotIn("SECRET VIDEO", output.read_text())
            digest = hashlib.sha256(output.read_bytes()).digest()
            with self.assertRaises(SystemExit):
                main(["--repo-root", str(root), "--roster", rel_roster,
                      "--secret-file", rel_secret, "--output", rel_output])
            self.assertEqual(hashlib.sha256(output.read_bytes()).digest(), digest)


if __name__ == "__main__":
    unittest.main()
