import copy
from contextlib import redirect_stdout
import csv
import hashlib
import hmac
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.audit_nextgqa_source_ledger_split import (
    audit_split_against_source_ledger, main,
)
from scripts.build_nextgqa_prior_exposure_source_ledger import (
    LEDGER_DOMAIN, ROSTER_PURPOSE, _canonical, build_source_ledger,
)


KEY = b"synthetic source ledger key with entropy 0123456789abcdef"
WRONG_KEY = b"another synthetic key with entropy 0123456789abcdef"


def raw(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")


class AuditSourceLedgerSplitTests(unittest.TestCase):
    def setUp(self):
        entries = [
            {"family": "sample-json", "namespace": "nextgqa", "relative_path": "p/a.json"},
            {"family": "lineage-json", "namespace": "nextgqa", "relative_path": "p/b.json"},
        ]
        roster = raw({"schema_version": 1, "purpose": ROSTER_PURPOSE,
                      "inputs": entries, "omitted_duplicate_content": []})
        self.source_raws = [
            raw([{"video_id": "prior-a", "qid": "q1"}, {"video_id": "prior-b", "qid": "q2"}]),
            raw({"records": [{"dataset_video_id": "prior-a",
                               "dataset_question_id": "q3"},
                              {"dataset_video_id": "lineage-only",
                               "dataset_question_id": "q4"}]})]
        self.ledger, _ = build_source_ledger(
            roster_raw=roster, source_raws=self.source_raws, secret=KEY)
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=["video_id", "qid", "question", "answer"])
        writer.writeheader()
        writer.writerows([
            {"video_id": "prior-a", "qid": "new-q", "question": "private", "answer": "secret"},
            {"video_id": "fresh-looking", "qid": "q4", "question": "private2", "answer": "secret2"},
        ])
        self.split = out.getvalue().encode()

    def test_exact_video_match_is_source_attributed_and_not_freshness(self):
        report = audit_split_against_source_ledger(
            split_raw=self.split, ledger_raw=raw(self.ledger), secret=KEY)
        self.assertEqual(report["split_unique_video_ids"], 2)
        self.assertEqual(report["prior_unique_nextgqa_video_id_tokens"], 3)
        self.assertEqual(report["matched_video_ids_in_source_union"], 1)
        self.assertEqual(report["matched_source_records_by_class"]["sample-json"], {
            "source_count": 1, "sources_with_video_id_match": 1,
            "matched_unique_video_ids": 1})
        self.assertEqual(report["matched_source_records_by_class"]["lineage-json"], {
            "source_count": 1, "sources_with_video_id_match": 1,
            "matched_unique_video_ids": 1})
        self.assertEqual(report["freshness_conclusion"], "not_made")
        self.assertFalse(report["historical_completeness_authenticated"])
        self.assertEqual(report["execution_authorization"], "not_granted")
        self.assertTrue(report["ledger_integrity_verified"])

    def test_exact_source_bytes_exclusion_removes_self_match_only(self):
        # lineage-only is present only in the second, lineage-json source.
        out = io.StringIO(newline="")
        writer = csv.DictWriter(out, fieldnames=["video_id", "qid"])
        writer.writeheader()
        writer.writerow({"video_id": "lineage-only", "qid": "new-q"})
        report = audit_split_against_source_ledger(
            split_raw=out.getvalue().encode(), ledger_raw=raw(self.ledger),
            secret=KEY, exclude_source_raws=[self.source_raws[1]])
        self.assertEqual(report["matched_video_ids_in_source_union"], 0)
        self.assertEqual(report["canonical_nextgqa_source_record_count"], 2)
        self.assertEqual(report["compared_nextgqa_source_record_count"], 1)
        self.assertEqual(report["excluded_source_record_count"], 1)
        self.assertEqual(report["matched_source_records_by_class"], {
            "sample-json": {"source_count": 1, "sources_with_video_id_match": 0,
                            "matched_unique_video_ids": 0},
        })

    def test_exclusion_requires_exactly_one_ledger_source(self):
        with self.assertRaisesRegex(ValueError, "excluded_source_not_in_ledger"):
            audit_split_against_source_ledger(
                split_raw=self.split, ledger_raw=raw(self.ledger), secret=KEY,
                exclude_source_raws=[b"not an exact source file"])
        with self.assertRaisesRegex(ValueError, "excluded_source_duplicate"):
            audit_split_against_source_ledger(
                split_raw=self.split, ledger_raw=raw(self.ledger), secret=KEY,
                exclude_source_raws=[self.source_raws[1], self.source_raws[1]])

    def test_excluded_source_bytes_are_bounded(self):
        with self.assertRaisesRegex(ValueError, "excluded_source_bytes_invalid"):
            audit_split_against_source_ledger(
                split_raw=self.split, ledger_raw=raw(self.ledger), secret=KEY,
                exclude_source_raws=["not bytes"])

    def test_cli_exact_source_exclusion_is_aggregate_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split = root / "split.csv"
            ledger = root / "ledger.json"
            key = root / "key.bin"
            source = root / "source.json"
            split.write_bytes(self.split)
            ledger.write_bytes(raw(self.ledger))
            key.write_bytes(KEY)
            source.write_bytes(self.source_raws[1])
            os.chmod(ledger, 0o600)
            os.chmod(key, 0o600)
            stdout = io.StringIO()
            with patch.object(sys, "argv", [
                    "audit_nextgqa_source_ledger_split.py",
                    "--split-csv", str(split), "--source-ledger", str(ledger),
                    "--hmac-key", str(key), "--exclude-source", str(source)]), \
                    redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            output = stdout.getvalue()
            report = json.loads(output)
            self.assertEqual(report["excluded_source_record_count"], 1)
            self.assertEqual(report["aggregation_scope"],
                             "explicit_sources_excluding_exact_source_bytes")
            self.assertNotIn(str(source), output)

    def test_wrong_key_nonmatch_remains_unverified(self):
        with self.assertRaisesRegex(ValueError, "ledger_authentication_failed"):
            audit_split_against_source_ledger(
                split_raw=self.split, ledger_raw=raw(self.ledger), secret=WRONG_KEY)

    def test_tampered_union_and_authenticated_coverage_reject(self):
        bad_union = copy.deepcopy(self.ledger)
        bad_union["known_identity_unions"]["dataset_video_hmac_sha256"] = []
        with self.assertRaisesRegex(ValueError, "ledger_authentication_failed"):
            audit_split_against_source_ledger(split_raw=self.split,
                                              ledger_raw=raw(bad_union), secret=KEY)
        bad_coverage = copy.deepcopy(self.ledger)
        bad_coverage["coverage"]["historical_completeness_authenticated"] = True
        bad_coverage.pop("ledger_integrity_hmac_sha256")
        bad_coverage["ledger_integrity_hmac_sha256"] = hmac.new(
            KEY, LEDGER_DOMAIN + _canonical(bad_coverage), hashlib.sha256).hexdigest()
        with self.assertRaisesRegex(ValueError, "ledger_coverage_flags_invalid"):
            audit_split_against_source_ledger(split_raw=self.split,
                                              ledger_raw=raw(bad_coverage), secret=KEY)

    def test_signed_unknown_source_class_is_rejected_without_echoing(self):
        bad = copy.deepcopy(self.ledger)
        bad["sources"][0]["source_class"] = "ATTACKER_CONTROLLED_CLASS"
        bad.pop("ledger_integrity_hmac_sha256")
        bad["ledger_integrity_hmac_sha256"] = hmac.new(
            KEY, LEDGER_DOMAIN + _canonical(bad), hashlib.sha256).hexdigest()
        with self.assertRaisesRegex(ValueError, "ledger_source_invalid"):
            audit_split_against_source_ledger(split_raw=self.split,
                                              ledger_raw=raw(bad), secret=KEY)


if __name__ == "__main__":
    unittest.main()
