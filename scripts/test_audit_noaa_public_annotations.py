"""Adversarial CPU tests for the privacy-safe NOAA annotation CSV audit."""

import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from audit_noaa_public_annotations import (
    EXPECTED_HEADER,
    HEADER_SHA256,
    AuditFailure,
    _validate_public_url,
    _write_exclusive,
    audit_csv_bytes,
)


def fixture_bytes(records=None, *, header=None, preamble="Logs for Dive 1105 [1]:"):
    header = list(EXPECTED_HEADER if header is None else header)
    records = records if records is not None else [default_record()]
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow([])
    writer.writerow([preamble])
    writer.writerow(header)
    for record in records:
        writer.writerow([record.get(column, "") for column in header])
    return buffer.getvalue().encode("utf-8")


def default_record(**overrides):
    record = {column: "" for column in EXPECTED_HEADER}
    record.update({
        "Start Date": "2025-04-13T19:25:00Z",
        "End Date": "2025-04-13T19:25:00Z",
        "Modified Date": "2025-04-13T20:00:00+00:00",
        "To Be Reviewed": "false",
        "Creator First Name": "SensitiveFirst",
        "Creator Last Name": "SensitiveLast",
        "Creator Email": "person@example.invalid",
        "Modifier First Name": "OtherFirst",
        "Modifier Last Name": "OtherLast",
        "Modifier Email": "other@example.invalid",
        "Annotation ID": "private-ann-123",
        "Resource ID": "private-resource-456",
        "Dive ID": "private-dive-789",
        "Comment": "Sensitive annotation narrative with\nembedded newline",
        "Taxonomy": "SensitiveTaxonomy",
        "Taxon": "SensitiveTaxon",
        "Taxon Common Names": "Sensitive common name",
    })
    record.update(overrides)
    return record


class NoaaPublicAnnotationsAuditTests(unittest.TestCase):
    def test_expected_schema_is_frozen(self):
        self.assertEqual(len(EXPECTED_HEADER), 114)
        self.assertEqual(len(set(EXPECTED_HEADER)), 114)
        self.assertEqual(
            HEADER_SHA256,
            hashlib.sha256("\0".join(EXPECTED_HEADER).encode("utf-8")).hexdigest(),
        )

    def test_counts_point_anchor_and_emits_no_row_values(self):
        raw = fixture_bytes()
        report = audit_csv_bytes(
            raw,
            segment_start_utc="2025-04-13T19:24:59Z",
            segment_end_utc="2025-04-13T19:29:59Z",
        )
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["start_end_timestamp_valid_records"], 1)
        self.assertEqual(report["start_equals_end_point_records"], 1)
        self.assertEqual(report["positive_duration_records"], 0)
        self.assertEqual(report["reversed_interval_records"], 0)
        self.assertEqual(report["segment"]["point_anchors_in_segment"], 1)
        self.assertEqual(report["segment"]["point_anchors_at_exclusive_end"], 0)
        self.assertFalse(report["privacy"]["person_names_or_emails_emitted"])
        rendered = json.dumps(report, sort_keys=True)
        for secret in (
            "SensitiveFirst", "SensitiveLast", "person@example.invalid",
            "other@example.invalid", "private-ann-123", "private-resource-456",
            "private-dive-789", "Sensitive annotation narrative", "SensitiveTaxon",
            "Sensitive common name",
        ):
            self.assertNotIn(secret, rendered)

    def test_segment_start_included_end_excluded(self):
        records = [
            default_record(**{"Start Date": "2025-04-13T19:24:59Z",
                              "End Date": "2025-04-13T19:24:59Z"}),
            default_record(**{"Start Date": "2025-04-13T19:29:59Z",
                              "End Date": "2025-04-13T19:29:59Z"}),
            default_record(**{"Start Date": "2025-04-13T19:24:58Z",
                              "End Date": "2025-04-13T19:25:01Z"}),
        ]
        report = audit_csv_bytes(
            fixture_bytes(records),
            segment_start_utc="2025-04-13T19:24:59Z",
            segment_end_utc="2025-04-13T19:29:59Z",
        )
        segment = report["segment"]
        self.assertEqual(segment["point_anchors_in_segment"], 1)
        self.assertEqual(segment["point_anchors_at_exclusive_end"], 1)
        self.assertEqual(segment["positive_duration_rows_overlapping_segment"], 1)

    def test_quoted_newlines_are_handled_without_row_corruption(self):
        raw = fixture_bytes([default_record(Comment="first line\nsecond line")])
        report = audit_csv_bytes(raw)
        self.assertEqual(report["record_count"], 1)
        self.assertEqual(report["start_equals_end_point_records"], 1)

    def test_taxonomy_output_is_populated_counts_only(self):
        report = audit_csv_bytes(fixture_bytes())
        self.assertEqual(report["taxonomy_populated_record_counts"]["Taxonomy"], 1)
        self.assertEqual(report["taxonomy_populated_record_counts"]["Taxon"], 1)
        self.assertEqual(report["taxonomy_populated_record_counts"]["Genus"], 0)
        self.assertNotIn("SensitiveTaxonomy", json.dumps(report))

    def test_review_flag_is_reduced_to_fixed_categories(self):
        records = [default_record(**{"To Be Reviewed": "TRUE"}),
                   default_record(**{"To Be Reviewed": ""}),
                   default_record(**{"To Be Reviewed": "unexpected internal text"})]
        report = audit_csv_bytes(fixture_bytes(records))
        self.assertEqual(report["review_flag_counts"], {
            "true": 1, "false": 0, "blank": 1, "other": 1,
        })
        self.assertNotIn("unexpected internal text", json.dumps(report))

    def test_wrong_preamble_rejected_without_echoing_it(self):
        secret = "private title with a name"
        with self.assertRaisesRegex(AuditFailure, "invalid_export_preamble") as caught:
            audit_csv_bytes(fixture_bytes(preamble=secret))
        self.assertNotIn(secret, str(caught.exception))

    def test_wrong_header_or_duplicate_header_rejected(self):
        wrong = list(EXPECTED_HEADER)
        wrong[10] = "unexpected identity field"
        with self.assertRaisesRegex(AuditFailure, "unexpected_export_header"):
            audit_csv_bytes(fixture_bytes(header=wrong))
        duplicate = list(EXPECTED_HEADER)
        duplicate[-1] = duplicate[0]
        with self.assertRaisesRegex(AuditFailure, "unexpected_export_header"):
            audit_csv_bytes(fixture_bytes(header=duplicate))

    def test_wrong_row_width_rejected(self):
        raw = fixture_bytes()
        rows = list(csv.reader(io.StringIO(raw.decode("utf-8"), newline="")))
        rows[3] = rows[3][:-1]
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerows(rows)
        with self.assertRaisesRegex(AuditFailure, "invalid_record_width"):
            audit_csv_bytes(buffer.getvalue().encode("utf-8"))

    def test_invalid_or_naive_timestamps_rejected(self):
        for bad in ("not a date", "2025-04-13T19:25:00"):
            with self.subTest(bad=bad):
                raw = fixture_bytes([default_record(**{"Start Date": bad})])
                reason = "invalid_timestamp" if bad == "not a date" else "timezone_missing"
                with self.assertRaisesRegex(AuditFailure, reason):
                    audit_csv_bytes(raw)

    def test_reversed_intervals_fail_closed(self):
        raw = fixture_bytes([default_record(
            **{"Start Date": "2025-04-13T19:26:00Z", "End Date": "2025-04-13T19:25:00Z"}
        )])
        with self.assertRaisesRegex(AuditFailure, "reversed_timestamp_interval"):
            audit_csv_bytes(raw)

    def test_invalid_utf8_and_malformed_csv_rejected(self):
        with self.assertRaisesRegex(AuditFailure, "invalid_utf8"):
            audit_csv_bytes(b"\xff")
        malformed = fixture_bytes().decode("utf-8").replace(
            "Sensitive annotation narrative with", '"unclosed narrative with', 1
        ).encode("utf-8")
        with self.assertRaisesRegex(AuditFailure, "malformed_csv"):
            audit_csv_bytes(malformed)

    def test_segment_bounds_require_both_valid_aware_ordered_values(self):
        valid = fixture_bytes()
        with self.assertRaisesRegex(AuditFailure, "incomplete_segment_bounds"):
            audit_csv_bytes(valid, segment_start_utc="2025-04-13T19:24:59Z")
        with self.assertRaisesRegex(AuditFailure, "timezone_missing"):
            audit_csv_bytes(valid, segment_start_utc="2025-04-13T19:24:59",
                            segment_end_utc="2025-04-13T19:29:59Z")
        with self.assertRaisesRegex(AuditFailure, "invalid_segment_bounds"):
            audit_csv_bytes(valid, segment_start_utc="2025-04-13T19:29:59Z",
                            segment_end_utc="2025-04-13T19:24:59Z")

    def test_source_url_allowlist_blocks_credentials_queries_and_other_domains(self):
        accepted = "https://oer.hpc.msstate.edu/okeanos/ex2503/Eventlogs/EX2503_DIVE03_ANNOTATIONS.csv"
        self.assertEqual(_validate_public_url(accepted), accepted)
        for rejected in (
            "http://oer.hpc.msstate.edu/file.csv",
            "https://user:secret@ncei.noaa.gov/file.csv",
            "https://ncei.noaa.gov/file.csv?token=secret",
            "https://example.org/file.csv",
        ):
            with self.subTest(url=rejected):
                with self.assertRaises(AuditFailure):
                    _validate_public_url(rejected)

    def test_source_urls_are_restricted_to_expected_public_endpoints(self):
        with self.assertRaisesRegex(AuditFailure, "source_url_not_allowlisted"):
            _validate_public_url(
                "https://ncei.noaa.gov/private.csv",
                allowed_hosts={"oer.hpc.msstate.edu"},
            )
        with self.assertRaisesRegex(AuditFailure, "source_url_not_allowlisted"):
            _validate_public_url(
                "https://oer.hpc.msstate.edu/video.html",
                allowed_hosts={"ncei.noaa.gov"},
            )

    def test_output_must_be_unique_results_json(self):
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp)
            output = _write_exclusive(Path("results/report.json"), {"ok": True}, repo)
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                _write_exclusive(Path("results/report.json"), {"ok": False}, repo)
            with self.assertRaises(ValueError):
                _write_exclusive(Path("other/report.json"), {"bad": True}, repo)


if __name__ == "__main__":
    unittest.main()
