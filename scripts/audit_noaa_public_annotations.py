"""Audit a public NOAA Okeanos annotation CSV without publishing row content.

The NOAA event-log export contains personal names and email addresses. This
CPU-only parser never places row values, annotation text, taxonomy labels,
identifiers, or local paths in its report. It records structural/time counts
and optional point-anchor membership in a caller-specified UTC video segment.

Example (after checking rights and source metadata):
    python scripts/audit_noaa_public_annotations.py \
      --input-csv /private/EX2503_DIVE03_ANNOTATIONS.csv \
      --segment-start-utc 2025-04-13T19:24:59Z \
      --segment-end-utc 2025-04-13T19:29:59Z \
      --source-url https://oer.hpc.msstate.edu/okeanos/ex2503/Eventlogs/EX2503_DIVE03_ANNOTATIONS.csv

The segment end is exclusive for anchor counting. This is a metadata audit,
not a determination that an annotation semantically supports a model claim.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit


EXPECTED_HEADER = (
    "Dive ID", "Dive Name", "Cruise Name", "Start Date", "End Date",
    "Annotation ID", "Annotation Source", "Modified Date", "Resource Type ID",
    "Resource ID", "Creator First Name", "Creator Last Name", "Creator Email",
    "Modifier First Name", "Modifier Last Name", "Modifier Email", "To Be Reviewed",
    "Comment", "TEMPPROBEDEEPDISCOVERERROV_23974_Temperature (C)",
    "TEMPPROBEDEEPDISCOVERERROV_23974_Temperature (C) Time",
    "DEEPDISCOVERERNAV01_23975_Latitude (deg)",
    "DEEPDISCOVERERNAV01_23975_Latitude (deg) Time",
    "DEEPDISCOVERERNAV01_23975_Pitch (deg)",
    "DEEPDISCOVERERNAV01_23975_Pitch (deg) Time",
    "DEEPDISCOVERERNAV01_23975_Longitude (deg)",
    "DEEPDISCOVERERNAV01_23975_Longitude (deg) Time",
    "DEEPDISCOVERERNAV01_23975_Roll (deg)",
    "DEEPDISCOVERERNAV01_23975_Roll (deg) Time",
    "DEEPDISCOVERERNAV01_23975_Altitude (m)",
    "DEEPDISCOVERERNAV01_23975_Altitude (m) Time",
    "DEEPDISCOVERERNAV01_23975_Heading (deg)",
    "DEEPDISCOVERERNAV01_23975_Heading (deg) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Oxidation Reduction Potential (mV)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Oxidation Reduction Potential (mV) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Oxygen Concentration (mg/l)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Oxygen Concentration (mg/l) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Temperature (C)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Temperature (C) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Depth (m)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Depth (m) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Practical Salinity (psu)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Practical Salinity (psu) Time",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Turbidity (FTU)",
    "SBECTD9PLUSDEEPDISCOVERER_23978_Turbidity (FTU) Time",
    "Taxonomy", "Taxon", "Taxon Common Names", "Total # Reviews",
    "% Positive Reviews", "Taxon Path", "Taxonomy Attributes", "Biota",
    "Kingdom", "Subkingdom", "Infrakingdom", "Phylum", "Division",
    "Subphylum", "Subdivision", "Infraphylum", "Infradivision", "Parvphylum",
    "Parvdivision", "Gigaclass", "Megaclass", "Superclass", "Class", "Subclass",
    "Infraclass", "Subterclass", "Superorder", "Order", "Suborder", "Infraorder",
    "Parvorder", "Section", "Subsection", "Superfamily", "Epifamily", "Family",
    "Subfamily", "Supertribe", "Tribe", "Subtribe", "Genus", "Subgenus",
    "Species", "Subspecies", "Natio", "Variety", "Subvariety", "Forma",
    "Subforma", "Component", "Mutatio", "Subcomponent", "Water Column Layer",
    "Hydroform Class", "Hydroform", "Hydroform Type", "Salinity", "Temperature",
    "Biogeochemical Feature", "Substrate Origin", "Substrate Class",
    "Substrate Subclass", "Substrate Group", "Substrate Subgroup",
    "Physiographic Setting", "Tectonic Setting", "Geoform Origin", "Geoform",
    "Geoform Type", "Unclassified",
)
HEADER_SHA256 = hashlib.sha256("\0".join(EXPECTED_HEADER).encode("utf-8")).hexdigest()
PREAMBLE_RE = re.compile(r"^Logs for Dive [0-9]+ \[[0-9]+\]:$")
REQUIRED_COLUMNS = {
    "Start Date", "End Date", "To Be Reviewed", "Taxonomy", "Taxon",
    "Taxon Common Names", "Annotation ID", "Creator Email", "Modifier Email",
}
TAXONOMY_COLUMNS = (
    "Taxonomy", "Taxon", "Taxon Common Names", "Taxon Path", "Biota",
    "Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species",
    "Geoform", "Geoform Type", "Substrate Class", "Water Column Layer",
)


class AuditFailure(ValueError):
    """Input cannot be safely interpreted under the known NOAA export schema."""


def _parse_aware_utc(value: str, field: str) -> datetime:
    text = value.strip()
    if not text:
        raise AuditFailure("missing_timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise AuditFailure("invalid_timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise AuditFailure("timezone_missing")
    return parsed.astimezone(timezone.utc)


def _segment_bounds(start_utc: str | None, end_utc: str | None) -> tuple[datetime, datetime] | None:
    if start_utc is None and end_utc is None:
        return None
    if start_utc is None or end_utc is None:
        raise AuditFailure("incomplete_segment_bounds")
    start = _parse_aware_utc(start_utc, "segment_start")
    end = _parse_aware_utc(end_utc, "segment_end")
    if end <= start:
        raise AuditFailure("invalid_segment_bounds")
    return start, end


def _parse_csv_records(raw: bytes) -> tuple[str, list[dict[str, str]]]:
    if not isinstance(raw, bytes):
        raise AuditFailure("input_not_bytes")
    source_hash = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        raise AuditFailure("invalid_utf8") from None
    try:
        parsed_rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    except csv.Error:
        raise AuditFailure("malformed_csv") from None
    if len(parsed_rows) < 3 or parsed_rows[0] != []:
        raise AuditFailure("invalid_export_preamble")
    if len(parsed_rows[1]) != 1 or not PREAMBLE_RE.fullmatch(parsed_rows[1][0].strip()):
        raise AuditFailure("invalid_export_preamble")
    header = parsed_rows[2]
    if len(header) != len(EXPECTED_HEADER) or tuple(header) != EXPECTED_HEADER:
        raise AuditFailure("unexpected_export_header")
    if len(set(header)) != len(header) or not REQUIRED_COLUMNS.issubset(header):
        raise AuditFailure("invalid_export_header")

    records: list[dict[str, str]] = []
    for row in parsed_rows[3:]:
        if len(row) != len(EXPECTED_HEADER):
            raise AuditFailure("invalid_record_width")
        records.append(dict(zip(EXPECTED_HEADER, row)))
    if not records:
        raise AuditFailure("no_records")
    return source_hash, records


def audit_csv_bytes(
    raw: bytes,
    *,
    segment_start_utc: str | None = None,
    segment_end_utc: str | None = None,
) -> dict[str, Any]:
    """Return aggregate-only counts; never return or interpolate source rows."""
    bounds = _segment_bounds(segment_start_utc, segment_end_utc)
    source_hash, records = _parse_csv_records(raw)
    review_counts = {"true": 0, "false": 0, "blank": 0, "other": 0}
    taxonomy_populated = {name: 0 for name in TAXONOMY_COLUMNS}
    point_records = 0
    positive_intervals = 0
    point_anchors_in_segment = 0
    point_anchors_at_segment_end = 0
    rows_overlapping_segment = 0

    for record in records:
        start = _parse_aware_utc(record["Start Date"], "start")
        end = _parse_aware_utc(record["End Date"], "end")
        if end == start:
            point_records += 1
        elif end > start:
            positive_intervals += 1
        else:
            raise AuditFailure("reversed_timestamp_interval")

        status = record["To Be Reviewed"].strip().casefold()
        if not status:
            review_counts["blank"] += 1
        elif status in {"true", "yes", "1"}:
            review_counts["true"] += 1
        elif status in {"false", "no", "0"}:
            review_counts["false"] += 1
        else:
            review_counts["other"] += 1

        for field in TAXONOMY_COLUMNS:
            if record[field].strip():
                taxonomy_populated[field] += 1

        if bounds is not None:
            seg_start, seg_end = bounds
            # Point observations are anchors: include [start, end), and count the
            # right boundary separately rather than treating it as an overlap.
            if end == start and seg_start <= start < seg_end:
                point_anchors_in_segment += 1
            elif end == start and start == seg_end:
                point_anchors_at_segment_end += 1
            elif start < seg_end and end > seg_start and end > start:
                rows_overlapping_segment += 1

    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "noaa_public_annotation_csv_structure_and_time_audit",
        "status": "complete",
        "source_sha256": source_hash,
        "source_bytes": len(raw),
        "export_header_columns": len(EXPECTED_HEADER),
        "export_header_sha256": HEADER_SHA256,
        "record_count": len(records),
        "start_end_timestamp_valid_records": len(records),
        "validated_timestamp_fields": ["Start Date", "End Date"],
        "start_equals_end_point_records": point_records,
        "positive_duration_records": positive_intervals,
        "reversed_interval_records": 0,
        "review_flag_counts": review_counts,
        "taxonomy_populated_record_counts": taxonomy_populated,
        "privacy": {
            "row_values_emitted": False,
            "person_names_or_emails_emitted": False,
            "annotation_text_or_identifiers_emitted": False,
            "taxonomy_labels_emitted": False,
            "absolute_input_path_emitted": False,
        },
        "annotation_semantics": (
            "Start/End timestamps are preserved as point anchors when equal; this audit "
            "does not determine semantic support for a model claim."
        ),
        "review_flag_semantics": (
            "Counts classify the export's To Be Reviewed field only; they do not certify "
            "annotation quality-control completion."
        ),
    }
    if bounds is not None:
        report["segment"] = {
            "start_utc": bounds[0].isoformat().replace("+00:00", "Z"),
            "end_utc_exclusive": bounds[1].isoformat().replace("+00:00", "Z"),
            "point_anchors_in_segment": point_anchors_in_segment,
            "point_anchors_at_exclusive_end": point_anchors_at_segment_end,
            "positive_duration_rows_overlapping_segment": rows_overlapping_segment,
            "membership_rule": "UTC point anchors in [segment_start, segment_end)",
        }
    return report


def _source_metadata(script_path: Path) -> dict[str, Any]:
    repo_root = script_path.resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True,
            check=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "runner_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
    }


def _write_exclusive(output: Path, report: dict[str, Any], repo_root: Path | None = None) -> Path:
    repo = Path(__file__).resolve().parents[1] if repo_root is None else repo_root.resolve()
    target = (repo / output).resolve() if not output.is_absolute() else output.resolve()
    if target.parent != (repo / "results").resolve() or target.suffix.lower() != ".json":
        raise ValueError("output must be a JSON file directly under repository results/")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return target


def _concise(report: dict[str, Any]) -> str:
    segment = report.get("segment")
    line = (
        f"NOAA annotations: records={report['record_count']}; "
        f"point={report['start_equals_end_point_records']}; "
        f"positive_intervals={report['positive_duration_records']}; "
        f"reversed={report['reversed_interval_records']}\n"
        f"review flags={report['review_flag_counts']}; "
        f"source_sha256={report['source_sha256']}"
    )
    if segment:
        line += (
            f"\nsegment point anchors={segment['point_anchors_in_segment']}; "
            f"at exclusive end={segment['point_anchors_at_exclusive_end']}"
        )
    return line


def _validate_public_url(value: str, *, allowed_hosts: set[str] | None = None) -> str:
    if allowed_hosts is None:
        allowed_hosts = {
            "oer.hpc.msstate.edu", "ncei.noaa.gov", "www.ncei.noaa.gov",
            "oceanexplorer.noaa.gov", "www.oceanexplorer.noaa.gov",
        }
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AuditFailure("invalid_source_url")
    try:
        parts = urlsplit(value)
    except ValueError:
        raise AuditFailure("invalid_source_url") from None
    if (parts.scheme != "https" or parts.hostname not in allowed_hosts
            or parts.username or parts.password or parts.query or parts.fragment):
        raise AuditFailure("source_url_not_allowlisted")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--segment-start-utc")
    parser.add_argument("--segment-end-utc")
    parser.add_argument("--source-url", help="public annotation-export URL; never a local path")
    parser.add_argument("--video-record-url", help="public NOAA/NCEI video metadata URL")
    parser.add_argument("--output", help="unique JSON output directly under results/")
    args = parser.parse_args(argv)
    input_path = Path(args.input_csv).expanduser()
    try:
        raw = input_path.read_bytes()
    except OSError:
        parser.error("could not read annotation CSV")
    try:
        report = audit_csv_bytes(
            raw,
            segment_start_utc=args.segment_start_utc,
            segment_end_utc=args.segment_end_utc,
        )
    except AuditFailure as error:
        parser.error(str(error))
    if args.source_url:
        try:
            report["declared_source_url"] = _validate_public_url(
                args.source_url, allowed_hosts={"oer.hpc.msstate.edu"})
        except AuditFailure as error:
            parser.error(str(error))
    if args.video_record_url:
        try:
            report["declared_video_record_url"] = _validate_public_url(
                args.video_record_url,
                allowed_hosts={"ncei.noaa.gov", "www.ncei.noaa.gov"},
            )
        except AuditFailure as error:
            parser.error(str(error))
    if args.source_url or args.video_record_url:
        report["declared_url_scope"] = (
            "caller-supplied provenance references; this parser hashes the local input bytes "
            "but does not fetch URLs or verify that they supplied those bytes"
        )
    report.update(_source_metadata(Path(__file__)))
    if args.output:
        output = Path(args.output).expanduser()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results") / f"noaa_public_annotation_audit_{stamp}.json"
    try:
        output_path = _write_exclusive(output, report)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(_concise(report))
    print(f"artifact: {output_path.relative_to(Path(__file__).resolve().parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
