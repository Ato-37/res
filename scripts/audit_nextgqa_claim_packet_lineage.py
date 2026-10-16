#!/usr/bin/env python3
"""Cross-check a private claim packet's 3 media/QID rows against local NExT-GQA source metadata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import tempfile

from run_nextgqa_gold_interval_margin_pilot import video_key
from private_output_paths import validate_private_output


PURPOSE = "private_local_nextgqa_model_generated_claim_review_packet"
DATASET_KEYS = ("test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json")
LINEAGE_KEYS = {"schema_version", "purpose", "created_by", "source_packet_sha256",
                "cohort_sha256", "dataset_inputs_sha256", "requested_records",
                "effective_records", "verification_scope", "status", "records"}
LINEAGE_ROW_KEYS = {"video_id", "claim_id", "dataset_question_id", "dataset_video_id",
                    "video_key", "relative_video_path", "video_sha256",
                    "question_sha256", "claim_text_sha256"}
SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


def audit(packet: dict, packet_raw: bytes, repo: Path, video_root: Path) -> tuple[dict, dict]:
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    hashes = {name: _sha_file(dataset / name) for name in DATASET_KEYS}
    if packet.get("dataset_inputs_sha256") != hashes:
        raise ValueError("dataset_input_hash_mismatch")
    if packet.get("purpose") != PURPOSE or packet.get("requested_videos") != 3 \
            or packet.get("effective_videos") != 3 or not isinstance(packet.get("records"), list) \
            or len(packet["records"]) != 3:
        raise ValueError("claim_packet_scope_invalid")
    video_map, _ = _read_json(dataset / "map_vid_vidorID.json")
    qid_rows = {}
    with (dataset / "test.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            qid = str(row.get("qid", ""))
            qid_rows.setdefault(qid, []).append(row)

    lineage_rows, seen_media, seen_claim = [], set(), set()
    for row in packet["records"]:
        qid = row.get("dataset_question_id")
        csv_rows = qid_rows.get(qid, [])
        media_path = Path(row.get("local_video_path", "")).resolve()
        try:
            relative = media_path.relative_to(video_root.resolve()).as_posix()
        except ValueError:
            raise ValueError("source_video_outside_frozen_root") from None
        if not csv_rows:
            raise ValueError("dataset_question_not_found")
        matching = [candidate for candidate in csv_rows
                    if str(candidate.get("video_id", "")) in video_map
                    and video_key(video_map[str(candidate.get("video_id", ""))]) == video_key(relative)]
        if len(matching) != 1:
            raise ValueError("dataset_video_question_binding_mismatch")
        csv_row = matching[0]
        vid = str(csv_row.get("video_id", ""))
        if str(csv_row.get("question", "")).strip() != row.get("question"):
            raise ValueError("dataset_video_question_binding_mismatch")
        media_sha = _sha_file(media_path)
        if media_sha != row.get("video_sha256") or media_sha in seen_media:
            raise ValueError("source_video_hash_mismatch_or_duplicate")
        seen_media.add(media_sha)
        claim = row.get("generated_parsed", {}).get("claim")
        if not isinstance(claim, str) or not claim:
            raise ValueError("parsed_claim_missing")
        claim_sha = hashlib.sha256(claim.encode("utf-8")).hexdigest()
        claim_id = "claim-" + hashlib.sha256(
            (media_sha + "\0" + qid + "\0" + claim_sha).encode("utf-8")
        ).hexdigest()
        if claim_id in seen_claim:
            raise ValueError("claim_id_duplicate")
        seen_claim.add(claim_id)
        lineage_rows.append({
            "video_id": "video-" + media_sha,
            "claim_id": claim_id,
            "dataset_question_id": qid,
            "dataset_video_id": vid,
            "video_key": video_key(relative),
            "relative_video_path": relative,
            "video_sha256": media_sha,
            "question_sha256": hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            "claim_text_sha256": claim_sha,
        })
    manifest = {
        "schema_version": 1,
        "purpose": "private_source_lineage_crosscheck_for_claim_review",
        "created_by": "audit_nextgqa_claim_packet_lineage.py",
        "source_packet_sha256": hashlib.sha256(packet_raw).hexdigest(),
        "cohort_sha256": packet.get("cohort_sha256"),
        "dataset_inputs_sha256": hashes,
        "requested_records": 3,
        "effective_records": len(lineage_rows),
        "verification_scope": "QID/video-key/question-hash/media-bytes cross-check; not semantic review, rights, or execution authorization",
        "status": "metadata_lineage_crosschecked",
        "records": lineage_rows,
    }
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n").encode("utf-8")
    return manifest, raw


def validate_for_worksheet(worksheet: dict, manifest: dict, manifest_raw: bytes) -> str:
    """Validate exact lineage-manifest bytes and bind its three rows to a draft worksheet."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    try:
        if len(manifest_raw) > 1_000_000:
            raise ValueError("lineage_manifest_too_large")
        parsed = json.loads(manifest_raw.decode("utf-8"), object_pairs_hook=unique,
                            parse_constant=lambda _: (_ for _ in ()).throw(
                                ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, ValueError):
        raise ValueError("lineage_manifest_invalid") from None
    if (not isinstance(manifest, dict) or parsed != manifest or set(manifest) != LINEAGE_KEYS
            or manifest.get("schema_version") != 1
            or manifest.get("purpose") != "private_source_lineage_crosscheck_for_claim_review"
            or manifest.get("status") != "metadata_lineage_crosschecked"
            or manifest.get("requested_records") != 3 or manifest.get("effective_records") != 3
            or manifest.get("source_packet_sha256") != worksheet.get("source_packet_sha256")
            or manifest.get("cohort_sha256") != worksheet.get("cohort_sha256")
            or not SHA256.fullmatch(str(manifest.get("source_packet_sha256", "")))
            or not SHA256.fullmatch(str(manifest.get("cohort_sha256", "")))
            or manifest.get("dataset_inputs_sha256") != worksheet.get("dataset_inputs_sha256")
            or not isinstance(manifest.get("records"), list) or len(manifest["records"]) != 3):
        raise ValueError("lineage_manifest_binding_invalid")
    for candidate, lineage in zip(worksheet["records"], manifest["records"], strict=True):
        if (not isinstance(candidate, dict) or not isinstance(lineage, dict)
                or set(lineage) != LINEAGE_ROW_KEYS
                or lineage.get("video_id") != candidate.get("video_id")
                or lineage.get("claim_id") != candidate.get("claim_id")
                or lineage.get("video_sha256") != candidate.get("video_sha256")
                or lineage.get("claim_text_sha256") != candidate.get("claim_text_sha256")
                or lineage.get("question_sha256") != hashlib.sha256(
                    candidate["question"].encode("utf-8")).hexdigest()
                or not isinstance(lineage.get("dataset_question_id"), str)
                or not lineage["dataset_question_id"].strip()
                or not isinstance(lineage.get("dataset_video_id"), str)
                or not lineage["dataset_video_id"].strip()
                or not isinstance(lineage.get("relative_video_path"), str)
                or "\\" in lineage["relative_video_path"]
                or PurePosixPath(lineage["relative_video_path"]).is_absolute()
                or ".." in PurePosixPath(lineage["relative_video_path"]).parts
                or not all(SHA256.fullmatch(str(lineage.get(k, ""))) for k in (
                    "video_sha256", "question_sha256", "claim_text_sha256"))
                or lineage.get("video_key") != video_key(lineage["relative_video_path"])):
            raise ValueError("lineage_row_binding_invalid")
        expected_claim_id = "claim-" + hashlib.sha256(
            (lineage["video_sha256"] + "\0" + lineage["dataset_question_id"] + "\0"
             + lineage["claim_text_sha256"]).encode("utf-8")
        ).hexdigest()
        if lineage["claim_id"] != expected_claim_id:
            raise ValueError("lineage_claim_identity_invalid")
    return hashlib.sha256(manifest_raw).hexdigest()


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    packet_path = Path(args.packet).resolve()
    try:
        output = validate_private_output(args.output, args.repo)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "private_output_path_invalid")
    try:
        if (packet_path.stat().st_mode & 0o777) != 0o600:
            parser.error("private_input_mode_invalid")
        value, raw = _read_json(packet_path)
        manifest, encoded = audit(value, raw, args.repo.resolve(), args.video_root.resolve())
        write_exclusive(output, encoded)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except OSError:
        parser.error("local_input_or_output_error")
    except (ValueError, TypeError, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "lineage_check_failed")
    print(f"Private metadata lineage cross-check: {len(manifest['records'])}/3; status={manifest['status']}; manifest_sha256={hashlib.sha256(encoded).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
