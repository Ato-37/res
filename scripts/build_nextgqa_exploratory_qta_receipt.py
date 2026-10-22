"""Build a private, CPU-only provenance receipt for an exploratory QTA cohort.

This binds selected evaluation rows to the local NExT-GQA test-associated
metadata and media bytes. It does not score a model and is not faithfulness
evidence, a clean holdout, or confirmatory evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

PROTOCOL = "nextgqa_test_associated_exploratory_qta_receipt_v1"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_key(row: dict[str, str]) -> tuple[str, str]:
    vid, qid = row.get("video_id"), row.get("qid")
    if not vid or not qid:
        raise ValueError("test_row_identifier_missing")
    return str(vid), str(qid)


def build_receipt(*, test_csv: Path, gsub_json: Path, frame2time_json: Path,
                  mapping_json: Path, split_plan: Path,
                  splitter_source: Path, base_plan: Path, v1_splitter: Path,
                  expected_v1_plan_sha256: str,
                  v1_freeze_receipt_sha256: str,
                  expected_partition_sizes: tuple[int, int] | None = (343, 147)) -> dict[str, Any]:
    """Validate joins and construct a privacy-minimal private receipt."""
    paths = {"test_csv": test_csv, "gsub_test_json": gsub_json,
             "frame2time_test_json": frame2time_json,
             "map_vid_vidorID_json": mapping_json,
             "split_plan": split_plan, "splitter_source": splitter_source,
             "v1_plan": base_plan, "v1_splitter_source": v1_splitter}
    if any(not p.is_file() for p in paths.values()):
        raise ValueError("required_input_missing")
    with test_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"video_id", "qid"}.issubset(reader.fieldnames or []):
            raise ValueError("test_csv_schema_invalid")
        csv_rows = list(reader)
    rows_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in csv_rows:
        key = _row_key(row)
        if key in rows_by_key:
            raise ValueError("duplicate_test_video_qid")
        rows_by_key[key] = row

    gsub, frame2time, mapping, plan = map(_load_json,
        (gsub_json, frame2time_json, mapping_json, split_plan))
    if not all(isinstance(x, dict) for x in (gsub, frame2time, mapping, plan)):
        raise ValueError("metadata_root_invalid")
    if plan.get("status") != "test_associated_exploratory_qid_valid_plan_not_evaluation":
        raise ValueError("split_plan_status_invalid")
    if plan.get("cohort") != "NExT-GQA official test-associated local media; not confirmatory":
        raise ValueError("split_plan_cohort_invalid")
    provenance = plan.get("provenance")
    if (not isinstance(provenance, dict)
            or provenance.get("v2_builder_sha256") != sha256_file(splitter_source)):
        raise ValueError("splitter_source_binding_mismatch")
    if sha256_file(base_plan) != expected_v1_plan_sha256:
        raise ValueError("v1_local_freeze_hash_mismatch")
    if provenance.get("v1_freeze_receipt_sha256") != v1_freeze_receipt_sha256:
        raise ValueError("v1_freeze_receipt_hash_mismatch")
    for field, path in (("test_csv_sha256", test_csv), ("gsub_test_sha256", gsub_json),
                         ("frame2time_test_sha256", frame2time_json),
                         ("map_vid_vidorID_sha256", mapping_json)):
        if provenance.get(field) != sha256_file(path):
            raise ValueError("v2_plan_input_hash_mismatch")
    from build_nextgqa_exploratory_video_split_v2 import derive_v2
    reconstructed = derive_v2(base_plan=base_plan, test_csv=test_csv, gsub_json=gsub_json,
        frame2time_json=frame2time_json, mapping_json=mapping_json,
        v1_splitter=v1_splitter, v2_source=splitter_source,
        expected_v1_plan_sha256=expected_v1_plan_sha256,
        v1_freeze_receipt_sha256=v1_freeze_receipt_sha256,
        expected_partition_sizes=expected_partition_sizes)
    if reconstructed != plan:
        raise ValueError("v2_plan_reconstruction_mismatch")
    calibration, evaluation = plan.get("calibration"), plan.get("evaluation")
    if not isinstance(calibration, list) or not isinstance(evaluation, list) or not evaluation:
        raise ValueError("split_plan_rows_invalid")

    def keys(rows: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
        result = []
        for item in rows:
            if not isinstance(item, dict):
                raise ValueError("split_entry_invalid")
            vid, qid, media_hash = item.get("video_id"), item.get("qid"), item.get("media_sha256")
            if not isinstance(vid, str) or not vid or not isinstance(qid, str) or not qid:
                raise ValueError("split_entry_identity_invalid")
            if not isinstance(media_hash, str) or len(media_hash) != 64:
                raise ValueError("split_entry_media_hash_invalid")
            result.append((vid, qid, media_hash))
        return result

    cal, ev = keys(calibration), keys(evaluation)
    cal_ids, eval_ids = {x[0] for x in cal}, {x[0] for x in ev}
    cal_hashes, eval_hashes = {x[2] for x in cal}, {x[2] for x in ev}
    selected_pairs = [(x[0], x[1]) for x in cal + ev]
    if (expected_partition_sizes is not None
            and (len(cal), len(ev)) != expected_partition_sizes):
        raise ValueError("split_plan_denominator_invalid")
    if (len(cal_ids) != len(cal) or len(eval_ids) != len(ev)
            or len(cal_hashes) != len(cal) or len(eval_hashes) != len(ev)
            or len(set(selected_pairs)) != len(selected_pairs)
            or cal_ids & eval_ids or cal_hashes & eval_hashes):
        raise ValueError("split_overlap_detected")
    entries = []
    valid_intervals = invalid_intervals = 0
    seen_eval = set()
    for item, (vid, qid, expected_media_sha) in zip(evaluation, ev, strict=True):
        key = (vid, qid)
        if key in seen_eval:
            raise ValueError("duplicate_evaluation_row")
        seen_eval.add(key)
        row = rows_by_key.get(key)
        if row is None:
            raise ValueError("evaluation_test_csv_join_missing")
        annotation = gsub.get(vid)
        if not isinstance(annotation, dict) or "duration" not in annotation or "location" not in annotation:
            raise ValueError("gsub_video_join_missing")
        location = annotation["location"]
        if not isinstance(location, dict) or qid not in location:
            raise ValueError("gsub_qid_spans_join_missing")
        spans = location[qid]
        duration = annotation["duration"]
        span_ok = (isinstance(spans, list) and bool(spans)
                   and isinstance(duration, (int, float)) and not isinstance(duration, bool)
                   and math.isfinite(float(duration)) and float(duration) > 0)
        if span_ok:
            for span in spans:
                if (not isinstance(span, list) or len(span) != 2
                        or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                               or not math.isfinite(float(x)) for x in span)):
                    span_ok = False
                    break
                start, end = map(float, span)
                if start < 0 or end <= start or end > float(duration):
                    span_ok = False
                    break
        if span_ok:
            valid_intervals += 1
        else:
            invalid_intervals += 1
        if (vid not in frame2time or not isinstance(frame2time[vid], list)
                or len(frame2time[vid]) != 32
                or any(not isinstance(value, (int, float)) or isinstance(value, bool)
                       or not math.isfinite(float(value)) for value in frame2time[vid])):
            raise ValueError("frame2time_video_join_missing")
        if vid not in mapping or not isinstance(mapping[vid], str) or not mapping[vid]:
            raise ValueError("video_mapping_join_missing")
        media_path_value = item.get("media_path")
        if not isinstance(media_path_value, str):
            raise ValueError("evaluation_media_path_missing")
        media_path = Path(media_path_value)
        if Path(mapping[vid]).stem != vid or media_path.stem != vid:
            raise ValueError("video_map_media_identity_mismatch")
        if not media_path.is_file() or sha256_file(media_path) != expected_media_sha:
            raise ValueError("evaluation_media_hash_mismatch")
        canonical_row = {
            "csv_row": row,
            "gsub": {"duration": duration, "spans": spans},
            "frame2time": frame2time[vid],
            "map_target": mapping[vid],
            "media_sha256": expected_media_sha,
        }
        row_digest = canonical_sha256(canonical_row)
        if item.get("selected_row_binding_sha256") != row_digest:
            raise ValueError("v2_selected_row_binding_mismatch")
        entries.append({"row_binding_sha256": row_digest})

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": ("preflight_receipt_only_no_model_scoring" if invalid_intervals == 0
                   else "no_go_invalid_annotation_interval_no_model_scoring"),
        "scope": "test-associated exploratory proxy only; not clean, fresh, confirmatory, or faithfulness evidence",
        "counts": {"test_csv_rows": len(csv_rows), "calibration_rows": len(cal),
                   "evaluation_rows": len(ev), "evaluation_rows_bound": len(entries),
                   "join_failures": 0, "valid_annotation_intervals": valid_intervals,
                   "invalid_annotation_intervals": invalid_intervals},
        "input_sha256": {**{name: sha256_file(path) for name, path in paths.items()},
                         "v1_freeze_receipt": v1_freeze_receipt_sha256},
        "evaluation_row_bindings": entries,
        "privacy": "no raw identifiers, paths, QA text, spans, mapping targets, or media bytes in receipt",
        "rights_status": "unverified; local-only protected data",
        "faithfulness_status": "not_measured",
    }


def write_private_exclusive(path: Path, payload: dict[str, Any], *, private_root: Path) -> None:
    root = private_root.resolve(strict=True)
    if not path.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("receipt_output_outside_private_root")
    root_stat = root.stat()
    if not stat.S_ISDIR(root_stat.st_mode) or stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise ValueError("private_root_permissions_invalid")
    if path.parent.resolve() != root:
        raise ValueError("receipt_must_be_direct_child_of_private_root")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
            json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def public_summary(receipt: dict[str, Any], receipt_sha256: str) -> dict[str, Any]:
    """Allow only aggregate counts and byte fingerprints in stdout."""
    return {"status": receipt["status"],
            "evaluation_rows": receipt["counts"]["evaluation_rows_bound"],
            "input_hashes": receipt["input_sha256"],
            "receipt_sha256": receipt_sha256,
            "privacy": "counts and hashes only; private receipt"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-name", required=True)
    a = p.parse_args(argv)
    repo = a.repo.resolve()
    data = repo / "NExT-GQA/datasets/nextgqa"
    private = repo / "data/private_nextgqa"
    output = private / a.output_name
    if Path(a.output_name).name != a.output_name:
        p.error("output_name_must_be_basename")
    try:
        from freeze_nextgqa_exploratory_v1_local import read_and_verify_freeze
        frozen = read_and_verify_freeze(repo)
        freeze_path = private/"nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"
        receipt = build_receipt(test_csv=data/"test.csv", gsub_json=data/"gsub_test.json",
            frame2time_json=data/"frame2time_test.json", mapping_json=data/"map_vid_vidorID.json",
            split_plan=private/"nextgqa_tlvs_exploratory_split_v2_20261018_05.json",
            splitter_source=repo/"scripts/build_nextgqa_exploratory_video_split_v2.py",
            base_plan=private/"nextgqa_tlvs_exploratory_split_20261018_01.json",
            v1_splitter=repo/"scripts/build_nextgqa_exploratory_video_split.py",
            expected_v1_plan_sha256=frozen["plan_sha256"],
            v1_freeze_receipt_sha256=sha256_file(freeze_path))
        write_private_exclusive(output, receipt, private_root=private)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(f"preflight_failed:{type(exc).__name__}:{str(exc)}")
    print(json.dumps(public_summary(receipt, sha256_file(output)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
