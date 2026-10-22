"""Derive a qid-valid exploratory plan without changing frozen v1 membership.

The only change from the v1 split is deterministic qid selection among rows
with valid gsub intervals. It remains entirely within the test-associated pool.
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

PROTOCOL = "nextgqa_test_associated_exploratory_video_split_v2_valid_qid"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def valid_spans(annotation: Any, qid: str) -> bool:
    if not isinstance(annotation, dict) or "duration" not in annotation:
        return False
    duration, loc = annotation["duration"], annotation.get("location")
    spans = loc.get(qid) if isinstance(loc, dict) else None
    if (not isinstance(duration, (int, float)) or isinstance(duration, bool)
            or not math.isfinite(float(duration)) or duration <= 0
            or not isinstance(spans, list) or not spans):
        return False
    for span in spans:
        if (not isinstance(span, list) or len(span) != 2
                or any(not isinstance(x, (int, float)) or isinstance(x, bool)
                       or not math.isfinite(float(x)) for x in span)):
            return False
        start, end = map(float, span)
        if start < 0 or end <= start or end > float(duration):
            return False
    return True


def derive_v2(*, base_plan: Path, test_csv: Path, gsub_json: Path,
              frame2time_json: Path, mapping_json: Path, v1_splitter: Path,
              v2_source: Path,
              expected_v1_plan_sha256: str,
              v1_freeze_receipt_sha256: str,
              expected_partition_sizes: tuple[int, int] | None = (343, 147)) -> dict[str, Any]:
    plan = json.loads(base_plan.read_text(encoding="utf-8"))
    data = {"gsub": json.loads(gsub_json.read_text(encoding="utf-8")),
            "frame2time": json.loads(frame2time_json.read_text(encoding="utf-8")),
            "mapping": json.loads(mapping_json.read_text(encoding="utf-8"))}
    if plan.get("status") != "exploratory_split_plan_not_evaluation":
        raise ValueError("base_plan_status_invalid")
    if plan.get("provenance", {}).get("source_script_sha256") != sha256_file(v1_splitter):
        raise ValueError("v1_splitter_source_binding_mismatch")
    if (not isinstance(expected_v1_plan_sha256, str) or len(expected_v1_plan_sha256) != 64
            or sha256_file(base_plan) != expected_v1_plan_sha256):
        raise ValueError("v1_local_freeze_hash_mismatch")
    if not isinstance(v1_freeze_receipt_sha256, str) or len(v1_freeze_receipt_sha256) != 64:
        raise ValueError("v1_local_freeze_receipt_hash_invalid")
    if plan.get("provenance", {}).get("test_csv_sha256") != sha256_file(test_csv):
        raise ValueError("test_csv_changed_since_v1_freeze")
    rows_by_video: dict[str, list[dict[str, str]]] = {}
    with test_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"video_id", "qid"}.issubset(reader.fieldnames or []):
            raise ValueError("test_csv_schema_invalid")
        for row in reader:
            vid, qid = row.get("video_id"), row.get("qid")
            if not vid or not qid:
                raise ValueError("test_csv_identifier_missing")
            rows_by_video.setdefault(vid, []).append(row)
    if not all(isinstance(x, dict) for x in data.values()):
        raise ValueError("metadata_root_invalid")
    valid_pairs_in_frozen_pool = 0
    videos_with_valid_qid_in_frozen_pool = 0
    for vid in {r["video_id"] for part in ("calibration", "evaluation") for r in plan[part]}:
        candidates = [r for r in rows_by_video.get(vid, [])
                      if valid_spans(data["gsub"].get(vid), r["qid"])]
        valid_pairs_in_frozen_pool += len(candidates)
        videos_with_valid_qid_in_frozen_pool += bool(candidates)

    def convert(part: str) -> list[dict[str, Any]]:
        output = []
        for old in plan[part]:
            vid = old["video_id"]
            candidates = [r for r in rows_by_video.get(vid, [])
                          if valid_spans(data["gsub"].get(vid), r["qid"])]
            if not candidates:
                raise ValueError("video_has_no_valid_test_qid")
            chosen = min(candidates, key=lambda r: hashlib.sha256(
                f"{PROTOCOL}\0qid\0{vid}\0{r['qid']}".encode()).hexdigest())
            if vid not in data["frame2time"] or vid not in data["mapping"]:
                raise ValueError("metadata_join_missing")
            times = data["frame2time"][vid]
            if (not isinstance(times, list) or len(times) != 32
                    or any(not isinstance(t, (int, float)) or isinstance(t, bool)
                           or not math.isfinite(float(t)) for t in times)):
                raise ValueError("frame2time_record_invalid")
            media = Path(old["media_path"])
            map_target = data["mapping"][vid]
            if (not isinstance(map_target, str) or Path(map_target).stem != vid
                    or media.stem != vid):
                raise ValueError("video_map_media_identity_mismatch")
            if not media.is_file() or sha256_file(media) != old["media_sha256"]:
                raise ValueError("media_hash_mismatch")
            binding = canonical_sha256({"csv_row": chosen,
                "gsub": {"duration": data["gsub"][vid]["duration"],
                         "spans": data["gsub"][vid]["location"][chosen["qid"]]},
                "frame2time": times, "map_target": map_target,
                "media_sha256": old["media_sha256"]})
            output.append({**old, "qid": chosen["qid"],
                           "valid_test_qid_count": len(candidates),
                           "selected_row_binding_sha256": binding})
        return output

    calibration, evaluation = convert("calibration"), convert("evaluation")
    cal_ids, ev_ids = {r["video_id"] for r in calibration}, {r["video_id"] for r in evaluation}
    cal_sha, ev_sha = {r["media_sha256"] for r in calibration}, {r["media_sha256"] for r in evaluation}
    if ((expected_partition_sizes is not None
            and (len(calibration), len(evaluation)) != expected_partition_sizes)
            or len(cal_ids) != len(calibration) or len(ev_ids) != len(evaluation)
            or len(cal_sha) != len(calibration) or len(ev_sha) != len(evaluation)
            or cal_ids & ev_ids or cal_sha & ev_sha):
        raise ValueError("split_overlap_detected")
    selected_pairs = [(r["video_id"], r["qid"]) for r in calibration + evaluation]
    if len(set(selected_pairs)) != len(selected_pairs):
        raise ValueError("selected_video_qid_pair_duplicate")
    return {
        "schema_version": 2, "protocol": PROTOCOL,
        "status": "test_associated_exploratory_qid_valid_plan_not_evaluation",
        "cohort": "NExT-GQA official test-associated local media; not confirmatory",
        "selection": "preserve v1 video/media partition; among each video's gsub-valid test qids choose minimum SHA256(protocol||NUL||qid||NUL||video_id||NUL||qid)",
        "provenance": {"v1_plan_sha256": sha256_file(base_plan),
            "v1_freeze_receipt_sha256": v1_freeze_receipt_sha256,
            "test_csv_sha256": sha256_file(test_csv), "gsub_test_sha256": sha256_file(gsub_json),
            "frame2time_test_sha256": sha256_file(frame2time_json),
            "map_vid_vidorID_sha256": sha256_file(mapping_json),
            "v1_splitter_sha256": sha256_file(v1_splitter),
            "v2_builder_sha256": sha256_file(v2_source)},
        "counts": {"calibration_videos": len(calibration), "evaluation_videos": len(evaluation),
                   "calibration_videos_with_valid_qid": len(calibration),
                   "evaluation_videos_with_valid_qid": len(evaluation),
                   "frozen_pool_videos_with_valid_qid": videos_with_valid_qid_in_frozen_pool,
                   "frozen_pool_valid_qid_pairs": valid_pairs_in_frozen_pool},
        "calibration": calibration, "evaluation": evaluation,
        "limitations": ["All records remain associated with the official test split.",
            "Not a clean holdout, fresh cohort, confirmatory test, or faithfulness evidence.",
            "Exact media hashes do not detect re-encodes or near-duplicate scenes.",
            "Private plan contains identifiers, paths, and media hashes; keep local."]}


def write_exclusive(path: Path, value: dict[str, Any], private_root: Path) -> None:
    root = private_root.resolve(strict=True)
    if path.parent.resolve(strict=True) != root or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("private_root_or_output_path_invalid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--output-name", required=True)
    a = p.parse_args(argv)
    if Path(a.output_name).name != a.output_name:
        p.error("output_name_must_be_basename")
    repo = a.repo.resolve(); private = repo / "data/private_nextgqa"
    ds = repo / "NExT-GQA/datasets/nextgqa"
    try:
        from freeze_nextgqa_exploratory_v1_local import read_and_verify_freeze
        frozen = read_and_verify_freeze(repo)
        freeze_path = private/"nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"
        out = derive_v2(base_plan=private/"nextgqa_tlvs_exploratory_split_20261018_01.json",
            test_csv=ds/"test.csv", gsub_json=ds/"gsub_test.json",
            frame2time_json=ds/"frame2time_test.json", mapping_json=ds/"map_vid_vidorID.json",
            v1_splitter=repo/"scripts/build_nextgqa_exploratory_video_split.py",
            v2_source=Path(__file__).resolve(),
            expected_v1_plan_sha256=frozen["plan_sha256"],
            v1_freeze_receipt_sha256=sha256_file(freeze_path))
        write_exclusive(private/a.output_name, out, private)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        p.error(f"v2_plan_failed:{type(exc).__name__}:{exc}")
    print(json.dumps({"status": out["status"], "counts": out["counts"],
        "input_hashes": out["provenance"], "privacy": "counts/hash only; private local plan"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
