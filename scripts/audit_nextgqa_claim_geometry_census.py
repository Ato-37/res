#!/usr/bin/env python3
"""Run an aggregate-only geometry census for local NExT-GQA videos.

This tests gold-interval sampling/control geometry before claim review. It does
not score faithfulness and writes only a private aggregate with input/code
hashes; no row IDs, questions, answers, paths, or media identifiers are saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import run_nextgqa_gold_interval_margin_pilot as pilot
from private_output_paths import validate_private_output


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _input_hashes(repo: Path) -> dict[str, str]:
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    paths = [dataset / "test.csv", dataset / "gsub_test.json",
             dataset / "map_vid_vidorID.json"]
    paths.extend(sorted((repo / "data").glob("nextgqa_samples_*.json")))
    return {str(path.relative_to(repo)): _sha(path.read_bytes()) for path in paths}


def build_summary(repo: Path, video_root: Path, exclusion_path: Path,
                  frame_count: int = 16) -> dict:
    exclusion_raw = exclusion_path.read_bytes()
    exclusion = json.loads(exclusion_raw.decode("utf-8"))
    hashes = exclusion.get("media_sha256") if isinstance(exclusion, dict) else None
    if (not isinstance(exclusion, dict) or exclusion.get("schema_version") != 1
            or not isinstance(hashes, list)
            or any(not isinstance(x, str) or len(x) != 64
                   or any(c not in "0123456789abcdef" for c in x) for x in hashes)
            or len(hashes) != len(set(hashes))):
        raise ValueError("private_media_exclusion_manifest_invalid")
    if frame_count < 2:
        raise ValueError("frame_count_must_be_at_least_two")

    pilot.MAX_VIDEOS = 1_000_000
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    selected, counts, meta, cohort_sha = pilot.freeze_cohort(
        repo, video_root, hashes, frame_count)
    elapsed = time.monotonic() - started
    candidate_count = int(meta["candidate_video_count"])
    candidate_rows_examined = int(counts["candidate_scan_count"])
    excluded_before_decode = int(counts["prior_cohort_media_excluded"])
    decode_attempts = sum(int(counts[key]) for key in (
        "timing_pass_videos", "timing_rejected_videos", "decode_failures"))
    if (candidate_rows_examined != excluded_before_decode + decode_attempts
            or candidate_rows_examined > candidate_count):
        raise ValueError("candidate_scan_accounting_inconsistent")
    input_hashes = _input_hashes(repo)
    selector_path = Path(pilot.__file__).resolve()
    return {
        "schema_version": 1,
        "purpose": "outcome_blind_full_gold_interval_geometry_census_no_faithfulness_scoring",
        "started_utc": started_utc,
        "candidate_count": candidate_count,
        "candidate_rows_examined": candidate_rows_examined,
        "candidates_excluded_before_decode": excluded_before_decode,
        "decode_attempts": decode_attempts,
        "unexamined_candidate_rows": max(0, candidate_count - candidate_rows_examined),
        "eligible_geometry_cohort": len(selected),
        "timing_pass_videos": int(counts["timing_pass_videos"]),
        "timing_rejected_videos": int(counts["timing_rejected_videos"]),
        "decoder_failures": int(counts["decode_failures"]),
        "no_equal_count_gold_geometry_control": int(
            counts["videos_without_interval_control_pair"]),
        "prior_claim_media_excluded": int(counts["prior_cohort_media_excluded"]),
        "questions_tested_for_assignment": int(counts["questions_tested_for_assignment"]),
        "distinct_selected_media": len({x["decode"]["source_sha256"] for x in selected}),
        "cohort_sha256": cohort_sha,
        "dataset_inputs_sha256": input_hashes,
        "selector_sha256": _sha(selector_path.read_bytes()),
        "census_runner_sha256": _sha(Path(__file__).resolve().read_bytes()),
        "exclusion_manifest_sha256": _sha(exclusion_raw),
        "frame_count": frame_count,
        "uniform_tolerance_seconds": float(pilot.UNIFORM_TOLERANCE),
        "elapsed_seconds": round(elapsed, 3),
        "peak_rss_kib": int(Path("/proc/self/status").read_text().split("VmHWM:")[1].split()[0]),
        "faithfulness_scores": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-media-sha256", required=True,
                        help="private local schema-1 media-hash exclusion manifest")
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    parser.add_argument("--frames", type=int, choices=(16, 32), default=16)
    parser.add_argument("--output", required=True,
                        help="new mode-0600 aggregate under data/private_nextgqa")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit_local_only_authorization_required")
    repo = Path(__file__).resolve().parents[1]
    try:
        output = validate_private_output(args.output, repo)
        summary = build_summary(repo, Path(args.video_root).resolve(strict=True),
                                Path(args.exclude_media_sha256).resolve(strict=True), args.frames)
        pilot.write_private_json_exclusive(output, summary)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError, KeyError, TypeError, UnicodeError,
            json.JSONDecodeError, IndexError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "geometry_census_failed")
    safe = {key: value for key, value in summary.items()
            if key not in {"dataset_inputs_sha256"}}
    safe["dataset_input_count"] = len(summary["dataset_inputs_sha256"])
    safe["status"] = "completed_aggregate_only_no_faithfulness_scores"
    print(json.dumps(safe, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
