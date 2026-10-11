"""Build a fresh NExT-GQA sample set, disjoint (by vidorID) from any sample
files already in this repo, for further layer=6/head=16 confirmation runs.

Motivation: the held-out significance test (see
experiments/2026-09-30_layer6_head16_heldout_significance.md) used the
already-existing `data/nextgqa_samples_big.json` (61) and
`data/nextgqa_samples_big_heldout.json` (25, n=22 actually processed) files.
Reusing the *screening* set (`..._big.json`) in a significance test would be
circular, since layer=6 was itself picked out on that data -- to honestly
strengthen the finding we need MORE samples that were never touched during
selection, i.e. a bigger, genuinely fresh held-out batch.

Also retries until `--n` samples are confirmed to have decodable video: the
2026-09-30 held-out run asked for 25 and only got 22 (3 videos failed to
open with OpenCV), so naively sampling exactly `--n` candidates tends to
under-deliver.

Requires the extracted NExT-GQA annotation files (test.csv, gsub_test.json,
map_vid_vidorID.json) and either an already-extracted video tree
(--video-root) or the original NExTVideo.zip (--zip-path) to pull individual
videos from on demand (this repo's convention throughout has been to extract
only the specific videos needed, not the full 24GB archive).

Per the project's standing rule (small-sample smoke test before a big job,
see CLAUDE.md/global policy), run this with --n 3 first and confirm it
produces 3 valid, readable videos before scaling up.

Usage:
    python scripts/build_samples.py \
        --test-csv datasets/nextgqa/test.csv \
        --gsub-json datasets/nextgqa/gsub_test.json \
        --map-json datasets/nextgqa/map_vid_vidorID.json \
        --video-root ~/nextgqa_sample/NExTVideo \
        --zip-path ~/nextgqa_sample/NExTVideo.zip \
        --n 3 --seed 1 \
        --exclude data/nextgqa_samples_big.json data/nextgqa_samples_big_heldout.json \
        --out data/nextgqa_samples_confirm2_smoketest.json
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import tempfile
import zipfile

import cv2


def normalize_answer(text):
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def row_choices(row):
    columns = [f"a{i}" for i in range(5)]
    present = [column in row and row[column].strip() for column in columns]
    if not any(present):
        return None, None, None
    if not all(present):
        raise RuntimeError(f"Incomplete NExT-GQA choice columns; expected {columns}")
    choices = [row[column].strip() for column in columns]
    matches = [i for i, choice in enumerate(choices)
               if normalize_answer(choice) == normalize_answer(row["answer"])]
    # NExT-GQA stores the correct answer as text rather than its original
    # answer-choice ID. If identical option strings occur more than once (or
    # the answer text matches none), the gold index cannot be recovered safely.
    return choices, matches[0] if len(matches) == 1 else None, len(matches)


def load_excluded_vidor_ids(exclude_files):
    excluded = set()
    for path in exclude_files:
        with open(path, encoding="utf-8") as stream:
            samples = json.load(stream)
        for s in samples:
            video = os.path.expanduser(s["video"])
            # "<video_root>/<vidorID>.mp4" -> vidorID (e.g. "1015/4631119360")
            base = os.path.splitext(video)[0]
            parts = base.split(os.sep)
            excluded.add("/".join(parts[-2:]))
    return excluded


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def validate_grounding(raw, duration):
    """Validate every interval; reject the whole annotation on any defect."""
    duration_value = _finite_number(duration)
    if duration_value is None or duration_value <= 0:
        return None, "invalid_video_duration"
    if not isinstance(raw, list):
        return None, "invalid_interval_container"
    if not raw:
        return None, "empty_interval_list"

    intervals = []
    for interval in raw:
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            return None, "malformed_interval_pair"
        start_raw, end_raw = interval
        start, end = _finite_number(start_raw), _finite_number(end_raw)
        if start is None or end is None:
            values = (start_raw, end_raw)
            if any(isinstance(value, float) and not math.isfinite(value)
                   for value in values):
                return None, "nonfinite_interval_endpoint"
            return None, "nonnumeric_interval_endpoint"
        if start < 0 or end < 0:
            return None, "negative_interval_endpoint"
        if end == start:
            return None, "zero_duration_interval"
        if end < start:
            return None, "reversed_interval"
        if end > duration_value:
            return None, "interval_past_video_duration"
        intervals.append((start, end))
    return intervals, None


def load_candidates(test_csv_path, gsub_json_path, map_json_path):
    with open(gsub_json_path, encoding="utf-8") as stream:
        gsub = json.load(stream)
    with open(map_json_path, encoding="utf-8") as stream:
        vid_to_vidor = json.load(stream)
    if not isinstance(gsub, dict) or not isinstance(vid_to_vidor, dict):
        raise ValueError("gsub and video map inputs must be JSON objects")

    candidates = []
    exclusion_counts = {}

    def exclude(reason):
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    with open(test_csv_path) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        required = {"video_id", "question", "answer", "qid"}
        missing = required - set(fieldnames)
        if missing:
            raise RuntimeError(
                f"test.csv is missing expected column(s) {missing}. "
                f"Actual columns found: {fieldnames}. "
                f"Share this with the assistant to fix the column names."
            )
        for row in reader:
            video_id = row["video_id"]
            qid = row["qid"]
            vidor_id = vid_to_vidor.get(video_id)
            if not vidor_id:
                exclude("missing_video_map")
                continue
            video_data = gsub.get(video_id)
            if not isinstance(video_data, dict):
                exclude("missing_gsub_video")
                continue
            location = video_data.get("location")
            if not isinstance(location, dict):
                exclude("missing_gsub_location")
                continue
            raw_grounding = location.get(str(qid))
            if raw_grounding is None:
                raw_grounding = location.get(qid)
            if raw_grounding is None:
                exclude("missing_gsub_annotation")
                continue
            grounding, grounding_error = validate_grounding(
                raw_grounding, video_data.get("duration")
            )
            if grounding_error:
                exclude(grounding_error)
                continue
            choices, answer_choice_index, answer_match_count = row_choices(row)
            if choices is None:
                exclude("missing_choices")
                continue
            if answer_choice_index is None:
                key = "ambiguous_answer" if answer_match_count > 1 else "unmatched_answer"
                exclude(key)
                continue
            candidates.append({
                "vidor_id": vidor_id,
                "video_id": video_id,
                # Keep the source-declared boundary distinct from measured
                # media duration; validate_grounding already checked it.
                "annotation_duration": _finite_number(video_data.get("duration")),
                "qid": qid,
                "question_type": row.get("type"),
                "question": row["question"],
                "answer": row["answer"],
                "choices": choices,
                "answer_choice_index": answer_choice_index,
                "grounding": grounding,
            })
    return candidates, exclusion_counts


def build_sample_record(candidate, video_root):
    """Serialize candidate fields while preserving annotation-declared time."""
    return {
        "video": f"{str(video_root).rstrip('/')}/{candidate['vidor_id']}.mp4",
        "video_id": candidate["video_id"],
        "annotation_duration": candidate["annotation_duration"],
        "qid": candidate["qid"],
        "question_type": candidate["question_type"],
        "question": candidate["question"],
        "answer": candidate["answer"],
        "choices": candidate["choices"],
        "answer_choice_index": candidate["answer_choice_index"],
        "grounding": candidate["grounding"],
    }


def ensure_video_available(vidor_id, video_root, zip_index, zip_path, tmp_extract_root):
    target = os.path.join(video_root, f"{vidor_id}.mp4")
    if os.path.exists(target):
        return target
    if zip_index is None:
        return None
    basename = os.path.basename(vidor_id) + ".mp4"
    member = zip_index.get(basename)
    if member is None:
        return None
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, path=tmp_extract_root)
    extracted_path = os.path.join(tmp_extract_root, member)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.move(extracted_path, target)
    return target


def is_video_readable(path):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--gsub-json", required=True)
    parser.add_argument("--map-json", required=True)
    parser.add_argument("--video-root", required=True, help="Directory videos live in (or get extracted into)")
    parser.add_argument("--zip-path", default=None, help="NExTVideo.zip, used to extract missing videos on demand")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--exclude", nargs="*", default=[],
                         help="Existing sample JSON files whose videos must NOT be reused")
    parser.add_argument("--unique-videos", action="store_true", default=True,
                         help="Don't reuse the same video twice within this new batch (default on)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if os.path.exists(args.out):
        parser.error(f"refusing to overwrite existing output: {os.path.basename(args.out)}")

    random.seed(args.seed)
    video_root = os.path.expanduser(args.video_root)
    excluded = load_excluded_vidor_ids(args.exclude)
    print(f"excluded previously used video count={len(excluded)}")

    candidates, exclusion_counts = load_candidates(args.test_csv, args.gsub_json, args.map_json)
    preexcluded = [c for c in candidates if c["vidor_id"] in excluded]
    exclusion_counts["previously_used_video"] = len(preexcluded)
    candidates = [c for c in candidates if c["vidor_id"] not in excluded]
    random.shuffle(candidates)
    print(f"eligible candidates after exclusions={len(candidates)}, seed={args.seed}")

    zip_index = None
    if args.zip_path:
        requested_zip_path = os.path.expanduser(args.zip_path)
        zip_path = requested_zip_path
        if not os.path.isfile(zip_path):
            home_archive = os.path.join(os.path.expanduser('~'),
                                        os.path.basename(requested_zip_path))
            if os.path.isfile(home_archive):
                print(f"NOTICE: requested archive missing; using same-named home archive: {home_archive}")
                zip_path = home_archive
            else:
                parser.error(
                    f"NExTVideo archive not found: {requested_zip_path}; "
                    f"also checked {home_archive}. Correct --zip-path or omit it "
                    "if all required videos are already under --video-root."
                )
        if not zipfile.is_zipfile(zip_path):
            parser.error(f"Archive is not a readable ZIP file: {zip_path}")
        print(f"indexing {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zip_index = {os.path.basename(n): n for n in zf.namelist() if n.endswith(".mp4")}
        print(f"  {len(zip_index)} mp4 entries found in archive")
    else:
        zip_path = None

    tmp_extract_root = tempfile.mkdtemp(prefix="build_samples_extract_")
    used_vidor_ids = set()
    results = []
    built_exclusions = {}

    def exclude_build_reason(reason):
        built_exclusions[reason] = built_exclusions.get(reason, 0) + 1

    try:
        for c in candidates:
            if len(results) >= args.n:
                break
            if args.unique_videos and c["vidor_id"] in used_vidor_ids:
                exclude_build_reason("duplicate_video_skipped")
                continue
            path = ensure_video_available(c["vidor_id"], video_root, zip_index, zip_path, tmp_extract_root)
            if path is None:
                exclude_build_reason("video_unavailable")
                continue
            if not is_video_readable(path):
                exclude_build_reason("video_unreadable")
                continue
            used_vidor_ids.add(c["vidor_id"])
            results.append(build_sample_record(c, args.video_root))
    finally:
        shutil.rmtree(tmp_extract_root, ignore_errors=True)

    for reason, count in built_exclusions.items():
        exclusion_counts[reason] = exclusion_counts.get(reason, 0) + count
    print(f"candidate_exclusion_reason_counts={json.dumps(dict(sorted(exclusion_counts.items())), sort_keys=True)}")

    if len(results) < args.n:
        print(f"\nWARNING: only found {len(results)} of {args.n} requested valid samples "
              f"(ran out of eligible candidates or too many were unreadable)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "x", encoding="utf-8") as stream:
        json.dump(results, stream, ensure_ascii=False, indent=2)
    print(f"wrote sample_count={len(results)} output_name={os.path.basename(args.out)}")


if __name__ == "__main__":
    main()
