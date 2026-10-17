#!/usr/bin/env python3
"""Exploratory NExT-GQA gold-interval QA-margin pilot on local video.

This is a benchmark-interval dependence diagnostic, not claim-specific
explanation faithfulness. It selects one QA per previously unused test video
using a deterministic outcome-blind rule, requires exact pixel/PTS provenance
and a fixed Qwen uniform-grid timing preflight, then compares full,
gold-interval-mask, and equal-count non-evidence-control inputs using both
black and Gaussian-blur fills. Raw dataset text and media are never written to
the summary. Network downloads and publication are off.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
import gc
import glob
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import time

from PIL import Image

from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
from necessity_choice_margin_utils import apply_mask_fill, nearest_control_indices
from presentation_time_utils import strict_evidence_indices_from_times

FRAMES = 16
UNIFORM_TOLERANCE = Fraction(1, 20)
SEED = 20261015
LABELS = "ABCDE"
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
MAX_VIDEOS = 10
CONDITION_NAMES = ("full", "black_gold_interval_mask", "black_equal_count_control_mask",
                   "blur_gold_interval_mask", "blur_equal_count_control_mask")
BLUR_RADIUS = 12.0
SOURCE_PATHS = (
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
)


def validate_cohort_cardinality(requested: int, effective: int) -> None:
    """Fail closed when the deterministic selector cannot meet the frozen n."""
    if effective != requested:
        raise ValueError(f"requested_{requested}_eligible_samples_but_found_{effective}")


def validate_preflight_cardinality(gate: dict, requested: int, selected_count: int) -> None:
    validate_cohort_cardinality(requested, selected_count)
    if (gate.get("requested_videos") != requested
            or gate.get("effective_videos") != selected_count):
        raise ValueError("preflight_requested_or_effective_count_mismatch")


def helper_source_hashes(repo: Path) -> dict:
    """Bind imported timing/control behavior into the preflight contract."""
    return {
        name: sha256_file(repo / "scripts" / name)
        for name in ("necessity_choice_margin_utils.py", "presentation_time_utils.py")
    }


def verify_committed_source_snapshot(repo: Path) -> tuple[str, dict[str, str]]:
    """Require every executed project-code source byte to match the recorded HEAD."""
    repo = Path(repo).resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    hashes = {}
    for relative in SOURCE_PATHS:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"], cwd=repo)
        current = (repo / relative).read_bytes()
        committed_hash = hashlib.sha256(committed).hexdigest()
        current_hash = hashlib.sha256(current).hexdigest()
        if current_hash != committed_hash:
            raise ValueError(f"uncommitted_research_source:{relative}")
        hashes[relative] = current_hash
    return commit, hashes


def validate_cohort_lock_payload(payload: dict, requested: int) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("private_cohort_lock_invalid")
    assignments = payload.get("assignments")
    if (set(payload) != {"schema_version", "assignments"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
            or not isinstance(assignments, list)
            or not assignments
            or any(not isinstance(item, dict)
                   or set(item) != {"media_sha256", "qid_sha256"}
                   or any(not isinstance(item[key], str) or len(item[key]) != 64
                          or any(c not in "0123456789abcdef" for c in item[key])
                          for key in ("media_sha256", "qid_sha256"))
                   for item in assignments)
            or len({item["media_sha256"] for item in assignments}) != len(assignments)):
        raise ValueError("private_cohort_lock_invalid")
    if len(assignments) != requested:
        raise ValueError("private_cohort_lock_count_must_equal_requested_n")
    return assignments


def build_cohort_lock_assignments(selected: list[dict]) -> list[dict]:
    """Build the private hash-only lock for the exact selected video/QID pairs."""
    assignments = [{
        "media_sha256": item["decode"]["source_sha256"],
        "qid_sha256": hashlib.sha256(str(item["qid"]).encode("utf-8")).hexdigest(),
    } for item in selected]
    hashes = [value for item in assignments for value in item.values()]
    media_hashes = [item["media_sha256"] for item in assignments]
    if (any(not isinstance(value, str) or len(value) != 64
            or any(char not in "0123456789abcdef" for char in value) for value in hashes)
            or len(set(media_hashes)) != len(media_hashes)):
        raise ValueError("selected_cohort_lock_assignments_invalid")
    return assignments


def validate_locked_selection_assignments(selected: list[dict],
                                         locked_assignments: list[dict]) -> None:
    """Require the selected records to consume each locked pair exactly once."""
    selected_pairs = build_cohort_lock_assignments(selected)
    locked_pairs = [{"media_sha256": item["media_sha256"],
                     "qid_sha256": item["qid_sha256"]}
                    for item in locked_assignments]
    if sorted((x["media_sha256"], x["qid_sha256"]) for x in selected_pairs) != sorted(
            (x["media_sha256"], x["qid_sha256"]) for x in locked_pairs):
        raise ValueError("cohort_lock_selected_assignment_mismatch")


def finalize_cuda_scoring(torch_module, records: list[dict],
                          failures: list[dict]) -> tuple[bool, int]:
    """Synchronize before publishing scores; discard them if device completion is unknown."""
    try:
        torch_module.cuda.synchronize()
    except Exception as exc:
        discarded = len(records)
        records.clear()
        failures.append({"video_index": None,
                         "failure_type": type(exc).__name__,
                         "failed_condition": "final_cuda_synchronize",
                         "completed_conditions": None})
        return False, discarded
    return True, 0


def validate_candidate_pair_filter(payload: dict) -> list[str]:
    """Validate the private SHA256 allowlist without accepting raw identifiers."""
    values = payload.get("pair_sha256") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict)
            or set(payload) != {"schema_version", "pair_sha256"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1 or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("private_candidate_pair_filter_invalid")
    return values


def validate_private_media_exclusion(payload: dict) -> list[str]:
    """Validate the local media exclusion manifest without raw identities."""
    values = payload.get("media_sha256") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict)
            or set(payload) != {"schema_version", "media_sha256"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1 or not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in values)
            or len(set(values)) != len(values)):
        raise ValueError("private_media_exclusion_manifest_invalid")
    return values


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("private_json_duplicate_key")
        result[key] = value
    return result


def read_private_json_snapshot(path: Path):
    """Read and hash one verified private-file byte snapshot."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(Path(path), flags)
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_json_file_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_json_invalid") from exc
    return payload, hashlib.sha256(raw).hexdigest()


def validate_new_output_path(path: Path, other_path: Path | None = None) -> Path:
    path = Path(os.path.abspath(path))
    if other_path is not None and path == Path(os.path.abspath(other_path)):
        raise ValueError("output_paths_must_be_different")
    if os.path.lexists(path):
        raise ValueError("refusing_to_overwrite_output")
    return path


def candidate_pair_sha256(video_id: str, qid: str) -> str:
    """Hash compact UTF-8 JSON for [contract, video_id, qid]."""
    if not isinstance(video_id, str) or not video_id or not isinstance(qid, str) or not qid:
        raise ValueError("candidate_pair_identity_invalid")
    canonical = json.dumps(["nextgqa-gold-interval-pair-v1", video_id, qid],
                           ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_candidate_filter_binding(gate: dict, expected_sha256: str | None) -> None:
    if gate.get("candidate_pair_filter_sha256") != expected_sha256:
        raise ValueError("candidate_pair_filter_binding_mismatch")


def filter_candidate_groups(grouped: dict, allowed_pair_sha256):
    if allowed_pair_sha256 is None:
        return grouped
    allowed_pair_sha256 = set(allowed_pair_sha256)
    filtered = {
        vid: [question for question in questions
              if candidate_pair_sha256(vid, question["qid"]) in allowed_pair_sha256]
        for vid, questions in grouped.items()
    }
    return {vid: questions for vid, questions in filtered.items() if questions}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def video_key(value: str) -> str:
    parts = [part for part in str(value).replace("\\", "/").split("/") if part]
    if parts:
        parts[-1] = parts[-1].rsplit(".", 1)[0]
    return "/".join(parts[-2:])


def norm(value: str) -> str:
    import re
    import unicodedata
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip().casefold()


def host_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            return int(fields[1]) * 1024
    raise RuntimeError("host_memavailable_missing")


def load_candidate_rows(repo: Path, video_root: Path, allowed_pair_sha256=None):
    dataset = repo / "NExT-GQA" / "datasets" / "nextgqa"
    video_map = json.loads((dataset / "map_vid_vidorID.json").read_text(encoding="utf-8"))
    gsub = json.loads((dataset / "gsub_test.json").read_text(encoding="utf-8"))
    paths = {}
    for path in video_root.rglob("*.mp4"):
        paths.setdefault(video_key(path.relative_to(video_root).as_posix()), []).append(path)
    by_key = {key: values[0] for key, values in paths.items() if len(values) == 1}
    used = set()
    exclusion_hashes = []
    for path in sorted((repo / "data").glob("nextgqa_samples_*.json")):
        raw = path.read_bytes()
        exclusion_hashes.append(hashlib.sha256(raw).hexdigest())
        try:
            records = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(records, list):
            used.update(video_key(record["video"]) for record in records
                        if isinstance(record, dict) and isinstance(record.get("video"), str))

    grouped = {}
    csv_path = dataset / "test.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "qid", "question", "answer", "a0", "a1", "a2", "a3", "a4"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("nextgqa_mcqa_schema_invalid")
        for row in reader:
            vid, qid = str(row["video_id"]), str(row["qid"])
            if vid not in video_map:
                continue
            key = video_key(video_map[vid])
            if key not in by_key or key in used:
                continue
            info = gsub.get(vid)
            loc = info.get("location") if isinstance(info, dict) else None
            raw_spans = loc.get(qid) if isinstance(loc, dict) else None
            if not isinstance(raw_spans, list) or not raw_spans:
                continue
            try:
                spans = [[float(span[0]), float(span[1])] for span in raw_spans]
                duration = float(info["duration"])
            except (TypeError, ValueError, IndexError, KeyError, OverflowError):
                continue
            if not math.isfinite(duration) or duration <= 0 or not all(
                    math.isfinite(a) and math.isfinite(b) and 0 <= a <= b <= duration
                    for a, b in spans):
                continue
            choices = [str(row[f"a{i}"]).strip() for i in range(5)]
            answers = [i for i, choice in enumerate(choices) if norm(choice) == norm(row["answer"])]
            if any(not choice for choice in choices) or len(answers) != 1:
                continue
            if not str(row["question"]).strip():
                continue
            grouped.setdefault(vid, []).append({
                "qid": qid,
                "question": str(row["question"]).strip(),
                "choices": choices,
                "gold": answers[0],
                "spans": spans,
                "video_path": str(by_key[key]),
                "video_key": key,
            })

    grouped = filter_candidate_groups(grouped, allowed_pair_sha256)

    ordered = []
    for vid, questions in grouped.items():
        q_order = sorted(questions, key=lambda record: hashlib.sha256(
            f"{vid}:{record['qid']}".encode("utf-8")).digest())
        ordered.append((vid, q_order))
    ordered.sort(key=lambda item: hashlib.sha256(item[0].encode("utf-8")).digest())
    meta = {
        "candidate_video_count": len(ordered),
        "ambiguous_local_media_key_count": sum(len(values) > 1 for values in paths.values()),
        "private_exclusion_manifest_count": len(exclusion_hashes),
        "private_exclusion_manifest_digest": _hash(sorted(exclusion_hashes)),
        "dataset": dataset,
        "video_root": video_root,
    }
    return ordered, meta


def freeze_cohort(repo: Path, video_root: Path, excluded_media_sha256=(),
                  frame_count=FRAMES, locked_assignments=None,
                  allowed_pair_sha256=None):
    candidates, meta = load_candidate_rows(repo, video_root, allowed_pair_sha256)
    excluded_media_sha256 = set(excluded_media_sha256)
    selected = []
    counts = {"timing_rejected_videos": 0, "timing_pass_videos": 0,
              "videos_without_interval_control_pair": 0, "decode_failures": 0,
              "questions_tested_for_assignment": 0, "candidate_scan_count": 0,
              "prior_cohort_media_excluded": 0}
    locked_by_media = ({item["media_sha256"]: item["qid_sha256"]
                       for item in locked_assignments}
                       if locked_assignments is not None else None)
    for vid, questions in candidates:
        if len(selected) >= MAX_VIDEOS:
            break
        counts["candidate_scan_count"] += 1
        video_path = Path(questions[0]["video_path"])
        media_sha = sha256_file(video_path) if locked_by_media is not None else None
        if excluded_media_sha256 and (media_sha or sha256_file(video_path)) in excluded_media_sha256:
            counts["prior_cohort_media_excluded"] += 1
            continue
        if locked_by_media is not None and media_sha not in locked_by_media:
            continue
        try:
            frames, decode = decode_sampled_video(
                video_path, num_frames=frame_count,
                uniform_tolerance_seconds=UNIFORM_TOLERANCE)
        except Exception as exc:
            # Keep failure detail private; no score is run for this video.
            if "uniform-grid tolerance" in str(exc):
                counts["timing_rejected_videos"] += 1
            else:
                counts["decode_failures"] += 1
            continue
        verify_sampled_frame_pixels(frames, decode)
        counts["timing_pass_videos"] += 1
        pts = [Fraction(*record["relative_presentation_time"])
               for record in decode["sampled_frames"]]
        indices = list(range(frame_count))
        chosen = None
        eligible_questions = questions
        if locked_by_media is not None:
            target_qid_sha = locked_by_media[media_sha]
            eligible_questions = [question for question in questions
                                  if hashlib.sha256(str(question["qid"]).encode("utf-8")).hexdigest()
                                  == target_qid_sha]
            if len(eligible_questions) != 1:
                raise ValueError("cohort_lock_question_binding_missing_or_ambiguous")
        for question in eligible_questions:
            counts["questions_tested_for_assignment"] += 1
            evidence = strict_evidence_indices_from_times(
                indices, pts, question["spans"])
            control = nearest_control_indices(indices, evidence)
            if evidence and len(control) == len(evidence) and not set(control) & set(evidence):
                chosen = dict(question)
                chosen.update({"frames": frames, "decode": decode,
                               "sampled_times": pts, "evidence_positions": evidence,
                               "control_positions": control})
                break
        if chosen is None:
            counts["videos_without_interval_control_pair"] += 1
            continue
        selected.append(chosen)
    cohort_fingerprint = _hash([
        {"qid": item["qid"], "video_key": item["video_key"],
         "video_sha256": item["decode"]["source_sha256"],
         "question_sha256": hashlib.sha256(item["question"].encode("utf-8")).hexdigest(),
         "choices_sha256": _hash(item["choices"]), "gold": item["gold"],
         "spans": item["spans"], "sampled_pts": [
             [t.numerator, t.denominator] for t in item["sampled_times"]],
         "evidence_positions": item["evidence_positions"],
         "control_positions": item["control_positions"]}
        for item in selected
    ])
    if locked_assignments is not None:
        validate_locked_selection_assignments(selected, locked_assignments)
    return selected, counts, meta, cohort_fingerprint


def prepare_conditions(selected, processor, torch, frame_count=FRAMES,
                       retain_batches=True):
    import numpy as np
    from qwen_vl_utils import process_vision_info
    prepared = []
    for sample_index, item in enumerate(selected):
        evidence = set(item["evidence_positions"])
        control = set(item["control_positions"])
        conditions = {"full": [frame.copy() for frame in item["frames"]]}
        for fill in ("black", "blur"):
            for name, masked in ((f"{fill}_gold_interval_mask", evidence),
                                 (f"{fill}_equal_count_control_mask", control)):
                conditions[name] = [
                    apply_mask_fill(frame, fill, BLUR_RADIUS) if i in masked else frame.copy()
                    for i, frame in enumerate(item["frames"])]
        condition_batches = {}
        condition_geometry = []
        sample_fps = item["decode"]["effective_sample_fps"]
        for condition, frames in conditions.items():
            text, messages = build_qa_prompt(processor, item, frames, sample_fps)
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise RuntimeError("qwen_video_utility_contract_failed")
            pixels, metadata = videos[0]
            if (not isinstance(pixels, torch.Tensor) or pixels.shape[0] != frame_count
                    or metadata.get("frames_indices") != list(range(frame_count))
                    or not math.isclose(float(metadata.get("fps", 0)),
                                         sample_fps[0] / sample_fps[1], rel_tol=1e-9)):
                raise RuntimeError("qwen_video_metadata_mismatch")
            source_signatures = torch.stack([
                torch.from_numpy(np.asarray(frame).copy())
                .float().mean(dim=(0, 1)) for frame in frames])
            tensor_signatures = pixels.float().mean(dim=(-1, -2))
            if (tensor_signatures.shape != source_signatures.shape
                    or not torch.allclose(tensor_signatures, source_signatures,
                                          rtol=0, atol=2.0)):
                raise RuntimeError("qwen_frame_pixel_identity_or_order_mismatch")
            batch = process_qa_video(processor, text, images, pixels, metadata)
            grid = batch["video_grid_thw"][0].tolist()
            seconds_per_grid = float(batch["second_per_grid_ts"][0])
            if grid[0] != frame_count // int(processor.video_processor.temporal_patch_size):
                raise RuntimeError("qwen_temporal_grid_mismatch")
            if abs(seconds_per_grid - 2 / (sample_fps[0] / sample_fps[1])) > 1e-6:
                raise RuntimeError("qwen_second_per_grid_mismatch")
            condition_geometry.append((grid, seconds_per_grid))
            if retain_batches:
                condition_batches[condition] = batch
            else:
                del batch, pixels, metadata, messages, images, videos, text
                gc.collect()
        if len(set((tuple(grid), value) for grid, value in condition_geometry)) != 1:
            raise RuntimeError("condition_video_timing_or_grid_mismatch")
        prepared.append({"batches": condition_batches if retain_batches else None,
                         "grid": condition_geometry[0][0],
                         "seconds_per_grid": condition_geometry[0][1],
                         "sample_index": sample_index})
    return prepared


def build_qa_prompt(processor, item, frames, sample_fps):
    """Build the QA margin runner's exact prompt and video-first messages."""
    options = "\n".join(f"{label}. {choice}"
                        for label, choice in zip(LABELS, item["choices"], strict=True))
    prompt_text = (f"{item['question']}\nOptions:\n{options}\n"
                   "Choose the best answer. Respond with one letter only.\nAnswer:")
    messages = [{"role": "user", "content": [
        {"type": "video", "video": frames,
         "sample_fps": sample_fps[0] / sample_fps[1]},
        {"type": "text", "text": prompt_text},
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True), messages


def process_qa_video(processor, text, images, pixels, metadata):
    """Apply the QA margin runner's fixed full/masked processor call."""
    return processor(
        text=[text], images=images, videos=[pixels], video_metadata=[metadata],
        do_sample_frames=False, cap_pixels_per_frame=True, padding=True,
        return_tensors="pt")


def summarize_values(values):
    # A small fixed-cohort summary has no population inferential target here;
    # report the mean, sample SD and observed range without a p-value/CI.
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(values), "mean": sum(values) / len(values),
            "sd": (sum((x - sum(values) / len(values)) ** 2 for x in values)
                   / (len(values) - 1)) ** .5 if len(values) > 1 else None,
            "min": min(values), "max": max(values)}


def gold_vs_best_foil_margin(scores, gold_index: int) -> float:
    """Return fixed-choice log-score margin; positive means gold beats every foil."""
    if (not isinstance(gold_index, int) or isinstance(gold_index, bool)
            or not 0 <= gold_index < 5 or not isinstance(scores, (list, tuple))
            or len(scores) != 5):
        raise ValueError("five_choice_margin_inputs_invalid")
    values = [float(value) for value in scores]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("five_choice_margin_nonfinite")
    return values[gold_index] - max(value for i, value in enumerate(values) if i != gold_index)


def control_minus_evidence_margin(control_margin: float, evidence_margin: float) -> float:
    """Positive values mean masking gold evidence hurts more than masking control."""
    control_margin = float(control_margin)
    evidence_margin = float(evidence_margin)
    if not math.isfinite(control_margin) or not math.isfinite(evidence_margin):
        raise ValueError("intervention_margin_nonfinite")
    return control_margin - evidence_margin


def write_private_json_exclusive(output: Path, payload):
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_name, output)  # atomic and fails if output already exists
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def main():
    global MAX_VIDEOS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "score"), required=True)
    parser.add_argument("--n", type=int, choices=(1, 3, 10), default=MAX_VIDEOS,
                        help="small exploratory cohort size; selection remains deterministic")
    parser.add_argument("--frames", type=int, choices=(16, 32), default=FRAMES,
                        help="endpoint-inclusive decoded frame count per video")
    parser.add_argument("--output", required=True, help="unique local summary JSON; no publishing")
    parser.add_argument("--preflight-summary", help="required in score mode")
    parser.add_argument("--exclude-media-sha256", required=True,
                        help="private JSON media-hash list to prevent reusing prior pilot videos")
    parser.add_argument("--cohort-lock",
                        help="optional private hash-only video/question assignment lock")
    parser.add_argument("--cohort-lock-output",
                        help="preflight-only path for a private hash-only selected cohort lock")
    parser.add_argument("--candidate-pair-filter",
                        help="optional private JSON allowlist of eligible pair SHA256 hashes")
    parser.add_argument("--user-authorized-local-only", action="store_true",
                        help="asserts the user's local-only scope for this exploratory run")
    parser.add_argument("--video-root", default=str(Path.home() / "nextgqa_sample/NExTVideo"))
    args = parser.parse_args()
    MAX_VIDEOS = args.n
    frame_count = args.frames
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    output = Path(os.path.abspath(args.output))
    if os.path.lexists(output):
        parser.error("refusing to overwrite output")
    if args.mode == "score" and not args.preflight_summary:
        parser.error("score mode requires an outcome-blind --preflight-summary")
    if args.mode == "score" and args.cohort_lock_output:
        parser.error("--cohort-lock-output is preflight-only")
    cohort_lock_output = None
    if args.cohort_lock_output:
        try:
            cohort_lock_output = validate_new_output_path(
                Path(args.cohort_lock_output), output)
        except ValueError as exc:
            parser.error(str(exc))
    repo = Path(__file__).resolve().parents[1]
    try:
        source_commit, source_hashes = verify_committed_source_snapshot(repo)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        parser.error(f"research_source_not_committed:{exc}")
    video_root = Path(args.video_root).resolve()
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        parser.error("pinned local checkpoint missing; downloads disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2_5_VLForConditionalGeneration)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA and BF16 required; CPU fallback disabled")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    host_free = host_available_bytes()
    if free_bytes < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
        parser.error("fixed VRAM/host-memory preflight failed")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    start = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    exclusion_path = Path(os.path.abspath(args.exclude_media_sha256))
    try:
        exclusion_payload, exclusion_manifest_sha256 = read_private_json_snapshot(
            exclusion_path)
        excluded_media_sha256 = validate_private_media_exclusion(exclusion_payload)
    except (OSError, ValueError) as exc:
        parser.error(f"private_media_exclusion_manifest_invalid:{exc}")
    excluded_media_digest = _hash(sorted(excluded_media_sha256))
    candidate_filter_path = (Path(os.path.abspath(args.candidate_pair_filter))
                             if args.candidate_pair_filter else None)
    allowed_pair_sha256 = None
    candidate_filter_sha256 = None
    candidate_filter_digest = None
    if candidate_filter_path:
        try:
            pair_filter_payload, candidate_filter_sha256 = read_private_json_snapshot(
                candidate_filter_path)
            allowed_pair_sha256 = validate_candidate_pair_filter(pair_filter_payload)
        except (OSError, ValueError) as exc:
            parser.error(f"private_candidate_pair_filter_invalid:{exc}")
        candidate_filter_digest = _hash(sorted(allowed_pair_sha256))
    cohort_lock_path = Path(os.path.abspath(args.cohort_lock)) if args.cohort_lock else None
    locked_assignments = None
    cohort_lock_input_sha256 = None
    if cohort_lock_path:
        try:
            lock_payload, cohort_lock_input_sha256 = read_private_json_snapshot(cohort_lock_path)
            locked_assignments = validate_cohort_lock_payload(lock_payload, args.n)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    selected, counts, candidate_meta, cohort_hash = freeze_cohort(
        repo, video_root, excluded_media_sha256, frame_count, locked_assignments,
        allowed_pair_sha256)
    try:
        validate_cohort_cardinality(args.n, len(selected))
    except ValueError as exc:
        parser.error(str(exc))
    # Outcome-blind processor check for the complete cohort and all conditions.
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    token_ids = []
    for label in LABELS:
        ids = processor.tokenizer(label, add_special_tokens=False).input_ids
        if len(ids) != 1:
            parser.error(f"choice_label_tokenization_invalid:{label}")
        token_ids.append(ids[0])
    if len(set(token_ids)) != 5:
        parser.error("choice_labels_not_distinct_tokens")
    prepared = prepare_conditions(selected, processor, torch, frame_count,
                                  retain_batches=False)
    preparation_seconds = time.monotonic() - start
    if args.mode == "preflight":
        cohort_lock_output_sha256 = None
        if cohort_lock_output:
            lock_payload = {"schema_version": 1,
                            "assignments": build_cohort_lock_assignments(selected)}
            write_private_json_exclusive(cohort_lock_output, lock_payload)
            cohort_lock_output_sha256 = sha256_file(cohort_lock_output)
        payload = {
            "schema_version": 1,
            "purpose": "exploratory_nextgqa_gold_interval_answer_margin_dependence",
            "mode": "outcome_blind_preflight",
            "status": "passed",
            "source_commit": source_commit,
            "runner_sha256": sha256_file(Path(__file__)),
            "source_file_sha256": source_hashes,
            "decoder_sha256": sha256_file(repo / "scripts/claim_mask_video_decoder.py"),
            "helper_source_sha256": helper_source_hashes(repo),
            "candidate_input_sha256": {
                name: sha256_file(candidate_meta["dataset"] / name) for name in
                ("test.csv", "gsub_test.json", "map_vid_vidorID.json")},
            "candidate_video_count": candidate_meta["candidate_video_count"],
            "candidate_videos_scanned": counts["candidate_scan_count"],
            "candidate_videos_unscanned": candidate_meta["candidate_video_count"]
                - counts["candidate_scan_count"],
            "prior_cohort_media_excluded": counts["prior_cohort_media_excluded"],
            "prior_cohort_exclusion_file_sha256": exclusion_manifest_sha256,
            "prior_cohort_exclusion_media_digest_sha256": excluded_media_digest,
            "cohort_lock_sha256": (cohort_lock_input_sha256
                                   if cohort_lock_path else cohort_lock_output_sha256),
            "candidate_pair_filter_sha256": candidate_filter_sha256,
            "candidate_pair_filter_entries": (len(allowed_pair_sha256)
                                              if allowed_pair_sha256 is not None else None),
            "candidate_pair_filter_digest_sha256": candidate_filter_digest,
            "ambiguous_media_keys_excluded": candidate_meta["ambiguous_local_media_key_count"],
            "private_exclusion_manifest_count": candidate_meta["private_exclusion_manifest_count"],
            "private_exclusion_manifest_digest_sha256": candidate_meta["private_exclusion_manifest_digest"],
            "requested_videos": MAX_VIDEOS,
            "selection_rule": f"first {MAX_VIDEOS} outcome-blind eligible videos in deterministic hash order after prior-media exclusion",
            "timing_rejected_videos": counts["timing_rejected_videos"],
            "timing_pass_videos": counts["timing_pass_videos"],
            "videos_without_interval_control_pair": counts["videos_without_interval_control_pair"],
            "questions_tested_for_assignment": counts["questions_tested_for_assignment"],
            "effective_videos": len(selected),
            "requested_frames_per_video": frame_count,
            "effective_frames_per_video": [len(item["frames"]) for item in selected],
            "uniform_tolerance_seconds": [UNIFORM_TOLERANCE.numerator,
                                          UNIFORM_TOLERANCE.denominator],
            "cohort_sha256": cohort_hash,
            "condition_grids": [item["grid"] for item in prepared],
            "condition_names": list(CONDITION_NAMES),
            "blur_radius_pixels": BLUR_RADIUS,
            "preflight_elapsed_seconds": preparation_seconds,
            "gpu_free_before_gib": free_bytes / 1024**3,
            "gpu_total_gib": total_bytes / 1024**3,
            "host_available_gib": host_free / 1024**3,
            "model_revision": MODEL_REVISION,
            "model_config_sha256": sha256_file(model_path / "config.json"),
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
            "authorization_scope": "user-directed local-only; rights not independently verified",
            "limitations": [
                "Exploratory benchmark gold-interval dependence, not generated-claim faithfulness.",
                "The first-decoded-PTS-relative origin is not calibrated to an annotation player/source edit; frame2time metadata is not read by this runner.",
                "Black masking can create out-of-distribution inputs; equal-count control does not eliminate all distribution shift.",
                "Gaussian blur radius 12 is a secondary fill sensitivity condition; it is not guaranteed to be in distribution.",
                f"Selection stops at the first {MAX_VIDEOS} eligible videos in deterministic order after excluding all media SHA-256 values in the private exclusion manifest; candidates remaining after the cap are unscanned, and the cohort is not a probability sample.",
            ],
            "privacy": "No QA text, choices, answer text, IDs, paths, media, or model-generated text emitted.",
        }
        write_private_json_exclusive(output, payload)
        print(f"Outcome-blind NExT-GQA preflight passed: {len(selected)} videos, "
              f"{frame_count} frames each, conditions={len(CONDITION_NAMES)}; summary={output.name}")
        return

    gate = json.loads(Path(args.preflight_summary).read_text(encoding="utf-8"))
    expected_inputs = {name: sha256_file(candidate_meta["dataset"] / name) for name in
                       ("test.csv", "gsub_test.json", "map_vid_vidorID.json")}
    expected_decoder = sha256_file(repo / "scripts/claim_mask_video_decoder.py")
    expected_helpers = helper_source_hashes(repo)
    expected_versions = {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")}
    expected_model_config = sha256_file(model_path / "config.json")
    expected_exclusion_sha = exclusion_manifest_sha256
    try:
        validate_candidate_filter_binding(gate, candidate_filter_sha256)
    except ValueError as exc:
        parser.error(f"frozen_preflight_binding_mismatch:{exc}; no score was run")
    if (gate.get("schema_version") != 1
            or gate.get("purpose") != "exploratory_nextgqa_gold_interval_answer_margin_dependence"
            or gate.get("mode") != "outcome_blind_preflight"
            or gate.get("status") != "passed"
            or gate.get("requested_videos") != MAX_VIDEOS
            or gate.get("selection_rule")
                != f"first {MAX_VIDEOS} outcome-blind eligible videos in deterministic hash order after prior-media exclusion"
            or gate.get("condition_names") != list(CONDITION_NAMES)
            or gate.get("blur_radius_pixels") != BLUR_RADIUS
            or gate.get("source_commit") != source_commit
            or gate.get("source_file_sha256") != source_hashes
            or gate.get("cohort_sha256") != cohort_hash
            or gate.get("effective_videos") != len(selected)
            or gate.get("runner_sha256") != sha256_file(Path(__file__))
            or gate.get("decoder_sha256") != expected_decoder
            or gate.get("helper_source_sha256") != expected_helpers
            or gate.get("candidate_input_sha256") != expected_inputs
            or gate.get("private_exclusion_manifest_digest_sha256")
                != candidate_meta["private_exclusion_manifest_digest"]
            or gate.get("candidate_videos_scanned") != counts["candidate_scan_count"]
            or gate.get("candidate_videos_unscanned")
                != candidate_meta["candidate_video_count"] - counts["candidate_scan_count"]
            or gate.get("prior_cohort_media_excluded")
                != counts["prior_cohort_media_excluded"]
            or gate.get("prior_cohort_exclusion_file_sha256") != expected_exclusion_sha
            or gate.get("prior_cohort_exclusion_media_digest_sha256") != excluded_media_digest
            or gate.get("cohort_lock_sha256") != (cohort_lock_input_sha256
                                                    if cohort_lock_path else None)
            or gate.get("candidate_pair_filter_entries")
                != (len(allowed_pair_sha256) if allowed_pair_sha256 is not None else None)
            or gate.get("candidate_pair_filter_digest_sha256") != candidate_filter_digest
            or gate.get("requested_frames_per_video") != frame_count
            or gate.get("uniform_tolerance_seconds") != [UNIFORM_TOLERANCE.numerator,
                                                           UNIFORM_TOLERANCE.denominator]
            or gate.get("model_revision") != MODEL_REVISION):
        parser.error("frozen_preflight_binding_mismatch; no score was run")
    try:
        validate_preflight_cardinality(gate, args.n, len(selected))
    except ValueError as exc:
        parser.error(f"frozen_preflight_binding_mismatch:{exc}; no score was run")
    if (gate.get("condition_grids") != [item["grid"] for item in prepared]
            or gate.get("versions") != expected_versions
            or gate.get("model_config_sha256") != expected_model_config):
        parser.error("processor_or_model_configuration_binding_mismatch; no score was run")
    del prepared
    gc.collect()
    free_before_load, total_before_load = torch.cuda.mem_get_info()
    host_free_before_load = host_available_bytes()
    if free_before_load < MIN_FREE_VRAM or host_free_before_load < MIN_HOST_AVAILABLE:
        parser.error("immediate_pre_model_load_resource_guard_failed; no score was run")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        parser.error("processor_checkpoint_video_token_mismatch")
    label_tensor = torch.tensor(token_ids, device="cuda")
    records, failures, attempted = [], [], 0
    torch.cuda.reset_peak_memory_stats()
    for sample_index, sample in enumerate(selected):
        metrics = {}
        prepared_item = None
        failed_condition = None
        free_before_condition_gib = None
        host_before_condition_gib = None
        try:
            prepared_item = prepare_conditions(
                [sample], processor, torch, frame_count, retain_batches=True)[0]
            if prepared_item["grid"] != gate["condition_grids"][sample_index]:
                raise RuntimeError("score_processor_grid_changed_after_preflight")
            for condition in CONDITION_NAMES:
                failed_condition = condition
                host_before_condition_gib = host_available_bytes() / 1024**3
                free_before_condition_gib, _ = torch.cuda.mem_get_info()
                free_before_condition_gib /= 1024**3
                if host_before_condition_gib * 1024**3 < MIN_HOST_AVAILABLE:
                    raise RuntimeError("per_forward_host_memory_guard_failed")
                if free_before_condition_gib * 1024**3 < 2 * 1024**3:
                    raise RuntimeError("per_forward_free_vram_guard_failed")
                attempted += 1
                batch = prepared_item["batches"][condition].to("cuda")
                with torch.inference_mode():
                    out = model(**batch, use_cache=False, logits_to_keep=1)
                logits = out.logits[0, -1].float()
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError("nonfinite_next_token_logits")
                log_probs = torch.log_softmax(logits, dim=-1)
                scores = log_probs[label_tensor].float().cpu().tolist()
                metrics[condition] = scores
                del out, batch, logits, log_probs
                torch.cuda.empty_cache()
            margins = {}
            predictions = {}
            for condition, scores in metrics.items():
                margins[condition] = gold_vs_best_foil_margin(scores, sample["gold"])
                predictions[condition] = max(range(5), key=lambda index: scores[index])
            black_effect = control_minus_evidence_margin(
                margins["black_equal_count_control_mask"],
                margins["black_gold_interval_mask"])
            blur_effect = control_minus_evidence_margin(
                margins["blur_equal_count_control_mask"],
                margins["blur_gold_interval_mask"])
            records.append({
                "video_index": len(records),
                "full_gold_vs_best_foil_margin": margins["full"],
                "black_gold_interval_mask_margin": margins["black_gold_interval_mask"],
                "black_equal_count_control_mask_margin": margins["black_equal_count_control_mask"],
                "primary_black_control_minus_gold_interval_margin": black_effect,
                "blur_gold_interval_mask_margin": margins["blur_gold_interval_mask"],
                "blur_equal_count_control_mask_margin": margins["blur_equal_count_control_mask"],
                "blur_control_minus_gold_interval_margin": blur_effect,
                "black_minus_blur_gold_interval_margin": (
                    margins["black_gold_interval_mask"] - margins["blur_gold_interval_mask"]),
                "baseline_restricted_choice_correct": predictions["full"] == sample["gold"],
                "evidence_frame_count": len(sample["evidence_positions"]),
                "control_frame_count": len(sample["control_positions"]),
                "video_grid_thw": prepared_item["grid"],
            })
        except Exception as exc:
            failures.append({"video_index": len(records),
                             "failure_type": type(exc).__name__,
                             "failed_condition": failed_condition,
                             "host_available_before_condition_gib": host_before_condition_gib,
                             "free_vram_before_condition_gib": free_before_condition_gib,
                             "completed_conditions": sorted(metrics)})
            break
        finally:
            del prepared_item
            gc.collect()
    final_cuda_sync_passed, discarded_records_after_sync_failure = finalize_cuda_scoring(
        torch, records, failures)
    elapsed = time.monotonic() - start
    effects = [record["primary_black_control_minus_gold_interval_margin"] for record in records]
    blur_effects = [record["blur_control_minus_gold_interval_margin"] for record in records]
    fill_differences = [record["black_minus_blur_gold_interval_margin"] for record in records]
    payload = {
        "schema_version": 1,
        "purpose": "exploratory_nextgqa_gold_interval_answer_margin_dependence",
        "mode": "score",
        "status": "passed" if not failures and len(records) == len(selected) else "partial",
        "started_utc": started_utc,
        "source_commit": source_commit,
        "runner_sha256": sha256_file(Path(__file__)),
        "source_file_sha256": source_hashes,
        "preflight_summary_sha256": sha256_file(Path(args.preflight_summary)),
        "cohort_sha256": cohort_hash,
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "attention_implementation": "eager",
        "seed": SEED,
        "requested_videos": args.n,
        "requested_frames_per_video": frame_count,
        "effective_frames_per_video": [len(item["frames"]) for item in selected],
        "candidate_video_count": candidate_meta["candidate_video_count"],
        "candidate_videos_scanned": counts["candidate_scan_count"],
        "candidate_videos_unscanned": candidate_meta["candidate_video_count"]
            - counts["candidate_scan_count"],
        "prior_cohort_media_excluded": counts["prior_cohort_media_excluded"],
        "prior_cohort_exclusion_file_sha256": expected_exclusion_sha,
        "prior_cohort_exclusion_media_digest_sha256": excluded_media_digest,
        "candidate_pair_filter_sha256": candidate_filter_sha256,
        "candidate_pair_filter_digest_sha256": candidate_filter_digest,
            "cohort_lock_sha256": (cohort_lock_input_sha256
                               if cohort_lock_path else None),
        "selection_rule": f"first {MAX_VIDEOS} outcome-blind eligible videos in deterministic hash order after prior-media exclusion",
        "effective_videos": len(records),
        "condition_names": list(CONDITION_NAMES),
        "blur_radius_pixels": BLUR_RADIUS,
        "requested_conditions_per_video": len(CONDITION_NAMES),
        "requested_model_forwards": len(selected) * len(CONDITION_NAMES),
        "attempted_model_forwards": attempted,
        "completed_full_condition_sets": len(records),
        "failures": failures,
        "final_cuda_synchronize_passed": final_cuda_sync_passed,
        "discarded_completed_records_after_sync_failure": discarded_records_after_sync_failure,
        "primary_effect_summary": summarize_values(effects),
        "primary_effect_definition": "black-control gold-vs-best-foil margin minus black-gold-interval-mask margin",
        "blur_sensitivity_effect_summary": summarize_values(blur_effects),
        "blur_sensitivity_effect_definition": "blur-control gold-vs-best-foil margin minus blur-gold-interval-mask margin; Gaussian radius 12 pixels",
        "black_minus_blur_gold_interval_margin_summary": summarize_values(fill_differences),
        "full_condition_correct_count": (
            sum(r["baseline_restricted_choice_correct"] for r in records)
            if records else None),
        "full_condition_correct_fraction": (
            sum(r["baseline_restricted_choice_correct"] for r in records) / len(records)
            if records else None),
        "records": records,
        "statistical_test": "none; exploratory fixed-cohort pilot",
        "p_value": None,
        "ci": None,
        "effect_definition": "black-fill gold-vs-best-foil next-token log-margin under equal-count control-mask minus gold-interval-mask",
        "full_condition_accuracy_definition": "argmax among five next-token A-E label tokens; prompt-dependent restricted-choice accuracy",
        "confidence_scope": "descriptive selected timing-eligible finite cohort; not population-generalizable",
        "elapsed_seconds": elapsed,
        "peak_allocated_gib": (torch.cuda.max_memory_allocated() / 1024**3
                                if final_cuda_sync_passed else None),
        "gpu": (torch.cuda.get_device_name(0) if final_cuda_sync_passed else None),
        "gpu_free_before_model_load_gib": free_before_load / 1024**3,
        "gpu_total_gib": total_before_load / 1024**3,
        "host_available_before_model_load_gib": host_free_before_load / 1024**3,
        "cuda_version": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "qa_text_used_for_inference": True,
        "authorization_scope": "user-directed local-only; rights not independently verified",
        "generated_text_saved": False,
        "claim_review_completed": False,
        "claim_specific_faithfulness_estimate": None,
        "limitations": [
            "This estimates benchmark gold-interval answer-margin dependence only; it does not assess model-generated explanation claims or explanation faithfulness.",
            f"Selection stops at the first {MAX_VIDEOS} outcome-blind eligible videos in deterministic hash order after excluding all media SHA-256 values in the private exclusion manifest; candidates remaining after the cap are unscanned, and the cohort is not a probability sample.",
            "The relative PTS origin is not independently calibrated to the annotator playback clock; frame2time metadata is not read by this runner.",
            "Black masking is potentially out-of-distribution, and matched frame count does not remove all distribution shift.",
            "Blur masking reduces the black-fill artifact but remains an intervention and does not establish an in-distribution counterfactual.",
            "No inferential test, population CI, or multiplicity-adjusted claim is made.",
        ],
        "privacy": "No QA/choice text, IDs, paths, generated text, raw media, or per-token logits emitted.",
    }
    write_private_json_exclusive(output, payload)
    print(f"NExT-GQA interval-margin pilot {payload['status']}: "
          f"{len(records)}/{len(selected)} videos, forwards={attempted}, "
          f"mean={payload['primary_effect_summary']['mean']}, "
          f"SD={payload['primary_effect_summary']['sd']}, output={output.name}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
