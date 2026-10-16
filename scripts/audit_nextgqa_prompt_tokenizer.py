#!/usr/bin/env python3
"""Private hash-only CPU audit of the frozen three-row NExT-GQA prompt path.

Uses the production QA-margin prompt construction, pinned local Qwen processor,
and the configured PyAV sampling/vision processing path (16 frames by default).
It loads no model
weights, requests no GPU, performs no scores, writes no QA text, and never
downloads. The result is diagnostic only and grants no execution authority.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import stat
import sys

from nextgqa_actual_qa_source_adapter import build_actual_qa_source
from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video

MODEL_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
DEFAULT_FRAME_COUNT = 16
TOLERANCE_NUMERATOR = 1
TOLERANCE_DENOMINATOR = 20


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def validate_answer_label_tokens(tokenizer) -> list[int]:
    """Validate the runner's exact A–E single-token answer labels."""
    token_ids = []
    for label in "ABCDE":
        encoded = tokenizer(label, add_special_tokens=False)
        ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if (not isinstance(ids, list) or len(ids) != 1
                or type(ids[0]) is not int or ids[0] < 0):
            raise ValueError("answer_label_must_be_one_token")
        token_ids.append(ids[0])
    if len(set(token_ids)) != 5:
        raise ValueError("answer_label_token_ids_must_be_distinct")
    return token_ids


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_source(csv_path: Path, lineage_path: Path, worksheet_path: Path):
    csv_raw, lineage_raw, worksheet_raw = (
        csv_path.read_bytes(), lineage_path.read_bytes(), worksheet_path.read_bytes())
    lineage = json.loads(lineage_raw)
    worksheet = json.loads(worksheet_raw)
    lineage_rows = lineage.get("records") if isinstance(lineage, dict) else None
    sheet_rows = worksheet.get("records") if isinstance(worksheet, dict) else None
    if (not isinstance(lineage, dict) or not isinstance(worksheet, dict)
            or not isinstance(lineage_rows, list) or len(lineage_rows) != 3
            or not isinstance(sheet_rows, list) or len(sheet_rows) != 3
            or worksheet.get("cohort_sha256") != lineage.get("cohort_sha256")
            or worksheet.get("execution_authorization") != "not_granted"
            or worksheet.get("review_completed") is not False):
        raise ValueError("frozen_lineage_worksheet_binding_invalid")
    selection = {"records": [{key: row[key] for key in (
        "video_id", "claim_id", "dataset_video_id", "dataset_question_id")}
        for row in lineage_rows]}
    target = {"records": [{key: row[key] for key in ("video_id", "claim_id")}
                          for row in lineage_rows]}
    qa_source, binding = build_actual_qa_source(
        csv_raw=csv_raw, selection=selection,
        target_manifest_raw=canonical(target), lineage_manifest_raw=lineage_raw)
    worksheet_by_video = {row.get("video_id"): row for row in sheet_rows
                          if isinstance(row, dict)}
    if len(worksheet_by_video) != 3:
        raise ValueError("worksheet_cohort_invalid")
    for qa in qa_source["records"]:
        sheet = worksheet_by_video.get(qa["video_id"])
        if (not isinstance(sheet, dict) or sheet.get("question") != qa["question"]
                or sha256(qa["question"].encode("utf-8"))
                != next(row["question_sha256"] for row in lineage_rows
                        if row["video_id"] == qa["video_id"])):
            raise ValueError("worksheet_source_question_mismatch")
    return csv_raw, lineage, worksheet, qa_source, binding


def audit_source(*, csv_path: Path, lineage_path: Path, worksheet_path: Path,
                 video_root: Path, model_path: Path,
                 frame_count: int = DEFAULT_FRAME_COUNT) -> dict:
    """Run actual local source, decoding, prompt, and processor checks."""
    if type(frame_count) is not int or frame_count < 2:
        raise ValueError("frame_count_invalid")
    # Disable hub access before importing Transformers; all files must be local.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    csv_raw, lineage, worksheet, qa_source, binding = _load_source(
        csv_path, lineage_path, worksheet_path)

    import torch
    from PIL import Image  # noqa: F401 - confirms the expected RGB image type
    import numpy as np
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from fractions import Fraction

    if torch.cuda.is_available():
        raise RuntimeError("unexpected_cuda_visible_cpu_audit_refuses_gpu")
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    answer_label_ids = validate_answer_label_tokens(processor.tokenizer)
    prompt_hashes, token_hashes, media_hashes, geometry = [], [], [], []
    total_tokens = 0
    for item in qa_source["records"]:
        lineage_row = next(row for row in lineage["records"]
                           if row["video_id"] == item["video_id"])
        relative = PurePosixPath(lineage_row["relative_video_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("lineage_media_path_invalid")
        video_path = video_root.joinpath(*relative.parts)
        if _file_sha256(video_path) != lineage_row["video_sha256"]:
            raise ValueError("lineage_media_bytes_mismatch")
        frames, decode = decode_sampled_video(
            video_path, num_frames=frame_count,
            uniform_tolerance_seconds=Fraction(TOLERANCE_NUMERATOR,
                                               TOLERANCE_DENOMINATOR))
        verify_sampled_frame_pixels(frames, decode)
        if (decode.get("source_sha256") != lineage_row["video_sha256"]
                or len(frames) != frame_count):
            raise ValueError("decoded_media_lineage_mismatch")
        sample_fps = decode["effective_sample_fps"]
        text, messages = build_qa_prompt(processor, item, frames, sample_fps)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or len(videos) != 1:
            raise ValueError("qwen_video_utility_contract_failed")
        pixels, metadata = videos[0]
        if (not isinstance(pixels, torch.Tensor) or pixels.shape[0] != frame_count
                or metadata.get("frames_indices") != list(range(frame_count))
                or not __import__("math").isclose(
                    float(metadata.get("fps", 0)), sample_fps[0] / sample_fps[1],
                    rel_tol=1e-9)):
            raise ValueError("qwen_video_metadata_mismatch")
        source_signatures = torch.stack([
            torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
            for frame in frames])
        tensor_signatures = pixels.float().mean(dim=(-1, -2))
        if (tensor_signatures.shape != source_signatures.shape
                or not torch.allclose(tensor_signatures, source_signatures,
                                      rtol=0, atol=2.0)):
            raise ValueError("qwen_frame_pixel_identity_or_order_mismatch")
        # This is the exact production processor call for its unmasked/full input.
        batch = process_qa_video(processor, text, images, pixels, metadata)
        ids = batch["input_ids"][0].tolist()
        grid = batch["video_grid_thw"][0].tolist()
        second_per_grid = float(batch["second_per_grid_ts"][0])
        if not ids or grid[0] != frame_count // int(
                processor.video_processor.temporal_patch_size):
            raise ValueError("qwen_processor_geometry_invalid")
        expected_second_per_grid = 2 / (sample_fps[0] / sample_fps[1])
        if abs(second_per_grid - expected_second_per_grid) > 1e-6:
            raise ValueError("qwen_second_per_grid_mismatch")
        prompt_hashes.append(sha256(text.encode("utf-8")))
        token_hashes.append(sha256(canonical(ids)))
        media_hashes.append(decode["source_sha256"])
        total_tokens += len(ids)
        geometry.append({"frame_count": len(frames), "video_grid_thw": grid,
                         "second_per_grid_ts": second_per_grid,
                         "token_count": len(ids)})
        del batch, pixels, metadata, messages, images, videos, text, frames

    processor_files = []
    for name in ("preprocessor_config.json", "processor_config.json",
                 "tokenizer_config.json", "tokenizer.json", "vocab.json",
                 "merges.txt", "chat_template.json", "special_tokens_map.json"):
        path = model_path / name
        if path.is_file():
            processor_files.append((name, _file_sha256(path)))
    if not processor_files:
        raise ValueError("pinned_processor_files_missing")
    repo = Path(__file__).resolve().parents[1]
    return {
        "schema_version": 1,
        "status": "diagnostic_passed",
        "diagnostic_only": True,
        "execution_authorization": "not_granted",
        "scope": "three_frozen_actual_QA_rows_full_unmasked_production_prompt_and_processor_input",
        "requested_rows": 3,
        "effective_rows": len(prompt_hashes),
        "requested_frames_per_row": frame_count,
        "effective_frames_per_row": [row["frame_count"] for row in geometry],
        "input_token_counts": [row["token_count"] for row in geometry],
        "total_input_tokens": total_tokens,
        "answer_label_token_count": len(answer_label_ids),
        "answer_label_token_ids_sha256": sha256(canonical(answer_label_ids)),
        "source_csv_sha256": sha256(csv_raw),
        "lineage_file_sha256": _file_sha256(lineage_path),
        "worksheet_file_sha256": _file_sha256(worksheet_path),
        "audit_script_sha256": _file_sha256(Path(__file__)),
        "production_runner_sha256": _file_sha256(
            repo / "scripts" / "run_nextgqa_gold_interval_margin_pilot.py"),
        "decoder_sha256": _file_sha256(repo / "scripts" / "claim_mask_video_decoder.py"),
        "source_adapter_sha256": _file_sha256(
            repo / "scripts" / "nextgqa_actual_qa_source_adapter.py"),
        "lineage_cohort_sha256": lineage["cohort_sha256"],
        "worksheet_cohort_matches_lineage": True,
        "source_row_binding_count": len(binding["source_row_bindings"]),
        "media_sha256_ordered_digest": sha256(canonical(media_hashes)),
        "prompt_sha256_ordered_digest": sha256(canonical(prompt_hashes)),
        "token_ids_sha256_ordered_digest": sha256(canonical(token_hashes)),
        "processor_files_sha256": sha256(canonical(processor_files)),
        "model_revision": MODEL_REVISION,
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "cuda_available": False,
        "model_weights_loaded": False,
        "network_downloads": 0,
        "score_calls": 0,
        "raw_QA_emitted": False,
        "decoded_frame_pixels_emitted": False,
        "limitations": [
            "no model forward or score was run",
            "the full unmasked processor path was checked; masked intervention tensors were not built",
            "the production QA-margin prompt was checked; claim-generation prompt tokens were not checked",
            "prompt/token hashes are diagnostics, not evidence of semantic validity or faithfulness",
            "rights status remains unverified; artifacts must remain local",
        ],
    }


def _write_exclusive(path: Path, payload: dict) -> None:
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[1]
    private = repo / "data" / "private_nextgqa"
    model_default = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                     / "snapshots" / MODEL_REVISION)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path,
                        default=repo / "NExT-GQA/datasets/nextgqa/test.csv")
    parser.add_argument("--lineage", type=Path,
                        default=private / "claim_lineage_sentence_exploratory_20261016_01.json")
    parser.add_argument("--worksheet", type=Path,
                        default=private / "claim_mapping_sentence_exploratory_20261016_01.json")
    parser.add_argument("--video-root", type=Path,
                        default=Path.home() / "nextgqa_sample/NExTVideo")
    parser.add_argument("--model-path", type=Path, default=model_default)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT,
                        help="frames per video; default matches the active margin runner")
    parser.add_argument("--output", type=Path,
                        help="optional unique private JSON output; default prints aggregate summary")
    args = parser.parse_args(argv)
    if not args.model_path.is_dir():
        parser.error("pinned_local_processor_missing; downloads_disabled")
    try:
        result = audit_source(csv_path=args.csv, lineage_path=args.lineage,
                              worksheet_path=args.worksheet, video_root=args.video_root,
                              model_path=args.model_path, frame_count=args.frames)
        if args.output is None:
            print(json.dumps(result, sort_keys=True))
        else:
            _write_exclusive(args.output, result)
            print(json.dumps({"status": result["status"], "output_written": True,
                              "output_mode": "0600", "output_sha256": _file_sha256(args.output),
                              "diagnostic_only": True,
                              "execution_authorization": "not_granted"}, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        # Do not print exception text: third-party errors may contain QA or paths.
        print(json.dumps({"status": "blocked", "failure_code": "audit_failed",
                          "diagnostic_only": True,
                          "execution_authorization": "not_granted",
                          "raw_QA_emitted": False}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
