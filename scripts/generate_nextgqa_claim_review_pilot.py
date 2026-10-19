#!/usr/bin/env python3
"""Generate a private NExT-GQA claim/frame review packet from real local videos.

This is a data-collection smoke, not a faithfulness estimate. It saves raw QA,
model claims, local video paths, and reference spans only to a new mode-0600
local JSON explicitly authorized by the user; never publish that file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from run_nextgqa_gold_interval_margin_pilot import (
    FRAMES, MODEL_REVISION, candidate_pair_sha256, freeze_cohort,
    host_available_bytes, sha256_file,
)
from private_output_paths import validate_private_output
from claim_text_quality import claim_text_quality_failure


SEED = 20261015
MIN_FREE_VRAM = 12 * 1024**3
MIN_HOST_AVAILABLE = 8 * 1024**3
PROMPT_SUFFIX = (
    "\nWrite exactly one short, standalone sentence about a concrete visible event "
    "that answers the question. Use only visible video content; do not infer from "
    "audio or text overlays. Return only that plain-English sentence, on one line. "
    "Do not include a label, answer letter, frame index, markdown, HTML, or template text."
)


def parse_claim_output(text: str, frame_count: int) -> dict:
    """Accept one claim-only line; do not require model-selected frames."""
    del frame_count  # retained for call compatibility; no frame list is requested
    match = re.fullmatch(r"[ \t]*(?:(?:CLAIM|C)[ \t]*[:=][ \t]*)?([^\r\n]+)[ \t]*", text,
                         flags=re.IGNORECASE)
    claim_text = match.group(1).strip() if match else ""
    quality_failure = claim_text_quality_failure(claim_text) if claim_text else "claim_missing"
    valid = bool(match and claim_text and quality_failure is None)
    return {"valid": valid,
            "claim_text_quality_failure": quality_failure,
            "answer_letter": None,
            "claim": claim_text or None,
            "evidence_frame_indices": []}


def candidate_pair_digest_for_sample(sample: dict) -> str:
    """Bind a private packet row to the exact dataset video/question pair."""
    return candidate_pair_sha256(sample["video_id"], sample["qid"])


def write_private_json_exclusive(output: Path, payload) -> None:
    import tempfile
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2,
                      sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_candidate_pair_filter(path: Path):
    """Read a private SHA256 allowlist; return ids and provenance digests."""
    import run_nextgqa_gold_interval_margin_pilot as pilot
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
        pairs = pilot.validate_candidate_pair_filter(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("private_candidate_pair_filter_invalid") from exc
    digest = pilot._hash(sorted(pairs))
    return pairs, hashlib.sha256(raw).hexdigest(), digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-media-sha256", required=True,
                        help="private local media-hash exclusion JSON")
    parser.add_argument("--n", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--frames", type=int, choices=(16, 32), default=16)
    parser.add_argument("--dataset-split", choices=("val", "test"), default="test",
                        help="split used for exploratory claim-generation review; default: test")
    parser.add_argument("--video-root", type=Path, default=None,
                        help="explicit local video root (default: ~/nextgqa_sample/NExTVideo)")
    parser.add_argument("--candidate-pair-filter", type=Path, default=None,
                        help="optional local JSON allowlist with schema_version and pair_sha256")
    parser.add_argument("--output", required=True,
                        help="new mode-0600 local JSON; overwrite refused")
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args()
    if not args.user_authorized_local_only:
        parser.error("explicit local-only user authorization flag required")
    repo = Path(__file__).resolve().parents[1]
    try:
        output = validate_private_output(args.output, repo)
    except FileExistsError:
        parser.error("refusing to overwrite output")
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "private_output_path_invalid")
    exclusion_path = Path(args.exclude_media_sha256).resolve()
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    hashes = exclusion.get("media_sha256")
    if (exclusion.get("schema_version") != 1 or not isinstance(hashes, list)
            or any(not isinstance(x, str) or len(x) != 64
                   or any(c not in "0123456789abcdef" for c in x) for x in hashes)):
        parser.error("private_media_exclusion_manifest_invalid")
    dataset = repo / "NExT-GQA/datasets/nextgqa"
    video_root = (args.video_root.expanduser().resolve() if args.video_root else
                  Path.home() / "nextgqa_sample/NExTVideo")
    candidate_pair_filter_sha256 = None
    candidate_pair_filter_digest = None
    allowed_pair_sha256 = None
    if args.candidate_pair_filter is not None:
        try:
            allowed_pair_sha256, candidate_pair_filter_sha256, candidate_pair_filter_digest = \
                read_candidate_pair_filter(args.candidate_pair_filter.expanduser().resolve())
        except (OSError, ValueError):
            parser.error("private_candidate_pair_filter_invalid")
    # The imported selector's fixed cap is lowered before selection so that
    # only the requested small review set is decoded and retained in memory.
    import run_nextgqa_gold_interval_margin_pilot as pilot
    pilot.MAX_VIDEOS = args.n
    selected, counts, meta, cohort_hash = freeze_cohort(
        repo, video_root, hashes, args.frames,
        allowed_pair_sha256=allowed_pair_sha256, dataset_split=args.dataset_split)
    if len(selected) != args.n:
        parser.error(f"requested_{args.n}_eligible_samples_but_found_{len(selected)}")

    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REVISION)
    if not model_path.is_dir():
        parser.error("pinned_local_checkpoint_missing; downloads_disabled")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import numpy as np
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("CUDA_and_BF16_required; CPU_fallback_disabled")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    host_free = host_available_bytes()
    if free_bytes < MIN_FREE_VRAM or host_free < MIN_HOST_AVAILABLE:
        parser.error("fixed_GPU_or_host_memory_preflight_failed")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    started_utc = datetime.now(timezone.utc).isoformat()
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(
        str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        parser.error("processor_checkpoint_video_token_mismatch")
    torch.cuda.reset_peak_memory_stats()

    records = []
    for index, sample in enumerate(selected):
        choices = [f"{chr(65+i)}. {choice}" for i, choice in enumerate(sample["choices"])]
        prompt = (sample["question"] + "\nOptions:\n" + "\n".join(choices)
                  + PROMPT_SUFFIX)
        messages = [{"role": "user", "content": [
            {"type": "video", "video": sample["frames"],
             "sample_fps": (sample["decode"]["effective_sample_fps"][0]
                            / sample["decode"]["effective_sample_fps"][1])},
            {"type": "text", "text": prompt},
        ]}]
        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, return_video_metadata=True)
        if images is not None or len(videos) != 1:
            parser.error("qwen_video_utility_contract_failed")
        pixels, video_metadata = videos[0]
        sample_fps = sample["decode"]["effective_sample_fps"]
        expected_fps = sample_fps[0] / sample_fps[1]
        if (pixels.shape[0] != args.frames
                or video_metadata.get("frames_indices") != list(range(args.frames))
                or abs(float(video_metadata.get("fps", 0)) - expected_fps) > 1e-7):
            parser.error("qwen_video_frame_count_mismatch")
        source_signatures = torch.stack([
            torch.from_numpy(np.asarray(frame).copy()).float().mean(dim=(0, 1))
            for frame in sample["frames"]])
        tensor_signatures = pixels.float().mean(dim=(-1, -2))
        if (tensor_signatures.shape != source_signatures.shape
                or not torch.allclose(tensor_signatures, source_signatures,
                                      rtol=0, atol=2.0)):
            parser.error("qwen_frame_pixel_order_or_identity_mismatch")
        batch = processor(
            text=[prompt_text], images=images, videos=[pixels],
            video_metadata=[video_metadata], do_sample_frames=False,
            cap_pixels_per_frame=True, padding=True, return_tensors="pt").to("cuda")
        if host_available_bytes() < MIN_HOST_AVAILABLE:
            parser.error("per_generation_host_memory_guard_failed")
        current_free, _ = torch.cuda.mem_get_info()
        if current_free < 2 * 1024**3:
            parser.error("per_generation_free_VRAM_guard_failed")
        with torch.inference_mode():
            generated = model.generate(
                **batch, do_sample=False, use_cache=True, max_new_tokens=128,
                return_dict_in_generate=True)
        tail = generated.sequences[0, batch["input_ids"].shape[1]:]
        generated_text = processor.tokenizer.decode(tail, skip_special_tokens=True).strip()
        if not generated_text:
            parser.error("empty_generated_review_candidate")
        records.append({
            "review_index": index,
            "local_video_path": sample["video_path"],
            "video_sha256": sample["decode"]["source_sha256"],
            "candidate_pair_sha256": candidate_pair_digest_for_sample(sample),
            "dataset_question_id": sample["qid"],
            "question": sample["question"],
            "choices": sample["choices"],
            "gold_answer": sample["choices"][sample["gold"]],
            "gold_interval_seconds": sample["spans"],
            "sampled_frame_relative_pts_seconds": [float(x) for x in sample["sampled_times"]],
            "generated_raw_text": generated_text,
            "generated_parsed": parse_claim_output(generated_text, args.frames),
            "human_review": {
                "claim_matches_question_and_answer": None,
                "claim_visually_supported_by_reported_frames": None,
                "reported_frames_intersect_gold_interval": None,
                "reviewer_notes": None,
            },
        })
        del generated, batch
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    structured_count = sum(record["generated_parsed"]["valid"] for record in records)
    payload = {
        "schema_version": 1,
        "purpose": "private_local_nextgqa_model_generated_claim_review_packet",
        "status": "passed" if structured_count == len(records) else "partial",
        "started_utc": started_utc,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
        "pilot_runner_sha256": sha256_file(repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py"),
        "dataset_split": args.dataset_split,
        "dataset_inputs_sha256": {name: sha256_file(dataset / name) for name in
            (f"{args.dataset_split}.csv", f"gsub_{args.dataset_split}.json",
             "map_vid_vidorID.json", f"frame2time_{args.dataset_split}.json")},
        "private_exclusion_manifest_sha256": sha256_file(exclusion_path),
        "candidate_pair_filter_sha256": candidate_pair_filter_sha256,
        "candidate_pair_filter_digest": candidate_pair_filter_digest,
        "candidate_pair_filter_entries": (len(allowed_pair_sha256)
                                          if allowed_pair_sha256 is not None else None),
        "cohort_sha256": cohort_hash,
        "requested_videos": args.n,
        "effective_videos": len(records),
        "fully_structured_claim_records": structured_count,
        "candidate_video_count": meta["candidate_video_count"],
        "candidate_videos_scanned": counts["candidate_scan_count"],
        "candidate_videos_unscanned": meta["candidate_video_count"] - counts["candidate_scan_count"],
        "frames_per_video": args.frames,
        "uniform_grid_tolerance_seconds": [1, 20],
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_revision": MODEL_REVISION,
        "quantization": "4bit",
        "compute_dtype": "bfloat16",
        "seed": SEED,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_free_before_load_gib": free_bytes / 1024**3,
        "gpu_total_gib": total_bytes / 1024**3,
        "host_available_before_load_gib": host_free / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "elapsed_seconds": time.monotonic() - start,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av")},
        "claim_review_completed": False,
        "faithfulness_effect_estimate": None,
        "privacy": "Raw QAs, paths, intervals and generations are retained only in this mode-0600 local review artifact.",
        "authorization_scope": "user-authorized local-only; rights not independently verified",
        "records": records,
    }
    write_private_json_exclusive(output, payload)
    print(f"Private claim review pilot {payload['status']}: {len(records)}/{args.n} videos, "
          f"structured={structured_count}/{len(records)}, frames={args.frames}, "
          f"elapsed={payload['elapsed_seconds']:.1f}s; output={output.name}; "
          "no faithfulness score computed")


if __name__ == "__main__":
    main()
