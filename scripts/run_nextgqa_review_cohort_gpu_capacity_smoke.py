#!/usr/bin/env python3
"""Capacity-only CUDA probe on the three privately staged review videos.

Runs exactly one unmasked 32-frame Qwen2.5-VL forward per custody-bound media
file. It does not use claims, questions, answers, review labels, or intervals.
The private output contains only aggregate/per-ordinal engineering diagnostics.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

import claim_mask_video_decoder as decoder
from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels

MODEL_REV = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
PROMPT = "Describe the visible scene briefly. Use only visual information."
FRAMES = 32
GRID_TOLERANCE = Fraction(1, 20)  # Existing exploratory engineering census: 50 ms.


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_asset_manifest(snapshot: Path) -> list[dict]:
    """Hash every cached model/processor file before loading the checkpoint."""
    rows = []
    for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
        rows.append({"relative_path": path.relative_to(snapshot).as_posix(),
                     "size_bytes": path.stat().st_size, "sha256": digest(path)})
    if not rows or not any(row["relative_path"].endswith(".safetensors") for row in rows):
        raise ValueError("pinned_model_snapshot_incomplete")
    return rows


def read_private_bytes(path: Path) -> bytes:
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or stat.S_IMODE(st.st_mode) != 0o600:
        raise ValueError("private_manifest_permissions_or_type_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as f:
        opened = os.fstat(f.fileno())
        if (not stat.S_ISREG(opened.st_mode) or opened.st_ino != st.st_ino
                or opened.st_dev != st.st_dev):
            raise ValueError("manifest_changed_during_open")
        return f.read()


def validate_cohort(custody_path: Path, plan_path: Path, provenance_path: Path,
                    media_dir: Path, expected: tuple[str, str, str],
                    scratch: Path) -> list[Path]:
    paths = (custody_path, plan_path, provenance_path)
    raw_manifests = [read_private_bytes(p) for p in paths]
    if tuple(hashlib.sha256(b).hexdigest() for b in raw_manifests) != expected:
        raise ValueError("manifest_sha256_pin_mismatch")
    dst = media_dir.lstat()
    if not stat.S_ISDIR(dst.st_mode) or stat.S_IMODE(dst.st_mode) != 0o700:
        raise ValueError("media_directory_not_private")
    custody, plan, prov = [json.loads(b.decode("utf-8")) for b in raw_manifests]
    if (custody.get("schema_version") != 1 or plan.get("schema_version") != 1
            or prov.get("schema_version") != 1
            or plan.get("status") != "predeclared_frozen"
            or prov.get("status") != "candidate_not_jointly_frozen"):
        raise ValueError("unexpected_cohort_manifest_contract")
    cr, pr, vr = custody.get("records"), plan.get("records"), prov.get("records")
    if not (isinstance(cr, list) and isinstance(pr, list) and isinstance(vr, list)
            and len(cr) == len(pr) == len(vr) == 3
            and custody.get("record_count") == 3):
        raise ValueError("exact_three_record_cohort_required")
    if prov.get("plan_sha256") != expected[1]:
        raise ValueError("provenance_plan_hash_link_mismatch")
    def by_token(rows):
        out = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid_manifest_row")
            token = row.get("review_token")
            if not isinstance(token, str) or not token or token in out:
                raise ValueError("missing_or_duplicate_review_token")
            out[token] = row
        return out
    cm, pm, vm = by_token(cr), by_token(pr), by_token(vr)
    if set(cm) != set(pm) or set(cm) != set(vm):
        raise ValueError("cohort_token_set_mismatch")
    files = []
    seen_names, seen_source_hashes, seen_staged_hashes = set(), set(), set()
    for c in cr:
        if not isinstance(c, dict):
            raise ValueError("invalid_custody_row")
        token = c["review_token"]
        p, v = pm[token], vm[token]
        if not isinstance(p, dict) or not isinstance(v, dict):
            raise ValueError("invalid_plan_or_provenance_row")
        if (c.get("source_media_sha256") != p.get("source_media_sha256")
                or c.get("staged_media_sha256") != v.get("staged_media_sha256")):
            raise ValueError("custody_plan_provenance_hash_disagreement")
        name = c.get("media_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("unsafe_media_filename")
        source_hash = c.get("source_media_sha256")
        staged_hash = c.get("staged_media_sha256")
        valid_hash = lambda x: isinstance(x, str) and len(x) == 64 and all(ch in "0123456789abcdef" for ch in x)
        if not valid_hash(source_hash) or not valid_hash(staged_hash):
            raise ValueError("invalid_media_sha256")
        if name in seen_names or source_hash in seen_source_hashes or staged_hash in seen_staged_hashes:
            raise ValueError("duplicate_staged_video_identity")
        seen_names.add(name); seen_source_hashes.add(source_hash); seen_staged_hashes.add(staged_hash)
        path = media_dir / name
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600:
            raise ValueError("staged_media_missing_or_not_private")
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "rb") as f:
            opened = os.fstat(f.fileno())
            if (not stat.S_ISREG(opened.st_mode) or opened.st_ino != before.st_ino
                    or opened.st_dev != before.st_dev):
                raise ValueError("media_changed_during_open")
            data = f.read()
        if len(data) != c.get("size_bytes") or hashlib.sha256(data).hexdigest() != c.get("staged_media_sha256"):
            raise ValueError("staged_media_hash_mismatch")
        snapshot = scratch / f"record_{len(files) + 1}.mp4"
        fd = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        files.append(snapshot)
    if len({p.resolve() for p in files}) != 3:
        raise ValueError("duplicate_staged_media_path")
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--custody", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--provenance", type=Path, required=True)
    ap.add_argument("--media-dir", type=Path, required=True)
    ap.add_argument("--expected-custody-sha256", required=True)
    ap.add_argument("--expected-plan-sha256", required=True)
    ap.add_argument("--expected-provenance-sha256", required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    out = a.output.resolve()
    if out.exists():
        ap.error("output exists; choose a new local path")
    # Verify every member before loading any media or model.
    out_parent = out.parent
    if out_parent.exists():
        parent_stat = out_parent.lstat()
        if not stat.S_ISDIR(parent_stat.st_mode) or stat.S_IMODE(parent_stat.st_mode) != 0o700:
            ap.error("output_parent_must_be_private_directory_mode_0700")
    else:
        ap.error("output_parent_must_exist_and_be_private")
    scratch_ctx = tempfile.TemporaryDirectory(prefix="nextgqa_capacity_", dir="/tmp")
    scratch = Path(scratch_ctx.name)
    os.chmod(scratch, 0o700)
    atexit.register(scratch_ctx.cleanup)
    try:
        files = validate_cohort(a.custody.absolute(), a.plan.absolute(), a.provenance.absolute(),
                                a.media_dir.absolute(), (a.expected_custody_sha256,
                                a.expected_plan_sha256, a.expected_provenance_sha256), scratch)
    except Exception:
        scratch_ctx.cleanup()
        raise
    repo = Path(__file__).resolve().parents[1]
    model_path = (Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct"
                  / "snapshots" / MODEL_REV)
    if not model_path.is_dir():
        ap.error("pinned_local_checkpoint_missing_downloads_disabled")
    asset_manifest = model_asset_manifest(model_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        ap.error("CUDA_and_BF16_required_no_CPU_fallback")
    device = torch.cuda.current_device()
    free0, total0 = torch.cuda.mem_get_info(device)
    if free0 < 12 * 1024**3:
        ap.error("at_least_12_GiB_free_VRAM_required_before_load")
    started = datetime.now(timezone.utc).isoformat()
    wall = time.monotonic()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True,
        min_pixels=256 * 28 * 28, max_pixels=256 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="eager", device_map="cuda",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
                                                bnb_4bit_compute_dtype=torch.bfloat16)).eval()
    if int(model.config.video_token_id) != int(processor.video_token_id):
        raise RuntimeError("model_processor_video_token_mismatch")
    rows = []
    failures = 0
    for ordinal, path in enumerate(files, 1):
        start = time.monotonic()
        try:
            frames, meta = decode_sampled_video(path, num_frames=FRAMES,
                uniform_tolerance_seconds=GRID_TOLERANCE)
            verify_sampled_frame_pixels(frames, meta)
            fps = meta["effective_sample_fps"]
            messages = [{"role": "user", "content": [
                {"type": "video", "video": frames, "sample_fps": fps[0] / fps[1]},
                {"type": "text", "text": PROMPT}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True)
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise RuntimeError("qwen_video_utility_contract_failed")
            pixels, metadata = videos[0]
            if (pixels.shape[0] != FRAMES or metadata.get("frames_indices") != list(range(FRAMES))
                    or abs(float(metadata.get("fps", 0)) - fps[0] / fps[1]) > 1e-7):
                raise RuntimeError("processor_frame_contract_failed")
            expected_sig = torch.stack([torch.from_numpy(np.asarray(f).copy()).float().mean((0, 1))
                                        for f in frames])
            if not torch.allclose(pixels.float().mean((-1, -2)), expected_sig, rtol=0, atol=2.0):
                raise RuntimeError("processor_pixel_order_or_identity_mismatch")
            batch = processor(text=[prompt], images=images, videos=[pixels],
                video_metadata=[metadata], do_sample_frames=False, cap_pixels_per_frame=True,
                padding=True, return_tensors="pt").to("cuda")
            torch.cuda.reset_peak_memory_stats(device)
            with torch.inference_mode():
                result = model(**batch, use_cache=False, logits_to_keep=1)
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(result.logits).all())
            if not finite:
                raise RuntimeError("nonfinite_logits")
            free_after, total_after = torch.cuda.mem_get_info(device)
            rows.append({"ordinal": ordinal, "status": "passed", "frames": len(frames),
                "finite_logits": True, "input_tokens": int(batch["input_ids"].shape[1]),
                "elapsed_seconds": time.monotonic() - start,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
                "driver_free_after_gib": free_after / 1024**3})
            del result, batch, pixels, frames, metadata, messages, images, videos
            torch.cuda.empty_cache()
        except Exception as e:
            failures += 1
            rows.append({"ordinal": ordinal, "status": "failed", "failure_code":
                         ("timing_grid_rejected" if isinstance(e, ValueError)
                          and "uniform-grid tolerance" in str(e)
                          else "decode_processor_or_forward_failed"),
                         "elapsed_seconds": time.monotonic() - start})
            break
    status = "passed" if len(rows) == 3 and failures == 0 else "incomplete"
    payload = {"schema_version": 1,
        "purpose": "exact_private_review_cohort_gpu_capacity_only_not_faithfulness",
        "status": status, "started_utc": started, "runner_sha256": digest(Path(__file__)),
        "cohort_identity_status": "caller_pinned_local_inputs_not_authenticated",
        "custody_sha256": a.expected_custody_sha256, "plan_sha256": a.expected_plan_sha256,
        "provenance_sha256": a.expected_provenance_sha256,
        "requested_videos": 3, "attempted_videos": len(rows),
        "effective_videos": sum(r["status"] == "passed" for r in rows), "failures": failures,
        "requested_frames_per_video": FRAMES, "uniform_tolerance_seconds": [1, 20],
        "records": rows, "elapsed_seconds": time.monotonic() - wall,
        "gpu": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "gpu_free_before_gib": free0 / 1024**3, "gpu_total_before_gib": total0 / 1024**3,
        "model_revision": MODEL_REV, "precision": "bf16_4bit_bf16_compute",
        "model_assets": asset_manifest, "random_seed": None,
        "stochastic_sampling": False,
        "decoder_helper_sha256": digest(Path(decoder.__file__)),
        "pyav_version": decoder.av.__version__,
        "ffmpeg_library_versions": {k: list(v) for k, v in sorted(decoder.av.library_versions.items())},
        "versions": {n: importlib.metadata.version(n) for n in
                     ("torch", "transformers", "bitsandbytes", "qwen-vl-utils")},
        "qa_text_used": False, "claims_or_intervals_used": False,
        "faithfulness_effect_estimate": None, "human_review_completed": False,
        "rights_status": "unverified_local_only",
        "privacy": "Opaque ordinals only; no media/question/answer/claim/reviewer identifiers or text."}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    scratch_ctx.cleanup()
    print(f"cohort capacity smoke {status}: {payload['effective_videos']}/3 completed; "
          f"failures={failures}; elapsed={payload['elapsed_seconds']:.1f}s")
    if status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
