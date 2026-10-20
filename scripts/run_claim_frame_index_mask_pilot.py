#!/usr/bin/env python3
"""Exploratory discrete frame-index claim interval dependence pilot.

Private local inference only. Receipts contain no source text or filesystem
paths. The score is machine-proposed interval dependence, not direct claim
faithfulness or human-reviewed evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
from fractions import Fraction

FRAME_COUNT = 16
MIN_INTERVAL = 1
MAX_INTERVAL = 4
SEED = 20261021
LOCALIZER_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
SCORER_REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
LOCALIZER_TEMPLATE = (
    "Select the shortest continuous interval in the labeled contact sheet that "
    "best shows the candidate claim. Frame indices are integers from 0 through "
    "15, and both endpoints are included. Return only one JSON object in this "
    "exact form, replacing the numbers with integer indices: "
    "{{\"start_frame_index\": 0, \"end_frame_index\": 0}}. "
    "Do not add an explanation, markdown, or any other text.\n"
    "Question: {question}\nCandidate claim: {claim}"
)


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def parse_frame_interval(text: str) -> tuple[int, int]:
    """Accept one unambiguous JSON object and a contiguous 1–4 frame span."""
    if not isinstance(text, str):
        raise ValueError("localization_not_text")
    decoder = json.JSONDecoder(object_pairs_hook=_no_duplicate_keys,
                               parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    candidates = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(obj, dict) and set(obj) == {"start_frame_index", "end_frame_index"}:
            candidates.append(obj)
    if len(candidates) != 1:
        raise ValueError("localization_json_invalid")
    obj = candidates[0]
    if not isinstance(obj, dict) or set(obj) != {"start_frame_index", "end_frame_index"}:
        raise ValueError("localization_schema_invalid")
    start, end = obj["start_frame_index"], obj["end_frame_index"]
    if (type(start) is not int or type(end) is not int or start < 0
            or end >= FRAME_COUNT or end < start
            or not MIN_INTERVAL <= end - start + 1 <= MAX_INTERVAL):
        raise ValueError("localization_bounds_invalid")
    return start, end


def interval_indices(interval: tuple[int, int]) -> list[int]:
    start, end = interval
    # Reuse strict validation by round-tripping only the integer geometry.
    if (type(start) is not int or type(end) is not int or start < 0
            or end >= FRAME_COUNT or end < start
            or end - start + 1 not in range(MIN_INTERVAL, MAX_INTERVAL + 1)):
        raise ValueError("localization_bounds_invalid")
    return list(range(start, end + 1))


def interval_json_token_sequences(tokenizer) -> list[list[int]]:
    """Build the finite JSON grammar for all inclusive 1–4 frame spans."""
    sequences = []
    for start in range(FRAME_COUNT):
        for end in range(start, min(start + MAX_INTERVAL, FRAME_COUNT) ):
            value = json.dumps({"start_frame_index": start, "end_frame_index": end},
                               separators=(",", ":"))
            tokens = tokenizer.encode(value, add_special_tokens=False)
            if not tokens:
                raise ValueError("localizer_json_tokenization_empty")
            sequences.append(tokens)
    return sequences


def allowed_next_tokens(sequences: list[list[int]], generated: list[int], eos_token_id: int) -> list[int]:
    """Return tokens that keep a generated suffix on one valid interval string."""
    matching = [sequence for sequence in sequences
                if len(generated) <= len(sequence) and sequence[:len(generated)] == generated]
    if not matching:
        return []
    allowed = {sequence[len(generated)] for sequence in matching
               if len(sequence) > len(generated)}
    if any(len(sequence) == len(generated) for sequence in matching):
        allowed.add(eos_token_id)
    return sorted(allowed)


def build_contact_sheet(frames, *, cell_size=(384, 256), columns=4):
    """Create deterministic 4×4 RGB contact sheet labeled by zero-based index."""
    from PIL import Image, ImageDraw, ImageFont
    if len(frames) != FRAME_COUNT or columns != 4:
        raise ValueError("contact_sheet_requires_exactly_16_frames")
    width, height = cell_size
    if width < 32 or height < 32:
        raise ValueError("contact_sheet_cell_too_small")
    sheet = Image.new("RGB", (columns * width, 4 * height), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, frame in enumerate(frames):
        image = frame.convert("RGB").copy()
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        x0, y0 = (index % columns) * width, (index // columns) * height
        x, y = x0 + (width - image.width) // 2, y0 + (height - image.height) // 2
        sheet.paste(image, (x, y))
        label = f"FRAME {index:02d}"
        draw.rectangle((x0 + 4, y0 + 4, x0 + 76, y0 + 21), fill=(0, 0, 0))
        draw.text((x0 + 8, y0 + 6), label, fill=(255, 255, 255), font=font)
    return sheet


def matched_control_indices(start: int, end: int) -> list[int]:
    from necessity_choice_margin_utils import nearest_control_indices
    evidence = interval_indices((start, end))
    control = nearest_control_indices(list(range(FRAME_COUNT)), evidence)
    if len(control) != len(evidence) or set(control) & set(evidence):
        raise ValueError("matched_control_invalid")
    return control


def complete_triplets(rows: list[dict], expected_rows: int) -> dict:
    """All-or-none aggregator; reject partial rows or condition sets."""
    if type(expected_rows) is not int or expected_rows not in (1, 3):
        raise ValueError("expected_row_count_invalid")
    if len(rows) != expected_rows:
        raise ValueError("incomplete_row_set")
    required = {"full", "claim_interval_black", "matched_control_black",
                "claim_interval_blur", "matched_control_blur"}
    effects_black, effects_blur = [], []
    for index, row in enumerate(rows):
        if row.get("row_index") != index or set(row.get("margins", {})) != required:
            raise ValueError("incomplete_condition_set")
        margins = row["margins"]
        if any(type(margins[name]) not in (int, float)
               or not math.isfinite(float(margins[name])) for name in required):
            raise ValueError("nonfinite_margin")
        effects_black.append(float(margins["matched_control_black"])
                             - float(margins["claim_interval_black"]))
        effects_blur.append(float(margins["matched_control_blur"])
                            - float(margins["claim_interval_blur"]))
    return {"n": expected_rows,
            "mean_control_minus_claim_black": sum(effects_black) / len(effects_black),
            "values_black": effects_black,
            "mean_control_minus_claim_blur": sum(effects_blur) / len(effects_blur),
            "values_blur": effects_blur}


def validate_smoke_forward_counts(receipt: dict) -> None:
    """Require a complete five-forward smoke before authorizing the full run."""
    if any(receipt.get(key) != 5 for key in (
            "requested_scorer_forwards", "attempted_scorer_forwards",
            "completed_scorer_forwards")):
        raise ValueError("smoke_scorer_forward_counts_invalid")


def _read_private_json(path: Path) -> tuple[dict, str]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o600):
            raise ValueError("private_input_permissions_invalid")
        chunks = []
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
    finally:
        os.close(fd)
    raw = b"".join(chunks)
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys), hashlib.sha256(raw).hexdigest()


def _write_private_json(path: Path, value: dict) -> None:
    import tempfile
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _validate_packet(payload: dict, mode: str) -> list[dict]:
    records = payload.get("records")
    expected = 1 if mode == "smoke" else 3
    if (payload.get("status") != "passed"
            or payload.get("purpose") != "private_local_nextgqa_model_generated_claim_review_packet"
            or not isinstance(records, list) or len(records) != 3):
        raise ValueError("packet_must_contain_exactly_three_records")
    checked, hashes = [], []
    for record in records:
        parsed = record.get("generated_parsed", {})
        digest, video = record.get("video_sha256"), record.get("local_video_path")
        if (not isinstance(parsed, dict) or parsed.get("valid") is not True
                or not isinstance(parsed.get("claim"), str) or not parsed["claim"].strip()
                or not isinstance(record.get("question"), str) or not record["question"].strip()
                or not isinstance(record.get("choices"), list) or len(record["choices"]) != 5
                or any(not isinstance(choice, str) for choice in record["choices"])
                or not isinstance(record.get("gold_answer"), str)
                or sum(choice == record["gold_answer"] for choice in record["choices"]) != 1
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or not isinstance(video, str)):
            raise ValueError("packet_record_contract_invalid")
        hashes.append(digest)
        checked.append(record)
    if len(set(hashes)) != len(hashes):
        raise ValueError("source_videos_not_unique")
    return checked[:expected]


def _resource_guard(torch, *, model_load=False, stage="inference"):
    free, _ = torch.cuda.mem_get_info()
    if free < (12 if model_load else 2) * 1024**3:
        raise RuntimeError(f"gpu_resource_guard_failed_{stage}")
    try:
        import psutil
        available = psutil.virtual_memory().available
    except Exception:
        available = 0
    if available < 8 * 1024**3:
        raise RuntimeError(f"host_resource_guard_failed_{stage}")


def _snapshot(model: str, revision: str) -> Path:
    path = (Path.home() / ".cache/huggingface/hub" /
            f"models--Qwen--{model}/snapshots/{revision}")
    if not path.is_dir():
        raise RuntimeError("pinned_local_checkpoint_missing")
    return path


def _model_hash(path: Path) -> str:
    config = path / "config.json"
    if not config.is_file():
        raise RuntimeError("checkpoint_config_missing")
    return hashlib.sha256(config.read_bytes()).hexdigest()


def _run(args) -> dict:
    import torch
    from PIL import Image
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from necessity_choice_margin_utils import apply_mask_fill
    from run_nextgqa_gold_interval_margin_pilot import build_qa_prompt, process_qa_video, gold_vs_best_foil_margin
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    packet, packet_sha = _read_private_json(Path(args.packet))
    started_monotonic = __import__("time").monotonic()
    records = _validate_packet(packet, args.mode)
    repo = Path(__file__).resolve().parents[1]
    source_files = (
        Path(__file__), repo / "scripts/claim_mask_video_decoder.py",
        repo / "scripts/necessity_choice_margin_utils.py",
        repo / "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    )
    source_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    runtime_versions = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "bitsandbytes", "qwen-vl-utils", "av", "Pillow")}
    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_memory_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    gpu_initial_free_bytes = int(torch.cuda.mem_get_info()[0])
    try:
        import psutil
        host_available_start_bytes = int(psutil.virtual_memory().available)
    except Exception:
        host_available_start_bytes = None
    smoke = None
    if args.mode == "full":
        if not args.smoke_receipt:
            raise ValueError("full_requires_passed_smoke_receipt")
        smoke, _ = _read_private_json(Path(args.smoke_receipt))
        validate_smoke_forward_counts(smoke)
        if (smoke.get("status") != "passed" or smoke.get("mode") != "smoke"
                or smoke.get("packet_sha256") != packet_sha
                or smoke.get("effective_rows") != 1
                or smoke.get("seed") != args.seed
                or smoke.get("localizer_revision") != LOCALIZER_REVISION
                or smoke.get("scorer_revision") != SCORER_REVISION
                or smoke.get("source_sha256") != source_hashes
                or smoke.get("runtime_versions") != runtime_versions
                or smoke.get("precision") != "bfloat16"
                or smoke.get("quantization") != "4bit"
                or smoke.get("gpu_name") != gpu_name
                or smoke.get("gpu_total_memory_bytes") != gpu_total_memory_bytes):
            raise ValueError("smoke_receipt_binding_invalid")

    localizer_path = _snapshot("Qwen2.5-VL-3B-Instruct", LOCALIZER_REVISION)
    scorer_path = _snapshot("Qwen2.5-VL-7B-Instruct", SCORER_REVISION)
    localizer_hash, scorer_hash = _model_hash(localizer_path), _model_hash(scorer_path)
    localizer_prompt_hash = hashlib.sha256(LOCALIZER_TEMPLATE.encode()).hexdigest()
    if smoke is not None and (
            smoke.get("localizer_config_sha256") != localizer_hash
            or smoke.get("scorer_config_sha256") != scorer_hash
            or smoke.get("localizer_prompt_template_sha256") != localizer_prompt_hash):
        raise ValueError("smoke_protocol_binding_invalid")
    frames_by_row, decode_by_row, intervals, controls = [], [], [], []
    for row_index, record in enumerate(records):
        frames, decoded = decode_sampled_video(record["local_video_path"], num_frames=FRAME_COUNT,
                                                uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, decoded)
        if len(frames) != FRAME_COUNT or decoded["source_sha256"] != record["video_sha256"]:
            raise ValueError("decoded_media_binding_invalid")
        packet_pts = record.get("sampled_frame_relative_pts_seconds")
        actual_pts = [float(item["relative_presentation_time"][0]) / item["relative_presentation_time"][1]
                      for item in decoded["sampled_frames"]]
        if (not isinstance(packet_pts, list) or len(packet_pts) != FRAME_COUNT
                or any(not math.isclose(float(a), b, rel_tol=0, abs_tol=1e-6)
                       for a, b in zip(packet_pts, actual_pts, strict=True))):
            raise ValueError("packet_frame_grid_binding_invalid")
        frames_by_row.append(frames)
        decode_by_row.append(decoded)

    _resource_guard(torch, model_load=True, stage="localizer_model")
    torch.cuda.reset_peak_memory_stats()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    localizer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        localizer_path, revision=LOCALIZER_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, quantization_config=quant,
        attn_implementation="eager", device_map="cuda")
    lp = AutoProcessor.from_pretrained(localizer_path, revision=LOCALIZER_REVISION,
                                       local_files_only=True)
    interval_sequences = interval_json_token_sequences(lp.tokenizer)
    for row_index, (record, frames) in enumerate(zip(records, frames_by_row, strict=True)):
        sheet = build_contact_sheet(frames)
        prompt_text = LOCALIZER_TEMPLATE.format(question=record["question"], claim=record["generated_parsed"]["claim"])
        messages = [{"role": "user", "content": [
            {"type": "image", "image": sheet}, {"type": "text", "text": prompt_text}]}]
        text = lp.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        if not images or videos:
            raise RuntimeError("localizer_image_processor_contract_failed")
        _resource_guard(torch, stage="localizer_input")
        batch = lp(text=[text], images=images, padding=True, return_tensors="pt").to("cuda")
        prompt_tokens = int(batch.input_ids.shape[1])
        eos_token_id = lp.tokenizer.eos_token_id
        if type(eos_token_id) is not int:
            raise RuntimeError("localizer_eos_token_invalid")
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated_ids = input_ids[prompt_tokens:].tolist()
            return allowed_next_tokens(interval_sequences, generated_ids, eos_token_id)
        with torch.inference_mode():
            generated = localizer.generate(**batch, do_sample=False, max_new_tokens=64,
                                           prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
        answer = lp.batch_decode(generated[:, batch.input_ids.shape[1]:], skip_special_tokens=True)[0]
        interval = parse_frame_interval(answer)
        if row_index == 0 and smoke is not None:
            prior = smoke.get("rows", [{}])[0]
            if [prior.get("start_frame_index"), prior.get("end_frame_index")] != list(interval):
                raise ValueError("smoke_localization_reproducibility_mismatch")
        intervals.append(interval)
        controls.append(matched_control_indices(*interval))
        del sheet, prompt_text, messages, text, images, videos, batch, generated, answer
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    del lp, localizer
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    _resource_guard(torch, model_load=True, stage="scorer_model")
    scorer = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        scorer_path, revision=SCORER_REVISION, local_files_only=True,
        torch_dtype=torch.bfloat16, quantization_config=quant,
        attn_implementation="eager", device_map="cuda")
    sp = AutoProcessor.from_pretrained(scorer_path, revision=SCORER_REVISION,
                                       local_files_only=True)
    labels = [sp.tokenizer.encode(label, add_special_tokens=False) for label in "ABCDE"]
    if any(len(tokens) != 1 for tokens in labels):
        raise RuntimeError("choice_label_tokenization_invalid")
    label_ids = torch.tensor([x[0] for x in labels], device="cuda")
    output_rows = []
    attempted = completed = 0
    for row_index, record in enumerate(records):
        evidence = set(interval_indices(intervals[row_index]))
        control = set(controls[row_index])
        conditions = {
            "full": [frame.copy() for frame in frames_by_row[row_index]],
            "claim_interval_black": [apply_mask_fill(frame, "black") if i in evidence else frame.copy()
                                     for i, frame in enumerate(frames_by_row[row_index])],
            "matched_control_black": [apply_mask_fill(frame, "black") if i in control else frame.copy()
                                       for i, frame in enumerate(frames_by_row[row_index])],
            "claim_interval_blur": [apply_mask_fill(frame, "blur") if i in evidence else frame.copy()
                                    for i, frame in enumerate(frames_by_row[row_index])],
            "matched_control_blur": [apply_mask_fill(frame, "blur") if i in control else frame.copy()
                                     for i, frame in enumerate(frames_by_row[row_index])],
        }
        margins = {}
        for name, frames in conditions.items():
            item = {"question": record["question"], "choices": record["choices"]}
            text, messages = build_qa_prompt(sp, item, frames, decode_by_row[row_index]["effective_sample_fps"])
            images, videos = process_vision_info(messages, return_video_metadata=True)
            if images is not None or len(videos) != 1:
                raise RuntimeError("scorer_video_processor_contract_failed")
            pixels, metadata = videos[0]
            _resource_guard(torch, stage="scorer_input")
            batch = process_qa_video(sp, text, images, pixels, metadata).to("cuda")
            attempted += 1
            with torch.inference_mode():
                logits = scorer(**batch).logits[0, -1].float()
                choice_logprobs = torch.log_softmax(logits, dim=-1)[label_ids].cpu().tolist()
            gold_index = record["choices"].index(record["gold_answer"])
            margins[name] = gold_vs_best_foil_margin(choice_logprobs, gold_index)
            completed += 1
            del text, messages, images, videos, pixels, metadata, batch, logits, choice_logprobs, frames
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        output_rows.append({"row_index": row_index, "start_frame_index": intervals[row_index][0],
                            "end_frame_index": intervals[row_index][1],
                            "control_frame_indices": controls[row_index], "margins": margins})
    summary = complete_triplets(output_rows, len(records))
    return {
        "schema_version": 1, "status": "passed", "mode": args.mode,
        "dataset_split": "private_claim_generation_packet", "packet_sha256": packet_sha,
        "source_video_sha256": [r["video_sha256"] for r in records],
        "requested_rows": 1 if args.mode == "smoke" else 3, "effective_rows": len(output_rows),
        "requested_scorer_forwards": 5 * len(records), "attempted_scorer_forwards": attempted,
        "completed_scorer_forwards": completed,
        "localizer_revision": LOCALIZER_REVISION, "localizer_config_sha256": localizer_hash,
        "scorer_revision": SCORER_REVISION, "scorer_config_sha256": scorer_hash,
        "localizer_checkpoint_hash": LOCALIZER_REVISION,
        "scorer_checkpoint_hash": SCORER_REVISION,
        "localizer_prompt_template_sha256": localizer_prompt_hash, "seed": args.seed,
        "source_sha256": source_hashes, "runtime_versions": runtime_versions,
        "precision": "bfloat16", "quantization": "4bit",
        "gpu_name": gpu_name, "gpu_total_memory_bytes": gpu_total_memory_bytes,
        "gpu_initial_free_bytes": gpu_initial_free_bytes,
        "host_available_start_bytes": host_available_start_bytes,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": round(__import__("time").monotonic() - started_monotonic, 3),
        "rows": output_rows, "finite_cohort_summary": summary,
        "metric_name": "machine_proposed_interval_dependence",
        "human_review_collection_waived": True,
        "human_review_actually_performed": False,
        "direct_claim_faithfulness_claimed": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smoke-receipt")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    try:
        if args.mode == "full" and not args.smoke_receipt:
            raise ValueError("full_requires_passed_smoke_receipt")
        from private_output_paths import validate_private_output
        repo = Path(__file__).resolve().parents[1]
        output = validate_private_output(args.output, repo)
        if output.parent != repo / "data" / "private_nextgqa":
            raise ValueError("local_private_output_required")
        args.output = str(output)
        result = _run(args)
        _write_private_json(output, result)
    except Exception as exc:
        # Only stable, non-data-bearing failure codes reach stdout.
        code = str(exc) if str(exc).isidentifier() else type(exc).__name__
        print(json.dumps({"status": "failed", "failure_code": code}), file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "mode": result["mode"],
                      "effective_rows": result["effective_rows"],
                      "completed_scorer_forwards": result["completed_scorer_forwards"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
