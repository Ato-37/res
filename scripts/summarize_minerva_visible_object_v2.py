#!/usr/bin/env python3
"""Verify and summarize the frozen Minerva visible-object v2 pilot receipts.

The output contains derived values and hashes only; it omits video IDs, claims,
questions, media paths, and image data. This is descriptive bookkeeping for a
three-row exploratory pilot, not a direct faithfulness estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
EXPECTED_BRANCHES = {
    f"{fill}_{condition}"
    for fill in ("support", "sham")
    for condition in ("none", "bbox_bias", "spatial_sham_bias")
}
EXPECTED_LAYERS = {str(layer): 1 for layer in range(5, 17)}
T_CRITICAL_95_DF1 = 12.706204736432095
SHARED_PREFLIGHT_FIELDS = (
    "schema_version", "status", "cohort_sha256", "manifest_sha256",
    "preflight_source_sha256", "decoder_source_sha256",
    "generated_runner_source_sha256", "bbox_mapper_source_sha256",
    "qwen_video_processor", "human_review_performed", "rights_status")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _same(a: Any, b: Any, *, tol: float = 1e-10) -> bool:
    if (isinstance(a, bool) or isinstance(b, bool)
            or not isinstance(a, (int, float)) or not isinstance(b, (int, float))
            or not math.isfinite(float(a)) or not math.isfinite(float(b))):
        return a == b
    return abs(float(a) - float(b)) <= tol


def _read_private_json(path: Path) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 50_000_000):
        raise ValueError("private_json_permissions_or_size_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("private_json_changed_during_read")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_json_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("private_json_shape_invalid")
    return raw, payload


def _resolve_private(relative: str, *, root: Path, private_root: Path) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("private_manifest_path_invalid")
    rel = Path(relative)
    if rel.is_absolute() or any(part in {"", ".", ".."} for part in rel.parts):
        raise ValueError("private_manifest_path_invalid")
    path = root
    for part in rel.parts:
        path = path / part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("private_symlink_rejected")
        except FileNotFoundError as exc:
            raise ValueError("private_input_missing") from exc
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(private_root.resolve()) or not resolved.is_file():
        raise ValueError("private_path_outside_root")
    return resolved


def _hash_file(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def _load_preflight_for_row(entry: dict, *, final: dict, final_raw: bytes,
                            cohort_sha: str, selected: dict,
                            row_index: int, root: Path,
                            private_root: Path) -> tuple[dict, bytes, str, dict]:
    if "cohort_preflight" in entry:
        path = _resolve_private(entry["cohort_preflight"], root=root,
                                 private_root=private_root)
        raw, preflight = _read_private_json(path)
        if (any(preflight.get(field) != final.get(field)
                for field in SHARED_PREFLIGHT_FIELDS)
                or preflight.get("cohort_sha256") != cohort_sha):
            raise ValueError("row_preflight_source_or_manifest_mismatch")
    else:
        raw, preflight = final_raw, final
    target = (selected.get("video_id"), selected.get("row_key_sha256"))
    final_rows = [record for record in final.get("records", [])
        if isinstance(record, dict)
        and (record.get("video_id"), record.get("row_key_sha256")) == target]
    selected_rows = [record for record in preflight.get("records", [])
        if isinstance(record, dict)
        and (record.get("video_id"), record.get("row_key_sha256")) == target]
    if (len(final_rows) != 1 or len(selected_rows) != 1
            or selected_rows[0] != final_rows[0]
            or selected_rows[0].get("status") != "pass"):
        raise ValueError("row_preflight_record_mismatch")
    return preflight, raw, _sha(raw), selected_rows[0]


def _validate_sources(source_hashes: dict, *, script_root: Path) -> str:
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("receipt_source_hashes_missing")
    for name, expected in source_hashes.items():
        if (not isinstance(name, str) or Path(name).name != name
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("receipt_source_hash_entry_invalid")
        source_path = script_root / name
        if source_path.is_symlink() or not source_path.is_file():
            raise ValueError("receipt_source_file_missing")
        if _sha(source_path.read_bytes()) != expected:
            raise ValueError("receipt_source_hash_mismatch")
    return _sha(json.dumps(source_hashes, sort_keys=True,
        separators=(",", ":")).encode("utf-8"))


def _validate_common_row(*, row_index: int, selected: dict, record: dict,
                         entry: dict, cpu: dict, cpu_raw: bytes, gpu: dict,
                         gpu_raw: bytes, preflight_sha: str,
                         cohort_sha: str, manifest_sha: str,
                         cohort_source_sha: dict,
                         final_preflight: dict,
                         root: Path, private_root: Path,
                         script_root: Path) -> dict:
    status = gpu.get("status")
    video_path = _resolve_private(entry.get("video"), root=root,
                                  private_root=private_root)
    time_path = _resolve_private(entry.get("time_map"), root=root,
                                 private_root=private_root)
    video_sha, video_md5 = _hash_file(video_path)
    time_sha, time_md5 = _hash_file(time_path)
    row_key_sha = selected.get("row_key_sha256")
    video_id_sha = _sha(str(selected.get("video_id", "")).encode("utf-8"))
    sources = cpu.get("source_sha256")
    source_bundle_sha = _validate_sources(sources, script_root=script_root)
    source_bindings = {
        "preflight_source_sha256": "preflight_minerva_grounding_cohort.py",
        "decoder_source_sha256": "run_minerva_ego_anchor_gpu_pilot.py",
        "generated_runner_source_sha256": "run_minerva_ego_generated_grounding_pilot.py",
        "bbox_mapper_source_sha256": "minerva_ego_anchor_utils.py"}
    if any(final_preflight.get(field) != sources.get(filename)
           for field, filename in source_bindings.items()):
        raise ValueError("final_preflight_source_hash_binding_invalid")
    expected = {
        "cohort_sha256": cohort_sha,
        "row_index": row_index,
        "row_key_sha256": row_key_sha,
        "video_sha256": video_sha,
        "video_md5": video_md5,
        "time_map_sha256": time_sha,
        "time_map_md5": time_md5,
        "manifest_sha256": manifest_sha,
        "cohort_cpu_preflight_sha256": preflight_sha,
    }
    if any(cpu.get(key) != value for key, value in expected.items()):
        raise ValueError("cpu_receipt_binding_invalid")
    if (cpu.get("status") != "completed_minerva_generated_object_pixel_cpu_preflight"
            or cpu.get("video_id_sha256") != video_id_sha
            or record.get("video_sha256") != video_sha
            or record.get("video_md5") != video_md5
            or record.get("time_map_sha256") != time_sha
            or record.get("time_map_md5") != time_md5):
        raise ValueError("cpu_preflight_media_binding_invalid")
    cpu_source_sha = cpu.get("source_sha256", {})
    annotation_sha = cpu.get("annotation_source_sha256")
    if annotation_sha != cohort_source_sha:
        raise ValueError("cpu_annotation_source_cohort_binding_invalid")
    runner_sha = cpu_source_sha.get("run_minerva_generated_object_pixel_pilot.py")
    if (not isinstance(annotation_sha, dict) or not annotation_sha
            or cpu.get("runner_sha256") != runner_sha
            or gpu.get("cohort_sha256") != cohort_sha
            or gpu.get("row_index") != row_index
            or gpu.get("row_key_sha256") != row_key_sha
            or gpu.get("cpu_pixel_receipt_sha256") != _sha(cpu_raw)
            or gpu.get("cohort_preflight_sha256") != preflight_sha
            or gpu.get("runner_sha256") != runner_sha
            or gpu.get("annotation_source_sha256", annotation_sha) != annotation_sha
            or gpu.get("video_sha256") != video_sha):
        raise ValueError("gpu_cpu_receipt_binding_invalid")
    if status == "completed_single_video_generated_object_pixel_exploratory_pilot":
        if (gpu.get("video_id_sha256") != video_id_sha
                or gpu.get("video_md5") != video_md5
                or gpu.get("time_map_sha256") != time_sha
                or gpu.get("source_sha256") != sources
                or gpu.get("pixel_tensor_sha256") != cpu.get("pixel_tensor_sha256")
                or gpu.get("condition_key_counts") != cpu.get("condition_key_counts")):
            raise ValueError("gpu_completed_source_binding_invalid")
    return {"cpu_receipt_sha256": _sha(cpu_raw),
        "gpu_receipt_sha256": _sha(gpu_raw),
        "cohort_preflight_sha256": preflight_sha,
        "source_bundle_sha256": source_bundle_sha,
        "runner_sha256": runner_sha,
        "annotation_source_sha256": annotation_sha}


def _score_row(gpu: dict, cpu: dict, base: dict) -> dict:
    generation = gpu.get("generation")
    model_revision = gpu.get("model_revision")
    if not isinstance(generation, dict) or not isinstance(model_revision, str) \
            or not model_revision:
        raise ValueError("gpu_generation_missing")
    token_ids = generation.get("generated_token_ids")
    token_hash = _sha(json.dumps(token_ids, separators=(",", ":"))
                      .encode("ascii")) if isinstance(token_ids, list) else ""
    text = generation.get("generated_text")
    if (not isinstance(text, str) or generation.get("generated_text_sha256")
            != _sha(text.encode("utf-8"))
            or not isinstance(token_ids, list) or not token_ids
            or any(type(token) is not int or token < 0 for token in token_ids)
            or generation.get("generated_token_ids_sha256") != token_hash
            or generation.get("target_token_count") != len(token_ids)):
        raise ValueError("gpu_generated_target_hash_binding_invalid")
    scores = generation.get("branch_mean_logprob_nats_per_token")
    audits = generation.get("branch_audit")
    if (not isinstance(scores, dict) or set(scores) != EXPECTED_BRANCHES
            or not isinstance(audits, dict) or set(audits) != EXPECTED_BRANCHES
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(float(value)) for value in scores.values())):
        raise ValueError("six_branch_score_set_invalid")
    counts = cpu.get("condition_key_counts")
    if (not isinstance(counts, dict)
            or type(counts.get("bbox")) is not int or counts["bbox"] <= 0
            or counts.get("bbox") != counts.get("spatial_sham")
            or gpu.get("condition_key_counts") != counts
            or generation.get("model_call_count") != {
                "generation_calls": 1, "teacher_forced_score_calls": 6,
                "completed_teacher_forced_score_calls": 6}):
        raise ValueError("six_branch_dose_or_call_count_invalid")
    for branch in sorted(EXPECTED_BRANCHES):
        audit = audits[branch]
        condition = branch.removeprefix("support_").removeprefix("sham_")
        expected_keys = (0 if condition == "none" else
            counts["bbox"] if condition == "bbox_bias" else counts["spatial_sham"])
        if (not isinstance(audit, dict)
                or audit.get("target_token_ids_sha256") != token_hash
                or audit.get("target_token_count") != len(token_ids)
                or audit.get("target_query_count") != len(token_ids)
                or audit.get("biased_key_count") != expected_keys
                or audit.get("deleted_key_count") != 0
                or audit.get("intervention") != ("none" if condition == "none"
                    else "QTA visual-key logit bias")
                or not _same(audit.get("logit_bias_nats"), math.log(2.0))
                or audit.get("layer_hook_calls") != EXPECTED_LAYERS):
            raise ValueError("branch_target_dose_or_hook_binding_invalid")
    d = {condition: float(scores[f"sham_{condition}"])
         - float(scores[f"support_{condition}"])
         for condition in ("none", "bbox_bias", "spatial_sham_bias")}
    primary = d["bbox_bias"] - d["spatial_sham_bias"]
    secondary = d["bbox_bias"] - d["none"]
    stored = generation.get("contrasts")
    expected_stored = {
        "D_no_bias_nats_per_token": d["none"],
        "D_bbox_bias_nats_per_token": d["bbox_bias"],
        "D_spatial_sham_bias_nats_per_token": d["spatial_sham_bias"],
        "primary_bbox_minus_spatial_sham_interaction_nats_per_token": primary,
        "secondary_bbox_minus_no_bias_interaction_nats_per_token": secondary}
    if (not isinstance(stored, dict)
            or any(not _same(stored.get(key), value)
                   for key, value in expected_stored.items())):
        raise ValueError("stored_contrast_recomputation_mismatch")
    return {**base, "row_index": gpu["row_index"], "status": "scored",
        "model_revision_sha256": _sha(model_revision.encode()),
        "target_token_count": len(token_ids),
        "target_token_ids_sha256": token_hash,
        "branch_count": len(scores), "score_forward_count": 6,
        "hook_calls_total": 6 * len(EXPECTED_LAYERS),
        "bbox_key_count": counts["bbox"],
        "spatial_sham_key_count": counts["spatial_sham"],
        "D_no_bias_nats_per_token": d["none"],
        "D_bbox_bias_nats_per_token": d["bbox_bias"],
        "D_spatial_sham_bias_nats_per_token": d["spatial_sham_bias"],
        "primary_interaction_nats_per_token": primary,
        "secondary_interaction_nats_per_token": secondary}


def _mean_t_interval(values: list[float]) -> dict:
    if len(values) != 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("descriptive_ci_requires_two_scores")
    mean = sum(values) / len(values)
    sample_sd = abs(values[0] - values[1]) / math.sqrt(2.0)
    half = T_CRITICAL_95_DF1 * sample_sd / math.sqrt(2.0)
    return {"n": 2, "df": 1, "mean": mean,
        "ci95_t": [mean - half, mean + half],
        "t_critical_df1": T_CRITICAL_95_DF1}


def build_summary(cohort_raw: bytes, final_preflight_raw: bytes,
                  items_raw: bytes, *, root: Path = ROOT,
                  private_root: Path = PRIVATE,
                  script_root: Path | None = None) -> dict:
    """Load, verify, and reduce the frozen three-row cohort to safe aggregates."""
    try:
        cohort = json.loads(cohort_raw)
        final = json.loads(final_preflight_raw)
        manifest = json.loads(items_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("input_json_invalid") from exc
    cohort_sha = _sha(cohort_raw)
    if (cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"
            or cohort.get("requested_rows") != 3
            or not isinstance(cohort.get("selection"), list)
            or len(cohort["selection"]) != 3
            or final.get("status") != "completed_frozen_cohort_cpu_preflight"
            or final.get("requested_rows") != 3 or final.get("effective_rows") != 3
            or final.get("cohort_sha256") != cohort_sha
            or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("items"), list)
            or len(manifest["items"]) != 3):
        raise ValueError("frozen_three_row_input_binding_invalid")
    selection, entries = cohort["selection"], manifest["items"]
    if ([entry.get("row_index") for entry in entries] != [0, 1, 2]
            or len({item.get("video_id") for item in selection}) != 3
            or len({item.get("row_key_sha256") for item in selection}) != 3):
        raise ValueError("frozen_row_order_or_uniqueness_invalid")
    preflight_rows = final.get("records")
    if not isinstance(preflight_rows, list) or len(preflight_rows) != 3:
        raise ValueError("final_preflight_row_count_invalid")
    if any(record.get("status") != "pass" for record in preflight_rows):
        raise ValueError("final_preflight_not_all_pass")
    if (len({(r.get("video_id"), r.get("row_key_sha256")) for r in preflight_rows})
            != 3):
        raise ValueError("final_preflight_rows_not_unique")
    final_by_key = {(r["video_id"], r["row_key_sha256"]): r for r in preflight_rows}
    common_manifest_sha = final.get("manifest_sha256")
    if not isinstance(common_manifest_sha, str) or len(common_manifest_sha) != 64:
        raise ValueError("official_manifest_hash_missing")
    script_root = script_root or (root / "scripts")
    rows = []
    for row_index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("row_index") != row_index:
            raise ValueError("items_manifest_row_entry_invalid")
        selected = selection[row_index]
        target = (selected.get("video_id"), selected.get("row_key_sha256"))
        record = final_by_key.get(target)
        if record is None:
            raise ValueError("selected_row_missing_from_final_preflight")
        _, _, preflight_sha, _ = _load_preflight_for_row(entry,
            final=final, final_raw=final_preflight_raw, cohort_sha=cohort_sha,
            selected=selected, row_index=row_index, root=root,
            private_root=private_root)
        cpu_path = _resolve_private(entry.get("cpu_receipt"), root=root,
                                    private_root=private_root)
        cpu_raw, cpu = _read_private_json(cpu_path)
        status = entry.get("status")
        if status == "review":
            if set(entry) != {"row_index", "status", "cohort_preflight",
                              "gpu_receipt", "cpu_receipt", "video", "time_map"}:
                if set(entry) != {"row_index", "status", "gpu_receipt",
                                  "cpu_receipt", "video", "time_map"}:
                    raise ValueError("review_entry_fields_invalid")
            gpu_path = _resolve_private(entry.get("gpu_receipt"), root=root,
                                        private_root=private_root)
        elif status == "gpu_failed":
            allowed = {"row_index", "status", "gpu_failure_receipt",
                       "cpu_receipt", "video", "time_map", "cohort_preflight"}
            required = allowed - {"cohort_preflight"}
            if not required.issubset(entry) or set(entry) - allowed:
                raise ValueError("failure_entry_fields_invalid")
            gpu_path = _resolve_private(entry.get("gpu_failure_receipt"), root=root,
                                        private_root=private_root)
        else:
            raise ValueError("unexpected_gpu_attrition_status")
        gpu_raw, gpu = _read_private_json(gpu_path)
        base = _validate_common_row(row_index=row_index, selected=selected,
            record=record, entry=entry, cpu=cpu, cpu_raw=cpu_raw, gpu=gpu,
            gpu_raw=gpu_raw, preflight_sha=preflight_sha, cohort_sha=cohort_sha,
            manifest_sha=common_manifest_sha,
            cohort_source_sha=cohort.get("source_sha256"), final_preflight=final,
            root=root,
            private_root=private_root, script_root=script_root)
        if row_index < 2:
            if (status != "review"
                    or gpu.get("status") != "completed_single_video_generated_object_pixel_exploratory_pilot"
                    or gpu.get("direct_generated_explanation_faithfulness") is not False
                    or gpu.get("human_review_performed") is not False):
                raise ValueError("scored_row_status_or_claim_invalid")
            rows.append(_score_row(gpu, cpu, base))
        else:
            progress = gpu.get("progress")
            if (status != "gpu_failed"
                    or gpu.get("status") != "failed_single_video_generated_object_pixel_exploratory_pilot"
                    or gpu.get("direct_generated_explanation_faithfulness") is not False
                    or gpu.get("human_review_performed") is not False
                    or gpu.get("failure_code") != "generated_target_not_short_single_line"
                    or not isinstance(progress, dict)
                    or progress.get("generation_attempted") is not True
                    or progress.get("generation_completed") is not True
                    or progress.get("teacher_forced_score_attempted") != 0
                    or progress.get("teacher_forced_score_completed") != 0):
                raise ValueError("row_two_structural_attrition_invalid")
            rows.append({**base, "row_index": row_index,
                "status": "structural_generation_attrition",
                "failure_code": "generated_target_not_short_single_line",
                "score_forward_count": 0, "branch_count": 0,
                "hook_calls_total": 0,
                "bbox_key_count": cpu["condition_key_counts"]["bbox"],
                "spatial_sham_key_count": cpu["condition_key_counts"]["spatial_sham"]})
    primary = _mean_t_interval([row["primary_interaction_nats_per_token"]
                                for row in rows[:2]])
    secondary = _mean_t_interval([row["secondary_interaction_nats_per_token"]
                                  for row in rows[:2]])
    if (rows[0]["model_revision_sha256"] != rows[1]["model_revision_sha256"]
            or rows[0]["runner_sha256"] != rows[1]["runner_sha256"]
            or rows[0]["source_bundle_sha256"] != rows[1]["source_bundle_sha256"]
            or rows[0]["annotation_source_sha256"]
                != rows[1]["annotation_source_sha256"]):
        raise ValueError("scored_rows_source_or_model_mismatch")
    return {
        "schema_version": 1,
        "purpose": "privacy_safe_descriptive_summary_of_minerva_visible_object_v2_pilot",
        "status": "verified_exploratory_two_of_three_scored",
        "requested_rows": 3, "effective_preflight_rows": 3,
        "scored_rows": 2, "structural_attrition_rows": 1,
        "input_sha256": {"frozen_cohort": cohort_sha,
            "final_preflight": _sha(final_preflight_raw),
            "items_manifest": _sha(items_raw)},
        "official_manifest_sha256": common_manifest_sha,
        "source_sha256": {"runner_sha256": rows[0]["runner_sha256"],
            "summarizer_sha256": _sha(Path(__file__).read_bytes()),
            "model_revision_sha256": rows[0]["model_revision_sha256"],
            "source_bundle_sha256": rows[0]["source_bundle_sha256"],
            "annotation_sources": rows[0]["annotation_source_sha256"]},
        "rows": rows,
        "primary_interaction_summary_nats_per_token": primary,
        "secondary_interaction_summary_nats_per_token": secondary,
        "estimand": {
            "D": "meanLP(spatial-sham-filled frame) - meanLP(annotation-bbox-filled frame)",
            "primary": "D_bbox_bias - D_spatial_sham_bias",
            "secondary": "D_bbox_bias - D_no_bias",
            "scores": "teacher-forced generated-token mean log probability in nats/token"},
        "limitations": [
            "Descriptive, post-feasibility, three-row exploratory pilot; two scored rows only.",
            "The 95% t intervals use df=1 and are unstable; no confirmatory or population claim is warranted.",
            "Conditional likelihood sensitivity is not human semantic support or direct explanation faithfulness.",
            "Object bbox mean fill can be out of distribution; bbox is not object segmentation.",
            "The row-2 generation-format failure remains structural attrition; no replacement was selected.",
            "Rights status remains unverified; outputs are derived local research summaries."],
        "direct_generated_explanation_faithfulness_measured": False,
        "rights_status": "unverified_local_only"}


def write_unique_json(output: Path, payload: dict) -> None:
    if (output.parent.resolve() != (ROOT / "results").resolve()
            or output.exists() or output.is_symlink()
            or output.suffix.lower() != ".json"):
        raise ValueError("results_output_contract_invalid")
    encoded = (json.dumps(payload, indent=2, sort_keys=True,
        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(encoded)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=Path(
        "data/private_minerva/minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json"))
    parser.add_argument("--final-preflight", type=Path, default=Path(
        "data/private_minerva/minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json"))
    parser.add_argument("--items-manifest", type=Path, default=Path(
        "data/private_minerva/minerva_visible_object_v2_review_items_20261022_01.json"))
    parser.add_argument("--output", type=Path, required=True,
        help="Unique JSON path directly under results/")
    args = parser.parse_args(argv)
    try:
        for path in (args.cohort, args.final_preflight, args.items_manifest):
            if path.is_symlink() or not path.is_file():
                raise ValueError("input_path_invalid")
            if not path.resolve().is_relative_to(PRIVATE.resolve()):
                raise ValueError("input_path_outside_private_root")
        cohort_raw, _ = _read_private_json(args.cohort)
        preflight_raw, _ = _read_private_json(args.final_preflight)
        items_raw, _ = _read_private_json(args.items_manifest)
        payload = build_summary(cohort_raw, preflight_raw, items_raw)
        write_unique_json(args.output, payload)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "summary_failed")
    print(f"検証済み: 3行中2行を採点、1行は生成形式失敗。"
        f"主相互作用の平均 {payload['primary_interaction_summary_nats_per_token']['mean']:+.6f} nat/token、"
        f"記述的95% t区間 [{payload['primary_interaction_summary_nats_per_token']['ci95_t'][0]:+.6f}, "
        f"{payload['primary_interaction_summary_nats_per_token']['ci95_t'][1]:+.6f}]。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
