"""Create a private, exclusive byte-pinning receipt for v2 proxy inputs.

This owner-local freeze is an integrity convenience, not a signature, human
review, or authorization to run direct claim-linked faithfulness scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from build_nextgqa_exploratory_qta_receipt import build_receipt, sha256_file
from freeze_nextgqa_exploratory_v1_local import read_and_verify_freeze

PLAN_NAME = "nextgqa_tlvs_exploratory_split_v2_20261018_05.json"
PREFLIGHT_NAME = "nextgqa_tlvs_exploratory_qta_preflight_receipt_v2_20261018_06.json"
GATE_NAME = "nextgqa_tlvs_qta_v2_run_input_freeze_local_20261018.json"
SOURCE_FILES = (
    "scripts/run_nextgqa_qta_exploratory_proxy_v2.py",
    "scripts/run_nextgqa_qta_v3_geometry_census.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/audit_qwen_checkpoint_bytes_cpu.py",
    "scripts/build_nextgqa_exploratory_video_split.py",
    "scripts/fixed_prediction_selective_dependence.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/nextgqa_atomic_batch_freeze.py",
    "scripts/presentation_time_utils.py",
    "scripts/qwen_internal_attention_ablation.py",
    "scripts/qwen_visual_attention_boost.py",
    "scripts/run_nextgqa_qta_v4_union_geometry_census.py",
    "scripts/run_nextgqa_qta_v4_union_scorer.py",
    "experiments/2026-10-18_qta_v4_union_treatment.md",
    "scripts/fixed_annotation_union_dependence.py",
    "scripts/run_nextgqa_gold_interval_margin_pilot.py",
    "scripts/claim_mask_video_decoder.py",
    "scripts/necessity_choice_margin_utils.py",
    "scripts/presentation_time_utils.py",
    "scripts/qwen_visual_attention_boost.py",
    "scripts/qwen_review_interval_token_groups.py",
    "scripts/audit_qwen_checkpoint_bytes_cpu.py",
    "scripts/build_nextgqa_exploratory_video_split.py",
    "docs/qwen_question_attention_faithfulness_protocol_v1_draft.md",
    "experiments/2026-10-18_qta_treatment_selection.md",
    "experiments/2026-10-18_qta_exploratory_qid_valid_preflight_v2.md",
    "scripts/build_nextgqa_exploratory_video_split.py",
    "scripts/build_nextgqa_exploratory_video_split_v2.py",
    "scripts/build_nextgqa_exploratory_qta_receipt.py",
    "scripts/freeze_nextgqa_exploratory_v1_local.py",
    "scripts/freeze_nextgqa_qta_v2_inputs_local.py",
    "scripts/run_nextgqa_validation_qta_annotation_proxy.py",
    "scripts/run_nextgqa_qta_exploratory_proxy_v2.py",
    "scripts/fixed_prediction_selective_dependence.py",
    "scripts/qwen_question_temporal_attention_selector.py",
    "scripts/qwen_internal_attention_ablation.py",
)
DATA_FILES = (
    "NExT-GQA/datasets/nextgqa/test.csv",
    "NExT-GQA/datasets/nextgqa/gsub_test.json",
    "NExT-GQA/datasets/nextgqa/frame2time_test.json",
    "NExT-GQA/datasets/nextgqa/map_vid_vidorID.json",
)


def read_private(path: Path, root: Path) -> bytes:
    real_root = root.resolve(strict=True)
    if stat.S_IMODE(real_root.stat().st_mode) != 0o700 or path.resolve(strict=True).parent != real_root:
        raise ValueError("private_input_root_invalid")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o600:
            raise ValueError("private_input_mode_invalid")
        pieces = []
        while block := os.read(fd, 1024 * 1024): pieces.append(block)
        return b"".join(pieces)
    finally:
        os.close(fd)


def build_gate(repo: Path) -> dict[str, Any]:
    private = repo / "data/private_nextgqa"
    freeze = read_and_verify_freeze(repo)
    plan_path, preflight_path = private / PLAN_NAME, private / PREFLIGHT_NAME
    plan_raw = read_private(plan_path, private)
    preflight_raw = read_private(preflight_path, private)
    plan, preflight = json.loads(plan_raw), json.loads(preflight_raw)
    if (sha256_file(private / "nextgqa_tlvs_exploratory_split_20261018_01.json")
            != freeze["plan_sha256"]):
        raise ValueError("v1_freeze_plan_changed")
    data = repo / "NExT-GQA/datasets/nextgqa"
    rebuilt = build_receipt(test_csv=data / "test.csv", gsub_json=data / "gsub_test.json",
        frame2time_json=data / "frame2time_test.json", mapping_json=data / "map_vid_vidorID.json",
        split_plan=plan_path, splitter_source=repo / "scripts/build_nextgqa_exploratory_video_split_v2.py",
        base_plan=private / "nextgqa_tlvs_exploratory_split_20261018_01.json",
        v1_splitter=repo / "scripts/build_nextgqa_exploratory_video_split.py",
        expected_v1_plan_sha256=freeze["plan_sha256"],
        v1_freeze_receipt_sha256=sha256_file(private / "nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"))
    if (rebuilt != preflight or plan.get("status") != "test_associated_exploratory_qid_valid_plan_not_evaluation"
            or len(plan.get("calibration", [])) != 343 or len(plan.get("evaluation", [])) != 147
            or preflight.get("counts", {}).get("invalid_annotation_intervals") != 0
            or preflight.get("counts", {}).get("evaluation_rows_bound") != 147):
        raise ValueError("v2_preflight_not_fully_reconstructed")
    return {"schema_version": 1,
        "status": "local_v2_inputs_frozen_integrity_only_not_authorization",
        "plan_sha256": hashlib.sha256(plan_raw).hexdigest(),
        "preflight_receipt_sha256": hashlib.sha256(preflight_raw).hexdigest(),
        "v1_freeze_receipt_sha256": sha256_file(private / "nextgqa_tlvs_exploratory_split_v1_freeze_local_20261018.json"),
        "v1_plan_sha256": freeze["plan_sha256"],
        "source_sha256": {name: sha256_file(repo / name) for name in SOURCE_FILES},
        "data_sha256": {name: sha256_file(repo / name) for name in DATA_FILES},
        "counts": {"calibration_videos": 343, "evaluation_videos": 147,
                   "evaluation_rows_bound": 147, "invalid_intervals": 0},
        "trust_scope": "mode-0600 owner-local integrity anchor; not a signature, human review, rights decision, or experiment authorization"}


def write_exclusive(path: Path, payload: dict, private: Path) -> None:
    root = private.resolve(strict=True)
    if path.parent.resolve(strict=True) != root or stat.S_IMODE(root.stat().st_mode) != 0o700:
        raise ValueError("private_output_root_invalid")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as f:
            json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-name", default=GATE_NAME)
    args = parser.parse_args(argv); repo = args.repo.resolve(); private = repo / "data/private_nextgqa"
    if Path(args.output_name).name != args.output_name:
        parser.error("output_name_must_be_basename")
    try:
        gate = build_gate(repo); write_exclusive(private / args.output_name, gate, private)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"local_freeze_failed:{type(exc).__name__}:{exc}")
    print(json.dumps({"status": gate["status"], "counts": gate["counts"],
        "privacy": "local-only; no cohort hashes in stdout"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
