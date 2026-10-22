#!/usr/bin/env python3
"""Build a private Japanese impression-review artifact for the frozen test cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat

from build_nextgqa_quick_review_artifact import (
    build, browser_compatible_bytes, private_output_path,
)
from run_test_split_internal_bias_frame_mean_interaction import (
    LOCK_REL, PAIR_FILTER_REL, PACKETS, _read_private_json, verify_locked_assignment,
)
from run_saved_claim_frame_permutation_attention_smoke import _read_packet, _sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_nextgqa"


def _write_exclusive(path: Path, raw: bytes, mode: int = 0o600) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), mode)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_artifact(*, output_rel: Path, package_rel: Path, translations_rel: Path) -> dict:
    output = private_output_path(ROOT / output_rel)
    package = private_output_path(ROOT / package_rel)
    translations = private_output_path(ROOT / translations_rel)
    if package.exists():
        raise FileExistsError("private_package_already_exists")
    if output.exists():
        raise FileExistsError("review_artifact_already_exists")
    tr_info = translations.lstat()
    if (not stat.S_ISREG(tr_info.st_mode) or stat.S_ISLNK(tr_info.st_mode)
            or tr_info.st_uid != os.getuid() or stat.S_IMODE(tr_info.st_mode) != 0o600):
        raise ValueError("private_translation_permissions_invalid")
    private_info = PRIVATE.lstat()
    if (stat.S_ISLNK(private_info.st_mode) or not stat.S_ISDIR(private_info.st_mode)
            or private_info.st_uid != os.getuid() or stat.S_IMODE(private_info.st_mode) != 0o700):
        raise ValueError("private_root_permissions_invalid")

    lock, lock_sha = _read_private_json(ROOT / LOCK_REL)
    _pair_filter, filter_sha = _read_private_json(ROOT / PAIR_FILTER_REL)
    rows = []
    expected_filter = lock.get("selected_pair_filter_sha256")
    for ordinal, (label, (packet_rel, record_index, baseline_rel)) in enumerate(PACKETS.items()):
        row, packet_sha = _read_packet(ROOT, packet_rel, record_index, expected_split="test")
        token_hash = _sha256_bytes((row["video_sha256"] + str(row.get("dataset_question_id", ""))).encode())
        verify_locked_assignment(lock, filter_sha, ordinal, row, expected_filter)
        interaction_path = PRIVATE / f"test_split_internal_bias_frame_mean_interaction_{label}_20261021_01.json"
        interaction, _interaction_sha = _read_private_json(interaction_path)
        baseline, baseline_sha = _read_private_json(ROOT / baseline_rel)
        claim = row.get("generated_parsed", {}).get("claim")
        question = row.get("question")
        video_path = row.get("local_video_path")
        if (interaction.get("status") != "completed_one_row_test_split_internal_bias_frame_mean_interaction"
                or interaction.get("row_token_sha256") != token_hash
                or interaction.get("private_input_packet_sha256") != packet_sha
                or interaction.get("baseline_receipt_sha256") != baseline_sha
                or baseline.get("saved_claim_text_sha256") != _sha256_bytes(claim.encode())
                or not isinstance(question, str) or not question.strip()
                or not isinstance(claim, str) or not claim.strip()
                or not isinstance(video_path, str) or not Path(video_path).is_file()):
            raise ValueError("review_source_row_binding_invalid")
        source_bytes = Path(video_path).read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != row["video_sha256"]:
            raise ValueError("review_source_video_digest_mismatch")
        opaque_token = hashlib.sha256((token_hash + "quick-review").encode()).hexdigest()[:24]
        rows.append({"review_token": opaque_token, "question": question, "claim_text": claim,
                     "video_path": Path(video_path), "source_video_sha256": row["video_sha256"],
                     "row_token_sha256": token_hash, "packet_sha256": packet_sha,
                     "baseline_receipt_sha256": baseline_sha})

    package.mkdir(mode=0o700)
    try:
        os.chmod(package, 0o700)
        index_data = {"records": [{key: row[key] for key in ("review_token", "question", "claim_text")}
                                  for row in rows]}
        source = ('<!doctype html><meta charset="utf-8"><script id="data" type="application/json">'
                  + json.dumps(index_data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
                  + "</script>")
        _write_exclusive(package / "index.html", source.encode("utf-8"))
        for row in rows:
            media = browser_compatible_bytes(row["video_path"])
            _write_exclusive(package / (row["review_token"] + ".mp4"), media)
        translations_payload = json.loads(translations.read_text(encoding="utf-8"))
        translation_items = translations_payload.get("items", {})
        for row in rows:
            key = hashlib.sha256((row["question"] + "\0" + row["claim_text"]).encode("utf-8")).hexdigest()
            if key not in translation_items:
                raise ValueError("review_translation_missing_for_locked_row")
        build(package, output, translations=__import__("build_nextgqa_quick_review_artifact").load_translations(translations))
        artifact_sha = hashlib.sha256(output.read_bytes()).hexdigest()
        if output.stat().st_uid != os.getuid() or stat.S_IMODE(output.stat().st_mode) != 0o600:
            raise ValueError("review_artifact_permissions_invalid")
        return {"status": "built_local_exploratory_review_artifact", "item_count": len(rows),
                "artifact": str(output.relative_to(ROOT)), "artifact_sha256": artifact_sha,
                "artifact_bytes": output.stat().st_size, "artifact_mode": "0600",
                "package_dir": str(package.relative_to(ROOT)), "package_mode": "0700",
                "cohort_lock_sha256": lock_sha,
                "items": [{"row_ordinal": i, "row_token_sha256": row["row_token_sha256"],
                           "packet_sha256": row["packet_sha256"],
                           "baseline_receipt_sha256": row["baseline_receipt_sha256"]}
                          for i, row in enumerate(rows)],
                "purpose": "informal clarity, answer-fit and evidence-sufficiency impressions only",
                "direct_faithfulness": False, "rights_status": "unverified; local-only"}
    except BaseException:
        import shutil
        shutil.rmtree(package, ignore_errors=True)
        if output.exists():
            output.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="relative path under data/private_nextgqa")
    parser.add_argument("--package-dir", type=Path, required=True, help="new relative directory under data/private_nextgqa")
    parser.add_argument("--translations", type=Path, required=True, help="private translation JSON relative to repository root")
    args = parser.parse_args()
    if (args.output.is_absolute() or args.package_dir.is_absolute() or args.translations.is_absolute()
            or not args.output.is_relative_to(Path("data/private_nextgqa"))
            or not args.package_dir.is_relative_to(Path("data/private_nextgqa"))
            or not args.translations.is_relative_to(Path("data/private_nextgqa"))):
        parser.error("all paths must remain under data/private_nextgqa")
    print(json.dumps(build_artifact(output_rel=args.output, package_rel=args.package_dir,
                                    translations_rel=args.translations), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
