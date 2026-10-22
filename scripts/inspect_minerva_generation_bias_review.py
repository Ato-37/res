#!/usr/bin/env python3
"""Read-only integrity and aggregate inspector for the private Minerva review export."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
DEFAULT_BUILDER = ROOT / "scripts" / "build_minerva_generation_bias_review.py"
VERDICTS = ("supports", "partially_supports", "contradicts", "not_visible", "cannot_judge")
BRANCHES = ("baseline", "bbox_key_bias", "spatial_sham_key_bias")
EXPECTED_ITEMS = 5
EXPECTED_FRAMES = 64
FINGERPRINT_RE = re.compile(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">')
BUILDER_RE = re.compile(rb'<meta name="review-builder-sha256" content="([0-9a-f]{64})">')
SOURCE_RE = re.compile(rb'<meta name="review-source-manifest-sha256" content="([0-9a-f]{64})">')
PAYLOAD_RE = re.compile(rb"const payload=(.*?);const \$=id=>", re.DOTALL)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _one(regex: re.Pattern[bytes], raw: bytes, error: str) -> str:
    matches = regex.findall(raw)
    if len(matches) != 1:
        raise ValueError(error)
    return matches[0].decode("ascii")


def _page_payload(page: bytes) -> tuple[dict, str, str, str]:
    fingerprint = _one(FINGERPRINT_RE, page, "page_fingerprint_slot_invalid")
    builder_hash = _one(BUILDER_RE, page, "page_builder_hash_missing")
    source_hash = _one(SOURCE_RE, page, "page_source_manifest_hash_missing")
    canonical = FINGERPRINT_RE.sub(
        b'<meta name="review-artifact-sha256" content="__PAGE_SHA__">', page)
    if sha(canonical) != fingerprint:
        raise ValueError("page_fingerprint_mismatch")
    match = PAYLOAD_RE.search(page)
    if not match:
        raise ValueError("page_payload_missing")
    try:
        payload = json.loads(match.group(1).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("page_payload_invalid") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list) \
            or not isinstance(payload.get("provenance"), dict):
        raise ValueError("page_payload_shape_invalid")
    return payload, fingerprint, builder_hash, source_hash


def _verify_manifest(provenance: dict, embedded_source_hash: str) -> None:
    if provenance.get("source_manifest_sha256") != embedded_source_hash:
        raise ValueError("page_source_manifest_hash_binding_invalid")
    material = dict(provenance)
    claimed = material.pop("source_manifest_sha256", None)
    digest = sha(json.dumps(material, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode("utf-8"))
    if digest != claimed:
        raise ValueError("source_manifest_content_hash_mismatch")


def _validate_items(items: list) -> dict[str, dict]:
    if len(items) != EXPECTED_ITEMS:
        raise ValueError("page_item_count_invalid")
    result = {}
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("item_id"), str)
                or not item["item_id"] or item["item_id"] in result
                or not isinstance(item.get("claim"), str)
                or not isinstance(item.get("question"), str)
                or not isinstance(item.get("frames"), list)
                or len(item["frames"]) != EXPECTED_FRAMES
                or not isinstance(item.get("source"), dict)):
            raise ValueError("page_item_invalid")
        source = item["source"]
        if (source.get("claim_sha256") != sha(item["claim"].encode("utf-8"))
                or source.get("question_sha256") != sha(item["question"].encode("utf-8"))):
            raise ValueError("page_text_hash_mismatch")
        indices, points, image_hashes = [], [], []
        for rank, frame in enumerate(item["frames"]):
            if (not isinstance(frame, dict) or frame.get("rank") != rank
                    or type(frame.get("source_frame_index")) is not int
                    or frame["source_frame_index"] < 0
                    or not isinstance(frame.get("time_seconds"), (int, float))
                    or isinstance(frame.get("time_seconds"), bool)
                    or not isinstance(frame.get("data_url"), str)
                    or not frame["data_url"].startswith("data:image/jpeg;base64,")):
                raise ValueError("page_frame_invalid")
            import base64
            try:
                raw = base64.b64decode(frame["data_url"].split(",", 1)[1], validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("page_frame_image_invalid") from exc
            indices.append(frame["source_frame_index"])
            points.append(frame["time_seconds"])
            image_hashes.append(sha(raw))
        if (indices != source.get("sample_frame_indices")
                or points != source.get("frame_pts_seconds")):
            raise ValueError("page_frame_source_binding_mismatch")
        if (source.get("frame_image_sha256") != image_hashes
                or source.get("frame_image_inventory_sha256") != sha(json.dumps(
                    image_hashes, separators=(",", ":")).encode())):
            raise ValueError("page_frame_hash_mismatch")
        result[item["item_id"]] = item
    return result


def _validate_mapping(mapping: dict, items: dict[str, dict], *, fingerprint: str,
                      builder_hash: str, source_hash: str) -> dict[str, dict]:
    if (mapping.get("purpose") != "private_blinded_review_item_to_generation_branch_map"
            or mapping.get("status") != "private_mapping_do_not_share_with_reviewers"
            or mapping.get("artifact_fingerprint") != fingerprint
            or mapping.get("builder_sha256") != builder_hash
            or mapping.get("source_manifest_sha256") != source_hash):
        raise ValueError("private_mapping_page_binding_invalid")
    mapped = mapping.get("items")
    if not isinstance(mapped, list) or len(mapped) != EXPECTED_ITEMS:
        raise ValueError("private_mapping_items_invalid")
    by_id = {}
    row_branch_counts = {i: {branch: 0 for branch in BRANCHES} for i in range(3)}
    row_items = {i: 0 for i in range(3)}
    for entry in mapped:
        if not isinstance(entry, dict):
            raise ValueError("private_mapping_entry_invalid")
        item_id = entry.get("item_id")
        row = entry.get("row_index")
        branches = entry.get("branches")
        item = items.get(item_id)
        if (item is None or item_id in by_id or type(row) is not int or row not in row_items
                or not isinstance(branches, list) or not branches
                or len(set(branches)) != len(branches)
                or any(branch not in BRANCHES for branch in branches)):
            raise ValueError("private_mapping_entry_invalid")
        source = item["source"]
        if (entry.get("claim_sha256") != source.get("claim_sha256")
                or entry.get("row_key_sha256") != source.get("row_key_sha256")
                or entry.get("cpu_receipt_sha256") != source.get("cpu_receipt_sha256")
                or entry.get("gpu_receipt_sha256") != source.get("gpu_receipt_sha256")
                or entry.get("cohort_preflight_sha256") != source.get("cohort_preflight_sha256")
                or entry.get("generated_token_ids_sha256") != source.get("generated_token_ids_sha256")):
            raise ValueError("private_mapping_item_binding_invalid")
        by_id[item_id] = entry
        row_items[row] += 1
        for branch in branches:
            row_branch_counts[row][branch] += 1
    if set(by_id) != set(items):
        raise ValueError("private_mapping_page_item_set_mismatch")
    if sorted(row_items.values()) != [1, 2, 2] or any(
            any(count != 1 for count in branch_counts.values())
            for branch_counts in row_branch_counts.values()):
        raise ValueError("private_mapping_branch_coverage_invalid")
    return by_id


def _verify_mapping_receipts(mapping_by_id: dict[str, dict], items: dict[str, dict],
                             gpu_receipts: list[tuple[bytes, dict]]) -> None:
    """Bind every blinded condition label to the immutable generation receipt."""
    if len(gpu_receipts) != 3:
        raise ValueError("three_gpu_receipts_required")
    receipts = {}
    for row, (raw, receipt) in enumerate(gpu_receipts):
        outcomes = receipt.get("outcomes") if isinstance(receipt, dict) else None
        if not isinstance(outcomes, dict) or set(outcomes) != set(BRANCHES):
            raise ValueError("gpu_receipt_branches_invalid")
        receipts[row] = (sha(raw), outcomes)
    for item_id, entry in mapping_by_id.items():
        receipt_sha, outcomes = receipts[entry["row_index"]]
        source = items[item_id]["source"]
        if receipt_sha != entry["gpu_receipt_sha256"]:
            raise ValueError("mapping_gpu_receipt_hash_mismatch")
        actual_branches = sorted(branch for branch, outcome in outcomes.items()
            if isinstance(outcome, dict)
            and outcome.get("generated_text_sha256") == source["claim_sha256"]
            and outcome.get("generated_token_ids_sha256")
                == source["generated_token_ids_sha256"])
        if sorted(entry["branches"]) != actual_branches:
            raise ValueError("mapping_gpu_receipt_branch_mismatch")


def inspect_bytes(*, page: bytes, mapping: dict, response: dict,
                   builder_bytes: bytes,
                   gpu_receipts: list[tuple[bytes, dict]] | None = None) -> dict:
    payload, fingerprint, embedded_builder, embedded_source = _page_payload(page)
    if sha(builder_bytes) != embedded_builder:
        raise ValueError("builder_hash_mismatch")
    provenance = payload["provenance"]
    if provenance.get("builder_sha256") != embedded_builder:
        raise ValueError("payload_builder_hash_mismatch")
    _verify_manifest(provenance, embedded_source)
    items = _validate_items(payload["items"])
    mapping_by_id = _validate_mapping(mapping, items, fingerprint=fingerprint,
        builder_hash=embedded_builder, source_hash=embedded_source)
    if gpu_receipts is not None:
        _verify_mapping_receipts(mapping_by_id, items, gpu_receipts)
    if (response.get("schema_version") != 1
            or response.get("purpose") != "private_blinded_minerva_generation_support_review"
            or response.get("status") != "informal_frame_grounded_verdicts_only"
            or response.get("artifact_fingerprint") != fingerprint
            or response.get("builder_sha256") != embedded_builder
            or response.get("source_manifest_sha256") != embedded_source):
        raise ValueError("response_page_binding_invalid")
    respondents = response.get("respondents")
    if not isinstance(respondents, list) or not respondents:
        raise ValueError("response_respondents_missing")
    nicknames = set()
    verdict_counts = {branch: {verdict: 0 for verdict in VERDICTS} for branch in BRANCHES}
    respondent_count = 0
    for respondent in respondents:
        if (not isinstance(respondent, dict) or not isinstance(respondent.get("nickname"), str)
                or not respondent["nickname"].strip()):
            raise ValueError("response_nickname_invalid")
        nicknames.add(respondent["nickname"].casefold())
        answers = respondent.get("answers")
        if not isinstance(answers, list) or len(answers) != EXPECTED_ITEMS:
            raise ValueError("response_incomplete_five_item_review")
        answer_by_id = {}
        for answer in answers:
            if not isinstance(answer, dict):
                raise ValueError("response_answer_invalid")
            item_id = answer.get("item_id")
            item = items.get(item_id)
            if (item is None or item_id in answer_by_id
                    or answer.get("verdict") not in VERDICTS):
                raise ValueError("response_answer_item_or_verdict_invalid")
            source = item["source"]
            expected = {"claim_sha256": source["claim_sha256"],
                "question_sha256": source["question_sha256"],
                "frame_image_inventory_sha256": source["frame_image_inventory_sha256"],
                "frame_image_sha256": source["frame_image_sha256"],
                "source_frame_indices": source["sample_frame_indices"],
                "frame_pts_seconds": source["frame_pts_seconds"]}
            if any(answer.get(key) != value for key, value in expected.items()):
                raise ValueError("response_answer_source_binding_mismatch")
            answer_by_id[item_id] = answer
        if set(answer_by_id) != set(items):
            raise ValueError("response_answer_item_set_mismatch")
        for item_id, answer in answer_by_id.items():
            for branch in mapping_by_id[item_id]["branches"]:
                verdict_counts[branch][answer["verdict"]] += 1
        respondent_count += 1
    multi_branch_items = sum(1 for entry in mapping_by_id.values()
        if len(entry["branches"]) > 1)
    branch_totals = {branch: sum(counts.values()) for branch, counts in verdict_counts.items()}
    return {"status": "valid_private_review_export",
        "respondents": respondent_count,
        "unique_nickname_count": len(nicknames),
        "distinct_claim_items": len(items),
        "items_with_multiple_generating_branches": multi_branch_items,
        "branch_verdict_counts": verdict_counts,
        "branch_review_totals": branch_totals,
        "interpretation": "Exploratory human judgments about support in sampled frames; not direct model faithfulness evidence."}


def _read_0600_json(path: Path) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 50_000_000):
        raise ValueError("private_input_permissions_or_size_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("private_input_changed_during_read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_input_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("private_input_json_shape_invalid")
    return raw, value


def _read_private_page(path: Path) -> bytes:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 100_000_000):
        raise ValueError("private_page_permissions_or_size_invalid")
    return path.read_bytes()


def _read_response_json(path: Path) -> dict:
    """Read a local browser download without requiring a private-dir copy."""
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_size > 50_000_000):
        raise ValueError("response_file_contract_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("response_file_changed_during_read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("response_json_shape_invalid")
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--builder", type=Path, default=DEFAULT_BUILDER)
    parser.add_argument("--gpu-receipt", type=Path, nargs=3, required=True,
        help="Three private generation receipts in row order")
    args = parser.parse_args(argv)
    try:
        for path in (args.page, args.mapping):
            if path.is_symlink() or not path.resolve().is_relative_to(PRIVATE.resolve()):
                raise ValueError("private_input_path_invalid")
        page = _read_private_page(args.page)
        _, mapping = _read_0600_json(args.mapping)
        gpu_receipts = [_read_0600_json(path) for path in args.gpu_receipt]
        response = _read_response_json(args.response)
        if args.builder.is_symlink() or not args.builder.is_file():
            raise ValueError("builder_source_path_invalid")
        result = inspect_bytes(page=page, mapping=mapping, response=response,
            builder_bytes=args.builder.read_bytes(), gpu_receipts=gpu_receipts)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "review_inspection_failed")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
