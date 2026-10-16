"""CPU-only adapter from local NExT-GQA CSV rows to the frozen QA target.

Private, unvalidated source binding only. This module does not authorize scoring,
load media/models, or verify the other source artifacts in a lineage manifest.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata

SHA256 = re.compile(r"^[a-f0-9]{64}$")
REQUIRED_CSV = {"video_id", "qid", "question", "answer", *(f"a{i}" for i in range(5))}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


def _strict_json(raw: bytes) -> dict:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("nonfinite_json")))
    except (AttributeError, UnicodeError, json.JSONDecodeError):
        raise ValueError("input_decode_invalid") from None
    return value


def build_actual_qa_source(*, csv_raw: bytes, selection: dict,
                           target_manifest_raw: bytes,
                           lineage_manifest_raw: bytes) -> tuple[dict, dict]:
    """Return QA source rows and target manifest bound to exact selected CSV rows.

    `selection.records` maps target video/claim IDs to dataset video/QID pairs.
    Source row hashes cover all columns in the parsed CSV row, preserving choice
    order and source metadata. The output intentionally remains unvalidated.
    """
    if not isinstance(csv_raw, bytes) or len(csv_raw) > 20_000_000:
        raise ValueError("csv_bytes_invalid")
    if not isinstance(selection, dict) or set(selection) != {"records"}:
        raise ValueError("selection_schema_invalid")
    records = selection["records"]
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("selection_count_invalid")
    lineage = _strict_json(lineage_manifest_raw)
    target = _strict_json(target_manifest_raw)
    try:
        text = csv_raw.decode("utf-8-sig")
    except UnicodeError:
        raise ValueError("input_decode_invalid") from None
    lineage_keys = {"schema_version", "purpose", "created_by", "source_packet_sha256",
                    "cohort_sha256", "dataset_inputs_sha256", "requested_records",
                    "effective_records", "verification_scope", "status", "records"}
    lineage_row_keys = {"video_id", "claim_id", "dataset_question_id", "dataset_video_id",
                        "video_key", "relative_video_path", "video_sha256",
                        "question_sha256", "claim_text_sha256"}
    if (not isinstance(lineage, dict) or set(lineage) != lineage_keys
            or type(lineage.get("schema_version")) is not int or lineage["schema_version"] != 1
            or lineage.get("purpose") != "private_source_lineage_crosscheck_for_claim_review"
            or lineage.get("status") != "metadata_lineage_crosschecked"
            or lineage.get("requested_records") != 3 or lineage.get("effective_records") != 3
            or not isinstance(lineage.get("records"), list) or len(lineage["records"]) != 3
            or any(not isinstance(row, dict) or set(row) != lineage_row_keys
                   for row in lineage["records"])):
        raise ValueError("lineage_schema_invalid")
    lineage_by_video = {r["video_id"]: r for r in lineage["records"]
                        if isinstance(r.get("video_id"), str)}
    if len(lineage_by_video) != 3:
        raise ValueError("lineage_cohort_invalid")
    lineage_order = [row["video_id"] for row in lineage["records"]]
    if (not isinstance(lineage.get("dataset_inputs_sha256"), dict)
            or lineage["dataset_inputs_sha256"].get("test.csv") != _sha(csv_raw)):
        raise ValueError("lineage_test_csv_hash_mismatch")
    if not isinstance(target, dict) or not isinstance(target.get("records"), list):
        raise ValueError("target_manifest_invalid")
    if len(target["records"]) != 3:
        raise ValueError("target_rows_invalid")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if (reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames))
            or not REQUIRED_CSV.issubset(reader.fieldnames)):
        raise ValueError("csv_columns_invalid")
    by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in reader:
        if (None in row or any(value is None for value in row.values())
                or any(not isinstance(value, str) for value in row.values())):
            raise ValueError("csv_row_width_invalid")
        pair = (str(row.get("video_id", "")), str(row.get("qid", "")))
        by_pair.setdefault(pair, []).append(row)
    if any(not isinstance(r, dict) for r in target["records"]):
        raise ValueError("target_rows_invalid")
    target_by_video = {r.get("video_id"): r for r in target["records"]}
    if (len(target_by_video) != 3
            or any(not isinstance(video_id, str) or not video_id for video_id in target_by_video)):
        raise ValueError("target_rows_invalid")
    seen_target: set[str] = set()
    seen_dataset_pairs: set[tuple[str, str]] = set()
    qa_rows, bound_rows = [], []
    for item in records:
        keys = {"video_id", "claim_id", "dataset_video_id", "dataset_question_id"}
        if not isinstance(item, dict) or set(item) != keys:
            raise ValueError("selection_row_invalid")
        video_id, claim_id = item["video_id"], item["claim_id"]
        dataset_video_id, qid = item["dataset_video_id"], item["dataset_question_id"]
        if (any(not isinstance(value, str) or not value.strip()
                for value in (video_id, claim_id, dataset_video_id, qid))
                or any(value != value.strip()
                       for value in (video_id, claim_id, dataset_video_id, qid))
                or video_id in seen_target or (dataset_video_id, qid) in seen_dataset_pairs):
            raise ValueError("selection_identity_invalid")
        lineage_row = lineage_by_video.get(video_id)
        if (lineage_row is None or lineage_row.get("claim_id") != claim_id
                or lineage_row.get("dataset_video_id") != dataset_video_id
                or lineage_row.get("dataset_question_id") != qid):
            raise ValueError("selection_lineage_binding_mismatch")
        seen_target.add(video_id)
        seen_dataset_pairs.add((dataset_video_id, qid))
        matches = by_pair.get((dataset_video_id, qid), [])
        if len(matches) != 1:
            raise ValueError("dataset_video_qid_not_unique")
        source = matches[0]
        question = source["question"]
        if hashlib.sha256(question.encode("utf-8")).hexdigest() != lineage_row["question_sha256"]:
            raise ValueError("source_question_lineage_mismatch")
        choices = [source[f"a{i}"].strip() for i in range(5)]
        answer = source["answer"]
        if not question.strip() or any(not c.strip() for c in choices):
            raise ValueError("dataset_text_empty")
        gold = [i for i, choice in enumerate(choices) if _norm(choice) == _norm(answer)]
        if len(gold) != 1:
            raise ValueError("dataset_gold_ambiguous_or_missing")
        qa = {"video_id": video_id, "dataset_video_id": dataset_video_id,
              "dataset_question_id": qid, "question": question,
              "choices": choices, "answer": answer}
        row_hash = _sha(_canonical(source))
        row = target_by_video.get(video_id)
        if not isinstance(row, dict) or row.get("claim_id") != claim_id:
            raise ValueError("target_claim_binding_mismatch")
        row.update({
            "dataset_qid_sha256": _sha(qid.encode("utf-8")),
            "dataset_source_row_sha256": row_hash,
            "question_sha256": _sha(question.encode("utf-8")),
            "ordered_choice_sha256": [_sha(c.encode("utf-8")) for c in choices],
            "gold_index": gold[0],
        })
        qa_rows.append(qa)
        bound_rows.append({"video_id": video_id, "dataset_video_id": dataset_video_id,
                           "dataset_question_id": qid, "source_row_sha256": row_hash,
                           "question_sha256": row["question_sha256"],
                           "ordered_choice_sha256": row["ordered_choice_sha256"],
                           "gold_index": gold[0]})
    if seen_target != set(target_by_video):
        raise ValueError("target_cohort_mismatch")
    if ([item["video_id"] for item in records] != lineage_order
            or [row["video_id"] for row in target["records"]] != lineage_order):
        raise ValueError("frozen_target_order_mismatch")
    source_doc = {"schema_version": 1,
                  "purpose": "private_actual_nextgqa_qa_source_rows_unvalidated",
                  "dataset_inputs_sha256": lineage["dataset_inputs_sha256"],
                  "records": qa_rows}
    # This aggregate is safe to inspect only as a private artifact: hashes can
    # still link back to benchmark records.
    target["dataset_source_manifest_sha256"] = _sha(_canonical(lineage["dataset_inputs_sha256"]))
    manifest = {"schema_version": 1,
                "purpose": "private_actual_nextgqa_source_binding_unvalidated",
                "source_csv_sha256": _sha(csv_raw),
                "source_row_bindings": bound_rows,
                "target_manifest": target}
    return source_doc, manifest
