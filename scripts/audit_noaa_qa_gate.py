"""Audit the private NOAA QA pilot before any QA inference.

This is a counts-only gate. It verifies three canonical video/sample/gold/audit
groups, keeps QA labels out of inference inputs, and requires an optional human
attestation before returning a go decision. It does not authorize or measure
faithfulness or QA efficacy.

Optional attestation JSON (kept private) schema, version 1::

    {
      "schema_version": 1,
      "signed": true,
      "signed_at_utc": "2026-10-12T12:00:00Z",
      "attestor": {"name": "Human name", "role": "annotator", "human": true},
      "groups": [{
        "group_id": "dive03",
        "media_sha256": "...", "samples_sha256": "...",
        "gold_sha256": "...", "audit_sha256": "...",
        "rights_sha256": "...", "reviewed_qids": ["Q001"],
        "checks": {
          "answer_checked_against_video": true,
          "evidence_supports_answer": true,
          "unique_best_choice": true,
          "stem_leakage_checked": true,
          "video_only_review": true,
          "no_audio_used": true,
          "rights_scope_confirmed": true,
          "credit_acknowledged": true
        },
        "rights_scope": {
          "video_only_model_inference": true,
          "audio_use": false,
          "external_media_publication": false,
          "required_credit": "NOAA Ocean Exploration"
        }
      }]
    }

The attestation is an accountable human record, not a cryptographic signature or
a legal opinion. The name and reviewed QIDs remain local; neither is included in
the summary output. It must be created by a human who actually checked every
listed answer and evidence interval against the video. An AI blind reviewer does
not satisfy this gate.

Usage::

    .venv/bin/python scripts/audit_noaa_qa_gate.py
    .venv/bin/python scripts/audit_noaa_qa_gate.py --attestation /private/path/attestation.json --require-go
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


GROUPS = (
    {
        "group_id": "dive03",
        "directory": ".",
        "audit_file": "annotation_audit_v2.json",
        "media_sha256": "13ab1d0e12d9c6b9e776890d123f363c12c572d52d1f58464b57a2e98667f126",
        "rights_sha256": "ef8e7c4f6d3081531b018ddab6e0ef7fc2dd1bb04a06abae55e68d8f666f97aa",
    },
    {
        "group_id": "dive04",
        "directory": "dive04",
        "audit_file": "annotation_audit_v2.json",
        "media_sha256": "0d68c44118067f640fa41b30c482170359d9846f597d73c6e3d573fe0c37fb03",
        "rights_sha256": "4202474c0d2a55294ad3eb243931ccf58bb3f60dc418a875e74a29e3463bb278",
    },
    {
        "group_id": "dive09",
        "directory": "dive09",
        "audit_file": "annotation_audit.json",
        "media_sha256": "59779cb92fd3bd0a66c2b4f5ef92e6915f32c80a9105d035c95a060559164676",
        "rights_sha256": "49c13de8c2b4c275d678b7aa5a60ce7af536825ec202d5fc7f5cd9f2528a5405",
    },
)

REQUIRED_CHECKS = (
    "answer_checked_against_video",
    "evidence_supports_answer",
    "unique_best_choice",
    "stem_leakage_checked",
    "video_only_review",
    "no_audio_used",
    "rights_scope_confirmed",
    "credit_acknowledged",
)
FORBIDDEN_SAMPLE_KEYS = {
    "answer", "answer_choice_index", "answer_id", "correct", "correct_answer",
    "correct_choice", "correct_choice_id", "evidence", "gold", "grounding",
    "label", "labels", "rationale", "target", "answer_index", "gold_answer",
    "ground_truth", "label_id", "is_correct",
}
ALLOWED_SAMPLE_KEYS = {
    "choice_order_original_indices", "choices", "qid", "question", "video",
    "video_id", "video_sha256", "window_end_seconds", "window_start_seconds",
}
ALLOWED_CHOICE_KEYS = {"id", "text"}
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class AuditFailure(Exception):
    """Input has failed a structural or provenance check."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditFailure("missing_source_file") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditFailure("invalid_json") from exc


def _as_records(value: Any, *, key: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = value.get(key)
    else:
        records = None
    if not isinstance(records, list) or any(not isinstance(x, dict) for x in records):
        raise AuditFailure("invalid_record_container")
    return records


def _qid_set(records: list[dict[str, Any]]) -> set[str]:
    values = [record.get("qid") for record in records]
    if any(not isinstance(value, (str, int)) or isinstance(value, bool) for value in values):
        raise AuditFailure("invalid_qid")
    qids = [str(value) for value in values]
    if len(set(qids)) != len(qids):
        raise AuditFailure("duplicate_qid")
    return set(qids)


def _has_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = _normalize_key(key)
            forbidden_prefixes = (
                "answer_", "correct_", "gold_", "label_", "evidence_",
                "grounding_", "rationale_",
            )
            if normalized in FORBIDDEN_SAMPLE_KEYS or normalized.startswith(forbidden_prefixes):
                return True
            if _has_forbidden_key(child):
                return True
    elif isinstance(value, list):
        return any(_has_forbidden_key(item) for item in value)
    return False


def _normalize_key(key: Any) -> str:
    """Normalize punctuation and camelCase so aliases cannot evade leak checks."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(key))
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _is_explicit_ai_role(value: Any) -> bool:
    """Recognize an explicit AI token or full phrase, not incidental substrings."""
    text = str(value).lower()
    tokens = re.findall(r"[a-z]+", text)
    return "ai" in tokens or "artificial intelligence" in text


def _audit_group(
    data_root: Path,
    group: dict[str, str],
    expected_media: str,
    expected_rights: str,
) -> tuple[dict[str, Any], set[str]]:
    group_root = data_root / group["directory"]
    private_root = group_root / "private"
    media_path = group_root / "media" / "source.mp4"
    sample_path = private_root / "samples.json"
    gold_path = private_root / "gold.json"
    audit_path = private_root / group["audit_file"]
    rights_path = private_root / "rights_review_v2.json"

    if not all(path.is_file() for path in (media_path, sample_path, gold_path, audit_path, rights_path)):
        raise AuditFailure("missing_source_file")

    actual_media_hash = sha256_file(media_path)
    rights_hash = sha256_file(rights_path)
    sample_hash = sha256_file(sample_path)
    gold_hash = sha256_file(gold_path)
    audit_hash = sha256_file(audit_path)
    if actual_media_hash != expected_media:
        raise AuditFailure("media_hash_mismatch")
    if rights_hash != expected_rights:
        raise AuditFailure("rights_hash_mismatch")

    rights = read_json(rights_path)
    if not isinstance(rights, dict) or rights.get("media_sha256") != expected_media:
        raise AuditFailure("rights_media_hash_mismatch")
    gate_status = rights.get("human_verified_manifest_status", "pending")
    if not isinstance(gate_status, str) or not gate_status.lower().startswith("pending"):
        # Any source edit to the pending flag is not itself an attestation.
        raise AuditFailure("rights_gate_status_changed")
    if rights.get("audio_used") is not False:
        raise AuditFailure("audio_scope_mismatch")

    samples = _as_records(read_json(sample_path), key="samples")
    gold = _as_records(read_json(gold_path), key="gold")
    audit = read_json(audit_path)
    if not isinstance(audit, dict):
        raise AuditFailure("invalid_audit_schema")
    audit_items = _as_records(audit, key="items")
    sample_qids = _qid_set(samples)
    gold_qids = _qid_set(gold)
    audit_qids = _qid_set(audit_items)
    if sample_qids != gold_qids:
        raise AuditFailure("gold_qid_join_mismatch")
    if sample_qids != audit_qids:
        raise AuditFailure("audit_qid_join_mismatch")
    if not samples:
        raise AuditFailure("empty_sample_group")

    video_ids = {str(record.get("video_id", "")) for record in samples}
    if len(video_ids) != 1 or not next(iter(video_ids)):
        raise AuditFailure("video_id_mismatch")
    rights_video_id = str(rights.get("video_id", ""))
    if not rights_video_id or video_ids not in ({rights_video_id}, {"NOAA_" + rights_video_id}):
        raise AuditFailure("rights_video_id_mismatch")

    canonical_media = media_path.resolve()
    for sample in samples:
        if _has_forbidden_key(sample):
            raise AuditFailure("label_leak_in_sample")
        if any(_normalize_key(key) not in ALLOWED_SAMPLE_KEYS for key in sample):
            raise AuditFailure("invalid_sample_schema")
        if not isinstance(sample.get("question"), str) or not sample["question"].strip():
            raise AuditFailure("invalid_question_schema")
        choices = sample.get("choices")
        if not isinstance(choices, list) or len(choices) != 5:
            raise AuditFailure("choice_count_mismatch")
        if all(isinstance(choice, str) for choice in choices):
            choice_texts = [choice.strip() for choice in choices]
            if any(not text for text in choice_texts) or len(set(choice_texts)) != len(choice_texts):
                raise AuditFailure("invalid_choice_schema")
        elif all(isinstance(choice, dict) for choice in choices):
            if any(_has_forbidden_key(choice) for choice in choices):
                raise AuditFailure("label_leak_in_sample")
            if any(set(choice) != ALLOWED_CHOICE_KEYS for choice in choices):
                raise AuditFailure("invalid_choice_schema")
            choice_ids = [choice.get("id") for choice in choices]
            choice_texts = [choice.get("text") for choice in choices]
            if (
                any(not isinstance(choice_id, str) or not choice_id for choice_id in choice_ids)
                or any(not isinstance(text, str) or not text.strip() for text in choice_texts)
                or len(set(choice_ids)) != len(choice_ids)
                or len({text.strip() for text in choice_texts}) != len(choice_texts)
            ):
                raise AuditFailure("invalid_choice_schema")
        else:
            raise AuditFailure("invalid_choice_schema")
        order = sample.get("choice_order_original_indices")
        if "choice_order_original_indices" in sample and (
            not isinstance(order, list)
            or len(order) != len(choices)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in order)
            or set(order) != set(range(len(choices)))
        ):
            raise AuditFailure("invalid_choice_order")
        start, end = sample.get("window_start_seconds"), sample.get("window_end_seconds")
        if (
            not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isinstance(end, (int, float)) or isinstance(end, bool)
            or not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end <= start
        ):
            raise AuditFailure("invalid_video_interval")
        if sample.get("video_sha256") != expected_media:
            raise AuditFailure("sample_media_hash_mismatch")
        sample_video = sample.get("video")
        if not isinstance(sample_video, str) or not sample_video:
            raise AuditFailure("sample_media_reference_missing")
        sample_media_path = Path(sample_video).expanduser()
        if not sample_media_path.is_absolute():
            sample_media_path = data_root.parent.parent / sample_media_path
        if sample_media_path.resolve() != canonical_media:
            raise AuditFailure("sample_media_reference_mismatch")

    gold_by_qid = {str(record["qid"]): record for record in gold}
    for sample in samples:
        gold_record = gold_by_qid[str(sample["qid"])]
        if str(gold_record.get("video_id", "")) != str(sample["video_id"]):
            raise AuditFailure("gold_video_id_join_mismatch")
        answer_index = gold_record.get("answer_choice_index")
        if not isinstance(answer_index, int) or isinstance(answer_index, bool):
            raise AuditFailure("invalid_gold_choice_index")
        if answer_index < 0 or answer_index >= len(sample["choices"]):
            raise AuditFailure("gold_choice_out_of_range")

    # Count explicitly AI-labelled reviews; missing/unspecified reviewer metadata
    # is not promoted to human review.
    audit_method = audit.get("method", "")
    ai_review_items = 0
    for item in audit_items:
        reviewer = item.get("independent_review") or item.get("blind_review") or {}
        role = reviewer.get("reviewer_role", "") if isinstance(reviewer, dict) else ""
        if _is_explicit_ai_role(audit_method) or _is_explicit_ai_role(role):
            ai_review_items += 1

    record = {
        "group_id": group["group_id"],
        "media_sha256": actual_media_hash,
        "samples_sha256": sample_hash,
        "gold_sha256": gold_hash,
        "audit_sha256": audit_hash,
        "rights_sha256": rights_hash,
        "sample_count": len(samples),
        "gold_count": len(gold),
        "audit_item_count": len(audit_items),
        "ai_blind_review_item_count": ai_review_items,
        "label_leak_count": 0,
        "qid_join_count": len(sample_qids),
    }
    return record, sample_qids


def _validate_attestation(
    value: Any,
    records: list[dict[str, Any]],
    qids_by_group: dict[str, set[str]],
) -> list[str]:
    failures: list[str] = []
    if not isinstance(value, dict):
        return ["attestation_invalid_schema"]
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version != 1 or value.get("signed") is not True:
        failures.append("attestation_not_signed_v1")
    attestor = value.get("attestor")
    if not isinstance(attestor, dict):
        failures.append("attestation_human_missing")
    elif (
        attestor.get("human") is not True
        or not isinstance(attestor.get("name"), str)
        or not attestor["name"].strip()
        or not isinstance(attestor.get("role"), str)
        or not attestor["role"].strip()
    ):
        failures.append("attestation_human_missing")
    signed_at = value.get("signed_at_utc")
    try:
        if not isinstance(signed_at, str) or not signed_at.endswith("Z"):
            raise ValueError
        datetime.fromisoformat(signed_at[:-1] + "+00:00")
    except ValueError:
        failures.append("attestation_timestamp_invalid")

    groups = value.get("groups")
    if not isinstance(groups, list) or any(not isinstance(item, dict) for item in groups):
        return failures + ["attestation_groups_invalid"]
    by_id = {str(item.get("group_id", "")): item for item in groups}
    if len(by_id) != len(groups) or set(by_id) != set(qids_by_group):
        failures.append("attestation_group_set_mismatch")

    record_by_id = {item["group_id"]: item for item in records}
    for group_id in sorted(set(record_by_id) & set(by_id)):
        item = by_id[group_id]
        record = record_by_id[group_id]
        for field in ("media_sha256", "samples_sha256", "gold_sha256", "audit_sha256", "rights_sha256"):
            digest = item.get(field)
            if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest) or digest != record[field]:
                failures.append("attestation_source_hash_mismatch")
                break
        reviewed = item.get("reviewed_qids")
        if (
            not isinstance(reviewed, list)
            or any(not isinstance(qid, (str, int)) or isinstance(qid, bool) for qid in reviewed)
            or len({str(qid) for qid in reviewed}) != len(reviewed)
            or {str(qid) for qid in reviewed} != qids_by_group[group_id]
        ):
            failures.append("attestation_reviewed_qids_mismatch")
        checks = item.get("checks")
        if not isinstance(checks, dict) or any(checks.get(key) is not True for key in REQUIRED_CHECKS):
            failures.append("attestation_checks_incomplete")
        scope = item.get("rights_scope")
        if not isinstance(scope, dict) or (
            scope.get("video_only_model_inference") is not True
            or scope.get("audio_use") is not False
            or scope.get("external_media_publication") is not False
            or scope.get("required_credit") != "NOAA Ocean Exploration"
        ):
            failures.append("attestation_rights_scope_invalid")
    return sorted(set(failures))


def audit_dataset(
    data_root: Path,
    *,
    attestation: Any = None,
    attestation_supplied: bool = False,
    expected_media_hashes: dict[str, str] | None = None,
    expected_rights_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return a publish-safe summary; only override hashes for fixture tests."""
    expected_media_hashes = expected_media_hashes or {
        group["group_id"]: group["media_sha256"] for group in GROUPS
    }
    expected_rights_hashes = expected_rights_hashes or {
        group["group_id"]: group["rights_sha256"] for group in GROUPS
    }
    records: list[dict[str, Any]] = []
    qids_by_group: dict[str, set[str]] = {}
    failures: list[str] = []
    for group in GROUPS:
        try:
            record, qids = _audit_group(
                data_root,
                group,
                expected_media_hashes[group["group_id"]],
                expected_rights_hashes[group["group_id"]],
            )
            records.append(record)
            qids_by_group[group["group_id"]] = qids
        except (AuditFailure, OSError) as exc:
            failures.append(exc.reason if isinstance(exc, AuditFailure) else "source_read_error")

    if len({record["media_sha256"] for record in records}) != len(records):
        failures.append("duplicate_media_across_groups")
    attestation_status = "missing"
    if attestation_supplied:
        if failures:
            attestation_status = "unvalidated"
        else:
            attestation_failures = _validate_attestation(attestation, records, qids_by_group)
            failures.extend(attestation_failures)
            attestation_status = "invalid" if attestation_failures else "valid"
    else:
        failures.append("human_attestation_missing")

    sample_count = sum(record["sample_count"] for record in records)
    gold_count = sum(record["gold_count"] for record in records)
    audit_item_count = sum(record["audit_item_count"] for record in records)
    label_leak_count_complete = len(records) == len(GROUPS)
    label_leak_count = sum(record["label_leak_count"] for record in records) if label_leak_count_complete else None
    structurally_complete = len(records) == len(GROUPS) and sample_count == gold_count == audit_item_count
    if not structurally_complete and not failures:
        failures.append("source_groups_incomplete")
    return {
        "schema_version": 1,
        "purpose": "noaa_manual_qa_preinference_gate",
        "decision": "go_for_video_only_qa_inference" if not failures and structurally_complete else "no_go",
        "scope_limit": "does_not_establish_qa_efficacy_or_explanation_faithfulness",
        "source_groups_requested": len(GROUPS),
        "source_groups_valid": len(records),
        "samples": sample_count,
        "gold_records": gold_count,
        "audit_records": audit_item_count,
        "joined_qids": sum(record["qid_join_count"] for record in records),
        "label_leak_count": label_leak_count,
        "label_leak_count_complete": label_leak_count_complete,
        "ai_blind_review_items": sum(record["ai_blind_review_item_count"] for record in records),
        "attestation_status": attestation_status,
        "failure_reasons": sorted(set(failures)),
        "source_sha256": {record["group_id"]: {
            key: record[key] for key in (
                "media_sha256", "samples_sha256", "gold_sha256", "audit_sha256", "rights_sha256"
            )
        } for record in records},
        "groups": records,
        "privacy": "No question text, choices, answers, qids, local paths, or attestor identity emitted.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets" / "noaa_qa_pilot_20261011",
        help="Private local pilot root (default: datasets/noaa_qa_pilot_20261011)",
    )
    parser.add_argument("--attestation", type=Path, help="Private human attestation JSON; never publish it")
    parser.add_argument(
        "--output", type=Path,
        help="Unique JSON summary path under repository results/ (default: timestamped path)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print the full JSON summary")
    parser.add_argument("--require-go", action="store_true", help="Exit 2 unless all source and human gates pass")
    args = parser.parse_args(argv)

    attestation_supplied = args.attestation is not None
    attestation = None
    if args.attestation is not None:
        try:
            attestation = read_json(args.attestation)
        except AuditFailure:
            attestation = None
    result = audit_dataset(
        args.root,
        attestation=attestation,
        attestation_supplied=attestation_supplied,
    )
    repo_root = Path(__file__).resolve().parents[1]
    output = args.output or (repo_root / "results" / f"noaa_qa_gate_{datetime.utcnow():%Y%m%dT%H%M%SZ}.json")
    try:
        artifact_path = _write_run_artifact(result, output, repo_root=repo_root)
    except (OSError, ValueError) as exc:
        print(f"NOAA QA gate artifact error: {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(artifact_path.read_text(encoding="utf-8"), end="")
    else:
        failed = ", ".join(result["failure_reasons"]) or "none"
        print(
            f"NOAA QA gate: {result['decision']} | groups "
            f"{result['source_groups_valid']}/{result['source_groups_requested']} | "
            f"samples {result['samples']} | attestation {result['attestation_status']} | "
            f"failures {failed} | artifact {artifact_path.relative_to(repo_root)}"
        )
    if args.require_go and result["decision"] != "go_for_video_only_qa_inference":
        return 2
    return 0


def _run_metadata(artifact_path: Path, repo_root: Path, runner_path: Path | None = None) -> dict[str, Any]:
    runner_path = runner_path or Path(__file__).resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "runner_sha256": sha256_file(runner_path),
        "checkout_head": head,
    }


def _write_run_artifact(result: dict[str, Any], output: Path, *, repo_root: Path) -> Path:
    repo_root = repo_root.resolve()
    output = output if output.is_absolute() else repo_root / output
    output = output.resolve()
    results_root = (repo_root / "results").resolve()
    if output.parent != results_root or output.suffix.lower() != ".json":
        raise ValueError("--output must be a .json file directly under repository results/")
    results_root.mkdir(parents=True, exist_ok=True)
    payload = result | {"run": _run_metadata(output, repo_root)}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")
    return output


if __name__ == "__main__":
    sys.exit(main())
