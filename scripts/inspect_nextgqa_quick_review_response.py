#!/usr/bin/env python3
"""Validate and summarize a local exploratory quick-review JSON without echoing PII."""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / (
    "data/private_nextgqa/quick_review_temporal_order_val_20261021_01.html"
)
LEGACY_SIMPLE_ARTIFACT = ROOT / (
    "data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v14.html"
)
MAX_JSON_BYTES = 1_000_000
MAX_ARTIFACT_BYTES = 64_000_000
JUDGMENTS = {"roughly_matches", "partly_differs", "cannot_tell"}
SUFFICIENCY = {"sufficient", "insufficient", "cannot_tell"}
CLARITY = {"clear", "unclear", "cannot_tell"}
ANSWER_FIT = {"fits", "does_not_fit", "cannot_tell"}
EXPECTED_LIMITATIONS = {
    "簡単な印象確認であり、完全な幻覚判定ではない",
    "item_idはローカルで回答を質問へ対応付けるための匿名化IDで、本人性や視聴を証明しない",
    "本人確認・独立性・動画の視聴完了は検証していない",
    "この結果だけからfaithfulnessの効果やモデル性能を結論づけない",
}


class ReviewError(ValueError):
    """The response is not a complete response for this exact artifact."""


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.in_script = False
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() == "script":
            self.in_script = True
            self.scripts.append("")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "script":
            self.in_script = False

    def handle_data(self, data: str) -> None:
        if self.in_script and self.scripts:
            self.scripts[-1] += data


def _read_bounded(path: Path, limit: int, code: str) -> bytes:
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ReviewError(code)
        raw = path.read_bytes()
    except OSError as exc:
        raise ReviewError(code) from exc
    if len(raw) > limit:
        raise ReviewError(code)
    return raw


def _read_artifact_snapshot(artifact_path: Path) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise ReviewError("artifact_nofollow_unavailable")
    fd = None
    try:
        nonblock = getattr(os, "O_NONBLOCK", 0)
        fd = os.open(artifact_path, os.O_RDONLY | nofollow | nonblock)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_ARTIFACT_BYTES:
            raise ReviewError("artifact_file_invalid")
        chunks = []
        remaining = MAX_ARTIFACT_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ReviewError("artifact_file_invalid")
        after = os.fstat(fd)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_size,
                                row.st_mtime_ns, row.st_ctime_ns)
        if (not stat.S_ISREG(after.st_mode) or identity(info) != identity(after)
                or len(raw) != after.st_size):
            raise ReviewError("artifact_file_changed_during_read")
        return raw
    except OSError as exc:
        raise ReviewError("artifact_file_invalid") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _item_ids_from_artifact(raw: bytes) -> list[str]:
    try:
        html = raw.decode("utf-8", errors="strict")
        parser = _Scripts()
        parser.feed(html)
        parser.close()
    except (UnicodeError, ValueError) as exc:
        raise ReviewError("artifact_html_invalid") from exc
    for script in parser.scripts:
        start_marker = "const items="
        start = script.find(start_marker)
        if start < 0:
            continue
        start += len(start_marker)
        end = script.find(";\nlet nickname", start)
        if end < 0:
            continue
        try:
            items = json.loads(script[start:end])
        except json.JSONDecodeError as exc:
            raise ReviewError("artifact_item_manifest_invalid") from exc
        if (not isinstance(items, list) or len(items) != 3
                or any(not isinstance(item, dict) for item in items)):
            raise ReviewError("artifact_item_manifest_invalid")
        ids = [item.get("item_id") for item in items]
        if (any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{20}", value)
                for value in ids) or len(set(ids)) != 3):
            raise ReviewError("artifact_item_ids_invalid")
        return ids
    raise ReviewError("artifact_item_manifest_missing")


_FINGERPRINT_META = re.compile(
    r'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">'
)


def _fingerprint_from_artifact(raw: bytes) -> str | None:
    """Verify the self-bound full HTML fingerprint; None means legacy unbound HTML."""
    try:
        html = raw.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ReviewError("artifact_html_invalid") from exc
    matches = list(_FINGERPRINT_META.finditer(html))
    if not matches:
        return None
    if len(matches) != 1:
        raise ReviewError("artifact_fingerprint_invalid")
    match = matches[0]
    canonical = html[:match.start(1)] + "__ARTIFACT_FINGERPRINT__" + html[match.end(1):]
    actual = __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest()
    if actual != match.group(1):
        raise ReviewError("artifact_fingerprint_mismatch")
    return actual


def expected_item_ids(artifact_path: Path) -> list[str]:
    return _item_ids_from_artifact(_read_artifact_snapshot(artifact_path))


def expected_artifact_fingerprint(artifact_path: Path) -> str | None:
    return _fingerprint_from_artifact(_read_artifact_snapshot(artifact_path))


def inspect_artifact(artifact_path: Path) -> tuple[list[str], str | None]:
    """Parse IDs and verify fingerprint from one no-follow file snapshot."""
    raw = _read_artifact_snapshot(artifact_path)
    return _item_ids_from_artifact(raw), _fingerprint_from_artifact(raw)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError("response_duplicate_json_key")
        result[key] = value
    return result


def summarize_response(response_raw: bytes, item_ids: list[str],
                       artifact_fingerprint: str | None = None) -> dict[str, Any]:
    try:
        response = json.loads(
            response_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ReviewError("response_nonfinite_number")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError("response_json_invalid") from exc

    base_keys = {"schema_version", "purpose", "status", "created_at_utc", "limitations"}
    if not isinstance(response, dict):
        raise ReviewError("response_schema_or_purpose_invalid")
    version = response.get("schema_version")
    if type(version) is not int:
        raise ReviewError("response_schema_or_purpose_invalid")
    top_keys = base_keys | ({"nickname", "answers"} if version in {2, 3}
                            else {"respondents"} if version in {4, 5} else
                            {"respondents", "artifact_fingerprint"} if version == 6 else set())
    if (set(response) != top_keys
            or type(response.get("schema_version")) is not int
            or version not in {2, 3, 4, 5, 6}
            or response.get("purpose") != "exploratory_human_impression_check"
            or response.get("status") != "informal_observations_only"):
        raise ReviewError("response_schema_or_purpose_invalid")

    if version == 6:
        response_fingerprint = response.get("artifact_fingerprint")
        if (artifact_fingerprint is None
                or not isinstance(response_fingerprint, str)
                or not re.fullmatch(r"[0-9a-f]{64}", response_fingerprint)
                or response_fingerprint != artifact_fingerprint):
            raise ReviewError("response_artifact_fingerprint_mismatch")

    if version in {2, 3}:
        nickname = response.get("nickname")
        if not isinstance(nickname, str) or not nickname.strip() or len(nickname) > 40:
            raise ReviewError("response_nickname_field_invalid")
    limitations = response.get("limitations")
    if (not isinstance(limitations, list) or len(limitations) != len(EXPECTED_LIMITATIONS)
            or any(not isinstance(value, str) for value in limitations)
            or set(limitations) != EXPECTED_LIMITATIONS):
        raise ReviewError("response_limitations_invalid")
    if version in {4, 5, 6}:
        respondents = response.get("respondents")
        if not isinstance(respondents, list) or not respondents:
            raise ReviewError("response_respondents_invalid")
        answer_groups = []
        for respondent in respondents:
            if (not isinstance(respondent, dict) or set(respondent) != {"nickname", "answers"}
                    or not isinstance(respondent["nickname"], str)
                    or not respondent["nickname"].strip() or len(respondent["nickname"]) > 40
                    or not isinstance(respondent["answers"], list)
                    or len(respondent["answers"]) != 3):
                raise ReviewError("response_respondent_schema_invalid")
            answer_groups.append(respondent["answers"])
    else:
        answers = response.get("answers")
        if not isinstance(answers, list) or len(answers) != 3:
            raise ReviewError("response_must_contain_three_answers")
        answer_groups = [answers]

    response_version = 3 if version == 4 else 4 if version in {5, 6} else version
    expected_answer_keys = ({"item_id", "item_number", "judgment",
                             "approximate_time", "note"} if response_version == 2 else
                            {"item_id", "item_number", "question_clarity",
                             "answer_question_fit", "video_evidence_sufficiency"}
                            if response_version == 4 else
                            {"item_id", "item_number", "claim_truth",
                             "explanation_sufficiency"})
    counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()
    clarity_counts: Counter[str] = Counter()
    answer_fit_counts: Counter[str] = Counter()
    item_judgments = []
    item_summary = [
        {"item": index, "question_clarity": Counter(),
         "answer_question_fit": Counter(), "video_evidence_sufficiency": Counter()}
        for index in range(1, 4)
    ]
    for respondent_index, answers in enumerate(answer_groups, start=1):
        for ordinal, (answer, item_id) in enumerate(zip(answers, item_ids), start=1):
            if (not isinstance(answer, dict) or set(answer) != expected_answer_keys
                    or answer.get("item_id") != item_id
                    or type(answer.get("item_number")) is not int
                    or answer["item_number"] != ordinal):
                raise ReviewError("response_item_mapping_or_judgment_invalid")
            if response_version == 2:
                if not isinstance(answer.get("judgment"), str) or answer["judgment"] not in JUDGMENTS:
                    raise ReviewError("response_item_mapping_or_judgment_invalid")
                for field, limit in (("approximate_time", 80), ("note", 500)):
                    value = answer[field]
                    if value is not None and (not isinstance(value, str) or len(value) > limit):
                        raise ReviewError("response_optional_field_invalid")
                counts[answer["judgment"]] += 1
                item_judgments.append({"respondent": respondent_index, "item": ordinal,
                                       "judgment": answer["judgment"]})
            elif response_version == 3:
                if (not isinstance(answer.get("claim_truth"), str)
                        or answer["claim_truth"] not in JUDGMENTS
                        or not isinstance(answer.get("explanation_sufficiency"), str)
                        or answer["explanation_sufficiency"] not in SUFFICIENCY):
                    raise ReviewError("response_item_mapping_or_judgment_invalid")
                counts[answer["claim_truth"]] += 1
                sufficiency_counts[answer["explanation_sufficiency"]] += 1
                item_judgments.append({"respondent": respondent_index, "item": ordinal,
                                       "claim_truth": answer["claim_truth"],
                                       "explanation_sufficiency": answer["explanation_sufficiency"]})
            else:
                if (not isinstance(answer.get("question_clarity"), str)
                        or answer["question_clarity"] not in CLARITY
                        or not isinstance(answer.get("answer_question_fit"), str)
                        or answer["answer_question_fit"] not in ANSWER_FIT
                        or not isinstance(answer.get("video_evidence_sufficiency"), str)
                        or answer["video_evidence_sufficiency"] not in SUFFICIENCY):
                    raise ReviewError("response_item_mapping_or_judgment_invalid")
                clarity_counts[answer["question_clarity"]] += 1
                answer_fit_counts[answer["answer_question_fit"]] += 1
                sufficiency_counts[answer["video_evidence_sufficiency"]] += 1
                item_summary[ordinal - 1]["question_clarity"][answer["question_clarity"]] += 1
                item_summary[ordinal - 1]["answer_question_fit"][answer["answer_question_fit"]] += 1
                item_summary[ordinal - 1]["video_evidence_sufficiency"][answer["video_evidence_sufficiency"]] += 1
                item_judgments.append({"respondent": respondent_index, "item": ordinal,
                                       "question_clarity": answer["question_clarity"],
                                       "answer_question_fit": answer["answer_question_fit"],
                                       "video_evidence_sufficiency": answer["video_evidence_sufficiency"]})

    return {
        "validated_exploratory_responses": sum(map(len, answer_groups)),
        "respondent_count": len(answer_groups),
        "judgment_counts": ({key: counts[key] for key in (
            "roughly_matches", "partly_differs", "cannot_tell")}
            if response_version != 4 else None),
        "sufficiency_counts": ({key: sufficiency_counts[key]
                                for key in ("sufficient", "insufficient", "cannot_tell")}
                               if response_version == 3 else None),
        "question_clarity_counts": ({key: clarity_counts[key]
                                     for key in ("clear", "unclear", "cannot_tell")}
                                    if response_version == 4 else None),
        "answer_question_fit_counts": ({key: answer_fit_counts[key]
                                        for key in ("fits", "does_not_fit", "cannot_tell")}
                                       if response_version == 4 else None),
        "video_evidence_sufficiency_counts": ({key: sufficiency_counts[key]
                                                for key in ("sufficient", "insufficient", "cannot_tell")}
                                               if response_version == 4 else None),
        "item_judgments": item_judgments,
        "item_summary": ([
            {"item": row["item"],
             "question_clarity_counts": {key: row["question_clarity"][key]
                                         for key in ("clear", "unclear", "cannot_tell")},
             "answer_question_fit_counts": {key: row["answer_question_fit"][key]
                                            for key in ("fits", "does_not_fit", "cannot_tell")},
             "video_evidence_sufficiency_counts": {key: row["video_evidence_sufficiency"][key]
                                                   for key in ("sufficient", "insufficient", "cannot_tell")}}
            for row in item_summary
        ] if response_version == 4 else None),
        "nickname_included_in_output": False,
        "free_text_included_in_output": False,
        "faithfulness_inference": "not_performed",
        "scoring_authorization": "not_granted",
        "artifact_binding": "verified" if version == 6 else "legacy_unbound",
    }


def inspect_response(response_path: Path, artifact_path: Path = DEFAULT_ARTIFACT, *,
                     allow_unbound_legacy: bool = False) -> dict[str, Any]:
    response_raw = _read_bounded(response_path, MAX_JSON_BYTES, "response_file_invalid")
    ids, fingerprint = inspect_artifact(artifact_path)
    report = summarize_response(response_raw, ids, fingerprint)
    if report["artifact_binding"] != "verified" and not allow_unbound_legacy:
        raise ReviewError("legacy_artifact_binding_requires_explicit_opt_in")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_json", type=Path,
                        help="artifactから保存した探索的回答JSON（ローカル限定）")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT,
                        help="照合対象HTML（既定は現在のGPU生成claim用artifact）")
    parser.add_argument("--show-items", action="store_true",
                        help="既定の集計に加え、参加者/項目別の回答も表示する")
    parser.add_argument("--show-item-summary", action="store_true",
                        help="回答者を分けず、各項目の3観点の件数を表示する")
    parser.add_argument("--allow-unbound-legacy", action="store_true",
                        help="旧schema v2〜v5をartifact版未検証の履歴として検査する")
    args = parser.parse_args(argv)
    try:
        report = inspect_response(args.response_json, args.artifact,
                                  allow_unbound_legacy=args.allow_unbound_legacy)
    except ReviewError as exc:
        parser.error(str(exc))
    counts = report["judgment_counts"]
    labels = {
        "roughly_matches": "だいたい合っている",
        "partly_differs": "一部ちがう",
        "cannot_tell": "動画だけではわからない",
    }
    print(f"形式確認: {report['respondent_count']}人・探索的回答 "
          f"{report['validated_exploratory_responses']}件")
    print("artifact HTML/JSON fingerprint一致: "
          + ("確認済み" if report["artifact_binding"] == "verified"
             else "未検証（旧形式JSON）"))
    if counts is not None:
        for key, label in labels.items():
            print(f"{label}: {counts[key]}件")
    for item in report["item_judgments"] if args.show_items else []:
        if "judgment" in item:
            print(f"{item['respondent']}人目・第{item['item']}件目: {labels[item['judgment']]}")
        elif "question_clarity" in item:
            clarity_labels = {"clear": "わかりやすい", "unclear": "わかりにくい",
                              "cannot_tell": "判断できない"}
            fit_labels = {"fits": "合っている", "does_not_fit": "違っている",
                          "cannot_tell": "判断できない"}
            evidence_labels = {"sufficient": "十分", "insufficient": "足りない点がある",
                               "cannot_tell": "判断できない"}
            print(f"{item['respondent']}人目・第{item['item']}件目: "
                  f"質問={clarity_labels[item['question_clarity']]}、"
                  f"答え={fit_labels[item['answer_question_fit']]}、"
                  f"動画根拠={evidence_labels[item['video_evidence_sufficiency']]}")
        else:
            sufficiency_labels = {"sufficient": "十分", "insufficient": "足りない",
                                  "cannot_tell": "判断できない"}
            print(f"{item['respondent']}人目・第{item['item']}件目: 主張={labels[item['claim_truth']]}、"
                  f"根拠={sufficiency_labels[item['explanation_sufficiency']]}")
    if report["sufficiency_counts"] is not None:
        suff = report["sufficiency_counts"]
        print(f"根拠の十分さ: 十分={suff['sufficient']}件、足りない={suff['insufficient']}件、"
              f"判断できない={suff['cannot_tell']}件")
    if report["question_clarity_counts"] is not None:
        clarity = report["question_clarity_counts"]
        fit = report["answer_question_fit_counts"]
        evidence = report["video_evidence_sufficiency_counts"]
        print(f"質問の明瞭さ: わかりやすい={clarity['clear']}件、わかりにくい={clarity['unclear']}件、"
              f"判断できない={clarity['cannot_tell']}件")
        print(f"答えの適合: 合っている={fit['fits']}件、違っている={fit['does_not_fit']}件、"
              f"判断できない={fit['cannot_tell']}件")
        print(f"動画根拠の十分さ: 十分={evidence['sufficient']}件、足りない={evidence['insufficient']}件、"
              f"判断できない={evidence['cannot_tell']}件")
    if args.show_item_summary and report["item_summary"] is not None:
        clarity_labels = {"clear": "明瞭", "unclear": "不明瞭", "cannot_tell": "判断不能"}
        fit_labels = {"fits": "適合", "does_not_fit": "不適合", "cannot_tell": "判断不能"}
        evidence_labels = {"sufficient": "十分", "insufficient": "不足", "cannot_tell": "判断不能"}
        for row in report["item_summary"]:
            c, f, e = (row["question_clarity_counts"], row["answer_question_fit_counts"],
                       row["video_evidence_sufficiency_counts"])
            print(f"第{row['item']}問（n={sum(c.values())}）: "
                  f"明瞭性 {', '.join(f'{clarity_labels[k]}={v}' for k,v in c.items())}; "
                  f"回答適合 {', '.join(f'{fit_labels[k]}={v}' for k,v in f.items())}; "
                  f"根拠十分性 {', '.join(f'{evidence_labels[k]}={v}' for k,v in e.items())}")
    print("ニックネーム・自由記述は表示していません。faithfulnessの推定・採点許可はありません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
