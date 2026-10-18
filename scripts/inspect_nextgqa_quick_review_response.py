#!/usr/bin/env python3
"""Validate and summarize a local exploratory quick-review JSON without echoing PII."""
from __future__ import annotations

import argparse
import json
import re
import stat
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / (
    "data/private_nextgqa/quick_review_ja_video_only_20261019_itemid_v3.html"
)
MAX_JSON_BYTES = 1_000_000
MAX_ARTIFACT_BYTES = 64_000_000
JUDGMENTS = {"roughly_matches", "partly_differs", "cannot_tell"}
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


def expected_item_ids(artifact_path: Path) -> list[str]:
    raw = _read_bounded(artifact_path, MAX_ARTIFACT_BYTES, "artifact_file_invalid")
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


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewError("response_duplicate_json_key")
        result[key] = value
    return result


def summarize_response(response_raw: bytes, item_ids: list[str]) -> dict[str, Any]:
    try:
        response = json.loads(
            response_raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ReviewError("response_nonfinite_number")),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewError("response_json_invalid") from exc

    top_keys = {"schema_version", "purpose", "status", "created_at_utc",
                "nickname", "answers", "limitations"}
    if (not isinstance(response, dict) or set(response) != top_keys
            or type(response.get("schema_version")) is not int
            or response["schema_version"] != 2
            or response.get("purpose") != "exploratory_human_impression_check"
            or response.get("status") != "informal_observations_only"):
        raise ReviewError("response_schema_or_purpose_invalid")

    nickname = response.get("nickname")
    if not isinstance(nickname, str) or not nickname.strip() or len(nickname) > 40:
        raise ReviewError("response_nickname_field_invalid")
    limitations = response.get("limitations")
    if (not isinstance(limitations, list) or len(limitations) != len(EXPECTED_LIMITATIONS)
            or any(not isinstance(value, str) for value in limitations)
            or set(limitations) != EXPECTED_LIMITATIONS):
        raise ReviewError("response_limitations_invalid")
    answers = response.get("answers")
    if not isinstance(answers, list) or len(answers) != 3:
        raise ReviewError("response_must_contain_three_answers")

    expected_answer_keys = {"item_id", "item_number", "judgment",
                            "approximate_time", "note"}
    counts: Counter[str] = Counter()
    for ordinal, (answer, item_id) in enumerate(zip(answers, item_ids), start=1):
        if (not isinstance(answer, dict) or set(answer) != expected_answer_keys
                or answer.get("item_id") != item_id
                or type(answer.get("item_number")) is not int
                or answer["item_number"] != ordinal
                or not isinstance(answer.get("judgment"), str)
                or answer["judgment"] not in JUDGMENTS):
            raise ReviewError("response_item_mapping_or_judgment_invalid")
        for field, limit in (("approximate_time", 80), ("note", 500)):
            value = answer[field]
            if value is not None and (not isinstance(value, str) or len(value) > limit):
                raise ReviewError("response_optional_field_invalid")
        counts[answer["judgment"]] += 1

    return {
        "validated_exploratory_responses": len(answers),
        "judgment_counts": {key: counts[key] for key in (
            "roughly_matches", "partly_differs", "cannot_tell")},
        "item_judgments": [
            {"item": index, "judgment": answer["judgment"]}
            for index, answer in enumerate(answers, start=1)
        ],
        "nickname_included_in_output": False,
        "free_text_included_in_output": False,
        "faithfulness_inference": "not_performed",
        "scoring_authorization": "not_granted",
    }


def inspect_response(response_path: Path, artifact_path: Path = DEFAULT_ARTIFACT) -> dict[str, Any]:
    response_raw = _read_bounded(response_path, MAX_JSON_BYTES, "response_file_invalid")
    ids = expected_item_ids(artifact_path)
    return summarize_response(response_raw, ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response_json", type=Path,
                        help="artifactから保存した探索的回答JSON（ローカル限定）")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT,
                        help="照合対象のitemid-v3 HTML")
    args = parser.parse_args(argv)
    try:
        report = inspect_response(args.response_json, args.artifact)
    except ReviewError as exc:
        parser.error(str(exc))
    counts = report["judgment_counts"]
    labels = {
        "roughly_matches": "だいたい合っている",
        "partly_differs": "一部ちがう",
        "cannot_tell": "動画だけではわからない",
    }
    print(f"形式確認: 探索的回答 {report['validated_exploratory_responses']}/3 件")
    for key, label in labels.items():
        print(f"{label}: {counts[key]}件")
    print("ニックネーム・自由記述は表示していません。faithfulnessの推定・採点許可はありません。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
