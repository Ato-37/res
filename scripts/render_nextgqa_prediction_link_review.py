"""Render an offline, answer-outcome-blinded QTA claim-to-prediction worksheet.

The page collects self-reported semantic judgments only. Its receipt is not
signed and cannot authorize model scoring.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import stat
import argparse
from typing import Any

from nextgqa_prediction_link_review import (
    BASELINE_MANIFEST_KEYS,
    BASELINE_RECORD_KEYS,
    BASELINE_SCORE_BINDING_RECORD_KEYS,
    MAX_BASELINE_MANIFEST_BYTES,
    _unique_object,
    displayed_fields_sha256,
)


def _open_private_child(root: Path, name: str, *, create: bool) -> tuple[int, int]:
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name or "\x00" in name):
        raise ValueError("private_filename_invalid")
    before = os.lstat(root)
    if (not stat.S_ISDIR(before.st_mode) or stat.S_IMODE(before.st_mode) & 0o077
            or before.st_uid != os.geteuid()):
        raise ValueError("private_root_permissions_invalid")
    root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                      | getattr(os, "O_NOFOLLOW", 0))
    try:
        current = os.fstat(root_fd)
        if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("private_root_changed")
        if create:
            fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o600, dir_fd=root_fd)
        else:
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), dir_fd=root_fd)
            info = os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_size > MAX_BASELINE_MANIFEST_BYTES
                    or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) & 0o077):
                os.close(fd)
                raise ValueError("baseline_manifest_file_invalid")
        return fd, root_fd
    except Exception:
        os.close(root_fd)
        raise


def render_prediction_link_review_page(
    *, baseline_manifest_raw: bytes, expected_baseline_manifest_sha256: str,
    protocol_sha256: str, video_claim_review_receipt_sha256: str,
) -> str:
    """Return a static HTML worksheet bound to exact private baseline bytes."""
    if (not isinstance(baseline_manifest_raw, bytes)
            or len(baseline_manifest_raw) > MAX_BASELINE_MANIFEST_BYTES
            or not isinstance(expected_baseline_manifest_sha256, str)
            or hashlib.sha256(baseline_manifest_raw).hexdigest()
            != expected_baseline_manifest_sha256):
        raise ValueError("baseline_manifest_bytes_hash_mismatch")
    if (not isinstance(protocol_sha256, str) or len(protocol_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in protocol_sha256)):
        raise ValueError("protocol_hash_invalid")
    if (not isinstance(video_claim_review_receipt_sha256, str)
            or len(video_claim_review_receipt_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in video_claim_review_receipt_sha256)):
        raise ValueError("video_review_hash_invalid")
    try:
        manifest = json.loads(baseline_manifest_raw.decode("utf-8"),
                              object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError("baseline_manifest_json_invalid") from None
    if (not isinstance(manifest, dict) or set(manifest) != BASELINE_MANIFEST_KEYS
            or type(manifest.get("schema_version")) is not int
            or (manifest.get("schema_version"), manifest.get("purpose")) not in (
                (1, "qta_baseline_prediction_link_manifest"),
                (2, "qta_baseline_prediction_link_manifest_v2"))):
        raise ValueError("baseline_manifest_schema_invalid")
    if manifest.get("video_claim_review_receipt_sha256") != video_claim_review_receipt_sha256:
        raise ValueError("baseline_video_review_receipt_binding_mismatch")
    rows = manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("baseline_manifest_records_invalid")
    seen = set()
    rendered = []
    for index, row in enumerate(rows):
        if (not isinstance(row, dict)
                or set(row) not in (BASELINE_RECORD_KEYS,
                                    BASELINE_SCORE_BINDING_RECORD_KEYS)):
            raise ValueError("baseline_manifest_record_schema_invalid")
        token = row.get("review_token")
        if not isinstance(token, str) or token in seen:
            raise ValueError("baseline_manifest_token_invalid")
        seen.add(token)
        digest = displayed_fields_sha256(
            row, video_claim_review_receipt_sha256=video_claim_review_receipt_sha256)
        options = row["options"]
        p, q = row["p_index"], row["q_index"]
        option_items = "".join(
            "<li{}>{}</li>".format(
                ' class="p-choice"' if opt_index == p else
                ' class="q-choice"' if opt_index == q else "",
                html.escape(option, quote=True))
            for opt_index, option in enumerate(options))
        rendered.append(
            '<article class="record">'
            f'<h2>記録 {index + 1} / {len(rows)}</h2>'
            f'<p class="token">記録token: {html.escape(token, quote=True)}</p>'
            f'<p><strong>質問</strong><br>{html.escape(row["question"], quote=True)}</p>'
            f'<p><strong>選択肢</strong>（緑=P、青=Q）</p><ol>{option_items}</ol>'
            f'<p><strong>固定された動画claim</strong><br>{html.escape(row["claim"], quote=True)}</p>'
            f'<p>先行動画レビュー: supports_claim（受領書SHA-256: '
            f'{html.escape(video_claim_review_receipt_sha256)}、区間SHA-256: '
            f'{html.escape(row["reviewed_intervals_sha256"])})</p>'
            '<p>この段階では動画を再評価しません。先行レビューが固定した動画claimを前提に、'
            'claim本文がPをQから区別するかを判定してください。</p>'
            '<label>先行動画レビューを前提に、claim本文はPをQから区別しますか？'
            f'<select data-index="{index}" data-token="{html.escape(token, quote=True)}" '
            f'data-displayed-sha256="{digest}"><option value="">選択してください</option>'
            '<option value="supports_p_over_q">支持する</option>'
            '<option value="compatible_not_discriminating">両立するがP/Qを区別しない</option>'
            '<option value="unrelated">無関係</option>'
            '<option value="contradicts_p">Pと矛盾する</option>'
            '<option value="ambiguous">曖昧</option></select></label>'
            f'<label>判断理由<textarea data-rationale="{index}" maxlength="4000"></textarea></label>'
            '</article>')
    page = """<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; img-src 'none'; media-src 'none';">
<title>QTA claim-prediction link review</title>
<style>body{font:16px sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#202124}.record{border:1px solid #aaa;border-radius:8px;padding:1rem;margin:1.2rem 0}.p-choice{background:#e5f5e0}.q-choice{background:#e3efff}select,textarea,input{font:inherit;display:block;margin:.4rem 0 1rem;max-width:100%}textarea{width:98%;min-height:4rem}.notice{background:#fff4d6;padding:1rem}button{font:inherit;padding:.7rem 1rem}</style></head>
<body><h1>claim と固定予測の対応確認</h1>
<p class="notice">この段階はbaselineのP/Qを固定した後、介入結果を見る前に行います。gold正答と介入・ablation後の得点は含みません。動画そのものや先行受領書の実体は再確認しません。信頼済み事前ゲートで検証される先行動画レビューのsupports_claim判定にhash-bound referenceで結び付け、その前提でclaim本文とP/Qの対応だけを判定します。このページは署名、本人認証、採点許可を与えません。</p>
<label>レビュー担当者のローカル識別子<input id="reviewer" maxlength="128"></label>
<label><input id="independent" type="checkbox">生成・採点・custody・第1段階の動画claim-reviewから独立して判断したことを自己申告する</label>
<label><input id="hidden" type="checkbox">gold正答と全介入結果を見ていないことを自己申告する</label>
__RECORDS__
<button id="export" disabled>prediction-link receiptを保存</button><span id="status" role="status"></span>
<script>'use strict';
const baselineHash='__BASELINE_HASH__',protocolHash='__PROTOCOL_HASH__',videoReviewHash='__VIDEO_REVIEW_HASH__';
const exportButton=document.getElementById('export'),status=document.getElementById('status');
function ready(){const choices=[...document.querySelectorAll('select[data-index]')];const rationales=[...document.querySelectorAll('textarea[data-rationale]')];exportButton.disabled=!document.getElementById('reviewer').value.trim()||!document.getElementById('independent').checked||!document.getElementById('hidden').checked||choices.some(x=>!x.value)||rationales.some(x=>!x.value.trim());}
document.addEventListener('input',ready);document.addEventListener('change',ready);
exportButton.addEventListener('click',()=>{if(exportButton.disabled)return;const records=[...document.querySelectorAll('select[data-index]')].map((choice,i)=>({review_token:choice.dataset.token,displayed_fields_sha256:choice.dataset.displayedSha256,judgment:choice.value,rationale:document.querySelector('textarea[data-rationale="'+i+'"]').value.trim()}));const receipt={schema_version:1,purpose:'qta_prediction_link_adjudication',status:'prediction_link_review_recorded_unvalidated',baseline_manifest_sha256:baselineHash,protocol_sha256:protocolHash,prior_video_review_receipt_sha256:videoReviewHash,reviewer_role:'independent_prediction_link_reviewer',reviewer_private_id:document.getElementById('reviewer').value.trim(),independence_attestation:'independent_from_generation_scoring_custody_and_video_review',gold_and_branch_outcomes_hidden:true,reviewed_at_utc:new Date().toISOString(),records,execution_authorization:'not_granted'};const blob=new Blob([JSON.stringify(receipt,null,2)+'\\n'],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='qta_prediction_link_review.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);status.textContent='ブラウザのダウンロード開始を要求しました。保存先でJSONの存在と権限を確認し、custodyへ渡してください。'});
</script></body></html>"""
    return (page.replace("__BASELINE_HASH__", expected_baseline_manifest_sha256)
            .replace("__PROTOCOL_HASH__", protocol_sha256)
            .replace("__VIDEO_REVIEW_HASH__", video_claim_review_receipt_sha256)
            .replace("__RECORDS__", "\n".join(rendered)))


def render_from_private_files(*, private_root: Path, manifest_name: str,
                              output_name: str, expected_manifest_sha256: str,
                              protocol_sha256: str,
                              video_claim_review_receipt_sha256: str) -> str:
    in_fd, root_fd = _open_private_child(private_root, manifest_name, create=False)
    try:
        chunks = []
        while True:
            part = os.read(in_fd, 1024 * 1024)
            if not part:
                break
            chunks.append(part)
            if sum(map(len, chunks)) > MAX_BASELINE_MANIFEST_BYTES:
                raise ValueError("baseline_manifest_file_invalid")
        raw = b"".join(chunks)
    finally:
        os.close(in_fd)
        os.close(root_fd)
    page = render_prediction_link_review_page(
        baseline_manifest_raw=raw,
        expected_baseline_manifest_sha256=expected_manifest_sha256,
        protocol_sha256=protocol_sha256,
        video_claim_review_receipt_sha256=video_claim_review_receipt_sha256)
    out_fd, root_fd = _open_private_child(private_root, output_name, create=True)
    try:
        data = page.encode("utf-8")
        view = memoryview(data)
        while view:
            count = os.write(out_fd, view)
            if count <= 0:
                raise OSError("private_output_short_write")
            view = view[count:]
        os.fsync(out_fd)
        if stat.S_IMODE(os.fstat(out_fd).st_mode) != 0o600:
            raise ValueError("private_output_mode_invalid")
    except Exception:
        os.close(out_fd)
        os.unlink(output_name, dir_fd=root_fd)
        os.close(root_fd)
        raise
    os.close(out_fd)
    os.close(root_fd)
    return hashlib.sha256(data).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-name", required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--video-claim-review-receipt-sha256", required=True)
    args = parser.parse_args(argv)
    private_root = Path(__file__).resolve().parents[1] / "data" / "private_nextgqa"
    digest = render_from_private_files(
        private_root=private_root, manifest_name=args.manifest_name,
        output_name=args.output_name,
        expected_manifest_sha256=args.expected_manifest_sha256,
        protocol_sha256=args.protocol_sha256,
        video_claim_review_receipt_sha256=args.video_claim_review_receipt_sha256)
    print(f"Private HTML rendered; sha256={digest}; mode=0600")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
