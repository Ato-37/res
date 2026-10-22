#!/usr/bin/env python3
"""Build an owner-only 3-item visual review artifact bound to two frozen results.

This exploratory artifact embeds protected local video and must remain under
data/private_nextgqa. It shows the 32 sampled model-input frames per item.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = (ROOT / "data/private_nextgqa").resolve()
DEFAULT_PACKET = PRIVATE / "claim_likelihood_claims_exploratory_32f_20261021_01.json"
DEFAULT_RESULT = PRIVATE / "claim_text_qa_exploratory_32f_shamv2_full_retry01_20261021_01.json"
DEFAULT_OUTPUT = PRIVATE / "claim_likelihood_human_review_v2_20261021_01.html"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_output(path: Path) -> Path:
    out = path.resolve()
    try:
        out.relative_to(PRIVATE)
    except ValueError as exc:
        raise ValueError("protected review artifact must remain under data/private_nextgqa") from exc
    return out


def frame_jpeg(video: Path, pts: float) -> bytes:
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{pts:.9f}", "-i", str(video),
                        "-frames:v", "1", "-vf", "scale=480:-2", "-f", "image2pipe",
                        "-vcodec", "mjpeg", "pipe:1"], check=True, capture_output=True)
    if not p.stdout.startswith(b"\xff\xd8"):
        raise ValueError("sampled frame extraction failed")
    return p.stdout


def _load(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("result JSON must be an object")
    return raw, value


def _write_owner_only(path: Path, content: bytes, *, overwrite: bool) -> None:
    """Publish a complete artifact atomically, without a permissive-mode window."""
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent)
    temp_path = Path(temp_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp_path, path)
        else:
            # link() is atomic and fails if another process created the target.
            os.link(temp_path, path, follow_symlinks=False)
            temp_path.unlink()
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        temp_path.unlink(missing_ok=True)
        raise


def build(packet_path: Path = DEFAULT_PACKET, result_path: Path = DEFAULT_RESULT,
          output: Path = DEFAULT_OUTPUT, *, overwrite: bool = False) -> None:
    output = safe_output(output)
    if output.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite existing review artifact")
    packet_raw, packet = _load(packet_path)
    result_raw, result = _load(result_path)
    if result.get("cohort_sha256") != sha256(packet_raw):
        raise ValueError("result is not bound to exact frozen packet bytes")
    records, result_rows = packet.get("records"), result.get("records")
    if (packet.get("schema_version") != 1 or packet.get("frames_per_video") != 32
            or packet.get("effective_videos") != 3 or not isinstance(records, list)
            or len(records) != 3 or result.get("completed_forwards") != 18
            or result.get("attempted_forwards") != 18 or result.get("frame_count") != 32
            or not isinstance(result_rows, list) or len(result_rows) != 3):
        raise ValueError("frozen input count/schema contract mismatch")
    items = []
    for ordinal, (row, scored) in enumerate(zip(records, result_rows), start=1):
        claim = row.get("generated_parsed", {}).get("claim")
        pts = row.get("sampled_frame_relative_pts_seconds")
        if (not isinstance(claim, str) or not claim.strip() or not isinstance(pts, list)
                or len(pts) != 32 or scored.get("source_video_sha256") != row.get("video_sha256")
                or scored.get("saved_claim_sha256") != sha256(claim.encode("utf-8"))
                or len(set(pts)) != 32):
            raise ValueError("packet/result item binding mismatch")
        video = Path(row["local_video_path"]).resolve()
        if not video.is_file() or sha256(video.read_bytes()) != row["video_sha256"]:
            raise ValueError("local video hash does not match frozen packet")
        frame_data = [base64.b64encode(frame_jpeg(video, float(t))).decode("ascii") for t in pts]
        video_data = base64.b64encode(video.read_bytes()).decode("ascii")
        pts_raw = json.dumps(pts, separators=(",", ":")).encode()
        items.append({
            "item_id": sha256((row["video_sha256"] + row["candidate_pair_sha256"]).encode())[:24],
            "item_number": ordinal, "question": row["question"], "claim": claim,
            "video_base64": video_data, "frames_base64": frame_data,
            "video_sha256": row["video_sha256"],
            "claim_sha256": sha256(claim.encode("utf-8")),
            "pts_sha256": sha256(pts_raw), "sampled_pts": pts,
            "model_frame_indices": list(range(32)),
        })
    manifest = {"schema_version": 1, "purpose": "exploratory_local_human_review",
                "packet_sha256": sha256(packet_raw), "result_sha256": sha256(result_raw),
                "items": [{k: item[k] for k in ("item_id", "video_sha256", "claim_sha256", "pts_sha256")}
                          for item in items]}
    data = json.dumps(items, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    manifest_json = json.dumps(manifest, separators=(",", ":"))
    html = r'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-artifact-sha256" content="__FP__"><title>動画・説明の予備確認</title>
<style>body{font:16px/1.55 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#183047}section{border:1px solid #ccd8df;border-radius:12px;padding:18px;margin:16px 0}video{width:100%;max-height:55vh;background:#111}.frames{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.frames img{width:100%}.choices label{display:block;padding:4px}button{font:inherit;padding:10px 16px}input,textarea{font:inherit;padding:8px;width:95%}.notice{background:#fff2d5;padding:12px}</style>
<main><section id="intro"><h1>動画と説明の予備確認</h1><p class="notice">各動画を見て、説明が動画に合うかを回答してください。これは探索的な印象確認で、厳密な正誤判定やモデル性能の評価ではありません。迷った場合は「判断できない」を選んでください。回答は外部送信されず、この端末にJSON保存されます。実名を使わず、JSONとHTMLを公開・共有しないでください。</p><label>ニックネーム <input id="nick" maxlength="40"></label><button id="start">開始</button></section><section id="task" hidden><p id="count"></p><h2>質問</h2><p id="question"></p><h2>説明（英語）</h2><p id="claim"></p><video id="video" controls></video><h3>32個のモデル入力フレーム</h3><p>各画像の番号は、モデルに渡したフレーム列の正確な0始まりindexです。表示時刻は元のサンプリング時刻です。</p><div class="frames" id="frames"></div><h3>この説明は動画に照らしてどう見えますか？</h3><div class="choices"><label><input type="radio" name="judgment" value="roughly_matches">だいたい合っている</label><label><input type="radio" name="judgment" value="partly_differs">一部異なる</label><label><input type="radio" name="judgment" value="cannot_tell">判断できない</label></div><label>任意のメモ<textarea id="note" maxlength="500"></textarea></label><button id="next">次へ</button></section><section id="done" hidden><h2>回答完了</h2><p id="done-count"></p><button id="another">次のレビュー担当者</button> <button id="save">全レビュー担当者の回答をJSON保存</button></section></main>
<script>const items=__ITEMS__,binding=__MANIFEST__;let idx=0,nick='',answers=[],respondents=[];const $=x=>document.getElementById(x);function start(){nick=$('nick').value.trim();if(!nick)return;idx=0;answers=[];show()}function show(){if(idx===items.length){respondents.push({nickname:nick,answers});$('task').hidden=true;$('done').hidden=false;$('done-count').textContent=`${respondents.length}人分の回答が完了しました`;$('another').hidden=false;return}let x=items[idx];$('intro').hidden=true;$('done').hidden=true;$('task').hidden=false;$('count').textContent=`${idx+1} / ${items.length}`;$('question').textContent=x.question;$('claim').textContent=x.claim;$('video').src=URL.createObjectURL(new Blob([Uint8Array.from(atob(x.video_base64),c=>c.charCodeAt(0))],{type:'video/mp4'}));let grid=$('frames');grid.replaceChildren();x.frames_base64.forEach((b,i)=>{let wrap=document.createElement('div'),im=document.createElement('img'),cap=document.createElement('div');im.src='data:image/jpeg;base64,'+b;cap.textContent=`model frame index ${x.model_frame_indices[i]} · ${Number(x.sampled_pts[i]).toFixed(3)} s`;wrap.append(im,cap);grid.append(wrap)});document.querySelectorAll('[name=judgment]').forEach(e=>e.checked=false);$('note').value='';$('next').textContent=idx===items.length-1?'回答を終える':'次へ'}$('start').onclick=start;$('another').onclick=()=>{$('done').hidden=true;$('intro').hidden=false;$('nick').value='';$('nick').focus()};$('next').onclick=()=>{let j=document.querySelector('[name=judgment]:checked');if(!j)return;answers.push({item_id:items[idx].item_id,item_number:idx+1,judgment:j.value,note:$('note').value||null});idx++;show()};$('save').onclick=()=>{let out={schema_version:1,purpose:'exploratory_local_human_review',status:'informal_observations_only',artifact_fingerprint:document.querySelector('meta[name="review-artifact-sha256"]').content,source_binding:binding,created_at_utc:new Date().toISOString(),respondents};let blob=new Blob([JSON.stringify(out,null,2)+'\n'],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='claim_likelihood_human_review_responses.json';document.body.appendChild(a);a.click();a.remove()};</script></html>'''
    html = html.replace("__ITEMS__", data).replace("__MANIFEST__", manifest_json)
    if html.count("__FP__") != 1:
        raise ValueError("fingerprint placeholder invalid")
    fingerprint = sha256(html.encode())
    html = html.replace("__FP__", fingerprint)
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_owner_only(output, html.encode("utf-8"), overwrite=overwrite)
    print(f"Wrote owner-only local review artifact ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.packet, args.result, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
