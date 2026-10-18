#!/usr/bin/env python3
"""Build a local-only, exploratory three-item NExT-GQA review HTML.

The output embeds protected local questions, claims, and videos. It is written
under data/private_nextgqa and must not be published or committed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "data/private_nextgqa/reviewer_package_v6_20261018_0243"
DEFAULT_OUTPUT = ROOT / "data/private_nextgqa/quick_review_ja_video_only_20261019.html"
PRIVATE_ROOT = (ROOT / "data/private_nextgqa").resolve()


def serialize_inline_json(value: object) -> str:
    """Serialize JSON without allowing data to terminate the surrounding script."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def private_output_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PRIVATE_ROOT)
    except ValueError as exc:
        raise ValueError("保護対象動画を含むartifactはdata/private_nextgqa内にのみ保存できます") from exc
    return resolved


def exploratory_item_id(review_token: str) -> str:
    """Return a stable, opaque local join key without exposing the raw token."""
    if not isinstance(review_token, str) or not review_token:
        raise ValueError("動画tokenが不正です")
    return hashlib.sha256(review_token.encode("utf-8")).hexdigest()[:20]


def browser_compatible_bytes(path: Path) -> bytes:
    """Return browser-playable video-only MP4 bytes, with audio/subtitles removed."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout).get("streams", [])
    video_codecs = [s.get("codec_name") for s in streams if s.get("codec_type") == "video"]
    has_nonvideo = any(s.get("codec_type") != "video" for s in streams)
    if video_codecs == ["h264"] and not has_nonvideo:
        return path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="res_quick_review_") as tmp:
        converted = Path(tmp) / "browser_compatible.mp4"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(converted)],
            check=True, capture_output=True, text=True,
        )
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "json", str(converted)],
            check=True, capture_output=True, text=True,
        )
        converted_streams = json.loads(result.stdout).get("streams", [])
        converted_video_codecs = [s.get("codec_name") for s in converted_streams
                                  if s.get("codec_type") == "video"]
        if converted_video_codecs != ["h264"] or any(
                s.get("codec_type") != "video" for s in converted_streams):
            raise ValueError(f"ブラウザー向け動画への変換に失敗しました: {path.name}")
        return converted.read_bytes()


def build(package_dir: Path, output: Path, *, overwrite: bool = False) -> None:
    output = private_output_path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"既存ファイルを上書きしません: {output}")
    source = package_dir / "index.html"
    html = source.read_text(encoding="utf-8")
    marker = '<script id="data" type="application/json">'
    start = html.index(marker) + len(marker)
    end = html.index("</script>", start)
    source_data = json.loads(html[start:end])
    records = source_data.get("records")
    if not isinstance(records, list) or len(records) != 3:
        raise ValueError("この簡易確認用artifactは3件の記録が必要です")

    items = []
    for row in records:
        token = row.get("review_token")
        if not isinstance(token, str) or not token or "/" in token or "\\" in token:
            raise ValueError("動画tokenが不正です")
        video_path = package_dir / f"{token}.mp4"
        raw = browser_compatible_bytes(video_path)
        items.append({
            "item_id": exploratory_item_id(token),
            "question": row["question"],
            "claim": row["claim_text"],
            "video_base64": base64.b64encode(raw).decode("ascii"),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    data = serialize_inline_json(items)
    document = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src blob:; connect-src 'none'; img-src data:; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>動画のかんたん確認</title>
<style>
:root{font:16px/1.6 system-ui,sans-serif;color:#183047;background:#f3f7fa}*{box-sizing:border-box}body{margin:0}main{max-width:820px;margin:32px auto;padding:0 18px 50px}.panel{background:white;border:1px solid #d7e2e8;border-radius:18px;padding:24px;box-shadow:0 8px 24px #1232}h1{font-size:1.8rem;line-height:1.25;margin:.2rem 0 1rem}h2{font-size:1.15rem}.muted{color:#536777}.notice{padding:12px 14px;border-radius:12px;background:#fff3d7}label{display:block;margin:14px 0 6px}input[type=text],textarea{font:inherit;width:100%;padding:10px;border:1px solid #9fb3c0;border-radius:9px}textarea{min-height:76px}video{width:100%;max-height:60vh;background:#111;border-radius:12px;margin:12px 0}.choices{display:grid;gap:8px;margin:14px 0}.choices label{margin:0;padding:11px;border:1px solid #c8d8e0;border-radius:10px}.choices input{margin-right:8px}button,.button{font:inherit;font-weight:700;border:0;border-radius:10px;padding:11px 17px;background:#176b87;color:white;cursor:pointer;text-decoration:none;display:inline-block}button:disabled{opacity:.45;cursor:default}.row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:18px}.hidden{display:none}.tag{font-size:.9rem;color:#536777}
</style></head><body><main>
<section id="welcome" class="panel"><p class="muted">ローカルで行う、3件だけの探索的な確認</p><h1>ニックネームを入力して始める</h1><p>動画と説明を見て、だいたい合っているかを答えます。迷ったら「わからない」で大丈夫です。厳密な幻覚判定や研究上の結論を出すための作業ではありません。</p><p class="notice">今回は映像だけを使い、音声は含めていません。ニックネームと回答はこの端末でJSONに保存します。実名を避け、JSONは共有しない場所に保存してください。質問・動画を外部へ送信する機能はありません。3件目まで回答して最後に保存してください。</p><label for="nickname">ニックネーム</label><input id="nickname" type="text" maxlength="40" autocomplete="nickname" placeholder="例：さくら"><div class="row"><span class="tag">所要時間の目安：数分</span><button id="start">質問を見る</button></div></section>
<section id="question" class="panel hidden"><div class="row"><span id="counter" class="tag"></span><span id="who" class="tag"></span></div><h1>この動画の説明は、だいたい合っていますか？</h1><video id="video" controls playsinline preload="metadata"></video><h2>質問</h2><p id="prompt"></p><h2>動画についての説明</h2><p id="claim"></p><div class="choices" id="choices"><label><input type="radio" name="judgment" value="roughly_matches">だいたい合っている</label><label><input type="radio" name="judgment" value="partly_differs">一部ちがうように見える</label><label><input type="radio" name="judgment" value="cannot_tell">動画だけではわからない</label></div><label for="time">気になったところ（任意・およその秒数）</label><input id="time" type="text" maxlength="80" placeholder="例：動画の後半、または 00:12 あたり"><label for="note">ひとことメモ（任意）</label><textarea id="note" maxlength="500" placeholder="気づいたことがあれば"></textarea><div class="row"><button id="next">次へ</button><span id="required" class="tag">回答を1つ選んでください</span></div></section>
<section id="done" class="panel hidden"><p class="muted">探索的な確認</p><h1>ありがとうございました</h1><p>回答をローカルJSONとして保存します。この結果だけで説明の完全な正しさや幻覚性を確定することはできません。</p><button id="save">回答を保存</button><p id="saved" class="tag" aria-live="polite"></p></section>
</main><script>
'use strict';
const items=__ITEMS__;
let nickname='';
let index=0;const answers=[];let activeVideoUrl='';
const $=id=>document.getElementById(id);
function begin(){nickname=$('nickname').value.trim();if(!nickname){$('nickname').focus();return}$('welcome').classList.add('hidden');showQuestion()}
function showQuestion(){if(index>=items.length){$('question').classList.add('hidden');$('done').classList.remove('hidden');if(activeVideoUrl){URL.revokeObjectURL(activeVideoUrl);activeVideoUrl=''}$('video').removeAttribute('src');$('video').load();return}const item=items[index];$('question').classList.remove('hidden');$('counter').textContent=`${index+1} / ${items.length} 件目`;$('who').textContent=`ニックネーム：${nickname}`;$('prompt').textContent=item.question;$('claim').textContent=item.claim;if(activeVideoUrl)URL.revokeObjectURL(activeVideoUrl);const binary=atob(item.video_base64);const bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);activeVideoUrl=URL.createObjectURL(new Blob([bytes],{type:'video/mp4'}));$('video').src=activeVideoUrl;$('video').load();document.querySelectorAll('[name=judgment]').forEach(x=>x.checked=false);$('time').value='';$('note').value='';$('next').textContent=index===items.length-1?'確認を終える':'次へ';$('required').textContent='回答を1つ選んでください'}
$('start').addEventListener('click',begin);$('nickname').addEventListener('keydown',e=>{if(e.key==='Enter')begin()});
$('next').addEventListener('click',()=>{const chosen=document.querySelector('[name=judgment]:checked');if(!chosen){$('required').textContent='どれか1つ選んでください';return}answers.push({item_id:items[index].item_id,item_number:index+1,judgment:chosen.value,approximate_time:$('time').value.trim()||null,note:$('note').value.trim()||null});index++;showQuestion()});
$('save').addEventListener('click',()=>{const result={schema_version:2,purpose:'exploratory_human_impression_check',status:'informal_observations_only',created_at_utc:new Date().toISOString(),nickname,answers,limitations:['簡単な印象確認であり、完全な幻覚判定ではない','item_idはローカルで回答を質問へ対応付けるための匿名化IDで、本人性や視聴を証明しない','本人確認・独立性・動画の視聴完了は検証していない','この結果だけからfaithfulnessの効果やモデル性能を結論づけない']};const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='exploratory_video_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1500);$('saved').textContent='ダウンロードを開始しました。保存先を確認してください。'});
</script></body></html>'''.replace("__ITEMS__", data)
    output.write_text(document, encoding="utf-8")
    output.chmod(0o600)
    print(f"Wrote local-only artifact: {output} ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.package_dir, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
