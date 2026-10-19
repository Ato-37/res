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
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "data/private_nextgqa/reviewer_package_v6_20261018_0243"
DEFAULT_OUTPUT = ROOT / "data/private_nextgqa/quick_review_ja_simple_20261019_itemid_v14.html"
DEFAULT_TRANSLATIONS = ROOT / "data/private_nextgqa/quick_review_ja_translations_20261019.json"
PRIVATE_ROOT = (ROOT / "data/private_nextgqa").resolve()


def translation_key(question: str, claim: str) -> str:
    """Bind a local translation to both protected source strings without exposing them."""
    if not isinstance(question, str) or not isinstance(claim, str):
        raise ValueError("訳文の照合元が不正です")
    return hashlib.sha256((question + "\0" + claim).encode("utf-8")).hexdigest()


def load_translations(path: Path = DEFAULT_TRANSLATIONS) -> dict[str, tuple[str, str]]:
    """Load local-only translations keyed by the question-and-claim source hash."""
    path = private_output_path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or set(payload) != {"schema_version", "items"}
            or type(payload.get("schema_version")) is not int
            or payload["schema_version"] != 1 or not isinstance(payload.get("items"), dict)):
        raise ValueError("日本語訳ファイルのschemaが不正です")
    result = {}
    for key, item in payload["items"].items():
        if (not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key)
                or not isinstance(item, dict)
                or set(item) != {"question_ja", "claim_ja"}
                or not all(isinstance(item[field], str) and item[field].strip()
                           for field in ("question_ja", "claim_ja"))):
            raise ValueError("日本語訳ファイルの項目が不正です")
        result[key] = (item["question_ja"], item["claim_ja"])
    return result


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


def build(package_dir: Path, output: Path, *, overwrite: bool = False,
          translations: dict[str, tuple[str, str]] | None = None) -> None:
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
    if translations is None:
        translations = load_translations()

    items = []
    for row in records:
        token = row.get("review_token")
        if not isinstance(token, str) or not token or "/" in token or "\\" in token:
            raise ValueError("動画tokenが不正です")
        video_path = package_dir / f"{token}.mp4"
        raw = browser_compatible_bytes(video_path)
        question = row["question"]
        claim = row["claim_text"]
        translated = translations.get(translation_key(question, claim))
        if translated is None:
            raise ValueError("固定3件の日本語訳がありません。内容を確認してから追加してください")
        question_ja, claim_ja = translated
        items.append({
            "item_id": exploratory_item_id(token),
            "question": question,
            "question_ja": question_ja,
            "claim": claim,
            "claim_ja": claim_ja,
            "video_base64": base64.b64encode(raw).decode("ascii"),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    data = serialize_inline_json(items)
    document = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-artifact-sha256" content="__ARTIFACT_FINGERPRINT__">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src blob:; connect-src 'none'; img-src data:; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>動画のかんたん確認</title>
<style>
:root{font:16px/1.6 system-ui,sans-serif;color:#183047;background:#f3f7fa}*{box-sizing:border-box}body{margin:0}main{max-width:820px;margin:32px auto;padding:0 18px 50px}.panel{background:white;border:1px solid #d7e2e8;border-radius:18px;padding:24px;box-shadow:0 8px 24px #1232}h1{font-size:1.8rem;line-height:1.25;margin:.2rem 0 1rem}h2{font-size:1.15rem}.muted{color:#536777}.notice{padding:12px 14px;border-radius:12px;background:#fff3d7}label{display:block;margin:14px 0 6px}input[type=text],textarea{font:inherit;width:100%;padding:10px;border:1px solid #9fb3c0;border-radius:9px}textarea{min-height:76px}video{width:100%;max-height:60vh;background:#111;border-radius:12px;margin:12px 0}.choices{display:grid;gap:8px;margin:14px 0}.choices label{margin:0;padding:11px;border:1px solid #c8d8e0;border-radius:10px}.choices input{margin-right:8px}button,.button{font:inherit;font-weight:700;border:0;border-radius:10px;padding:11px 17px;background:#176b87;color:white;cursor:pointer;text-decoration:none;display:inline-block}button:disabled{opacity:.45;cursor:default}.row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:18px}.hidden{display:none}.tag{font-size:.9rem;color:#536777}
</style></head><body><main>
<section id="welcome" class="panel"><p class="muted">動画についての短い確認（3問）</p><h1 id="welcome-title">ニックネームを入力</h1><p id="welcome-instruction">入力して「質問へ」を押すと、すぐ動画と質問が表示されます。ファイル選択や時刻入力はありません。</p><p class="notice">動画を見て、説明がだいたい合っているか、違うように感じるかを答えるだけです。気軽な予備確認で、厳密な幻覚判定や研究上の性能評価ではありません。実名は避けてください。回答は外へ送信されません。必要なら最後に、この端末へJSONで保存できます。</p><form id="welcome-form"><label for="nickname">ニックネーム</label><input id="nickname" type="text" maxlength="40" autocomplete="nickname" placeholder="例：さくら"><div class="row"><span class="tag">目安：数分</span><button id="start" type="submit">質問へ</button></div></form><button id="cancel-next" class="hidden" type="button">前の画面に戻る</button></section>
<section id="question" class="panel hidden"><div class="row"><span id="counter" class="tag"></span><span id="who" class="tag"></span></div><h1>動画と説明を見てください</h1><video id="video" controls playsinline preload="metadata"></video><h2>質問</h2><p id="prompt"></p><h2>動画についての説明</h2><p id="claim"></p><details id="source-language"><summary>元の英語の質問・説明を見る</summary><p><strong>質問（英語）</strong></p><p id="prompt-source"></p><p><strong>説明（英語）</strong></p><p id="claim-source"></p></details><aside class="notice" id="rating-guide">②は動画と見比べた答えの正しさ、③は説明が動画中の出来事を根拠として示しているかを見ます。③では、質問の答えに必要な行動や、必要な場合はその順序が説明に含まれているか確認してください。②と③は別の観点です。質問の意味や動画の場面がはっきりしないときは推測せず、①は「わかりにくい」、②③は「判断できない」を選べます。</aside><p>① 質問を読んで、何を答えればよいか分かりますか？</p><div class="choices" id="clarity-choices"><label><input type="radio" name="question_clarity" value="clear">わかる</label><label><input type="radio" name="question_clarity" value="unclear">わかりにくい</label><label><input type="radio" name="question_clarity" value="cannot_tell">判断できない</label></div><p>② 動画を見て、説明に書かれた答えは正しいと思いますか？</p><div class="choices" id="answer-fit-choices"><label><input type="radio" name="answer_question_fit" value="fits">合っている</label><label><input type="radio" name="answer_question_fit" value="does_not_fit">違っている</label><label><input type="radio" name="answer_question_fit" value="cannot_tell">判断できない</label></div><p>③ 説明は、答えの根拠となる動画中の出来事（必要なら動きの順序）を十分に示していますか？</p><div class="choices" id="evidence-choices"><label><input type="radio" name="video_evidence_sufficiency" value="sufficient">十分</label><label><input type="radio" name="video_evidence_sufficiency" value="insufficient">足りない点がある</label><label><input type="radio" name="video_evidence_sufficiency" value="cannot_tell">判断できない</label></div><div class="row"><button id="next">次へ</button><span id="required" class="tag">3つそれぞれ1つ選んでください</span></div></section>
<section id="done" class="panel hidden"><p class="muted">簡単な予備確認</p><h1>ありがとうございました</h1><p id="reviewer-count">1人目の回答が完了しました。</p><p>次の人は「次の人へ」を押し、自分のニックネームを入力してから質問に進んでください。全員が終わったら「全員分を保存」を押してください。</p><p>これは短い印象確認で、説明の正しさや幻覚性を確定する検査ではありません。</p><p class="notice">JSONには各人のニックネームが含まれます。実名は使わず、ファイルを共有・公開しないでください。</p><div class="row"><button id="next-reviewer">次の人へ（ニックネーム入力）</button><button id="save">全員分を保存</button></div><p id="saved" class="tag" aria-live="polite"></p><a id="saved-file" class="button hidden" rel="noopener">JSONをもう一度ダウンロード</a><section id="summary" class="hidden" aria-live="polite"><h2>今回の回答</h2><p id="summary-count"></p><ul id="summary-items"></ul><p class="muted">回答の一覧です。正式なレビューやfaithfulness評価ではありません。</p></section></section>
</main><script>
'use strict';
const items=__ITEMS__;
let nickname='';
let index=0;let answers=[];const respondents=[];let activeVideoUrl='';let exportUrl='';
const $=id=>document.getElementById(id);
function begin(){nickname=$('nickname').value.trim();if(!nickname){$('nickname').focus();return}index=0;answers=[];$('nickname').value='';$('welcome').classList.add('hidden');$('cancel-next').classList.add('hidden');showQuestion()}
function showQuestion(){if(index>=items.length){respondents.push({nickname,answers:answers.slice()});$('question').classList.add('hidden');$('done').classList.remove('hidden');$('reviewer-count').textContent=`${respondents.length}人分の回答が完了しました。`;if(activeVideoUrl){URL.revokeObjectURL(activeVideoUrl);activeVideoUrl=''}$('video').removeAttribute('src');$('video').load();showSummary();return}const item=items[index];$('question').classList.remove('hidden');$('counter').textContent=`${index+1} / ${items.length} 件目`;$('who').textContent=`ニックネーム：${nickname}`;$('prompt').textContent=item.question_ja;$('claim').textContent=item.claim_ja;$('prompt-source').textContent=item.question;$('claim-source').textContent=item.claim;$('source-language').open=false;if(activeVideoUrl)URL.revokeObjectURL(activeVideoUrl);const binary=atob(item.video_base64);const bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);activeVideoUrl=URL.createObjectURL(new Blob([bytes],{type:'video/mp4'}));$('video').src=activeVideoUrl;$('video').load();document.querySelectorAll('[name=question_clarity],[name=answer_question_fit],[name=video_evidence_sufficiency]').forEach(x=>x.checked=false);$('next').textContent=index===items.length-1?'確認を終える':'次へ';$('required').textContent='3つそれぞれ1つ選んでください'}
$('welcome-form').addEventListener('submit',event=>{event.preventDefault();begin()});
$('next').addEventListener('click',()=>{const clarity=document.querySelector('[name=question_clarity]:checked');const fit=document.querySelector('[name=answer_question_fit]:checked');const evidence=document.querySelector('[name=video_evidence_sufficiency]:checked');if(!clarity||!fit||!evidence){$('required').textContent='3つそれぞれ1つ選んでください';return}answers.push({item_id:items[index].item_id,item_number:index+1,question_clarity:clarity.value,answer_question_fit:fit.value,video_evidence_sufficiency:evidence.value});index++;showQuestion()});
function showSummary(){const clarityLabels={clear:'質問：わかりやすい',unclear:'質問：わかりにくい',cannot_tell:'判断できない'};const fitLabels={fits:'答え：合っている',does_not_fit:'答え：違っている',cannot_tell:'判断できない'};const evidenceLabels={sufficient:'動画根拠：十分',insufficient:'動画根拠：足りない',cannot_tell:'動画根拠：判断できない'};const current=respondents[respondents.length-1];$('summary-count').textContent=`${current.answers.length} / ${items.length} 件に回答済み`;$('summary-items').replaceChildren(...current.answers.map(answer=>{const li=document.createElement('li');li.textContent=`${answer.item_number}件目：${clarityLabels[answer.question_clarity]}／${fitLabels[answer.answer_question_fit]}／${evidenceLabels[answer.video_evidence_sufficiency]}`;return li}));$('summary').classList.remove('hidden')}
function prepareNextReviewer(){const nextNumber=respondents.length+1;nickname='';index=0;answers=[];$('nickname').value='';$('welcome-title').textContent=`${nextNumber}人目のニックネームを入力`;$('welcome-instruction').textContent='ニックネームを入力して「質問へ」を押すと、この人の1問目が始まります。';$('cancel-next').classList.remove('hidden');$('done').classList.add('hidden');$('summary').classList.add('hidden');$('welcome').classList.remove('hidden');$('nickname').focus()}
$('next-reviewer').addEventListener('click',prepareNextReviewer);
$('cancel-next').addEventListener('click',()=>{$('welcome').classList.add('hidden');$('cancel-next').classList.add('hidden');$('done').classList.remove('hidden');$('nickname').value=''});
 $('save').addEventListener('click',()=>{const result={schema_version:6,artifact_fingerprint:document.querySelector('meta[name="review-artifact-sha256"]').content,purpose:'exploratory_human_impression_check',status:'informal_observations_only',created_at_utc:new Date().toISOString(),respondents,limitations:['簡単な印象確認であり、完全な幻覚判定ではない','item_idはローカルで回答を質問へ対応付けるための匿名化IDで、本人性や視聴を証明しない','本人確認・独立性・動画の視聴完了は検証していない','この結果だけからfaithfulnessの効果やモデル性能を結論づけない']};const blob=new Blob([JSON.stringify(result,null,2)+'\n'],{type:'application/json'});if(exportUrl)URL.revokeObjectURL(exportUrl);exportUrl=URL.createObjectURL(blob);const a=$('saved-file');const stamp=new Date().toISOString().replace(/[:.]/g,'-');a.href=exportUrl;a.download='exploratory_video_review_'+stamp+'.json';a.classList.remove('hidden');a.click();$('saved').textContent=`${respondents.length}人分のJSONのダウンロードを開始しました。完了したとは限らないため、ブラウザのダウンロード一覧で保存を確認してください。できていない場合は下のリンクを押してください。`});
</script></body></html>'''
    document = document.replace("__ITEMS__", data)
    fingerprint_placeholder = "__ARTIFACT_FINGERPRINT__"
    if document.count(fingerprint_placeholder) != 1:
        raise ValueError("artifact fingerprint placeholder must occur exactly once")
    fingerprint = hashlib.sha256(document.encode("utf-8")).hexdigest()
    document = document.replace(fingerprint_placeholder, fingerprint)
    output.write_bytes(document.encode("utf-8"))
    output.chmod(0o600)
    print(f"Wrote local-only artifact: {output} ({output.stat().st_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--translations", type=Path, default=DEFAULT_TRANSLATIONS,
                        help="ローカル限定の日本語訳JSON（保護対象のため公開しない）")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build(args.package_dir, args.output, overwrite=args.overwrite,
          translations=load_translations(args.translations))


if __name__ == "__main__":
    main()
