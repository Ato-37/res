#!/usr/bin/env python3
"""Create a private, low-burden claim-to-model-frame human review artifact."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import stat

from run_test_split_internal_bias_frame_mean_interaction import (
    LOCK_REL, PAIR_FILTER_REL, PACKETS, _read_private_json, verify_locked_assignment,
)
from run_saved_claim_frame_permutation_attention_smoke import _read_packet, _sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_nextgqa"
FRAME_COUNT = 16
TOLERANCE_SECONDS = 0.05


def _private_file(path: Path) -> bytes:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600):
        raise ValueError("private_input_permissions_invalid")
    return path.read_bytes()


def _safe_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = path.parent.lstat()
    if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid()):
        raise ValueError("private_output_parent_invalid")
    os.chmod(path.parent, 0o700)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def frame_jpeg_data_url(frame) -> str:
    """Downscale an exact model-input RGB frame for a private review thumbnail."""
    from PIL import Image
    if not isinstance(frame, Image.Image) or frame.mode != "RGB":
        raise ValueError("review_frame_invalid")
    image = frame.copy()
    image.thumbnail((720, 480), Image.Resampling.LANCZOS)
    import io
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_decision(decision: str, frame_indices: list[int]) -> bool:
    if (not isinstance(decision, str)
            or decision not in {"supports", "contradicts", "not_visible_in_model_frames", "cannot_judge"}):
        return False
    if (not isinstance(frame_indices, list)
            or any(type(index) is not int or not 0 <= index < FRAME_COUNT for index in frame_indices)
            or len(frame_indices) != len(set(frame_indices))):
        return False
    return bool(frame_indices) if decision in {"supports", "contradicts"} else not frame_indices


def render(items: list[dict], provenance: dict | None = None,
           title: str = "動画説明の根拠フレーム確認") -> bytes:
    if not 1 <= len(items) <= 3:
        raise ValueError("review_item_count_must_be_one_to_three")
    source_data = json.dumps(items, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    provenance_data = json.dumps(provenance or {}, ensure_ascii=False,
                                 separators=(",", ":")).replace("<", "\\u003c")
    doc = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-artifact-sha256" content="__FINGERPRINT__">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; media-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>動画説明の根拠確認</title><style>
:root{font:16px/1.55 system-ui,sans-serif;color:#173249;background:#f1f6f8}*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:20px auto;padding:0 14px 36px}.card{background:#fff;border:1px solid #cfdae0;border-radius:16px;padding:20px;box-shadow:0 5px 18px #17324918}h1{font-size:1.55rem;line-height:1.3;margin:.3rem 0 .8rem}h2{font-size:1.05rem}.notice{background:#fff4d9;padding:12px;border-radius:10px}.muted{color:#526777}.hidden{display:none}input[type=text]{font:inherit;padding:10px;border:1px solid #9eb1bd;border-radius:8px;width:100%}button{font:inherit;font-weight:700;border:0;border-radius:9px;padding:10px 16px;background:#126b83;color:white;cursor:pointer}button:disabled{opacity:.45;cursor:default}.row{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-top:16px}.tag{font-size:.9rem;color:#516777}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frame{border:1px solid #c9d5db;border-radius:9px;padding:5px;min-width:0}.frame img{display:block;width:100%;aspect-ratio:16/10;object-fit:contain;background:#111;border-radius:5px}.frame label{display:flex;align-items:center;gap:6px;font-size:.86rem}.frame input{width:18px;height:18px}.choices{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}.choices label{border:1px solid #c9d5db;border-radius:9px;padding:10px}.claim{background:#f4f8fa;border-left:4px solid #26849a;padding:10px 14px}.pill{display:inline-block;background:#e4f0f3;border-radius:20px;padding:2px 9px;margin:3px;font-size:.85rem}@media(max-width:620px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}.choices{grid-template-columns:1fr}}
.player{display:grid;grid-template-columns:minmax(0,1fr);gap:8px;margin:12px 0 18px;padding:12px;border:1px solid #c9d5db;border-radius:10px;background:#f8fbfc}.player img{display:block;width:100%;max-height:480px;object-fit:contain;background:#111;border-radius:6px}.player-controls{display:grid;grid-template-columns:auto auto 1fr auto auto;gap:8px;align-items:center}.player input[type=range]{width:100%}.frame.current{outline:3px solid #e08a21}.player-note{margin:0}@media(max-width:620px){.player-controls{grid-template-columns:auto auto 1fr auto}.player-time{grid-column:1/-1}}</style></head><body><main>
<section id="welcome" class="card"><p class="muted">__COUNT__件の動画説明を確認します（1人あたり数分）</p><h1 id="welcome-title">ニックネームを入力</h1><p>動画と説明を見比べ、根拠になる場面のコマを選んでください。ファイル選択や時刻入力はありません。回答はこの端末内だけに保存できます。</p><div class="notice"><b>見方：</b>各説明が、質問への答えに必要な動画内の行動や出来事に支えられているかを判断します。下の16枚は動画から抽出したデコード済みフレームの縮小表示です（全動画でも、processorが作るcrop/patchでもありません）。支持・矛盾を選ぶ場合、そう判断したフレームを1枚以上選択してください。どれにも現れない／判断できない場合は、その選択肢を使えます。正解回答は表示されません。<br>この画面の回答だけでは、本人の独立性や視聴を証明しません。ニックネームを使い、実名を入力しないでください。</div><form id="welcome-form"><label for="nickname">ニックネーム</label><input id="nickname" maxlength="40" autocomplete="nickname" placeholder="例：さくら"><div class="row"><span class="tag">各人が__COUNT__件すべてに回答します</span><button type="submit">質問へ</button></div></form></section>
<section id="question" class="card hidden"><div class="row"><span id="counter" class="tag"></span><span id="who" class="tag"></span></div><h1>説明が動画の根拠に支えられているか</h1><h2>質問</h2><p id="question-text"></p><h2>動画の説明</h2><p id="claim" class="claim"></p><div class="notice">まずサンプルの流れを見てください。ここに表示するのは動画全体ではなく、モデルに入力された16枚を時刻順に並べたものです。再生・コマ送りで確認し、根拠だと判断したコマは下の一覧で選んでください。</div><section class="player" aria-label="モデル入力フレームの時刻順表示"><img id="player-image" alt="モデル入力フレーム"><p class="player-note muted">サンプルフレームのスライド表示（動画の連続再生ではありません）</p><div class="player-controls"><button type="button" id="frame-prev" aria-label="前のフレーム">前へ</button><button type="button" id="player-toggle">再生</button><input id="player-seek" type="range" min="0" max="15" value="0" aria-label="表示するフレーム"><button type="button" id="frame-next" aria-label="次のフレーム">次へ</button><span id="player-time" class="tag player-time"></span></div><span id="player-position" class="tag" aria-live="polite"></span></section><div id="frames" class="frames"></div><h2>この説明と動画の関係</h2><div class="choices"><label><input type="radio" name="decision" value="supports">説明の内容は動画に支持されている</label><label><input type="radio" name="decision" value="contradicts">動画が説明の内容と矛盾している</label><label><input type="radio" name="decision" value="not_visible_in_model_frames">根拠になる場面がこの16枚には見当たらない</label><label><input type="radio" name="decision" value="cannot_judge">判断できない</label></div><p id="validation" class="tag">判定を選んでください。</p><div class="row"><button id="next">次へ</button><span class="tag">1人目・全__COUNT__件</span></div></section>
<section id="done" class="card hidden"><h1>この人の確認が終わりました</h1><p id="reviewer-count"></p><p>次の人は「次の人へ」を押して、自分のニックネームを入力してください。最後に全員分を1つのJSONとして保存します。</p><div class="notice">回答JSONにはニックネームと選択フレームが含まれます。ファイルは研究担当者へローカルで渡し、公開しないでください。まだ保存されていない場合、ページを閉じると回答は失われます。</div><div class="row"><button id="next-reviewer">次の人へ</button><button id="save">全員分をJSONで保存</button></div><p id="saved" class="tag" aria-live="polite"></p><a id="saved-file" class="hidden" download="claim_support_frame_review.json">JSONを再ダウンロード</a><div id="summary"></div></section>
</main><script>__SCRIPT__</script></body></html>'''
    doc = doc.replace("__COUNT__", str(len(items)))
    doc = doc.replace("__ITEMS__", source_data)
    doc = doc.replace("__PROVENANCE__", provenance_data)
    script = r'''const items=__ITEMS__;const sourceProvenance=__PROVENANCE__;let nickname='',itemIndex=0,answers=[],reviewers=[],exportUrl='',playerTimer=null,playerFrameIndex=0;const $=id=>document.getElementById(id);function stopPlayer() {if(playerTimer!==null){clearInterval(playerTimer);playerTimer=null}if($('player-toggle'))$('player-toggle').textContent='再生'}function setPlayerFrame(index){if(!$('player-image'))return;const frames=items[itemIndex].frames;playerFrameIndex=Math.max(0,Math.min(frames.length-1,index));const f=frames[playerFrameIndex];$('player-image').src=f.data;$('player-image').alt=`モデル入力フレーム ${playerFrameIndex}`;$('player-position').textContent=`${playerFrameIndex+1} / ${frames.length} 枚目`;$('player-time').textContent=`${f.time.toFixed(2)}秒`;$('player-seek').value=String(playerFrameIndex);document.querySelectorAll('#frames .frame').forEach((el,i)=>el.classList.toggle('current',i===playerFrameIndex))}function start(){nickname=$('nickname').value.trim();if(!nickname){$('nickname').focus();return}answers=[];itemIndex=0;$('nickname').value='';$('welcome').classList.add('hidden');showItem()}function showItem(){stopPlayer();const x=items[itemIndex];$('question').classList.remove('hidden');$('counter').textContent=`${itemIndex+1} / ${items.length} 件目`;$('who').textContent=`ニックネーム：${nickname}`;$('question-text').textContent=x.question;$('claim').textContent=x.claim;const grid=$('frames');grid.replaceChildren();x.frames.forEach((f,i)=>{const cell=document.createElement('div');cell.className='frame';const img=document.createElement('img');img.src=f.data;img.alt=`モデル入力フレーム ${i}`;const label=document.createElement('label');const check=document.createElement('input');check.type='checkbox';check.value=String(i);const text=document.createElement('span');text.textContent=`フレーム ${String(i).padStart(2,'0')}（${f.time.toFixed(2)}秒）`;label.append(check,text);cell.append(img,label);img.addEventListener('click',()=>setPlayerFrame(i));grid.append(cell)});setPlayerFrame(0);document.querySelectorAll('[name=decision]').forEach(x=>x.checked=false);$('validation').textContent='判定を選んでください。';$('next').textContent=itemIndex===items.length-1?'確認を終える':'次へ'}if($('frame-prev')){$('frame-prev').addEventListener('click',()=>{stopPlayer();setPlayerFrame(playerFrameIndex-1)});$('frame-next').addEventListener('click',()=>{stopPlayer();setPlayerFrame(playerFrameIndex+1)});$('player-seek').addEventListener('input',e=>{stopPlayer();setPlayerFrame(Number(e.target.value))});$('player-toggle').addEventListener('click',()=>{if(playerTimer!==null){stopPlayer();return}$('player-toggle').textContent='一時停止';playerTimer=setInterval(()=>{if(playerFrameIndex>=items[itemIndex].frames.length-1){stopPlayer();return}setPlayerFrame(playerFrameIndex+1)},800)})}$('welcome-form').addEventListener('submit',e=>{e.preventDefault();start()});$('next').addEventListener('click',()=>{stopPlayer();const selected=document.querySelector('[name=decision]:checked');const frames=[...document.querySelectorAll('#frames input:checked')].map(x=>Number(x.value));if(!selected){$('validation').textContent='判定を選んでください。';return}const needs=['supports','contradicts'].includes(selected.value);if(needs!==Boolean(frames.length)){$('validation').textContent=needs?'根拠としたフレームを1枚以上選んでください。':'この判定ではフレームを選ばずに進んでください。';return}answers.push({item_id:items[itemIndex].item_id,decision:selected.value,frame_indices:frames});if(++itemIndex<items.length){showItem();return}reviewers.push({nickname,answers:answers.slice()});$('question').classList.add('hidden');$('done').classList.remove('hidden');$('reviewer-count').textContent=`${reviewers.length}人分の回答を一時保存しました。`;const p=document.createElement('p');p.textContent=`今回 ${answers.length} 件を確認しました。`;$('summary').replaceChildren(p)});$('next-reviewer').addEventListener('click',()=>{if(exportUrl){URL.revokeObjectURL(exportUrl);exportUrl=''}$('saved-file').classList.add('hidden');$('saved').textContent='';nickname='';answers=[];itemIndex=0;$('welcome-title').textContent=`${reviewers.length+1}人目のニックネームを入力`;$('welcome').classList.remove('hidden');$('done').classList.add('hidden');$('nickname').focus()});$('save').addEventListener('click',()=>{const payload={schema_version:1,purpose:'local_claim_support_frame_review',status:'human_responses_self_reported_unvalidated',artifact_fingerprint:document.querySelector('meta[name=review-artifact-sha256]').content,created_at_utc:new Date().toISOString(),reviewers,limitations:['回答と選択フレームは自己申告であり、本人性・独立性・動画の視聴を証明しない','表示は動画から抽出した16枚のデコード済みフレームの縮小画像であり、モデルprocessorのcrop/patchそのものではない','16フレームに写らない場面について動画全体に証拠がないとは結論できない','結果は研究担当者がartifact・動画・protocolに照合するまで未検証']};if(exportUrl)URL.revokeObjectURL(exportUrl);exportUrl=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}));const a=$('saved-file');a.href=exportUrl;a.download=`claim_support_frame_review_${new Date().toISOString().replace(/[:.]/g,'-')}.json`;a.classList.remove('hidden');a.click();$('saved').textContent='JSONの保存を開始しました。Downloadsで保存完了を確認してください。保存できない場合は「JSONを再ダウンロード」を押してください。'});'''
    script = script.replace("__ITEMS__", source_data).replace("__PROVENANCE__", provenance_data)
    doc = doc.replace("__SCRIPT__", script)
    if doc.count("__FINGERPRINT__") != 1 or doc.count("__PROVENANCE__") != 0:
        raise ValueError("review_fingerprint_placeholder_invalid")
    fingerprint = hashlib.sha256(doc.encode("utf-8")).hexdigest()
    return doc.replace("__FINGERPRINT__", fingerprint).encode("utf-8")


def build(output_rel: Path, translations_rel: Path) -> dict:
    output = (ROOT / output_rel).resolve()
    private_root = PRIVATE.resolve()
    if not output.is_relative_to(private_root) or output.suffix != ".html":
        raise ValueError("private_review_output_invalid")
    if output.exists():
        raise FileExistsError("review_artifact_already_exists")
    translations_path = (ROOT / translations_rel).resolve()
    if not translations_path.is_relative_to(private_root):
        raise ValueError("private_translation_path_invalid")
    translations = json.loads(_private_file(translations_path).decode("utf-8")).get("items")
    if not isinstance(translations, dict):
        raise ValueError("review_translations_invalid")
    lock, lock_sha = _read_private_json(ROOT / LOCK_REL)
    _, pair_filter_sha = _read_private_json(ROOT / PAIR_FILTER_REL)
    from claim_mask_video_decoder import decode_sampled_video, verify_sampled_frame_pixels
    from fractions import Fraction

    items = []
    source_manifest = {"schema_version": 1, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "locked_assignment_sha256": lock_sha,
                       "pair_filter_sha256": pair_filter_sha,
                       "translation_sha256": _sha256_bytes(_private_file(translations_path)),
                       "decoder_sha256": hashlib.sha256((ROOT / "scripts" / "claim_mask_video_decoder.py").read_bytes()).hexdigest(),
                       "items": []}
    for ordinal, (label, (packet_rel, record_index, baseline_rel)) in enumerate(PACKETS.items()):
        row, packet_sha = _read_packet(ROOT, packet_rel, record_index, expected_split="test")
        verify_locked_assignment(lock, pair_filter_sha, ordinal, row, lock.get("selected_pair_filter_sha256"))
        claim = row.get("generated_parsed", {}).get("claim")
        question, video_path = row.get("question"), row.get("local_video_path")
        if (not isinstance(claim, str) or not claim.strip() or not isinstance(question, str)
                or not isinstance(video_path, str) or not Path(video_path).is_file()):
            raise ValueError("review_packet_row_invalid")
        baseline, baseline_sha = _read_private_json(ROOT / baseline_rel)
        interaction_path = PRIVATE / f"test_split_internal_bias_frame_mean_interaction_{label}_20261021_01.json"
        interaction, interaction_sha = _read_private_json(interaction_path)
        row_token = _sha256_bytes((row["video_sha256"] + str(row.get("dataset_question_id", ""))).encode())
        if (baseline.get("private_input_packet_sha256") != packet_sha
                or baseline.get("source_video_sha256") != row["video_sha256"]
                or baseline.get("saved_claim_text_sha256") != _sha256_bytes(claim.encode())
                or interaction.get("row_token_sha256") != row_token
                or interaction.get("private_input_packet_sha256") != packet_sha
                or row.get("human_review", {}).get("claim_visually_supported_by_reported_frames") is not None):
            raise ValueError("review_packet_provenance_join_invalid")
        frame_pts = row.get("sampled_frame_relative_pts_seconds")
        if not isinstance(frame_pts, list) or len(frame_pts) != FRAME_COUNT:
            raise ValueError("review_frame_grid_missing")
        frames, metadata = decode_sampled_video(video_path, num_frames=FRAME_COUNT,
            uniform_tolerance_seconds=Fraction(1, 20))
        verify_sampled_frame_pixels(frames, metadata)
        if (metadata.get("source_sha256") != row["video_sha256"]
                or any(abs(float(pt) - float(Fraction(*sample["relative_presentation_time"]))) > 1e-6
                       for pt, sample in zip(frame_pts, metadata["sampled_frames"], strict=True))):
            raise ValueError("review_frame_grid_video_binding_invalid")
        translation_key = hashlib.sha256((question + "\0" + claim).encode("utf-8")).hexdigest()
        translated = translations.get(translation_key)
        if (not isinstance(translated, dict) or not isinstance(translated.get("question_ja"), str)
                or not isinstance(translated.get("claim_ja"), str)):
            raise ValueError("review_translation_missing")
        item_token = hashlib.sha256((row_token + "claim-support-frame-review-v1").encode()).hexdigest()[:24]
        items.append({
            "item_id": item_token, "question": translated["question_ja"],
            "claim": translated["claim_ja"],
            "frames": [{"data": frame_jpeg_data_url(frame), "time": float(pt)}
                       for frame, pt in zip(frames, frame_pts, strict=True)],
        })
        source_manifest["items"].append({"item_id": item_token, "packet_sha256": packet_sha,
            "video_sha256": row["video_sha256"], "baseline_sha256": baseline_sha,
            "interaction_sha256": interaction_sha, "row_token_sha256": row_token})
    raw = render(items, provenance=source_manifest)
    _safe_write(output, raw)
    return {"status": "built_private_claim_support_frame_review", "item_count": len(items),
            "frame_count_per_item": FRAME_COUNT, "artifact": str(output.relative_to(ROOT)),
            "artifact_sha256": hashlib.sha256(raw).hexdigest(), "artifact_bytes": len(raw),
            "artifact_mode": "0600", "purpose": "claim-to-model-input-frame human annotation handoff only",
            "source_manifest_sha256": hashlib.sha256(json.dumps(source_manifest, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "direct_faithfulness_measured": False, "rights_status": "unverified_local_only"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
        help="new HTML path relative to data/private_nextgqa")
    parser.add_argument("--translations", type=Path,
        default=Path("data/private_nextgqa/test_split_internal_bias_review_translations_20261021_01.json"))
    args = parser.parse_args(argv)
    if args.output.is_absolute() or not args.output.is_relative_to(Path("data/private_nextgqa")):
        parser.error("output must be a relative child of data/private_nextgqa")
    if args.translations.is_absolute() or not args.translations.is_relative_to(Path("data/private_nextgqa")):
        parser.error("translations must be a relative child of data/private_nextgqa")
    print(json.dumps(build(args.output, args.translations), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
