#!/usr/bin/env python3
"""Render an offline, private human-review page from a frozen 3-row worksheet.

The page contains protected question/claim text. It performs no network calls,
does not embed video bytes, and cannot authorize scoring. Browser downloads may
inherit permissive modes; move the exported JSON to a private directory and set
mode 0600 before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from build_nextgqa_claim_mapping_packet import FROZEN_POLICY_SHA256
from audit_nextgqa_claim_packet_lineage import validate_for_worksheet
from claim_text_quality import claim_text_quality_failure
from private_output_paths import validate_private_output

MAX_WORKSHEET_BYTES = 1_000_000
MAX_MEDIA_BYTES = 64_000_000
TOP_KEYS = {
    "schema_version", "purpose", "status", "created_at_utc", "source_packet_sha256",
    "cohort_sha256", "model_revision", "source_commit", "claim_packet_runner_sha256",
    "cohort_selector_runner_sha256", "dataset_inputs_sha256", "occurrence_policy",
    "review_procedure_status", "requested_records", "effective_records",
    "claim_records_available", "review_completed", "execution_authorization", "records",
    "claim_review_policy_id", "claim_review_policy_sha256",
}
RECORD_KEYS = {"video_path", "video_sha256", "video_id", "claim_id", "review_token",
               "question", "claim_text", "claim_text_sha256", "review"}
REVIEW_KEYS = {
    "source_identity_verified", "full_source_video_reviewed",
    "question_and_claim_only_context_shown", "gold_and_seed_hidden_until_mapping_frozen",
    "outcome_scores_not_accessed", "occurrence_policy_applied", "support_decision",
    "evidence_intervals_seconds", "rationale", "reviewer_role",
    "review_record_reference", "review_started_at_utc", "reviewed_at_utc", "review_status",
}
DATASET_KEYS = {"test.csv", "gsub_test.json", "map_vid_vidorID.json", "frame2time_test.json"}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate_json_key")
        obj[key] = value
    return obj


def render(payload: Any, raw: bytes, lineage: Any, lineage_raw: bytes,
           expected_lineage_sha256: str, custody: Any,
           custody_raw: bytes,
           expected_custody_sha256: str) -> bytes:
    parsed = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if (len(raw) > MAX_WORKSHEET_BYTES or not isinstance(payload, dict) or parsed != payload
            or set(payload) != TOP_KEYS
            or payload.get("status") != "draft_not_reviewed"
            or payload.get("review_completed") is not False
            or payload.get("execution_authorization") != "not_granted"
            or payload.get("effective_records") != 3
            or payload.get("requested_records") != 3
            or payload.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
            or payload.get("review_procedure_status") != "frozen_pre_review"
            or payload.get("occurrence_policy") != "all_relevant_occurrences"
            or not isinstance(payload.get("records"), list)
            or len(payload["records"]) != 3):
        raise ValueError("worksheet_not_review_ready")
    if (not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("cohort_sha256", "")))
            or not re.fullmatch(r"[a-f0-9]{64}", str(payload.get("source_packet_sha256", "")))
            or not isinstance(payload.get("model_revision"), str)
            or not payload["model_revision"].strip()
            or payload.get("claim_review_policy_id") != "nextgqa_claim_review_v5"
            or not isinstance(payload.get("dataset_inputs_sha256"), dict)
            or set(payload["dataset_inputs_sha256"]) != DATASET_KEYS
            or any(not re.fullmatch(r"[a-f0-9]{64}", str(x))
                   for x in payload["dataset_inputs_sha256"].values())):
        raise ValueError("worksheet_provenance_invalid")
    seen_videos, seen_claims, seen_tokens = set(), set(), set()
    for row in payload["records"]:
        if (not isinstance(row, dict) or set(row) != RECORD_KEYS
                or not isinstance(row.get("review"), dict)
                or set(row["review"]) != REVIEW_KEYS
                or any(row["review"].get(k) is not None for k in REVIEW_KEYS - {"review_status"})
                or not isinstance(row.get("question"), str)
                or not isinstance(row.get("claim_text"), str)
                or row.get("review", {}).get("review_status") != "not_started"):
            raise ValueError("worksheet_record_invalid")
        video_hash = row.get("video_sha256")
        claim_hash = row.get("claim_text_sha256")
        try:
            actual_claim_hash = hashlib.sha256(row["claim_text"].encode("utf-8")).hexdigest()
        except UnicodeEncodeError:
            raise ValueError("worksheet_record_invalid") from None
        if (not isinstance(video_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", video_hash)
                or not isinstance(claim_hash, str)
                or not re.fullmatch(r"[a-f0-9]{64}", claim_hash)
                or claim_hash != actual_claim_hash
                or not row["question"].strip() or not row["claim_text"].strip()
                or claim_text_quality_failure(row["claim_text"]) is not None
                or len(row["question"]) > 20_000 or len(row["claim_text"]) > 20_000
                or not isinstance(row.get("video_id"), str)
                or row["video_id"] != "video-" + video_hash
                or not isinstance(row.get("claim_id"), str)
                or not re.fullmatch(r"claim-[a-f0-9]{64}", row["claim_id"])
                or not isinstance(row.get("review_token"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{24}", row["review_token"])
                or row.get("video_id") in seen_videos or row.get("claim_id") in seen_claims
                or row.get("review_token") in seen_tokens
                or row.get("review")["review_status"] != "not_started"):
            raise ValueError("worksheet_record_invalid")
        seen_videos.add(row.get("video_id")); seen_claims.add(row.get("claim_id"))
        seen_tokens.add(row.get("review_token"))
    lineage_sha = validate_for_worksheet(payload, lineage, lineage_raw)
    if (not re.fullmatch(r"[a-f0-9]{64}", expected_lineage_sha256)
            or lineage_sha != expected_lineage_sha256):
        raise ValueError("lineage_manifest_sha256_mismatch")
    if (custody is None or custody_raw is None
            or not re.fullmatch(r"[a-f0-9]{64}", str(expected_custody_sha256))):
        raise ValueError("custody_manifest_required")
    if custody is not None or custody_raw is not None or expected_custody_sha256 is not None:
        custody_keys = {"schema_version", "purpose", "created_at_utc", "worksheet_sha256",
                        "source_lineage_manifest_sha256", "claim_review_policy_sha256", "record_count", "records"}
        try:
            custody_parsed = json.loads(custody_raw.decode("utf-8"), object_pairs_hook=_unique_object)
        except (AttributeError, UnicodeError, json.JSONDecodeError, ValueError):
            raise ValueError("custody_manifest_invalid") from None
        if (custody_parsed != custody or not isinstance(custody, dict) or set(custody) != custody_keys
                or hashlib.sha256(custody_raw).hexdigest() != expected_custody_sha256
                or custody.get("schema_version") != 1 or custody.get("record_count") != 3
                or custody.get("purpose") != "private_custodian_token_to_verified_media_manifest"
                or custody.get("worksheet_sha256") != hashlib.sha256(raw).hexdigest()
                or custody.get("source_lineage_manifest_sha256") != lineage_sha
                or custody.get("claim_review_policy_sha256") != FROZEN_POLICY_SHA256
                or not isinstance(custody.get("records"), list) or len(custody["records"]) != 3):
            raise ValueError("custody_manifest_binding_invalid")
        for row, entry in zip(payload["records"], custody["records"], strict=True):
            if (not isinstance(entry, dict)
                    or set(entry) != {"review_token", "row_ordinal", "media_file", "source_media_sha256",
                                      "staged_media_sha256", "size_bytes", "media_check_salt", "media_check_sha256"}
                    or entry.get("row_ordinal") != payload["records"].index(row) + 1
                    or entry.get("source_media_sha256") != row["video_sha256"]
                    or entry.get("staged_media_sha256") != row["video_sha256"]
                    or not isinstance(entry.get("size_bytes"), int) or entry["size_bytes"] <= 0
                    or entry["size_bytes"] > 64_000_000
                    or entry.get("review_token") != row["review_token"]
                    or entry.get("media_file") != row["review_token"] + ".mp4"
                    or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("media_check_salt", "")))
                    or not re.fullmatch(r"[a-f0-9]{64}", str(entry.get("media_check_sha256", "")))):
                raise ValueError("custody_media_token_binding_invalid")
    # HTML script data is escaped so claims cannot terminate the data element.
    page_payload = {
        "schema_version": 1,
        "purpose": "anonymous_local_claim_video_review",
        "claim_review_policy_id": payload["claim_review_policy_id"],
        "claim_review_policy_sha256": payload["claim_review_policy_sha256"],
        "occurrence_policy": payload["occurrence_policy"],
        "records": [],
    }
    page_payload["records"] = []
    for row in payload["records"]:
        item = {key: row[key] for key in ("review_token", "question", "claim_text")}
        custody_entry = custody["records"][len(page_payload["records"])]
        item["media_check_salt"] = custody_entry["media_check_salt"]
        item["media_check_digest"] = custody_entry["media_check_sha256"]
        page_payload["records"].append(item)
    data = json.dumps(page_payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    data = data.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = r'''<!doctype html>
<html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src blob:; connect-src 'none'; img-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>Private offline video review</title>
<style>
body{font:16px system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#18202a}
.notice{background:#fff2cc;padding:1rem;border:1px solid #d6a900} article{border:1px solid #aaa;padding:1rem;margin:1rem 0}
 video{display:block;max-width:100%;max-height:65vh;background:#111;margin:1rem 0} label{display:block;margin:.65rem 0}
textarea{width:100%;min-height:4rem} input[type=number]{width:8rem} button{margin:.25rem;padding:.45rem}
.claim{white-space:pre-wrap;background:#eef5ff;padding:.7rem}.bad{color:#a00}.ok{color:#075}
</style>
<h1>動画主張の人手レビュー</h1>
<p class="notice">この画面はローカル専用です。質問とモデル生成主張だけを表示し、QA正解・選択肢・gold区間・モデルの選択フレーム・スコアは表示しません。動画は画面に表示されたランダムtoken名のファイルを選択します。ページ内に安定した動画SHAやデータセットIDを含みません。公開データセットの索引や回答を調べず、見た内容だけで判定してください。レビューJSON保存後に、完全性indexも別ボタンで保存してください。どちらも採点許可にはなりません。</p>
<p class="notice">各動画の開始時刻は最初の再生イベント、完了時刻はその行の必須レビュー入力が揃った時点で記録します。いずれもブラウザの自己申告ログであり、実際の視聴や時系列を独立に証明しません。</p>
<p class="notice">音声と字幕トラックは無効化します。映像に焼き付いた字幕・キャプション・テロップは残ることがありますが、根拠として使わないでください。文字 overlay だけが主張を支える場合、その主張は不支持として判定します。</p>
<p>方針 <code>nextgqa_claim_review_v5</code> / occurrence: <code>all_relevant_occurrences</code> / policy SHA-256: <code>POLICY</code></p>
<div id="rows"></div>
<h2>レビュー担当者</h2>
<label>氏名または非公開の担当者識別 <input id="reviewer" autocomplete="off"></label>
<label>役割 <select id="role"><option value="">選択</option><option value="independent_human">claim生成・採点から独立した人間レビュアー</option></select></label>
<label><input type="checkbox" id="attest"> 私は人間であり、claim生成・採点・custodyから独立しています。照合manifestを見ておらず、QA正解やgold区間にもアクセスせず、方針に従い全3動画を自分でレビューしました。</label>
<button id="export" disabled>レビューJSONを保存</button><button id="export-index" disabled>完全性indexを保存</button><button id="export-clock" disabled>HTML5時計観測ログを保存</button><span id="global"></span>
<p>保存後のJSONは private な場所へ移し、ファイル権限を 0600 にしてください。人手記録はモデル入力・動画の真正性・権利・PTS時計校正・SESOI・採点許可を証明しません。</p>
<p>HTML5時計観測ログはclaim根拠区間と別に保存します。アンカーを記録するときは、動画を一時停止しseek中でない状態にしてください。対応ブラウザならrequestVideoFrameCallbackの直近提示フレーム時刻・連番・提示時刻も記録しますが、直近callback値がアンカー時点の表示フレームと一致する保証はなく、decoder PTSの同一フレーム対応も証明しません。currentTimeだけの観測、未取得metadata、またはdecoderフレームとの照合不能は未校正として扱い、fit/held-out分割・事前固定済みの対応証拠・不確かさ評価がそろうまで校正値として使用できません。</p>
<script id="data" type="application/json">DATA</script>
<script>
'use strict';
const base=JSON.parse(document.getElementById('data').textContent);
const MAX_MEDIA_BYTES=64000000;
const rows=document.getElementById('rows'), state=base.records.map(()=>({tokenOK:false,full:false,context:false,seedHidden:false,outcomeHidden:false,decision:'',intervals:[],clockEvents:[],duration:null,mime:'',reviewStartedAt:null,reviewedAt:null}));
const decisions=[['supports_claim','主張を支持'],['partial','一部のみ支持'],['contradicted','動画と矛盾'],['unsupported','支持なし'],['ambiguous','曖昧'],['unlocalizable','時刻特定不能']];
function el(tag,attrs={},text=''){const e=document.createElement(tag);for(const [k,v] of Object.entries(attrs)){if(k==='className')e.className=v;else e.setAttribute(k,v)}if(text)e.textContent=text;return e}
function checkReady(){for(const s of state){const done=s.tokenOK&&Number.isFinite(s.duration)&&s.duration>0&&s.reviewStartedAt&&s.full&&s.context&&s.seedHidden&&s.outcomeHidden&&s.decision&&(s.decision!=='supports_claim'||s.intervals.length>0);if(done&&!s.reviewedAt)s.reviewedAt=new Date().toISOString();if(!done)s.reviewedAt=null}const ready=state.every(s=>s.tokenOK&&Number.isFinite(s.duration)&&s.duration>0&&s.reviewStartedAt&&s.reviewedAt&&s.full&&s.context&&s.seedHidden&&s.outcomeHidden&&s.decision)&&document.getElementById('reviewer').value.trim()&&document.getElementById('role').value==='independent_human'&&document.getElementById('attest').checked;document.getElementById('export').disabled=!ready;document.getElementById('global').textContent=ready?' 全項目の入力がそろいました。':' 3件すべてのレビュー記録が必要です。'}
base.records.forEach((r,i)=>{
 const s=state[i],box=el('article');box.append(el('h2',{},`匿名動画 ${i+1} — ${r.review_token}.mp4`));
 box.append(el('h3',{},'質問'));box.append(el('p',{className:'claim'},r.question));box.append(el('h3',{},'固定されたモデル生成主張'));box.append(el('p',{className:'claim'},r.claim_text));
 const file=el('input',{type:'file',accept:'video/mp4'});box.append(el('label',{},'表示されたtoken名のcustodian動画ファイルを選択'));box.lastChild.append(file);
 const status=el('p',{},'動画未選択');box.append(status);const video=el('video',{playsinline:''});video.muted=true;video.volume=0;video.addEventListener('volumechange',()=>{if(!video.muted||video.volume!==0){video.muted=true;video.volume=0}});video.textTracks?.addEventListener?.('addtrack',e=>{e.track.mode='disabled'});let objectUrl='',selection=0,metadataSelection=null,lastPresentedFrame=null;function observePresentedFrame(){if(typeof video.requestVideoFrameCallback!=='function')return;const scheduledSelection=selection;video.requestVideoFrameCallback((callbackNow,meta)=>{if(scheduledSelection===selection&&objectUrl&&video.currentSrc===objectUrl){lastPresentedFrame={selection:scheduledSelection,media_time_seconds:meta.mediaTime,presented_frames:meta.presentedFrames,presentation_time_ms:meta.presentationTime,expected_display_time_ms:meta.expectedDisplayTime,width:meta.width,height:meta.height,callback_now_ms:callbackNow}}observePresentedFrame()})}observePresentedFrame();video.addEventListener('loadedmetadata',()=>{if(metadataSelection===selection&&s.tokenOK&&objectUrl&&video.currentSrc===objectUrl){s.duration=video.duration;checkReady()}});box.append(video);
 const play=el('button',{},'再生 / 一時停止');play.type='button';play.onclick=()=>video.paused?video.play():video.pause();video.addEventListener('play',()=>{if(s.tokenOK&&s.reviewStartedAt===null)s.reviewStartedAt=new Date().toISOString();checkReady()});const seek=el('input',{type:'range',min:'0',max:'1000',value:'0'});seek.addEventListener('input',()=>{if(video.duration>0)video.currentTime=video.duration*(Number(seek.value)/1000)});video.addEventListener('timeupdate',()=>{if(video.duration>0)seek.value=String(Math.round(video.currentTime/video.duration*1000))});box.append(play,seek,el('p',{},'音声は強制ミュート。字幕トラックは無効。再生速度は通常速度に固定。'));
 const flagInputs=[];
 file.addEventListener('change',async()=>{const current=++selection;metadataSelection=null;lastPresentedFrame=null;s.tokenOK=false;s.duration=null;s.mime='';s.full=false;s.context=false;s.seedHidden=false;s.outcomeHidden=false;s.decision='';s.intervals=[];s.clockEvents=[];s.reviewStartedAt=null;s.reviewedAt=null;for(const cb of flagInputs)cb.checked=false;if(typeof select!=='undefined')select.value='';start.value='';end.value='';reason.value='';anchorId.value='';redraw();refreshClockButton();status.textContent='匿名ファイル名とcustodian照合値を確認中…';video.pause();if(objectUrl){URL.revokeObjectURL(objectUrl);objectUrl=''}video.removeAttribute('src');video.load();checkReady();try{const f=file.files[0];if(!f||f.size>MAX_MEDIA_BYTES||f.name!==r.review_token+'.mp4')throw Error('opaque_token_filename_mismatch');const payload=new Uint8Array(await f.arrayBuffer()),salt=Uint8Array.from(r.media_check_salt.match(/../g),x=>parseInt(x,16)),joined=new Uint8Array(salt.length+payload.length);joined.set(salt);joined.set(payload,salt.length);const actual=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',joined))).map(x=>x.toString(16).padStart(2,'0')).join('');if(current!==selection)return;if(actual!==r.media_check_digest)throw Error('custody_media_digest_mismatch');objectUrl=URL.createObjectURL(f);metadataSelection=current;video.src=objectUrl;video.load();video.playbackRate=1;video.muted=true;s.mime=f.type;s.tokenOK=true;status.textContent='匿名tokenとcustodian照合値が一致しました。';status.className='ok'}catch(e){if(current!==selection)return;video.removeAttribute('src');video.load();s.tokenOK=false;status.textContent='ファイル名・内容照合に失敗、または動画が64 MBを超過しました。選択し直してください。';status.className='bad'}checkReady()});
 const flags=[['full','動画全体を通常速度で視聴した'],['context','表示された質問と主張だけを回答コンテキストとして用いた'],['seedHidden','gold回答とQA seed区間をマッピング凍結まで見なかった'],['outcomeHidden','採点結果を見ていない']];
 for(const [key,label] of flags){const cb=el('input',{type:'checkbox'});flagInputs.push(cb);cb.addEventListener('change',()=>{s[key]=cb.checked;s.reviewedAt=null;checkReady()});box.append(el('label'));box.lastChild.append(cb,document.createTextNode(' '+label))}
 const select=el('select');select.append(el('option',{value:''},'判定を選択'));for(const [v,t] of decisions)select.append(el('option',{value:v},t));select.addEventListener('change',()=>{s.decision=select.value;s.reviewedAt=null;checkReady()});box.append(el('label',{},'主張全体の視覚的支持判定'));box.lastChild.append(select);
 box.append(el('p',{},'supports_claimの場合、全ての重要な事実要素が直接見える時間を記録します。反復する根拠は全て別区間で記録します。入力値は0.001秒単位ですが、映像上の根拠の時間分解能はフレーム間隔（この3件は約33–40 ms）に制約されるため、1 ms精度の測定を意味しません。'));const intervals=el('div');box.append(intervals);
 function redraw(){intervals.replaceChildren();s.intervals.forEach((it,j)=>{const row=el('div');row.append(document.createTextNode(`区間${j+1}: ${it.start.toFixed(3)}–${it.end.toFixed(3)}秒 — ${it.rationale} `));const del=el('button',{},'削除');del.type='button';del.onclick=()=>{s.intervals.splice(j,1);s.reviewedAt=null;redraw();checkReady()};row.append(del);intervals.append(row)})}
 const start=el('input',{type:'number',min:'0',step:'0.001'}),end=el('input',{type:'number',min:'0',step:'0.001'}),reason=el('textarea');
 const anchorId=el('input',{type:'text',maxLength:'40',autocomplete:'off',class:'clock-control'}),clockButton=el('button',{class:'clock-control'},'HTML5時計アンカー観測を追加');clockButton.type='button';clockButton.addEventListener('click',()=>{const id=anchorId.value.trim(),frame=lastPresentedFrame&&lastPresentedFrame.selection===selection?{media_time_seconds:lastPresentedFrame.media_time_seconds,presented_frames:lastPresentedFrame.presented_frames,presentation_time_ms:lastPresentedFrame.presentation_time_ms,expected_display_time_ms:lastPresentedFrame.expected_display_time_ms,width:lastPresentedFrame.width,height:lastPresentedFrame.height,callback_now_ms:lastPresentedFrame.callback_now_ms}:null;if(!s.tokenOK||!Number.isFinite(video.duration)||video.currentSrc!==objectUrl||video.playbackRate!==1||!video.paused||video.seeking||video.readyState<2||!(/^[A-Za-z0-9_-]{1,40}$/).test(id)||s.clockEvents.some(x=>x.anchor_id===id)){alert('照合済み動画を一時停止し、seek完了・表示可能状態で、未使用の事前定義アンカーIDを指定してください。');return}s.clockEvents.push({anchor_id:id,sequence:s.clockEvents.length+1,html5_current_time_seconds:video.currentTime,request_video_frame_callback_metadata:frame,frame_match_status:frame?'last_presented_frame_metadata_unverified':'presented_frame_metadata_missing',performance_now_ms:performance.now(),media_duration_seconds:video.duration,playback_rate:video.playbackRate,muted:video.muted,volume:video.volume,text_tracks_disabled:Array.from(video.textTracks||[]).every(t=>t.mode==='disabled'),paused:video.paused,seeking:video.seeking,ready_state:video.readyState,current_src_is_staged_media:video.currentSrc===objectUrl,visibility_state:document.visibilityState,captured_at_utc:new Date().toISOString()});anchorId.value='';refreshClockButton()});box.append(el('label',{},'事前定義した時計アンカーID'));box.lastChild.append(anchorId,clockButton);const anchorStatus=el('p',{},'時計観測 0 件（claim根拠区間とは別記録）');box.append(anchorStatus);function refreshClockButton(){clockButton.disabled=!s.tokenOK;anchorStatus.textContent=`時計観測 ${s.clockEvents.length} 件（claim根拠区間とは別記録）`;document.getElementById('export-clock').disabled=!state.some(x=>x.clockEvents.length>0)}refreshClockButton();
 const capture=el('button',{},'再生時刻を開始へ');capture.type='button';capture.onclick=()=>{start.value=video.currentTime.toFixed(3)};const captureEnd=el('button',{},'再生時刻を終了へ');captureEnd.type='button';captureEnd.onclick=()=>{end.value=video.currentTime.toFixed(3)};
 const add=el('button',{},'区間を追加');add.type='button';add.onclick=()=>{const a=Number(start.value),b=Number(end.value),why=reason.value.trim(),duration=video.duration;const ms=x=>Math.abs(x*1000-Math.round(x*1000))<1e-6;if(!Number.isFinite(a)||!Number.isFinite(b)||!Number.isFinite(duration)||a<0||b<=a||b>duration||!ms(a)||!ms(b)||!why||s.intervals.some(x=>a<=x.end&&b>=x.start)){alert('動画長内の正の区間（ミリ秒精度）と根拠メモを入力してください。既存区間との重複も不可です。');return}s.intervals.push({start:a,end:b,rationale:why});s.intervals.sort((x,y)=>x.start-y.start);s.reviewedAt=null;redraw();checkReady();start.value='';end.value='';reason.value=''};
 box.append(el('label',{},'開始秒（inclusive・0.001秒精度）'));box.lastChild.append(start,capture);box.append(el('label',{},'終了秒（inclusive・0.001秒精度）'));box.lastChild.append(end,captureEnd);box.append(el('label',{},'この区間がどの主張要素を支持するか（非公開メモ）'));box.lastChild.append(reason);box.append(add);rows.append(box)
});
for(const id of ['reviewer','role','attest'])document.getElementById(id).addEventListener('input',checkReady);for(const id of ['role','attest'])document.getElementById(id).addEventListener('change',checkReady);
document.getElementById('export').onclick=async()=>{
 if(!state.every(s=>s.decision!=='supports_claim'||s.intervals.length>0)){alert('支持判定の各項目には根拠区間が必要です。');return}
 const now=new Date().toISOString(), policySha=base.claim_review_policy_sha256;
 const out={schema_version:2,purpose:'anonymous_local_claim_video_human_review',status:'human_review_recorded_unvalidated',review_completed:true,execution_authorization:'not_granted',claim_review_policy_id:base.claim_review_policy_id,claim_review_policy_sha256:policySha,human_reviewer_role:'independent_human',human_reviewer_private_id:document.getElementById('reviewer').value.trim(),review_environment:{browser_user_agent:navigator.userAgent,offline_static_page:true,network_fetch_available:false},independence_attestation:'reviewer_asserts_independence_from_generation_scoring_and_custody_and_did_not_view_custody_crosswalk',records:base.records.map((r,i)=>{const s=state[i];return {review_token:r.review_token,review:{full_source_video_reviewed:s.full,question_and_claim_only_context_shown:s.context,gold_and_seed_hidden_until_mapping_frozen:s.seedHidden,outcome_scores_not_accessed:s.outcomeHidden,media_bytes_verified:s.tokenOK,occurrence_policy_applied:base.occurrence_policy,support_decision:s.decision,evidence_intervals_seconds:s.intervals.map(x=>[x.start,x.end]),interval_rationales:s.intervals.map(x=>x.rationale),media_duration_seconds:s.duration,reviewer_role:'independent_human',review_started_at_utc:s.reviewStartedAt,reviewed_at_utc:s.reviewedAt,review_status:'completed',display_player:'HTML5 video; audio forced muted; text/caption tracks disabled; playback rate 1.0; UA='+navigator.userAgent,claim_review_policy_sha256:policySha}}})};
 const raw=JSON.stringify(out,null,2)+'\n',bytes=new TextEncoder().encode(raw),digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))).map(x=>x.toString(16).padStart(2,'0')).join(''),index={schema_version:1,purpose:'anonymous_unvalidated_human_review_handoff',review_payload_sha256:digest,claim_review_policy_sha256:policySha,created_at_utc:now,review_status:'unvalidated_requires_custodian_audit',execution_authorization:'not_granted'};
 function download(text,name,type){const blob=new Blob([text],{type}),a=el('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
 const stem='nextgqa_human_review_'+now.replace(/[:.]/g,'-');window.pendingReviewIndex=JSON.stringify(index,null,2)+'\n';download(raw,stem+'.json','application/json');for(const control of rows.querySelectorAll('input:not(.clock-control),select,textarea,button:not(.clock-control)'))control.disabled=true;document.getElementById('reviewer').disabled=true;document.getElementById('role').disabled=true;document.getElementById('attest').disabled=true;document.getElementById('export').disabled=true;document.getElementById('export-index').disabled=false
};
document.getElementById('export-index').onclick=()=>{if(window.pendingReviewIndex){const blob=new Blob([window.pendingReviewIndex],{type:'application/json'}),a=el('a');a.href=URL.createObjectURL(blob);a.download='nextgqa_human_review_index.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
document.getElementById('export-clock').onclick=async()=>{const now=new Date().toISOString(),payload={schema_version:2,purpose:'local_unvalidated_html5_player_clock_observations',status:'raw_player_clock_observations_only',player_clock:'html5_video_currentTime_seconds',presented_frame_metadata:'requestVideoFrameCallback_when_available_not_decoder_matched',decoder_frame_identity:'not_verified',created_at_utc:now,browser_user_agent:navigator.userAgent,anchor_id_validation:'format_and_per_video_uniqueness_only_unverified',predeclared_plan_binding:'not_checked',fit_heldout_split:'not_assigned',decoder_pts_pairing:'not_provided',execution_authorization:'not_granted',records:base.records.map((r,i)=>({review_token:r.review_token,events:state[i].clockEvents}))};const raw=JSON.stringify(payload,null,2)+'\n',digest=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',new TextEncoder().encode(raw)))).map(x=>x.toString(16).padStart(2,'0')).join(''),index={schema_version:1,purpose:'unvalidated_html5_clock_observation_integrity_index',payload_sha256:digest,created_at_utc:now,status:'raw_observations_only',execution_authorization:'not_granted'};function download(text,name){const blob=new Blob([text],{type:'application/json'}),a=el('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}const stem='nextgqa_html5_clock_observations_'+now.replace(/[:.]/g,'-');download(raw,stem+'.json');download(JSON.stringify(index,null,2)+'\n',stem+'_index.json')};
checkReady();
</script></html>'''
    html_doc = (template.replace("POLICY", payload["claim_review_policy_sha256"])
                .replace("DATA", data)).encode("utf-8")
    return html_doc


def write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(raw); f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--lineage-manifest", required=True)
    parser.add_argument("--expected-lineage-sha256", required=True,
                        help="independently recorded digest of the audited lineage manifest")
    parser.add_argument("--custody-manifest", required=True)
    parser.add_argument("--expected-custody-sha256", required=True,
                        help="independently recorded digest of the private custody manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--user-authorized-local-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.user_authorized_local_only:
        parser.error("explicit local-only authorization flag required")
    source = Path(args.input).resolve()
    lineage_source = Path(args.lineage_manifest).resolve()
    custody_source = Path(args.custody_manifest).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    try:
        target = validate_private_output(args.output, repo_root)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except (OSError, ValueError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "private_output_path_invalid")
    if any((path.stat().st_mode & 0o777) != 0o600
           for path in (source, lineage_source, custody_source)):
        parser.error("private_input_mode_invalid")
    try:
        raw = source.read_bytes(); payload = json.loads(raw.decode("utf-8"))
        lineage_raw = lineage_source.read_bytes()
        lineage = json.loads(lineage_raw.decode("utf-8"))
        custody_raw = custody_source.read_bytes()
        custody = json.loads(custody_raw.decode("utf-8"))
        result = render(payload, raw, lineage, lineage_raw, args.expected_lineage_sha256,
                        custody, custody_raw, args.expected_custody_sha256)
        write_exclusive(target, result)
    except FileExistsError:
        parser.error("refusing_to_overwrite_output")
    except OSError:
        parser.error("input_or_output_io_error")
    except (UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "worksheet_invalid")
    print("Private offline reviewer page created; protected content remains local; scoring authorization=not_granted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
