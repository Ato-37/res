#!/usr/bin/env python3
"""Build a private Japanese review page for up to three generated Minerva claims.

The page embeds the exact 64 source frames sampled for each Qwen input. It is
local-only and exports verdicts as one JSON batch per reviewer. Review and
GPU-failed manifest entries may name a private row-specific cohort-preflight
receipt whose selected row must match the final aggregate receipt exactly.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
ITEM_COUNT = 3
FRAME_COUNT = 64
MAX_JSON_BYTES = 50_000_000
MAX_VIDEO_BYTES = 4_000_000_000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finalize_page(page: bytes) -> tuple[bytes, str]:
    """Fill the self-excluding fingerprint slot and return finalized bytes/hash."""
    marker = b'<meta name="review-artifact-sha256" content="__PAGE_FINGERPRINT__">'
    if page.count(marker) != 1:
        raise ValueError("review_page_fingerprint_slot_invalid")
    fingerprint = _sha(page)
    finalized = page.replace(marker, marker.replace(b"__PAGE_FINGERPRINT__",
        fingerprint.encode("ascii")), 1)
    return finalized, fingerprint


def verify_page_fingerprint(page: bytes) -> tuple[str, str, str]:
    """Return fingerprint, builder hash, and source-manifest hash after verification."""
    pattern = re.compile(
        rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">')
    matches = list(pattern.finditer(page))
    builder_matches = re.findall(
        rb'<meta name="review-builder-sha256" content="([0-9a-f]{64})">', page)
    source_matches = re.findall(
        rb'<meta name="review-source-manifest-sha256" content="([0-9a-f]{64})">', page)
    if len(matches) != 1 or len(builder_matches) != 1 or len(source_matches) != 1:
        raise ValueError("review_page_binding_metadata_invalid")
    fingerprint = matches[0].group(1).decode("ascii")
    canonical = pattern.sub(
        b'<meta name="review-artifact-sha256" content="__PAGE_FINGERPRINT__">',
        page, count=1)
    if _sha(canonical) != fingerprint:
        raise ValueError("review_page_fingerprint_mismatch")
    return (fingerprint, builder_matches[0].decode("ascii"),
            source_matches[0].decode("ascii"))


def validate_response_binding(page: bytes, response: dict) -> None:
    """Check an exported response against its exact page, builder, and source manifest."""
    fingerprint, builder_sha256, source_manifest_sha256 = verify_page_fingerprint(page)
    if (not isinstance(response, dict)
            or response.get("artifact_fingerprint") != fingerprint
            or response.get("builder_sha256") != builder_sha256
            or response.get("source_manifest_sha256") != source_manifest_sha256):
        raise ValueError("review_response_page_binding_mismatch")


def _private_root() -> Path:
    if (not PRIVATE.is_dir() or PRIVATE.is_symlink()
            or PRIVATE.stat().st_uid != os.getuid()
            or stat.S_IMODE(PRIVATE.stat().st_mode) != 0o700):
        raise ValueError("private_root_contract_invalid")
    return PRIVATE.resolve()


def _resolve_private(relative: str | Path, *, must_exist: bool = True) -> Path:
    root = _private_root()
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or any(part in {"..", "."} for part in rel.parts):
        raise ValueError("private_relative_path_invalid")
    path = ROOT
    for part in rel.parts:
        path = path / part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("private_symlink_rejected")
        except FileNotFoundError:
            if must_exist or path != ROOT / rel:
                raise ValueError("private_path_component_missing") from None
    resolved = path.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise ValueError("private_path_outside_root")
    if must_exist and not resolved.is_file():
        raise ValueError("private_input_not_regular_file")
    if not must_exist and (resolved.parent != root or resolved.exists()):
        raise ValueError("private_output_contract_invalid")
    return resolved


def _read_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > max_bytes):
        raise ValueError("private_json_permissions_or_size_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("private_json_changed_during_read")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_json_invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError("private_json_shape_invalid")
    return raw, payload


def _file_hashes(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        total = 0
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            total += len(block)
            if total > MAX_VIDEO_BYTES:
                raise ValueError("private_video_too_large")
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def validate_receipt_pair(gpu: dict, cpu: dict, gpu_raw: bytes, cpu_raw: bytes,
                          *, cohort_sha256: str, row_index: int,
                          video_sha256: str, video_md5: str,
                          time_map_sha256: str) -> None:
    """Bind a completed GPU result to its exact CPU receipt and source media."""
    generation = gpu.get("generation")
    from run_minerva_generated_object_pixel_pilot import MODEL_REVISION
    generated_ids = generation.get("generated_token_ids") if isinstance(generation, dict) else None
    generated_text = generation.get("generated_text") if isinstance(generation, dict) else None
    if (gpu.get("status") != "completed_single_video_generated_object_pixel_exploratory_pilot"
            or gpu.get("direct_generated_explanation_faithfulness") is not False
            or gpu.get("cohort_sha256") != cohort_sha256
            or gpu.get("row_index") != row_index
            or gpu.get("video_sha256") != video_sha256
            or gpu.get("video_md5") != video_md5
            or gpu.get("time_map_sha256") != time_map_sha256
            or gpu.get("model_revision") != MODEL_REVISION
            or gpu.get("source_sha256", {}).get(
                "run_minerva_generated_object_pixel_pilot.py") != gpu.get("runner_sha256")
            or gpu.get("cpu_pixel_receipt_sha256") != _sha(cpu_raw)
            or gpu.get("cohort_preflight_sha256") != cpu.get("cohort_cpu_preflight_sha256")
            or not isinstance(generated_text, str) or not generated_text.strip()
            or not isinstance(generated_ids, list) or not generated_ids
            or len(generated_ids) > 64
            or any(type(token_id) is not int or token_id < 0 for token_id in generated_ids)
            or generation.get("generated_text_sha256") != _sha(generated_text.encode("utf-8"))
            or generation.get("generated_token_ids_sha256") != _sha(
                json.dumps(generated_ids, separators=(",", ":")).encode("ascii"))):
        raise ValueError("gpu_cpu_source_binding_invalid")
    if (cpu.get("status") != "completed_minerva_generated_object_pixel_cpu_preflight"
            or cpu.get("cohort_sha256") != cohort_sha256
            or cpu.get("row_index") != row_index
            or cpu.get("video_sha256") != video_sha256
            or cpu.get("video_md5") != video_md5
            or cpu.get("time_map_sha256") != time_map_sha256
            or gpu.get("row_key_sha256") != cpu.get("row_key_sha256")
            or gpu.get("annotation_source_sha256") != cpu.get("annotation_source_sha256")
            or gpu.get("runner_sha256") != cpu.get("runner_sha256")):
        raise ValueError("cpu_receipt_source_binding_invalid")


def _jpeg_data_url(frame: Image.Image) -> str:
    image = frame.convert("RGB")
    image.thumbnail((480, 360), Image.Resampling.LANCZOS)
    from io import BytesIO
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def validate_review_items(items: list[dict]) -> None:
    if not isinstance(items, list) or not 1 <= len(items) <= ITEM_COUNT:
        raise ValueError("review_requires_one_to_three_items")
    ids = []
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("item_id"), str)
                or not item["item_id"] or not isinstance(item.get("question"), str)
                or not item["question"].strip() or not isinstance(item.get("claim"), str)
                or not item["claim"].strip() or not isinstance(item.get("frames"), list)
                or len(item["frames"]) != FRAME_COUNT):
            raise ValueError("review_item_contract_invalid")
        source = item.get("source")
        if (not isinstance(source, dict)
                or source.get("question_sha256") != _sha(item["question"].encode("utf-8"))
                or source.get("claim_sha256") != _sha(item["claim"].encode("utf-8"))):
            raise ValueError("review_text_hash_binding_invalid")
        ids.append(item["item_id"])
        frame_hashes = []
        for rank, frame in enumerate(item["frames"]):
            if (not isinstance(frame, dict) or frame.get("rank") != rank
                    or type(frame.get("source_frame_index")) is not int
                    or frame["source_frame_index"] < 0
                    or not isinstance(frame.get("time_seconds"), (int, float))
                    or isinstance(frame.get("time_seconds"), bool)
                    or not isinstance(frame.get("data_url"), str)
                    or not frame["data_url"].startswith("data:image/jpeg;base64,")):
                raise ValueError("review_frame_contract_invalid")
            try:
                image_bytes = base64.b64decode(
                    frame["data_url"].removeprefix("data:image/jpeg;base64,"), validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("review_frame_image_invalid") from exc
            frame_hashes.append(_sha(image_bytes))
        inventory_sha256 = _sha(json.dumps(frame_hashes,
            separators=(",", ":")).encode("ascii"))
        if (source.get("frame_image_sha256") != frame_hashes
                or source.get("frame_image_inventory_sha256") != inventory_sha256):
            raise ValueError("review_image_inventory_binding_invalid")
    if len(set(ids)) != len(items):
        raise ValueError("review_item_ids_must_be_unique")


def _script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")


def _row_preflight(entry: dict, *, final_preflight: dict,
                   final_preflight_raw: bytes, cohort_sha256: str,
                   row_index: int, selected: dict) -> tuple[dict, bytes, str, dict]:
    """Select a per-row receipt, requiring exact agreement with final cohort row."""
    path_value = entry.get("cohort_preflight")
    if path_value is None:
        preflight, raw = final_preflight, final_preflight_raw
    else:
        if not isinstance(path_value, str):
            raise ValueError("row_preflight_path_invalid")
        path = _resolve_private(path_value)
        raw, preflight = _read_json(path)
        shared_fields = (
            "schema_version", "status", "cohort_sha256", "manifest_sha256",
            "preflight_source_sha256",
            "decoder_source_sha256", "generated_runner_source_sha256",
            "bbox_mapper_source_sha256", "qwen_video_processor",
            "human_review_performed", "rights_status")
        if (any(preflight.get(key) != final_preflight.get(key) for key in shared_fields)
                or preflight.get("cohort_sha256") != cohort_sha256):
            raise ValueError("row_preflight_cohort_source_manifest_mismatch")
    digest = _sha(raw)
    target = (selected.get("video_id"), selected.get("row_key_sha256"))
    final_records = [record for record in final_preflight.get("records", [])
        if isinstance(record, dict)
        and (record.get("video_id"), record.get("row_key_sha256")) == target]
    row_records = [record for record in preflight.get("records", [])
        if isinstance(record, dict)
        and (record.get("video_id"), record.get("row_key_sha256")) == target]
    if (len(final_records) != 1 or len(row_records) != 1
            or row_records[0] != final_records[0]
            or row_records[0].get("row_index", row_index) != row_index):
        raise ValueError("row_preflight_record_mismatch")
    return preflight, raw, digest, row_records[0]


def render(items: list[dict], provenance: dict) -> str:
    validate_review_items(items)
    embedded = _script_json({"items": items, "provenance": provenance})
    doc = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-artifact-sha256" content="__PAGE_FINGERPRINT__"><meta name="review-builder-sha256" content="__BUILDER_SHA256__"><meta name="review-source-manifest-sha256" content="__SOURCE_MANIFEST_SHA256__">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>生成された動画説明の確認</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#172b3a;font:16px/1.55 system-ui,sans-serif}main{max-width:900px;margin:24px auto;padding:0 16px 48px}.card{background:#fff;border:1px solid #d3dde5;border-radius:14px;padding:20px;margin:14px 0}h1{font-size:1.55rem}h2{font-size:1.15rem}.note{background:#fff4d6;padding:12px;border-radius:10px}.muted{color:#526575}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frame{margin:0}.frame img{display:block;width:100%;background:#111;border-radius:6px}.frame figcaption{font-size:.75rem}.options label{display:block;padding:11px;border:1px solid #ced9e0;border-radius:9px;margin:8px 0}.options input{margin-right:8px}button{border:0;background:#176b87;color:#fff;font:inherit;font-weight:700;border-radius:9px;padding:10px 16px;cursor:pointer}input{font:inherit;padding:9px;border:1px solid #9fb3c0;border-radius:8px}input[type=text]{width:min(100%,400px)}.hidden{display:none}.row{display:flex;gap:10px;align-items:center;justify-content:space-between;flex-wrap:wrap}@media(max-width:600px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main>
<section id="start" class="card"><h1>動画と短い説明の確認（<span id="total-items"></span>件）</h1><p class="note">各動画の64枚の入力フレームを確認し、説明が動画の見える内容を支持するか選んでください。64枚に写っていない内容や情報不足は「判断できない」を選んでください。これは少人数の参考確認であり、説明の正しさやfaithfulnessを確定する判定ではありません。実名は入力しないでください。入力と回答はこの端末内で処理され、送信されません。</p><label for="nickname">ニックネーム（この回答者につき1つ）</label><br><input id="nickname" type="text" maxlength="40" autocomplete="nickname"><div class="row"><span class="muted">動画内で見えない・分からない場合は推測しないでください。</span><button id="begin">確認を始める</button></div></section>
<section id="task" class="card hidden"><div class="row"><span id="counter"></span><span id="reviewer"></span></div><h2>質問</h2><p id="question"></p><h2>生成された短い説明</h2><p id="claim"></p><h2>Qwen入力に使った64フレーム（時刻順）</h2><p class="muted">元動画から同じフレーム番号でデコードした画像です。Qwen内部のリサイズやpatch表現ではなく、モデルへ渡す前のフレームです。</p><div id="frames" class="frames"></div><h2>この説明と動画の関係</h2><div class="options"><label><input type="radio" name="verdict" value="supports">支持する：説明の内容が動画に見える</label><label><input type="radio" name="verdict" value="contradicts">矛盾する：動画に見える内容と説明が食い違う</label><label><input type="radio" name="verdict" value="not_visible">見えない：説明の対象が動画に現れない／確認できない</label><label><input type="radio" name="verdict" value="cannot_judge">判断できない：映像が不鮮明、または情報が足りない</label></div><div class="row"><span id="required" class="muted">1つ選んでください</span><button id="next">次へ</button></div></section>
<section id="done" class="card hidden"><h1>回答者 <span id="reviewer-count"></span> 人分を記録しました</h1><p>次の人は別のニックネームを入力してください。全員が終わったら、回答バッチをまとめてJSONで保存できます。JSONには各人のニックネームが入るため、共有や公開を避けてください。</p><div class="row"><button id="next-reviewer">次の回答者</button><button id="save">全回答者分をJSONで保存</button></div><p id="saved" aria-live="polite"></p><a id="download" class="hidden">JSONをもう一度保存</a></section>
</main><script>
'use strict';const payload=__PAYLOAD__;const items=payload.items;let nickname='',index=0,answers=[];const respondents=[];const $=id=>document.getElementById(id);$('total-items').textContent=items.length;const labels={supports:'supports',contradicts:'contradicts',not_visible:'not_visible',cannot_judge:'cannot_judge'};
function begin(){nickname=$('nickname').value.trim();if(!nickname){$('nickname').focus();return}index=0;answers=[];$('nickname').value='';$('start').classList.add('hidden');show()}
function show(){if(index>=items.length){respondents.push({reviewer_nickname:nickname,items:answers.slice()});$('task').classList.add('hidden');$('done').classList.remove('hidden');$('reviewer-count').textContent=respondents.length;return}const item=items[index];$('task').classList.remove('hidden');$('counter').textContent=`${index+1} / ${items.length}`;$('reviewer').textContent=`ニックネーム：${nickname}`;$('question').textContent=item.question;$('claim').textContent=item.claim;$('frames').replaceChildren(...item.frames.map(f=>{const fig=document.createElement('figure');fig.className='frame';const img=document.createElement('img');img.loading='lazy';img.alt=`入力フレーム ${f.rank+1}`;img.src=f.data_url;const cap=document.createElement('figcaption');cap.textContent=`${f.rank+1}枚目 · ${Number(f.time_seconds).toFixed(2)}秒`;fig.append(img,cap);return fig}));document.querySelectorAll('[name=verdict]').forEach(x=>x.checked=false);$('next').textContent=index===items.length-1?'確認を終える':'次へ'}
 $('begin').addEventListener('click',begin);$('next').addEventListener('click',()=>{const selected=document.querySelector('[name=verdict]:checked');if(!selected){$('required').textContent='回答するには1つ選んでください';return}const item=items[index];answers.push({item_id:item.item_id,verdict:labels[selected.value],gpu_receipt_sha256:item.source.gpu_receipt_sha256,cpu_receipt_sha256:item.source.cpu_receipt_sha256,cohort_preflight_sha256:item.source.cohort_preflight_sha256,question_sha256:item.source.question_sha256,claim_sha256:item.source.claim_sha256,frame_image_inventory_sha256:item.source.frame_image_inventory_sha256,frame_image_sha256:item.source.frame_image_sha256,source_frame_indices:item.frames.map(f=>f.source_frame_index),frame_pts_seconds:item.frames.map(f=>f.time_seconds)});index++;show()});
$('next-reviewer').addEventListener('click',()=>{$('done').classList.add('hidden');$('start').classList.remove('hidden');$('nickname').focus()});
$('save').addEventListener('click',()=>{const output={schema_version:1,purpose:'private_exploratory_minerva_generated_claim_review',status:'informal_human_verdicts_only',created_at_utc:new Date().toISOString(),artifact_fingerprint:document.querySelector('meta[name="review-artifact-sha256"]').content,builder_sha256:payload.provenance.builder_sha256,source_manifest_sha256:payload.provenance.source_manifest_sha256,attrition:payload.provenance.attrition,respondents,limitations:['この簡易レビューは動画中の見え方に関する印象記録であり、真偽やfaithfulnessを確定しない','回答者の独立性、本人性、動画を全て見たことは検証していない','少数のレビューから統計的な結論を導かない']};const blob=new Blob([JSON.stringify(output,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=$('download');a.href=url;a.download='minerva_generated_claim_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.classList.remove('hidden');a.click();$('saved').textContent='JSONのダウンロードを開始しました。ブラウザのダウンロード一覧で保存を確認してください。'});
</script></body></html>'''
    if (not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("builder_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}",
                str(provenance.get("source_manifest_sha256", "")))):
        raise ValueError("review_provenance_hashes_missing")
    return (doc.replace("__PAYLOAD__", embedded)
        .replace("__BUILDER_SHA256__", provenance["builder_sha256"])
        .replace("__SOURCE_MANIFEST_SHA256__", provenance["source_manifest_sha256"]))


def _prepare_item(entry: dict, *, cohort: dict, cohort_sha256: str,
                  preflight: dict, preflight_raw: bytes, preflight_sha256: str,
                  minerva_rows: list[dict],
                  minerva_json_sha256: str, manifest: dict,
                  manifest_sha256: str) -> tuple[dict, dict]:
    from preflight_minerva_grounding_cohort import load_time_map_seconds, normalize_manifest_path
    from run_minerva_ego_anchor_gpu_pilot import decode_uniform_video

    required = {"row_index", "status", "gpu_receipt", "cpu_receipt", "video", "time_map"}
    if (not isinstance(entry, dict) or set(entry) - required - {"cohort_preflight"}
            or not required.issubset(entry)
            or entry.get("status") != "review"):
        raise ValueError("item_input_manifest_entry_invalid")
    paths = {key: _resolve_private(entry[key])
             for key in ("gpu_receipt", "cpu_receipt", "video", "time_map")}
    gpu_raw, gpu = _read_json(paths["gpu_receipt"])
    cpu_raw, cpu = _read_json(paths["cpu_receipt"])
    row_index = cpu.get("row_index")
    selection = cohort.get("selection")
    if (type(row_index) is not int or not isinstance(selection, list)
            or not 0 <= row_index < len(selection)):
        raise ValueError("receipt_cohort_row_index_invalid")
    selected = selection[row_index]
    if (row_index != entry["row_index"]
            or cpu.get("cohort_sha256") != cohort_sha256
            or selected.get("row_key_sha256") != cpu.get("row_key_sha256")
            or _sha(str(selected.get("row_key", "")).encode()) != cpu.get("row_key_sha256")
            or _sha(str(selected.get("video_id", "")).encode()) != cpu.get("video_id_sha256")):
        raise ValueError("receipt_frozen_cohort_binding_invalid")
    _, _, row_preflight_sha256, record = _row_preflight(
        entry, final_preflight=preflight, final_preflight_raw=preflight_raw,
        cohort_sha256=cohort_sha256, row_index=row_index, selected=selected)
    if cpu.get("cohort_cpu_preflight_sha256") != row_preflight_sha256:
        raise ValueError("receipt_preflight_binding_invalid")
    if record.get("status") != "pass":
        raise ValueError("preflight_record_missing_or_ambiguous")
    video_sha, video_md5 = _file_hashes(paths["video"])
    time_sha, time_md5 = _file_hashes(paths["time_map"])
    validate_receipt_pair(gpu, cpu, gpu_raw, cpu_raw,
        cohort_sha256=cohort_sha256, row_index=row_index,
        video_sha256=video_sha, video_md5=video_md5, time_map_sha256=time_sha)
    if (record.get("video_sha256") != video_sha or record.get("video_md5") != video_md5
            or record.get("time_map_sha256") != time_sha
            or record.get("time_map_md5") != time_md5
            or cpu.get("time_map_md5") != time_md5
            or cpu.get("manifest_sha256") != manifest_sha256):
        raise ValueError("official_media_or_time_map_binding_invalid")
    video_id = selected["video_id"]
    expected_video = normalize_manifest_path(f"Videos/{video_id[:3]}/{video_id}.mp4")
    expected_map = normalize_manifest_path(
        f"Videos/{video_id[:3]}/{video_id}_mp4_to_vrs_time_ns.csv")
    if manifest.get(expected_video) != video_md5 or manifest.get(expected_map) != time_md5:
        raise ValueError("official_manifest_md5_mismatch")
    sampling = cpu.get("sampling")
    record_sampling = record.get("clip_sampling", {})
    indices = sampling.get("sample_frame_indices") if isinstance(sampling, dict) else None
    grid_error = sampling.get("maximum_grid_time_error_seconds") if isinstance(sampling, dict) else None
    if (not isinstance(indices, list) or len(indices) != FRAME_COUNT
            or len(set(indices)) != FRAME_COUNT
            or any(type(index) is not int or index < 0 for index in indices)
            or sampling.get("anchor_sample_rank") != cpu.get("anchor_sample_rank")
            or any(sampling.get(key) != record_sampling.get(key) for key in (
                "sample_frame_indices", "sample_fps", "clip_start_source_seconds",
                "clip_end_source_seconds", "anchor_sample_rank",
                "maximum_grid_time_error_seconds"))
            or isinstance(grid_error, bool) or not isinstance(grid_error, (int, float))
            or not math.isfinite(float(grid_error)) or grid_error > 0.25):
        raise ValueError("sampled_frame_index_receipt_invalid")
    times = load_time_map_seconds(paths["time_map"])
    frames, decoded = decode_uniform_video(paths["video"], times,
        sample_frame_indices=indices, expected_sample_count=FRAME_COUNT)
    anchor_rank = cpu["anchor_sample_rank"]
    if (decoded[anchor_rank]["decoded_frame_index"] != cpu.get("anchor_frame_number")
            or abs(decoded[anchor_rank]["time_seconds"] - cpu.get("anchor_pts_seconds", -1)) > .001):
        raise ValueError("sampled_anchor_pts_binding_invalid")
    row_key = selected["row_key"]
    row_matches = [row for row in minerva_rows if row.get("key") == row_key]
    if (len(row_matches) != 1 or row_matches[0].get("video_id") != video_id
            or cpu.get("annotation_source_sha256", {}).get("minerva_json")
            != minerva_json_sha256):
        raise ValueError("question_source_binding_invalid")
    sampled = [{"rank": rank, "source_frame_index": point["decoded_frame_index"],
                "time_seconds": point["time_seconds"],
                "data_url": _jpeg_data_url(frame)}
               for rank, (frame, point) in enumerate(zip(frames, decoded, strict=True))]
    frame_image_sha256 = [_sha(base64.b64decode(
        frame["data_url"].removeprefix("data:image/jpeg;base64,"), validate=True))
        for frame in sampled]
    question = row_matches[0]["question"]
    claim = gpu["generation"]["generated_text"]
    item_id = _sha((gpu.get("row_key_sha256", "") + gpu.get("video_sha256", "")
                    + "minerva-generated-object-pixel-review-v1").encode())[:24]
    source = {"item_id": item_id, "row_index": row_index,
        "video_sha256": video_sha,
        "time_map_sha256": time_sha, "gpu_receipt_sha256": _sha(gpu_raw),
        "cpu_receipt_sha256": _sha(cpu_raw), "row_key_sha256": cpu["row_key_sha256"],
        "cohort_preflight_sha256": row_preflight_sha256,
        "runner_sha256": gpu["runner_sha256"], "model_revision": gpu["model_revision"],
        "pixel_tensor_sha256": cpu["pixel_tensor_sha256"],
        "rendered_prompt_sha256": cpu["rendered_prompt_sha256"],
        "processor_video_grid_thw": cpu["processor_video_grid_thw"],
        "question_sha256": _sha(question.encode("utf-8")),
        "claim_sha256": _sha(claim.encode("utf-8")),
        "frame_image_sha256": frame_image_sha256,
        "frame_image_inventory_sha256": _sha(json.dumps(
            frame_image_sha256, separators=(",", ":")).encode("ascii")),
        "sample_frame_indices": indices,
        "frame_pts_seconds": [point["time_seconds"] for point in decoded],
        "frame_count": len(frames)}
    item = {"item_id": item_id, "question": question,
            "claim": claim, "frames": sampled,
            "source": source}
    provenance = source
    return item, provenance


def _validate_attrition(entry: dict, *, row_index: int, selected: dict,
                        record: dict, cohort_sha256: str,
                        preflight: dict, preflight_raw: bytes,
                        preflight_sha256: str, manifest: dict,
                        manifest_sha256: str, minerva_json_sha256: str) -> dict:
    status = entry.get("status") if isinstance(entry, dict) else None
    if status == "preflight_failed":
        if (set(entry) != {"row_index", "status", "reason_code"}
                or record.get("status") != "fail"
                or entry.get("reason_code") != record.get("failure_code")
                or not isinstance(entry.get("reason_code"), str)
                or not re.fullmatch(r"[A-Za-z0-9_]{1,120}", entry["reason_code"])):
            raise ValueError("preflight_failure_attrition_binding_invalid")
        return {"row_index": row_index, "video_id_sha256": _sha(
            str(selected["video_id"]).encode()), "row_key_sha256": selected["row_key_sha256"],
            "status": status, "reason_code": entry["reason_code"],
            "cohort_preflight_sha256": preflight_sha256}
    if status == "missing_output":
        if (set(entry) != {"row_index", "status", "reason_code"}
                or record.get("status") != "pass"
                or not isinstance(entry.get("reason_code"), str)
                or not re.fullmatch(r"[A-Za-z0-9_]{1,80}", entry["reason_code"])):
            raise ValueError("missing_output_attrition_binding_invalid")
        return {"row_index": row_index, "video_id_sha256": _sha(
            str(selected["video_id"]).encode()), "row_key_sha256": selected["row_key_sha256"],
            "status": status, "reason_code": entry["reason_code"],
            "cohort_preflight_sha256": preflight_sha256}
    if status != "gpu_failed":
        raise ValueError("attrition_status_invalid")
    required = {"row_index", "status", "gpu_failure_receipt", "cpu_receipt",
                "video", "time_map"}
    if (set(entry) - required - {"cohort_preflight"}
            or not required.issubset(entry)):
        raise ValueError("gpu_failure_attrition_manifest_invalid")
    _, _, row_preflight_sha256, record = _row_preflight(entry,
        final_preflight=preflight, final_preflight_raw=preflight_raw,
        cohort_sha256=cohort_sha256, row_index=row_index, selected=selected)
    if record.get("status") != "pass":
        raise ValueError("gpu_failure_attrition_manifest_invalid")
    paths = {key: _resolve_private(entry[key]) for key in required - {"row_index", "status"}}
    gpu_raw, gpu = _read_json(paths["gpu_failure_receipt"])
    cpu_raw, cpu = _read_json(paths["cpu_receipt"])
    video_sha, video_md5 = _file_hashes(paths["video"])
    time_sha, time_md5 = _file_hashes(paths["time_map"])
    if (gpu.get("status") != "failed_single_video_generated_object_pixel_exploratory_pilot"
            or gpu.get("row_index") != row_index
            or gpu.get("cohort_sha256") != cohort_sha256
            or gpu.get("cohort_preflight_sha256") != row_preflight_sha256
            or gpu.get("cpu_pixel_receipt_sha256") != _sha(cpu_raw)
            or gpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or gpu.get("video_sha256") != video_sha
            or cpu.get("status") != "completed_minerva_generated_object_pixel_cpu_preflight"
            or cpu.get("row_index") != row_index
            or cpu.get("cohort_sha256") != cohort_sha256
            or cpu.get("cohort_cpu_preflight_sha256") != row_preflight_sha256
            or cpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or cpu.get("video_sha256") != video_sha
            or cpu.get("video_md5") != video_md5
            or cpu.get("time_map_sha256") != time_sha
            or cpu.get("time_map_md5") != time_md5
            or cpu.get("manifest_sha256") != manifest_sha256
            or record.get("video_sha256") != video_sha
            or record.get("video_md5") != video_md5
            or record.get("time_map_sha256") != time_sha
            or record.get("time_map_md5") != time_md5
            or cpu.get("video_id_sha256") != _sha(str(selected["video_id"]).encode())
            or cpu.get("annotation_source_sha256", {}).get("minerva_json")
            != minerva_json_sha256):
        raise ValueError("gpu_failure_receipt_binding_invalid")
    video_id = selected["video_id"]
    video_key = f"Videos/{video_id[:3]}/{video_id}.mp4"
    time_key = f"Videos/{video_id[:3]}/{video_id}_mp4_to_vrs_time_ns.csv"
    from preflight_minerva_grounding_cohort import normalize_manifest_path
    if (manifest.get(normalize_manifest_path(video_key)) != video_md5
            or manifest.get(normalize_manifest_path(time_key)) != time_md5):
        raise ValueError("gpu_failure_official_media_binding_invalid")
    return {"row_index": row_index, "video_id_sha256": _sha(video_id.encode()),
        "row_key_sha256": selected["row_key_sha256"], "status": status,
        "reason_code": (gpu.get("failure_type") if isinstance(gpu.get("failure_type"), str)
                        and re.fullmatch(r"[A-Za-z0-9_]{1,80}", gpu["failure_type"])
                        else "gpu_failure"),
        "gpu_failure_receipt_sha256": _sha(gpu_raw),
        "cpu_receipt_sha256": _sha(cpu_raw), "video_sha256": video_sha,
        "time_map_sha256": time_sha,
        "cohort_preflight_sha256": row_preflight_sha256}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(*, items_manifest_rel: Path, cohort_rel: Path, preflight_rel: Path,
          minerva_json_path: Path, official_manifest_path: Path,
          output_rel: Path) -> dict:
    items_path = _resolve_private(items_manifest_rel)
    cohort_path = _resolve_private(cohort_rel)
    preflight_path = _resolve_private(preflight_rel)
    items_raw, manifest_input = _read_json(items_path)
    cohort_raw, cohort = _read_json(cohort_path)
    preflight_raw, preflight = _read_json(preflight_path)
    selection = cohort.get("selection")
    input_entries = manifest_input.get("items")
    if (set(manifest_input) != {"schema_version", "items"}
            or type(manifest_input.get("schema_version")) is not int
            or manifest_input.get("schema_version") != 1
            or not isinstance(input_entries, list)
            or len(manifest_input["items"]) != ITEM_COUNT
            or not isinstance(selection, list) or len(selection) != ITEM_COUNT
            or cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"
            or preflight.get("status") != "completed_frozen_cohort_cpu_preflight"
            or preflight.get("cohort_sha256") != _sha(cohort_raw)):
        raise ValueError("review_manifest_or_frozen_cohort_invalid")
    if (any(not isinstance(row, dict) for row in selection)
            or any(not isinstance(entry, dict) for entry in input_entries)
            or len({row.get("video_id") for row in selection}) != ITEM_COUNT
            or len({row.get("row_key_sha256") for row in selection}) != ITEM_COUNT
            or [entry.get("row_index") for entry in input_entries]
            != list(range(ITEM_COUNT))
            or any(type(entry.get("row_index")) is not int for entry in input_entries)):
        raise ValueError("frozen_three_row_attrition_manifest_invalid")
    if minerva_json_path.is_symlink() or not minerva_json_path.is_file():
        raise ValueError("minerva_question_source_invalid")
    minerva_json_path = minerva_json_path.resolve(strict=True)
    minerva_rows = json.loads(minerva_json_path.read_text(encoding="utf-8"))
    minerva_json_sha256 = _sha_file(minerva_json_path)
    from download_frozen_hd_epic_videos import EXPECTED_MANIFEST_SHA256
    from preflight_minerva_grounding_cohort import parse_manifest
    manifest, official_sha = parse_manifest(official_manifest_path)
    if official_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError("official_manifest_binding_invalid")
    preflight_rows = {}
    for row in preflight.get("records", []):
        key = (row.get("video_id"), row.get("row_key_sha256"))
        if key in preflight_rows:
            raise ValueError("preflight_rows_not_unique")
        preflight_rows[key] = row
    items, provenance_items, attrition = [], [], []
    for row_index, entry in enumerate(input_entries):
        selected = selection[row_index]
        record = preflight_rows.get((selected.get("video_id"),
                                     selected.get("row_key_sha256")))
        if record is None:
            raise ValueError("frozen_row_missing_from_preflight")
        if entry.get("status") == "review":
            item, provenance = _prepare_item(entry, cohort=cohort,
                cohort_sha256=_sha(cohort_raw), preflight=preflight,
                preflight_raw=preflight_raw,
                preflight_sha256=_sha(preflight_raw), minerva_rows=minerva_rows,
                minerva_json_sha256=minerva_json_sha256,
                manifest=manifest, manifest_sha256=official_sha)
            items.append(item)
            provenance_items.append(provenance)
        else:
            attrition.append(_validate_attrition(entry, row_index=row_index,
                selected=selected, record=record,
                cohort_sha256=_sha(cohort_raw),
                preflight=preflight, preflight_raw=preflight_raw,
                preflight_sha256=_sha(preflight_raw), manifest=manifest,
                manifest_sha256=official_sha,
                minerva_json_sha256=minerva_json_sha256))
    if not items:
        raise ValueError("no_successful_gpu_rows_for_review")
    validate_review_items(items)
    model_revisions = {item["model_revision"] for item in provenance_items}
    if len(model_revisions) != 1:
        raise ValueError("review_items_must_share_model_revision")
    provenance = {"schema_version": 1,
        "purpose": "private_minerva_generated_object_pixel_review_source_manifest",
        "cohort_sha256": _sha(cohort_raw),
        "cohort_preflight_sha256": _sha(preflight_raw),
        "items_manifest_sha256": _sha(items_raw),
        "minerva_question_source_sha256": minerva_json_sha256,
        "manifest_sha256": official_sha, "items": provenance_items,
        "requested_rows": ITEM_COUNT, "reviewable_rows": len(items),
        "attrition": attrition,
        "row_cohort_preflight_sha256": {
            str(row["row_index"]): row["cohort_preflight_sha256"]
            for row in [*provenance_items, *attrition]},
        "frames_per_item": FRAME_COUNT, "item_count": len(items),
        "rights_status": "unverified_local_only",
        "direct_faithfulness_measured": False}
    builder_sha256 = _sha_file(Path(__file__))
    provenance["builder_sha256"] = builder_sha256
    source_manifest_sha256 = _sha(_script_json(provenance).encode("utf-8"))
    provenance["source_manifest_sha256"] = source_manifest_sha256
    html_doc = render(items, provenance)
    encoded, artifact_fingerprint = finalize_page(html_doc.encode("utf-8"))
    output = _resolve_private(output_rel, must_exist=False)
    fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(encoded)
        while view:
            count = os.write(fd, view)
            view = view[count:]
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"status": "built_private_minerva_generated_object_pixel_review",
        "requested_rows": ITEM_COUNT, "reviewable_rows": len(items),
        "attrition_rows": len(attrition),
        "attrition_row_indices": [row["row_index"] for row in attrition],
        "frames_per_item": FRAME_COUNT,
        "artifact": str(output.relative_to(ROOT)), "artifact_sha256": _sha(encoded),
        "artifact_mode": "0600",
        "artifact_fingerprint": artifact_fingerprint,
        "builder_sha256": builder_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "rights_status": "unverified_local_only", "direct_faithfulness_measured": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items-manifest", required=True, type=Path,
                        help="0600 JSON; review/gpu_failed entries may set cohort_preflight to a private per-row receipt path")
    parser.add_argument("--cohort", required=True, type=Path)
    parser.add_argument("--cohort-preflight", required=True, type=Path)
    parser.add_argument("--minerva-json", type=Path, default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--official-manifest", type=Path,
                        default=Path("/tmp/hd-epic-downloader/data/md5.txt"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = build(items_manifest_rel=args.items_manifest,
            cohort_rel=args.cohort, preflight_rel=args.cohort_preflight,
            minerva_json_path=args.minerva_json,
            official_manifest_path=args.official_manifest, output_rel=args.output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "minerva_review_build_failed")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
