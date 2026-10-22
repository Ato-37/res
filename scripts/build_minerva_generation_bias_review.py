#!/usr/bin/env python3
"""Build a private blinded Japanese review page for Minerva generation-bias outputs.

The page deduplicates identical generated text within a row and shows its exact
64 source frames. Treatment-to-item associations are written to a separate
0600 private mapping manifest and are never embedded in the page.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "data" / "private_minerva"
ITEM_COUNT = 5
FRAME_COUNT = 64
BRANCH_ORDER = ("baseline", "bbox_key_bias", "spatial_sham_key_bias")
LAYERS_ZERO_BASED = {str(layer): 1 for layer in range(4, 16)}
PREFLIGHT_SHARED_FIELDS = ("schema_version", "status", "cohort_sha256",
    "manifest_sha256", "preflight_source_sha256", "decoder_source_sha256",
    "generated_runner_source_sha256", "bbox_mapper_source_sha256",
    "qwen_video_processor", "human_review_performed", "rights_status")
DEFAULT_GPU_RECEIPTS = [PRIVATE / f"minerva_visible_object_v2_generation_bias_row{i}_gpu_20261022_01.json"
                        for i in range(3)]
DEFAULT_CPU_RECEIPTS = [PRIVATE / f"minerva_visible_object_v2_fresh_row{i}_cpu_20261022_01.json"
                        for i in range(3)]
DEFAULT_PREFLIGHTS = [
    PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_01.json",
    PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json",
    PRIVATE / "minerva_visible_object_v2_cohort_cpu_preflight_20261022_02.json"]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_private(path: Path) -> tuple[bytes, dict]:
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)
            or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_size > 50_000_000):
        raise ValueError("private_json_permissions_or_size_invalid")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise ValueError("private_json_changed_during_read")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("private_json_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("private_json_shape_invalid")
    return raw, value


def _resolve_private(relative: str | Path, *, root: Path = ROOT,
                     private_root: Path = PRIVATE) -> Path:
    rel = Path(relative)
    if (rel.is_absolute() or not rel.parts
            or any(part in {"", ".", ".."} for part in rel.parts)):
        raise ValueError("private_relative_path_invalid")
    path = root
    for part in rel.parts:
        path = path / part
        try:
            if stat.S_ISLNK(path.lstat().st_mode):
                raise ValueError("private_symlink_rejected")
        except FileNotFoundError as exc:
            raise ValueError("private_path_component_missing") from exc
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(private_root.resolve()) or not resolved.is_file():
        raise ValueError("private_path_outside_root")
    return resolved


def _file_hashes(path: Path) -> tuple[str, str]:
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return sha.hexdigest(), md5.hexdigest()


def _validate_media_path(path: Path, private_root: Path) -> Path:
    if path.is_symlink():
        raise ValueError("private_media_symlink_rejected")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(private_root.resolve()) or not resolved.is_file():
        raise ValueError("private_media_path_invalid")
    current = resolved
    while current != private_root.resolve():
        if current.is_symlink():
            raise ValueError("private_media_symlink_rejected")
        current = current.parent
    return resolved


def _jpeg_data_url(frame: Image.Image) -> str:
    from io import BytesIO
    image = frame.convert("RGB")
    image.thumbnail((480, 360), Image.Resampling.LANCZOS)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _script_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) \
        .replace("<", "\\u003c")


def validate_review_items(items: list[dict]) -> None:
    if not isinstance(items, list) or len(items) != ITEM_COUNT:
        raise ValueError("generation_review_requires_five_items")
    ids = set()
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("item_id"), str)
                or not item["item_id"] or item["item_id"] in ids
                or not isinstance(item.get("question"), str)
                or not item["question"].strip()
                or not isinstance(item.get("claim"), str) or not item["claim"].strip()
                or not isinstance(item.get("frames"), list)
                or len(item["frames"]) != FRAME_COUNT):
            raise ValueError("generation_review_item_invalid")
        ids.add(item["item_id"])
        source = item.get("source")
        if (not isinstance(source, dict)
                or source.get("claim_sha256") != _sha(item["claim"].encode())
                or source.get("question_sha256") != _sha(item["question"].encode())):
            raise ValueError("generation_review_text_binding_invalid")
        image_hashes = []
        for rank, frame in enumerate(item["frames"]):
            if (not isinstance(frame, dict) or frame.get("rank") != rank
                    or type(frame.get("source_frame_index")) is not int
                    or frame["source_frame_index"] < 0
                    or not isinstance(frame.get("time_seconds"), (int, float))
                    or isinstance(frame.get("time_seconds"), bool)
                    or not isinstance(frame.get("data_url"), str)
                    or not frame["data_url"].startswith("data:image/jpeg;base64,")):
                raise ValueError("generation_review_frame_invalid")
            try:
                raw = base64.b64decode(frame["data_url"].split(",", 1)[1], validate=True)
            except (ValueError, base64.binascii.Error) as exc:
                raise ValueError("generation_review_frame_data_invalid") from exc
            image_hashes.append(_sha(raw))
        if ([frame["source_frame_index"] for frame in item["frames"]]
                != source.get("sample_frame_indices")
                or [frame["time_seconds"] for frame in item["frames"]]
                != source.get("frame_pts_seconds")):
            raise ValueError("generation_review_frame_source_binding_invalid")
        if (source.get("frame_image_sha256") != image_hashes
                or source.get("frame_image_inventory_sha256") != _sha(
                    json.dumps(image_hashes, separators=(",", ":")).encode())):
            raise ValueError("generation_review_frame_inventory_invalid")


def _render(items: list[dict], provenance: dict) -> str:
    validate_review_items(items)
    if (not re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("builder_sha256", "")))
            or not re.fullmatch(r"[0-9a-f]{64}",
                str(provenance.get("source_manifest_sha256", "")))):
        raise ValueError("review_provenance_hashes_missing")
    payload = _script_json({"items": items, "provenance": provenance})
    return r'''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="review-artifact-sha256" content="__PAGE_SHA__">
<meta name="review-builder-sha256" content="__BUILDER_SHA__">
<meta name="review-source-manifest-sha256" content="__SOURCE_SHA__">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>動画説明の確認</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#172b3a;font:16px/1.55 system-ui,sans-serif}main{max-width:1000px;margin:20px auto;padding:0 14px 40px}.card{background:white;border:1px solid #d3dde5;border-radius:12px;padding:18px;margin:12px 0}.note{background:#fff4d6;padding:12px;border-radius:8px}.muted{color:#526575}.frames{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.frame{margin:0}.frame img{display:block;width:100%;background:#111;border-radius:5px}.frame figcaption{font-size:.75rem}.options label{display:block;padding:10px;border:1px solid #ced9e0;border-radius:8px;margin:7px 0}button{border:0;background:#176b87;color:#fff;font:inherit;font-weight:700;border-radius:8px;padding:10px 15px;cursor:pointer}input{font:inherit;padding:9px;border:1px solid #9fb3c0;border-radius:7px}.hidden{display:none}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.error{color:#a11}@media(max-width:650px){.frames{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style></head><body><main><h1>生成された動画説明の確認</h1>
<div class="card note">各説明が表示された64枚の映像にどの程度支持されるかを判断してください。説明は同じ動画からの別生成の場合があります。生成条件やモデルの採点結果は表示しません。この記録だけで説明の真偽やfaithfulnessが確定するものではありません。</div>
<section id="start" class="card"><label>確認者のニックネーム（各自1つ）<br><input id="nickname" maxlength="40" autocomplete="off"></label><button id="begin">確認を始める</button></section>
<section id="task" class="hidden"><div class="card row"><strong id="counter"></strong><span id="reviewer"></span></div><div class="card"><h2>質問</h2><p id="question"></p><h2>生成された説明</h2><p id="claim"></p></div><div class="card"><h2>元動画からの64フレーム</h2><div id="frames" class="frames"></div></div><div class="card options"><h2>この映像は説明を支持しますか</h2>
<label><input type="radio" name="verdict" value="supports">支持する</label><label><input type="radio" name="verdict" value="partially_supports">一部を支持する</label><label><input type="radio" name="verdict" value="contradicts">矛盾する</label><label><input type="radio" name="verdict" value="not_visible">必要な情報が映っていない</label><label><input type="radio" name="verdict" value="cannot_judge">判断できない</label><p id="required" class="error"></p><button id="next">次へ</button></div></section>
<section id="done" class="hidden card"><p>この確認者の回答を追加しました。続けて別の確認者が回答できます。</p><div class="row"><button id="another">別の確認者</button><button id="save">全回答をJSONで保存</button></div><a id="download" class="hidden">JSONをダウンロード</a><p id="saved" class="muted"></p></section><section class="card"><p class="muted">レビュー可能な項目：5。回答はこのブラウザ内でまとめられます。ファイルはローカルで保管してください。</p></section>
<script>'use strict';const payload=__PAYLOAD__;const $=id=>document.getElementById(id);const items=payload.items;const labels={supports:'supports',partially_supports:'partially_supports',contradicts:'contradicts',not_visible:'not_visible',cannot_judge:'cannot_judge'};let answers=[],respondents=[],index=0,nickname='';function start(){const n=$('nickname').value.trim();if(!n){$('nickname').focus();return}nickname=n;index=0;answers=[];$('start').classList.add('hidden');show()}function show(){if(index>=items.length){respondents.push({nickname,answers:[...answers]});$('task').classList.add('hidden');$('done').classList.remove('hidden');return}const x=items[index];$('task').classList.remove('hidden');$('counter').textContent=`${index+1} / ${items.length}`;$('reviewer').textContent=`ニックネーム：${nickname}`;$('question').textContent=x.question;$('claim').textContent=x.claim;$('frames').replaceChildren(...x.frames.map(f=>{const fig=document.createElement('figure');fig.className='frame';const im=document.createElement('img');im.loading='lazy';im.alt=`入力フレーム ${f.rank+1}`;im.src=f.data_url;const cap=document.createElement('figcaption');cap.textContent=`${f.rank+1}枚目 · ${Number(f.time_seconds).toFixed(2)}秒`;fig.append(im,cap);return fig}));document.querySelectorAll('[name=verdict]').forEach(x=>x.checked=false);$('required').textContent='';$('next').textContent=index===items.length-1?'確認を終える':'次へ'}$('begin').addEventListener('click',start);$('next').addEventListener('click',()=>{const v=document.querySelector('[name=verdict]:checked');if(!v){$('required').textContent='回答するには1つ選んでください';return}const x=items[index];answers.push({item_id:x.item_id,verdict:labels[v.value],claim_sha256:x.source.claim_sha256,question_sha256:x.source.question_sha256,frame_image_inventory_sha256:x.source.frame_image_inventory_sha256,frame_image_sha256:x.source.frame_image_sha256,source_frame_indices:x.frames.map(f=>f.source_frame_index),frame_pts_seconds:x.frames.map(f=>f.time_seconds)});index++;show()});$('another').addEventListener('click',()=>{$('done').classList.add('hidden');$('start').classList.remove('hidden');$('nickname').value='';$('nickname').focus()});$('save').addEventListener('click',()=>{const out={schema_version:1,purpose:'private_blinded_minerva_generation_support_review',status:'informal_frame_grounded_verdicts_only',created_at_utc:new Date().toISOString(),artifact_fingerprint:document.querySelector('meta[name=review-artifact-sha256]').content,builder_sha256:payload.provenance.builder_sha256,source_manifest_sha256:payload.provenance.source_manifest_sha256,respondents,limitations:['少数のレビューから統計的結論を導かない','映像中の見え方に関する判断であり、真偽やfaithfulnessを確定しない','回答者の独立性、本人性、64枚全てを見たことは検証していない']};const blob=new Blob([JSON.stringify(out,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=$('download');a.href=url;a.download='minerva_generation_bias_review_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json';a.classList.remove('hidden');a.click();$('saved').textContent='JSONのダウンロードを開始しました。保存先を確認してください。'});</script></body></html>'''.replace("__PAYLOAD__", payload).replace("__BUILDER_SHA__", provenance["builder_sha256"]).replace("__SOURCE_SHA__", provenance["source_manifest_sha256"])


def _finalize_page(page: bytes) -> tuple[bytes, str]:
    marker = b'<meta name="review-artifact-sha256" content="__PAGE_SHA__">'
    if page.count(marker) != 1:
        raise ValueError("page_fingerprint_slot_invalid")
    fingerprint = _sha(page)
    return page.replace(marker, marker.replace(b"__PAGE_SHA__", fingerprint.encode()), 1), fingerprint


def _verify_preflight_row(*, row_preflight: dict, row_preflight_raw: bytes,
                          final_preflight: dict, cohort_sha: str,
                          selected: dict, row_index: int) -> tuple[str, dict]:
    if (any(row_preflight.get(key) != final_preflight.get(key)
            for key in PREFLIGHT_SHARED_FIELDS)
            or row_preflight.get("cohort_sha256") != cohort_sha):
        raise ValueError("row_preflight_source_or_manifest_mismatch")
    target = (selected.get("video_id"), selected.get("row_key_sha256"))
    alt_rows = [r for r in row_preflight.get("records", [])
        if isinstance(r, dict) and (r.get("video_id"), r.get("row_key_sha256")) == target]
    final_rows = [r for r in final_preflight.get("records", [])
        if isinstance(r, dict) and (r.get("video_id"), r.get("row_key_sha256")) == target]
    if (len(alt_rows) != 1 or len(final_rows) != 1
            or alt_rows[0] != final_rows[0] or alt_rows[0].get("status") != "pass"):
        raise ValueError("row_preflight_record_mismatch")
    return _sha(row_preflight_raw), final_rows[0]


def _validate_generation_receipt(gpu: dict, *, row_index: int,
                                 expected_cpu_sha: str, cohort_sha: str,
                                 row_preflight_sha: str, selected: dict,
                                 cpu: dict, video_sha: str, time_sha: str,
                                 source_root: Path) -> list[dict]:
    if (gpu.get("status") != "completed_minerva_visible_object_generation_bias_v1"
            or gpu.get("direct_faithfulness_validated") is not False
            or gpu.get("row_index") != row_index
            or gpu.get("cohort_sha256") != cohort_sha
            or gpu.get("cohort_preflight_sha256") != row_preflight_sha
            or gpu.get("cpu_receipt_sha256") != expected_cpu_sha
            or gpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or gpu.get("video_sha256") != video_sha
            or gpu.get("time_map_sha256") != time_sha
            or gpu.get("frame_count") != FRAME_COUNT
            or gpu.get("branch_order") != list(BRANCH_ORDER)
            or gpu.get("failure_code") is not None
            or cpu.get("cohort_sha256") != cohort_sha
            or cpu.get("cohort_cpu_preflight_sha256") != row_preflight_sha
            or cpu.get("row_index") != row_index
            or cpu.get("row_key_sha256") != selected.get("row_key_sha256")
            or cpu.get("video_sha256") != video_sha
            or cpu.get("time_map_sha256") != time_sha
            or gpu.get("base_pixel_tensor_sha256")
                != cpu.get("pixel_tensor_sha256", {}).get("base")):
        raise ValueError("generation_cpu_gpu_binding_invalid")
    from run_minerva_generated_object_pixel_pilot import MODEL_REVISION
    if gpu.get("model_revision") != MODEL_REVISION:
        raise ValueError("generation_model_revision_mismatch")
    sources = gpu.get("source_sha256")
    expected_sources = {"runner": "run_minerva_visible_object_generation_bias_v1.py",
        "pilot": "run_minerva_generated_object_pixel_pilot.py",
        "generation_hooks": "qwen_visual_attention_boost.py",
        "generation_audit": "run_qta_intervened_interval_generation.py"}
    if not isinstance(sources, dict) or set(sources) != set(expected_sources):
        raise ValueError("generation_source_bundle_invalid")
    for key, filename in expected_sources.items():
        path = source_root / filename
        if (path.is_symlink() or not path.is_file()
                or _sha(path.read_bytes()) != sources[key]):
            raise ValueError("generation_source_hash_mismatch")
    cpu_sources = cpu.get("source_sha256")
    if not isinstance(cpu_sources, dict):
        raise ValueError("cpu_source_bundle_missing")
    for filename, digest in cpu_sources.items():
        source_path = source_root / filename
        if (source_path.is_symlink() or not source_path.is_file()
                or _sha(source_path.read_bytes()) != digest):
            raise ValueError("cpu_source_hash_mismatch")
    outcomes = gpu.get("outcomes")
    dose = gpu.get("key_dose")
    if (not isinstance(outcomes, dict) or set(outcomes) != set(BRANCH_ORDER)
            or not isinstance(dose, dict) or dose.get("bbox") != dose.get("spatial_sham")
            or type(dose.get("bbox")) is not int or dose["bbox"] <= 0
            or dose.get("bbox") != cpu.get("condition_key_counts", {}).get("bbox")
            or dose.get("spatial_sham")
                != cpu.get("condition_key_counts", {}).get("spatial_sham")
            or gpu.get("prompt_sha256") != cpu.get("rendered_prompt_sha256")
            or gpu.get("biased_layers_1_based") != list(range(5, 17))):
        raise ValueError("generation_branch_or_dose_contract_invalid")
    grouped: dict[str, dict] = {}
    for branch in BRANCH_ORDER:
        outcome = outcomes[branch]
        text = outcome.get("generated_text") if isinstance(outcome, dict) else None
        text_hash = outcome.get("generated_text_sha256") if isinstance(outcome, dict) else None
        token_hash = outcome.get("generated_token_ids_sha256") if isinstance(outcome, dict) else None
        token_count = outcome.get("generated_token_count") if isinstance(outcome, dict) else None
        if (not isinstance(text, str) or not text.strip()
                or text_hash != _sha(text.encode("utf-8"))
                or not isinstance(token_hash, str) or len(token_hash) != 64
                or type(token_count) is not int or token_count < 1
                or not isinstance(outcome.get("shape_trace"), list)
                or not isinstance(outcome.get("mechanical_diagnostic"), dict)):
            raise ValueError("generated_text_integrity_invalid")
        audit = outcome.get("mask_audit")
        biased = branch != "baseline"
        expected_keys = dose["bbox"] if branch == "bbox_key_bias" else dose["spatial_sham"]
        if (not isinstance(audit, dict) or audit.get("applied") is not biased
                or audit.get("all_selected_cells_match_bias") is not True
                or audit.get("hook_calls") != (token_count * 12 if biased else 0)
                or (not biased and (audit.get("verified_cells") != 0
                    or audit.get("calls_per_layer") not in (None, {})))
                or (biased and (audit.get("verified_cells") != token_count * expected_keys * 12
                    or audit.get("calls_per_layer") != {
                        str(layer): token_count for layer in range(4, 16)}))):
            raise ValueError("generation_branch_hook_audit_invalid")
        if biased and abs(float(audit.get("effective_mask_bias", 0)) - 0.69140625) > 1e-9:
            raise ValueError("generation_effective_bias_mismatch")
        grouped.setdefault(text_hash, {"claim": text, "token_hash": token_hash,
            "token_count": token_count, "branches": []})
        item = grouped[text_hash]
        if (item["claim"] != text or item["token_hash"] != token_hash
                or item["token_count"] != token_count):
            raise ValueError("duplicate_text_token_binding_mismatch")
        item["branches"].append(branch)
    return list(grouped.values())


def build_artifacts(*, cohort_path: Path, final_preflight_path: Path,
                    minerva_json_path: Path, gpu_paths: list[Path],
                    cpu_paths: list[Path], row_preflight_paths: list[Path],
                    video_paths: list[Path], time_map_paths: list[Path],
                    output_path: Path, mapping_path: Path,
                    source_root: Path = ROOT / "scripts",
                    private_root: Path = PRIVATE,
                    decoder=None) -> dict:
    if not all(len(paths) == 3 for paths in (gpu_paths, cpu_paths,
            row_preflight_paths, video_paths, time_map_paths)):
        raise ValueError("exactly_three_row_inputs_required")
    cohort_raw, cohort = _read_private(cohort_path)
    final_raw, final = _read_private(final_preflight_path)
    if (cohort.get("status") != "frozen_pre_outcome_exploratory_cohort"
            or cohort.get("requested_rows") != 3
            or not isinstance(cohort.get("selection"), list)
            or len(cohort["selection"]) != 3
            or final.get("status") != "completed_frozen_cohort_cpu_preflight"
            or final.get("requested_rows") != 3 or final.get("effective_rows") != 3
            or final.get("cohort_sha256") != _sha(cohort_raw)
            or len(final.get("records", [])) != 3
            or any(r.get("status") != "pass" for r in final["records"])):
        raise ValueError("cohort_or_final_preflight_invalid")
    minerva_json_path = minerva_json_path.resolve(strict=True)
    minerva_raw = minerva_json_path.read_bytes()
    minerva_rows = json.loads(minerva_raw.decode("utf-8"))
    from preflight_minerva_grounding_cohort import load_time_map_seconds
    from run_minerva_ego_anchor_gpu_pilot import decode_uniform_video
    decoder = decoder or decode_uniform_video
    all_items, map_items, item_sources = [], [], []
    for row_index in range(3):
        selected = cohort["selection"][row_index]
        video_id = selected.get("video_id")
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("selected_video_id_invalid")
        cpu_raw, cpu = _read_private(cpu_paths[row_index])
        gpu_raw, gpu = _read_private(gpu_paths[row_index])
        row_preflight_raw, row_preflight = _read_private(row_preflight_paths[row_index])
        preflight_sha, preflight_record = _verify_preflight_row(
            row_preflight=row_preflight, row_preflight_raw=row_preflight_raw,
            final_preflight=final, cohort_sha=_sha(cohort_raw),
            selected=selected, row_index=row_index)
        video_path = _validate_media_path(video_paths[row_index], private_root)
        time_path = _validate_media_path(time_map_paths[row_index], private_root)
        video_sha, video_md5 = _file_hashes(video_path)
        time_sha, time_md5 = _file_hashes(time_path)
        if (cpu.get("status") != "completed_minerva_generated_object_pixel_cpu_preflight"
                or cpu.get("cohort_sha256") != _sha(cohort_raw)
                or cpu.get("row_index") != row_index
                or cpu.get("row_key_sha256") != selected.get("row_key_sha256")
                or cpu.get("video_id_sha256") != _sha(video_id.encode())
                or cpu.get("video_sha256") != video_sha
                or cpu.get("video_md5") != video_md5
                or cpu.get("time_map_sha256") != time_sha
                or cpu.get("time_map_md5") != time_md5
                or cpu.get("manifest_sha256") != final.get("manifest_sha256")
                or preflight_record.get("video_sha256") != video_sha
                or preflight_record.get("video_md5") != video_md5
                or preflight_record.get("time_map_sha256") != time_sha
                or preflight_record.get("time_map_md5") != time_md5
                or cpu.get("annotation_source_sha256", {}).get("minerva_json")
                    != _sha(minerva_raw)):
            raise ValueError("cpu_preflight_media_or_annotation_binding_invalid")
        generation_groups = _validate_generation_receipt(gpu, row_index=row_index,
            expected_cpu_sha=_sha(cpu_raw), cohort_sha=_sha(cohort_raw),
            row_preflight_sha=preflight_sha, selected=selected, cpu=cpu,
            video_sha=video_sha, time_sha=time_sha, source_root=source_root)
        row = preflight_record
        sample = cpu.get("sampling")
        record_sampling = row.get("clip_sampling", {})
        sample_indices = sample.get("sample_frame_indices") if isinstance(sample, dict) else None
        if (not isinstance(sample_indices, list) or len(sample_indices) != FRAME_COUNT
                or len(set(sample_indices)) != FRAME_COUNT
                or any(type(i) is not int or i < 0 for i in sample_indices)
                or any(sample.get(k) != record_sampling.get(k) for k in (
                    "sample_frame_indices", "sample_fps", "clip_start_source_seconds",
                    "clip_end_source_seconds", "anchor_sample_rank",
                    "maximum_grid_time_error_seconds"))
                or sample.get("maximum_grid_time_error_seconds", 999) > .25):
            raise ValueError("generation_review_sampling_binding_invalid")
        anchor = row.get("rationale_mask_anchor", {})
        if (sample.get("anchor_sample_rank") != cpu.get("anchor_sample_rank")
                or cpu.get("anchor_frame_number") != anchor.get("mask_frame_number")
                or abs(float(cpu.get("anchor_pts_seconds", -1))
                    - float(anchor.get("mask_time_seconds", -2))) > .001):
            raise ValueError("cpu_preflight_anchor_binding_invalid")
        times = load_time_map_seconds(time_map_paths[row_index])
        frames, decoded = decoder(video_paths[row_index], times,
            sample_frame_indices=sample_indices, expected_sample_count=FRAME_COUNT)
        if (len(frames) != FRAME_COUNT or len(decoded) != FRAME_COUNT
                or [p.get("decoded_frame_index") for p in decoded] != sample_indices
                or any(not isinstance(p.get("time_seconds"), (int, float))
                    for p in decoded)):
            raise ValueError("decoded_review_frames_not_sample_binding")
        record_pts = record_sampling.get("sample_pts_seconds")
        if isinstance(record_pts, list) and (len(record_pts) != FRAME_COUNT
                or any(abs(float(p["time_seconds"]) - t) > .001
                    for p, t in zip(decoded, record_pts, strict=True))):
            raise ValueError("decoded_review_pts_differ_from_preflight")
        row_matches = [row for row in minerva_rows if row.get("key") == selected.get("row_key")
                       and row.get("video_id") == video_id]
        if len(row_matches) != 1 or not isinstance(row_matches[0].get("question"), str):
            raise ValueError("review_question_source_binding_invalid")
        question = row_matches[0]["question"]
        frame_items = []
        for rank, (frame, point) in enumerate(zip(frames, decoded, strict=True)):
            frame_items.append({"rank": rank,
                "source_frame_index": point["decoded_frame_index"],
                "time_seconds": point["time_seconds"],
                "data_url": _jpeg_data_url(frame)})
        frame_hashes = [_sha(base64.b64decode(x["data_url"].split(",", 1)[1]))
                        for x in frame_items]
        rowkey_sha = selected["row_key_sha256"]
        for group in generation_groups:
            item_id = _sha((rowkey_sha + group["claim"]
                + "minerva-generation-bias-review-v1").encode("utf-8"))[:24]
            source = {"item_id": item_id,
                "video_sha256": video_sha, "time_map_sha256": time_sha,
                "row_key_sha256": rowkey_sha,
                "cpu_receipt_sha256": _sha(cpu_raw),
                "gpu_receipt_sha256": _sha(gpu_raw),
                "cohort_preflight_sha256": preflight_sha,
                "claim_sha256": _sha(group["claim"].encode()),
                "question_sha256": _sha(question.encode()),
                "generated_token_ids_sha256": group["token_hash"],
                "frame_image_sha256": frame_hashes,
                "frame_image_inventory_sha256": _sha(json.dumps(frame_hashes,
                    separators=(",", ":")).encode()),
                "sample_frame_indices": sample_indices,
                "frame_pts_seconds": [p["time_seconds"] for p in decoded]}
            all_items.append({"item_id": item_id, "question": question,
                "claim": group["claim"], "frames": frame_items, "source": source})
            map_items.append({"item_id": item_id, "row_index": row_index,
                "branches": group["branches"],
                "claim_sha256": source["claim_sha256"],
                "generated_token_ids_sha256": group["token_hash"],
                "cpu_receipt_sha256": _sha(cpu_raw),
                "gpu_receipt_sha256": _sha(gpu_raw),
                "cohort_preflight_sha256": preflight_sha,
                "row_key_sha256": rowkey_sha})
            item_sources.append(source)
    row_item_counts = [sum(1 for item in map_items if item["row_index"] == i) for i in range(3)]
    if (len(all_items) != ITEM_COUNT or len({item["item_id"] for item in all_items}) != ITEM_COUNT
            or row_item_counts != [2, 1, 2]):
        raise ValueError("expected_five_distinct_review_items")
    # Deterministically mix items without revealing row group/order.
    all_items.sort(key=lambda item: _sha((item["item_id"] + "review-order-v1").encode()))
    validate_review_items(all_items)
    builder_sha = _sha(Path(__file__).read_bytes())
    manifest = {"schema_version": 1,
        "purpose": "private_blinded_minerva_generation_bias_review_source_manifest",
        "cohort_sha256": _sha(cohort_raw),
        "final_preflight_sha256": _sha(final_raw),
        "minerva_source_sha256": _sha(minerva_raw),
        "items": [{k: v for k, v in source.items() if k != "item_id"}
                  | {"item_id": source["item_id"]} for source in item_sources],
        "requested_rows": 3, "distinct_review_items": len(all_items),
        "frames_per_item": FRAME_COUNT,
        "rights_status": "unverified_local_only",
        "branches_blinded_in_review_page": True,
        "direct_faithfulness_measured": False,
        "builder_sha256": builder_sha}
    source_manifest_sha = _sha(_script_json(manifest).encode())
    manifest["source_manifest_sha256"] = source_manifest_sha
    page_html = _render(all_items, manifest)
    page_bytes, fingerprint = _finalize_page(page_html.encode("utf-8"))
    mapping = {"schema_version": 1,
        "purpose": "private_blinded_review_item_to_generation_branch_map",
        "status": "private_mapping_do_not_share_with_reviewers",
        "artifact_fingerprint": fingerprint,
        "builder_sha256": builder_sha,
        "source_manifest_sha256": source_manifest_sha,
        "cohort_sha256": _sha(cohort_raw),
        "final_preflight_sha256": _sha(final_raw),
        "items": map_items,
        "branch_definitions": {"baseline": "no attention bias",
            "bbox_key_bias": "annotation bbox visual-key +ln(2)",
            "spatial_sham_key_bias": "equal-dose spatial-sham visual-key +ln(2)"},
        "rights_status": "unverified_local_only"}
    output_path = output_path.resolve(strict=False)
    mapping_path = mapping_path.resolve(strict=False)
    if (private_root.is_symlink() or output_path.parent != private_root.resolve()
            or mapping_path.parent != private_root
            or mapping_path.parent.resolve() != private_root.resolve()
            or output_path == mapping_path or output_path.exists()
            or mapping_path.exists() or output_path.is_symlink()
            or mapping_path.is_symlink()):
        raise ValueError("private_review_output_contract_invalid")
    mapping_bytes = (json.dumps(mapping, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    _write_exclusive(output_path, page_bytes, 0o600)
    try:
        _write_exclusive(mapping_path, mapping_bytes, 0o600)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
    return {"status": "built_private_blinded_generation_review",
        "requested_rows": 3, "distinct_review_items": ITEM_COUNT,
        "frames_per_item": FRAME_COUNT,
        "page_sha256": _sha(page_bytes), "artifact_fingerprint": fingerprint,
        "mapping_sha256": _sha(mapping_bytes),
        "builder_sha256": builder_sha,
        "source_manifest_sha256": source_manifest_sha,
        "rights_status": "unverified_local_only"}


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        view = memoryview(data)
        while view:
            n = os.write(fd, view)
            view = view[n:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=PRIVATE / "minerva_metadata_proxy_anchor_cohort_v2_20261022_03.json")
    parser.add_argument("--final-preflight", type=Path, default=DEFAULT_PREFLIGHTS[1])
    parser.add_argument("--minerva-json", type=Path, default=Path("/tmp/minerva_ego_20260205.json"))
    parser.add_argument("--output", type=Path, required=True,
        help="New .html path directly under data/private_minerva/")
    parser.add_argument("--mapping-output", type=Path, required=True,
        help="New .json path directly under data/private_minerva/; keep private")
    parser.add_argument("--user-authorized-local-only", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        for path in (args.cohort, args.final_preflight):
            if path.is_symlink() or not path.resolve().is_relative_to(PRIVATE.resolve()):
                raise ValueError("private_cohort_input_invalid")
        result = build_artifacts(cohort_path=args.cohort,
            final_preflight_path=args.final_preflight,
            minerva_json_path=args.minerva_json,
            gpu_paths=DEFAULT_GPU_RECEIPTS, cpu_paths=DEFAULT_CPU_RECEIPTS,
            row_preflight_paths=DEFAULT_PREFLIGHTS,
            video_paths=[PRIVATE / "range_download" / "HD-EPIC" / "Videos"
                / selected_prefix / f"{video_id}.mp4"
                for selected_prefix, video_id in _selected_media(args.cohort)[0]],
            time_map_paths=[PRIVATE / "range_download" / "HD-EPIC" / "Videos"
                / selected_prefix / f"{video_id}_mp4_to_vrs_time_ns.csv"
                for selected_prefix, video_id in _selected_media(args.cohort)[0]],
            output_path=args.output, mapping_path=args.mapping_output)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc).split(":", 1)[0] or "generation_review_build_failed")
    print(json.dumps({"status": result["status"],
        "requested_rows": result["requested_rows"],
        "distinct_review_items": result["distinct_review_items"],
        "frames_per_item": result["frames_per_item"],
        "artifact_fingerprint": result["artifact_fingerprint"],
        "builder_sha256": result["builder_sha256"],
        "source_manifest_sha256": result["source_manifest_sha256"]}))
    return 0


def _selected_media(cohort_path: Path):
    _, cohort = _read_private(cohort_path)
    selection = cohort.get("selection")
    if not isinstance(selection, list) or len(selection) != 3:
        raise ValueError("frozen_selection_invalid")
    return ([(str(row["video_id"])[:3], row["video_id"]) for row in selection], cohort)


if __name__ == "__main__":
    raise SystemExit(main())
