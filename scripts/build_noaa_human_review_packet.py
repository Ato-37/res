"""Build a private, local-only human QA review packet for the NOAA pilot.

The packet contains the six QA items and videos only when the canonical source
audit passes apart from its intentionally missing human attestation. It never
creates an attestation or advances the QA gate. A human can download a private
attestation only after choosing an answer before each gold reveal, marking all
items reviewed, completing every required group check, and explicitly claiming
human review with a name and role.

Usage::

    .venv/bin/python scripts/build_noaa_human_review_packet.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

import audit_noaa_qa_gate as gate


RIGHTS_SCOPE = {
    "video_only_model_inference": True,
    "audio_use": False,
    "external_media_publication": False,
    "required_credit": "NOAA Ocean Exploration",
}

CHECK_LABELS = {
    "answer_checked_against_video": "I checked every answer against the video.",
    "evidence_supports_answer": "The annotated evidence window supports the answer.",
    "unique_best_choice": "The keyed answer is the unique best choice among the five options.",
    "stem_leakage_checked": "I checked for answer leakage in the question stem and choices.",
    "video_only_review": "I reviewed the video evidence only; I did not use audio.",
    "no_audio_used": "No audio was used for this video-only QA review.",
    "rights_scope_confirmed": "I confirm the exact video-only rights scope shown below.",
    "credit_acknowledged": "I acknowledge the required NOAA Ocean Exploration credit.",
}
HASH_FIELDS = ("media_sha256", "samples_sha256", "gold_sha256", "audit_sha256", "rights_sha256")
HEX_HASH = re.compile(r"^[a-f0-9]{64}$")


class PacketError(ValueError):
    """The validated source set cannot safely produce a review packet."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _run_capture(command):
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PacketError("local ffmpeg/ffprobe media validation failed") from exc


def _probe_streams(path, ffprobe_bin="ffprobe"):
    result = _run_capture([
        ffprobe_bin, "-v", "error", "-show_entries", "stream=codec_type,start_time,duration,nb_frames",
        "-of", "json", str(path),
    ])
    try:
        streams = json.loads(result.stdout)["streams"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise PacketError("ffprobe returned invalid stream metadata") from exc
    if not isinstance(streams, list):
        raise PacketError("ffprobe returned invalid stream metadata")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1:
        raise PacketError("review media must have exactly one video stream")
    video = videos[0]
    try:
        start = float(video["start_time"])
        duration = float(video["duration"])
        frames = int(video["nb_frames"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise PacketError("ffprobe video timing metadata is incomplete") from exc
    if not (math.isfinite(start) and math.isfinite(duration) and duration > 0 and frames > 0):
        raise PacketError("ffprobe video timing metadata is invalid")
    return {"start_time": start, "duration": duration, "nb_frames": frames,
            "audio_streams": len(audios), "video_streams": len(videos)}


def _probe_video_pts(path, ffprobe_bin="ffprobe"):
    result = _run_capture([
        ffprobe_bin, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path),
    ])
    timestamps = []
    try:
        for line in result.stdout.splitlines():
            value = line.strip().split(",", 1)[0]
            if value:
                timestamps.append(float(value))
    except ValueError as exc:
        raise PacketError("ffprobe returned invalid video frame timestamps") from exc
    if not timestamps or any(not math.isfinite(value) for value in timestamps):
        raise PacketError("ffprobe returned missing or non-finite video frame timestamps")
    return timestamps


def _packet_sha256(path, ffmpeg_bin="ffmpeg"):
    result = _run_capture([
        ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-c:v", "copy", "-f", "hash", "-hash", "sha256", "-",
    ])
    match = re.fullmatch(r"\s*SHA256=([a-fA-F0-9]{64})\s*", result.stdout)
    if not match:
        raise PacketError("ffmpeg returned invalid video packet hash output")
    return match.group(1).lower()


def _prepare_silent_video(source_path, output_path, ffmpeg_bin="ffmpeg", ffprobe_bin="ffprobe"):
    source_path, output_path = Path(source_path), Path(output_path)
    source_meta = _probe_streams(source_path, ffprobe_bin)
    source_pts = _probe_video_pts(source_path, ffprobe_bin)
    _run_capture([
        ffmpeg_bin, "-nostdin", "-hide_banner", "-v", "error", "-n", "-copyts",
        "-i", str(source_path), "-map", "0:v:0", "-c:v", "copy", "-an",
        "-map_metadata", "-1", str(output_path),
    ])
    if not output_path.is_file():
        raise PacketError("ffmpeg did not create the silent review copy")
    output_meta = _probe_streams(output_path, ffprobe_bin)
    output_pts = _probe_video_pts(output_path, ffprobe_bin)
    if output_meta["audio_streams"] != 0 or output_meta["video_streams"] != 1:
        raise PacketError("silent review copy must contain one video stream and no audio")
    if (source_meta["nb_frames"] != output_meta["nb_frames"]
            or len(source_pts) != len(output_pts)
            or any(abs(left - right) > 1e-6 for left, right in zip(source_pts, output_pts))
            or abs(source_meta["start_time"] - output_meta["start_time"]) > 1e-6
            or abs(source_meta["duration"] - output_meta["duration"]) > 1e-6):
        raise PacketError("silent review copy changed video timing or frame timestamps")
    source_packet_hash = _packet_sha256(source_path, ffmpeg_bin)
    output_packet_hash = _packet_sha256(output_path, ffmpeg_bin)
    if source_packet_hash != output_packet_hash:
        raise PacketError("silent review copy changed video packet payload")
    return {
        "original_media_sha256": _sha256_file(source_path),
        "silent_copy_sha256": _sha256_file(output_path),
        "source_video_packet_sha256": source_packet_hash,
        "copy_video_packet_sha256": output_packet_hash,
        "source_video_start_time": source_meta["start_time"],
        "copy_video_start_time": output_meta["start_time"],
        "source_video_duration": source_meta["duration"],
        "copy_video_duration": output_meta["duration"],
        "source_video_nb_frames": source_meta["nb_frames"],
        "copy_video_nb_frames": output_meta["nb_frames"],
        "video_pts_count": len(source_pts),
        "source_audio_streams_removed": source_meta["audio_streams"],
        "copy_audio_streams": output_meta["audio_streams"],
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _records(value, key):
    if isinstance(value, list):
        records = value
    elif isinstance(value, dict):
        records = value.get(key)
    else:
        records = None
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise PacketError("validated source record container is invalid")
    return records


def _safe_json(value):
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _read_verified_sources(data_root, gate_report):
    if (gate_report.get("decision") != "no_go"
            or gate_report.get("attestation_status") != "missing"
            or gate_report.get("failure_reasons") != ["human_attestation_missing"]):
        raise PacketError("source gate must pass with only the human-attestation gate pending")
    groups = gate.GROUPS
    if (gate_report.get("source_groups_requested") != len(groups)
            or gate_report.get("source_groups_valid") != len(groups)
            or gate_report.get("samples") != gate_report.get("gold_records")
            or gate_report.get("samples") != gate_report.get("audit_records")):
        raise PacketError("source gate groups or record counts are incomplete")
    records_by_id = {item.get("group_id"): item for item in gate_report.get("groups", [])
                     if isinstance(item, dict)}
    source_hashes = gate_report.get("source_sha256")
    if set(records_by_id) != {group["group_id"] for group in groups}:
        raise PacketError("source gate group set does not match canonical groups")
    if not isinstance(source_hashes, dict):
        raise PacketError("source gate hashes are missing")

    prepared = []
    for group in groups:
        group_id = group["group_id"]
        gate_record = records_by_id[group_id]
        hashes = source_hashes.get(group_id)
        if not isinstance(hashes, dict) or any(
                not isinstance(hashes.get(field), str) or not HEX_HASH.fullmatch(hashes[field])
                for field in HASH_FIELDS):
            raise PacketError("source gate hashes are invalid")
        group_root = data_root / group["directory"]
        private_root = group_root / "private"
        media_path = group_root / "media" / "source.mp4"
        sample_path = private_root / "samples.json"
        gold_path = private_root / "gold.json"
        audit_path = private_root / group["audit_file"]
        rights_path = private_root / "rights_review_v2.json"
        paths = {
            "media_sha256": media_path,
            "samples_sha256": sample_path,
            "gold_sha256": gold_path,
            "audit_sha256": audit_path,
            "rights_sha256": rights_path,
        }
        raw_by_field = {}
        for field, path in paths.items():
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise PacketError("a validated source file became unreadable") from exc
            if _sha256_bytes(raw) != hashes[field] or gate_record.get(field) != hashes[field]:
                raise PacketError("source file bytes do not match the validated gate hashes")
            raw_by_field[field] = raw

        try:
            samples_value = json.loads(raw_by_field["samples_sha256"].decode("utf-8"))
            gold_value = json.loads(raw_by_field["gold_sha256"].decode("utf-8"))
            rights = json.loads(raw_by_field["rights_sha256"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PacketError("validated JSON sources could not be decoded") from exc
        samples = _records(samples_value, "samples")
        gold_records = _records(gold_value, "gold")
        samples_by_qid = {}
        for sample in samples:
            qid = sample.get("qid")
            if isinstance(qid, bool) or not isinstance(qid, (str, int)) or str(qid) in samples_by_qid:
                raise PacketError("sample QID set is invalid")
            samples_by_qid[str(qid)] = sample
        gold_by_qid = {}
        for item in gold_records:
            qid = item.get("qid")
            if isinstance(qid, bool) or not isinstance(qid, (str, int)) or str(qid) in gold_by_qid:
                raise PacketError("gold QID set is invalid")
            gold_by_qid[str(qid)] = item
        if set(samples_by_qid) != set(gold_by_qid):
            raise PacketError("sample and gold QID joins do not match")
        if (not isinstance(rights, dict)
                or rights.get("media_sha256") != hashes["media_sha256"]
                or rights.get("audio_used") is not False):
            raise PacketError("rights source no longer matches the validated video-only scope")

        items = []
        for qid, sample in samples_by_qid.items():
            choices = sample.get("choices")
            if not isinstance(choices, list) or len(choices) != 5:
                raise PacketError("validated QA choice list is invalid")
            choice_texts = []
            for choice in choices:
                if isinstance(choice, str):
                    choice_texts.append(choice)
                elif isinstance(choice, dict) and isinstance(choice.get("text"), str):
                    choice_texts.append(choice["text"])
                else:
                    raise PacketError("validated QA choice text is invalid")
            start, end = sample.get("window_start_seconds"), sample.get("window_end_seconds")
            if (isinstance(start, bool) or not isinstance(start, (int, float))
                    or isinstance(end, bool) or not isinstance(end, (int, float))
                    or not (0 <= start < end)):
                raise PacketError("validated evidence window is invalid")
            answer_index = gold_by_qid[qid].get("answer_choice_index")
            if (isinstance(answer_index, bool) or not isinstance(answer_index, int)
                    or not 0 <= answer_index < len(choices)):
                raise PacketError("validated gold choice index is invalid")
            items.append({
                "qid": sample["qid"],
                "question": sample.get("question", ""),
                "choices": choice_texts,
                "window_start_seconds": start,
                "window_end_seconds": end,
                "gold_choice_index": answer_index,
            })

        media_canonical = media_path.resolve()
        if not media_canonical.is_relative_to(data_root.resolve()):
            raise PacketError("canonical video media resolves outside the private dataset root")
        prepared.append({
            "group_id": group_id,
            "hashes": {field: hashes[field] for field in HASH_FIELDS},
            "media_path": media_canonical,
            "items": items,
        })
    return prepared


def _render_packet(prepared, packet_dir):
    review_groups = []
    attestation_groups = []
    html_groups = []
    video_index = 0
    question_index = 0
    for group in prepared:
        relative_video = os.path.relpath(group["review_video_path"], packet_dir)
        if Path(packet_dir, relative_video).resolve() != group["review_video_path"].resolve():
            raise PacketError("relative silent video reference did not resolve to review copy")
        review_items = []
        body_items = []
        for item in group["items"]:
            choices = [{"label": chr(ord("A") + index), "text": text}
                       for index, text in enumerate(item["choices"])]
            review_items.append({
                "qid": item["qid"],
                "question": item["question"],
                "choices": choices,
                "start": item["window_start_seconds"],
                "end": item["window_end_seconds"],
                "question_index": question_index,
            })
            choice_html = "".join(
                f'<label><input type="radio" name="choice-{question_index}" '
                f'value="{index}"> {choice["label"]}. {html.escape(choice["text"])}'
                "</label>"
                for index, choice in enumerate(choices)
            )
            body_items.append(
                f'<article class="question" data-question-index="{question_index}">'
                f'<h4>Review item {question_index + 1} · QID {html.escape(str(item["qid"]))}</h4>'
                f'<p class="question-text">{html.escape(item["question"])}</p>'
                f'<p>Annotated evidence window: {item["window_start_seconds"]:.3f}–'
                f'{item["window_end_seconds"]:.3f} seconds.</p>'
                f'<button type="button" class="seek" data-video="{video_index}" '
                f'data-start="{item["window_start_seconds"]}">Seek video to window start</button>'
                f'<fieldset><legend>Record your first-pass answer</legend>{choice_html}</fieldset>'
                f'<label class="reviewed-label"><input type="checkbox" class="qid-reviewed" '
                f'data-question="{question_index}" disabled> I watched the video and reviewed '
                "this QA item’s question, choices, and evidence window.</label>"
                "</article>"
            )
            question_index += 1
        review_groups.append({
            "group_id": group["group_id"],
            "video_src": relative_video,
            "video_index": video_index,
            "items": review_items,
        })
        attestation_groups.append({
            "group_id": group["group_id"],
            **group["hashes"],
            "reviewed_qids": [],
            "checks": {key: False for key in gate.REQUIRED_CHECKS},
            "rights_scope": RIGHTS_SCOPE,
        })
        checkboxes = "".join(
            f'<label><input type="checkbox" class="group-check" '
            f'data-group="{html.escape(group["group_id"])}" data-check="{key}"> '
            f'{html.escape(CHECK_LABELS[key])}</label>'
            for key in gate.REQUIRED_CHECKS
        )
        rights_text = (
            "<li>Video-only model inference: allowed within this review scope.</li>"
            "<li>Audio use: not allowed and not used.</li>"
            "<li>External media publication: not allowed.</li>"
            "<li>Required credit: NOAA Ocean Exploration.</li>"
        )
        html_groups.append(
            f'<section class="group" data-group="{html.escape(group["group_id"])}">'
            f'<h2>{html.escape(group["group_id"])}</h2>'
            '<p>Review each question independently. Watch the relevant video evidence; '
            'AI blind reviews already in the source audit are not human attestation.</p>'
            f'<video controls preload="metadata" id="video-{video_index}" '
            f'src="{html.escape(relative_video, quote=True)}">Your browser cannot play this local video.</video>'
            f'{"".join(body_items)}'
            f'<fieldset class="required-checks"><legend>Required group attestations '
            f'(all begin unchecked)</legend>{checkboxes}</fieldset>'
            f'<h3>Exact rights scope</h3><ul>{rights_text}</ul>'
            '</section>'
        )
        video_index += 1

    template = {
        "schema_version": 1,
        "signed": False,
        "signed_at_utc": None,
        "attestor": {"name": "", "role": "", "human": False},
        "groups": attestation_groups,
        "first_pass": None,
        "review_media": [group["media_provenance"] for group in prepared],
    }
    review_data = {
        "groups": review_groups,
        "source_sha256": {group["group_id"]: group["hashes"] for group in prepared},
        "review_media": [group["media_provenance"] for group in prepared],
    }
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self' file: blob:; img-src 'self' data:; connect-src 'none'; form-action 'none'; base-uri 'none'">
<title>Private NOAA human QA review</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17202a}}
.notice{{background:#fff4d6;border:1px solid #c59427;padding:1rem}} section{{border-top:2px solid #555;margin-top:2rem;padding-top:1rem}}
video{{display:block;width:min(100%,800px);max-height:70vh;background:#111;margin:1rem 0}} article{{border:1px solid #bbb;padding:1rem;margin:1rem 0}}
article label,.required-checks label{{display:block;margin:.4rem 0}} button{{padding:.45rem .7rem;margin:.3rem 0}} .gold{{font-weight:700;color:#174a24}}
.required-checks{{margin:1rem 0}} #download{{font-size:1rem}} .signer{{display:grid;gap:.5rem;max-width:32rem}} 
</style></head><body>
<h1>Private NOAA human QA review</h1>
<div class="notice"><strong>Local private review only.</strong> This page is self-contained and makes no network requests.
Review all six QA items against their videos and annotated windows. Existing AI reviews are not human review and do not satisfy the gate.
Formal checkboxes record your attestation; they do not prove semantic correctness. Do not publish this page, its videos, or the downloaded JSON.</div>
<p><strong>Procedural blinding only; this is not secure blinding.</strong> Do not open <code>gold_reference.html</code> until you have made and exported your own choices for all six questions. The page cannot verify that you watched or assessed the video.</p>
<p>For each item, watch the relevant silent video and select your independent answer. Check answer support, a unique best choice, and leakage from the question stem.</p>
{"".join(html_groups)}
<section><h2>First-pass record and gold reference</h2>
<p>First export your choices. That private JSON records every selected answer before you open the separate gold reference. Opening the reference is a procedural step, not proof of blinding.</p>
<button id="export-first-pass" type="button" disabled>Export first-pass choices before gold review</button>
<a id="open-gold-reference" href="gold_reference.html" target="_blank" hidden>Open separate gold reference</a>
<p id="first-pass-status" aria-live="polite">No first-pass choices exported.</p>
</section>
<section><h2>Human attestation</h2>
<p>The JSON is downloaded only after every group check and item review, plus your explicit human claim and name/role. It includes the first-pass choices.</p>
<div class="signer"><label>Name <input id="attestor-name" type="text" autocomplete="name" value=""></label>
<label>Role <input id="attestor-role" type="text" value=""></label>
<label><input id="human-claim" type="checkbox"> I am a human reviewer and personally completed the checks above.</label>
<button id="download" type="button" disabled>Download private human attestation JSON</button></div></section>
<script id="review-data" type="application/json">{_safe_json(review_data)}</script>
<script id="attestation-template" type="application/json">{_safe_json(template)}</script>
<script>
const reviewData=JSON.parse(document.getElementById('review-data').textContent);
const template=JSON.parse(document.getElementById('attestation-template').textContent);
const all=(selector)=>Array.from(document.querySelectorAll(selector));
let firstPassState=null;let goldReferenceOpened=false;
function saveJson(payload,filename){{
  const blob=new Blob([JSON.stringify(payload,null,2)+'\\n'],{{type:'application/json'}});const url=URL.createObjectURL(blob);
  const link=document.createElement('a');link.href=url;link.download=filename;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}}
function selectedChoices(){{
  const records=[];for(const group of reviewData.groups){{
    const choices=[];for(const item of group.items){{
      const selected=document.querySelector('input[name="choice-'+item.question_index+'"]:checked');if(!selected)return null;
      choices.push({{qid:item.qid,choice_index:Number(selected.value),choice_label:item.choices[Number(selected.value)].label}});
    }}records.push({{group_id:group.group_id,source_sha256:reviewData.source_sha256[group.group_id],choices}});
  }}return records;
}}
function updateGate(){{
  const selections=selectedChoices();document.getElementById('export-first-pass').disabled=!(selections&&!firstPassState);
  let ready=Boolean(firstPassState&&goldReferenceOpened);
  for(const group of reviewData.groups){{
    const section=document.querySelector('.group[data-group="'+group.group_id+'"]');
    const reviewed=all('.qid-reviewed').filter(x=>group.items.some(i=>i.question_index===Number(x.dataset.question)));
    const groupReviewed=reviewed.length===group.items.length&&reviewed.every(x=>x.checked);
    for(const box of reviewed)box.disabled=!(firstPassState&&goldReferenceOpened);
    const checks=Array.from(section.querySelectorAll('.group-check'));
    for(const check of checks)check.disabled=!groupReviewed;
    if(!groupReviewed)for(const check of checks)check.checked=false;
    if(!groupReviewed||checks.some(x=>!x.checked))ready=false;
  }}
  const identity=document.getElementById('attestor-name').value.trim()&&document.getElementById('attestor-role').value.trim();
  document.getElementById('download').disabled=!(ready&&identity&&document.getElementById('human-claim').checked);
}}
document.addEventListener('change',updateGate);document.addEventListener('input',updateGate);
for(const button of all('.seek'))button.addEventListener('click',()=>{{
  const video=document.getElementById('video-'+button.dataset.video);video.currentTime=Number(button.dataset.start);
}});
for(const radio of all('input[type="radio"][name^="choice-"]'))radio.addEventListener('change',updateGate);
document.getElementById('export-first-pass').addEventListener('click',()=>{{
  const choices=selectedChoices();if(!choices||firstPassState)return;
  firstPassState={{schema_version:1,signed:false,exported_at_utc:new Date().toISOString(),
    attestor:{{name:'',role:'',human:false}},source_sha256:reviewData.source_sha256,
    review_media:reviewData.review_media,groups:choices}};
  for(const radio of all('input[type="radio"][name^="choice-"]'))radio.disabled=true;
  saveJson(firstPassState,'noaa_first_pass_choices_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json');
  document.getElementById('open-gold-reference').hidden=false;
  document.getElementById('first-pass-status').textContent='First-pass choices exported. You may now open the separate gold reference.';
  updateGate();
}});
document.getElementById('open-gold-reference').addEventListener('click',()=>{{goldReferenceOpened=true;updateGate();}});
document.getElementById('download').addEventListener('click',()=>{{
  updateGate();if(document.getElementById('download').disabled)return;
  const payload=JSON.parse(JSON.stringify(template));payload.signed=true;payload.signed_at_utc=new Date().toISOString();
  payload.attestor={{name:document.getElementById('attestor-name').value.trim(),role:document.getElementById('attestor-role').value.trim(),human:true}};
  payload.first_pass=firstPassState;
  for(const group of payload.groups){{
    const source=reviewData.groups.find(x=>x.group_id===group.group_id);group.reviewed_qids=source.items.map(x=>x.qid);
    const section=document.querySelector('.group[data-group="'+group.group_id+'"]');
    group.checks={{}};for(const check of section.querySelectorAll('.group-check'))group.checks[check.dataset.check]=check.checked;
    group.rights_scope={{video_only_model_inference:true,audio_use:false,external_media_publication:false,required_credit:'NOAA Ocean Exploration'}};
  }}
  saveJson(payload,'noaa_human_review_attestation_'+new Date().toISOString().replace(/[:.]/g,'-')+'.json');
}});
updateGate();
</script></body></html>"""
    return page


def _render_gold_reference(prepared):
    reference_groups = []
    for group in prepared:
        reference_groups.append({
            "group_id": group["group_id"],
            "source_sha256": group["hashes"],
            "answers": [{
                "qid": item["qid"],
                "answer_text": item["choices"][item["gold_choice_index"]],
            } for item in group["items"]],
        })
    gold_data = {"groups": reference_groups}
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; form-action 'none'; base-uri 'none'">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Private gold reference</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem}}.warning{{background:#fff4d6;padding:1rem}}section{{border-top:1px solid #777;margin-top:1rem}}</style>
</head><body><h1>Separate private gold reference</h1>
<p class="warning"><strong>Procedural, not secure blinding.</strong> This file contains keyed answers. Open it only after exporting the first-pass choices JSON from index.html. The file cannot prove that the first pass happened first.</p>
<label>Select the first-pass JSON you just exported: <input id="first-pass-file" type="file" accept="application/json"></label>
<p id="status" aria-live="polite">Gold answers are hidden until a matching first-pass JSON is selected.</p><div id="answers" hidden></div>
<script id="gold-data" type="application/json">{_safe_json(gold_data)}</script>
<script>
const goldData=JSON.parse(document.getElementById('gold-data').textContent);
document.getElementById('first-pass-file').addEventListener('change',async(event)=>{{
 const status=document.getElementById('status'),container=document.getElementById('answers');container.replaceChildren();container.hidden=true;
 try{{
  const first=JSON.parse(await event.target.files[0].text());
  if(first.schema_version!==1||first.signed!==false||!first.source_sha256||!Array.isArray(first.groups))throw new Error();
  if(first.groups.length!==goldData.groups.length)throw new Error();
  for(const group of goldData.groups){{
   const supplied=first.groups.find(x=>x.group_id===group.group_id);if(!supplied||JSON.stringify(supplied.source_sha256)!==JSON.stringify(group.source_sha256))throw new Error();
   if(!Array.isArray(supplied.choices)||supplied.choices.length!==group.answers.length)throw new Error();
   const qids=new Set(supplied.choices.map(x=>String(x.qid)));
   if(qids.size!==group.answers.length||group.answers.some(x=>!qids.has(String(x.qid))||!supplied.choices.some(c=>String(c.qid)===String(x.qid)&&Number.isInteger(c.choice_index)&&c.choice_index>=0&&c.choice_index<5)))throw new Error();
  }}
  for(const group of goldData.groups){{
   const section=document.createElement('section');const heading=document.createElement('h2');heading.textContent=group.group_id;section.appendChild(heading);
   for(const item of group.answers){{const p=document.createElement('p');p.textContent='QID '+item.qid+': keyed answer — '+item.answer_text;section.appendChild(p);}}
   container.appendChild(section);
  }}
  container.hidden=false;status.textContent='First-pass record matched. Keyed answers are now shown for human comparison.';
 }}catch(_error){{status.textContent='The selected file did not match the complete first-pass record; answers remain hidden.';}}
}});
</script></body></html>"""
    return page


def _validate_media_provenance(group, output_path, provenance):
    if not isinstance(provenance, dict):
        raise PacketError("silent review copy provenance is missing")
    expected_source = group["hashes"]["media_sha256"]
    try:
        source_start = float(provenance["source_video_start_time"])
        copy_start = float(provenance["copy_video_start_time"])
        source_duration = float(provenance["source_video_duration"])
        copy_duration = float(provenance["copy_video_duration"])
        numeric_times_valid = all(math.isfinite(value) for value in (
            source_start, copy_start, source_duration, copy_duration))
        source_packet = provenance["source_video_packet_sha256"]
        copy_packet = provenance["copy_video_packet_sha256"]
        frames = provenance["source_video_nb_frames"]
        copy_frames = provenance["copy_video_nb_frames"]
        pts_count = provenance["video_pts_count"]
        source_audio = provenance["source_audio_streams_removed"]
        copy_audio = provenance["copy_audio_streams"]
        frame_count_valid = (isinstance(frames, int) and not isinstance(frames, bool)
                             and frames > 0
                             and isinstance(copy_frames, int) and not isinstance(copy_frames, bool)
                             and copy_frames == frames
                             and isinstance(pts_count, int) and not isinstance(pts_count, bool)
                             and pts_count == frames)
        audio_counts_valid = (
            isinstance(source_audio, int) and not isinstance(source_audio, bool)
            and source_audio >= 0
            and isinstance(copy_audio, int) and not isinstance(copy_audio, bool)
            and copy_audio == 0
        )
        hashes_valid = (HEX_HASH.fullmatch(str(source_packet)) is not None
                        and HEX_HASH.fullmatch(str(copy_packet)) is not None
                        and source_packet == copy_packet
                        and provenance["silent_copy_sha256"] == _sha256_file(output_path))
        timing_valid = (numeric_times_valid and source_duration > 0
                        and abs(source_start - copy_start) <= 1e-6
                        and abs(source_duration - copy_duration) <= 1e-6)
    except (KeyError, TypeError, ValueError, OverflowError):
        raise PacketError("silent review copy provenance failed validation") from None
    if (provenance.get("original_media_sha256") != expected_source
            or not hashes_valid or not timing_valid or not frame_count_valid
            or not audio_counts_valid):
        raise PacketError("silent review copy provenance failed validation")
    return {"group_id": group["group_id"], **provenance}


def build_review_packet(data_root=None, packet_name=None, media_preparer=None):
    repo_root = Path(__file__).resolve().parents[1]
    data_root = (repo_root / "datasets" / "noaa_qa_pilot_20261011"
                 if data_root is None else Path(data_root)).resolve()
    private_root = (data_root / "private").resolve()
    if private_root.parent != data_root or not private_root.is_dir():
        raise PacketError("canonical private dataset directory is missing")
    report = gate.audit_dataset(data_root, attestation_supplied=False)
    prepared = _read_verified_sources(data_root, report)

    name = packet_name or (
        "review_packet_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        + "_" + uuid.uuid4().hex[:8]
    )
    if not re.fullmatch(r"review_packet_[A-Za-z0-9_-]+", name):
        raise PacketError("packet directory name is invalid")
    packet_dir = private_root / name
    prepare_media = _prepare_silent_video if media_preparer is None else media_preparer
    packet_dir.mkdir(mode=0o700, exist_ok=False)
    try:
        assets_dir = packet_dir / "assets"
        assets_dir.mkdir(mode=0o700)
        for group in prepared:
            output_path = assets_dir / f"{group['group_id']}_silent.mp4"
            provenance = prepare_media(group["media_path"], output_path)
            if not output_path.is_file():
                raise PacketError("silent review copy was not created")
            output_path.chmod(0o600)
            group["review_video_path"] = output_path.resolve()
            group["media_provenance"] = _validate_media_provenance(
                group, output_path, provenance)
        html_text = _render_packet(prepared, packet_dir)
        gold_text = _render_gold_reference(prepared)
        index_path = packet_dir / "index.html"
        descriptor = os.open(index_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(html_text)
        gold_path = packet_dir / "gold_reference.html"
        descriptor = os.open(gold_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(gold_text)
    except Exception:
        shutil.rmtree(packet_dir, ignore_errors=True)
        raise
    counts = {
        "groups": len(prepared),
        "review_items": sum(len(group["items"]) for group in prepared),
        "attestation_status": report["attestation_status"],
        "silent_copy_audio_streams": sum(
            group["media_provenance"]["copy_audio_streams"] for group in prepared),
    }
    return index_path, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args(argv)
    try:
        index_path, counts = build_review_packet()
    except (PacketError, gate.AuditFailure, OSError) as exc:
        parser.error(f"review packet not created: {exc}")
    repo_root = Path(__file__).resolve().parents[1]
    try:
        display_path = index_path.relative_to(repo_root)
    except ValueError:
        display_path = Path("<private-local-packet>")
    print(f"Private human review packet: groups={counts['groups']}; "
          f"review_items={counts['review_items']}; human_attestation=pending; "
          f"index={display_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
