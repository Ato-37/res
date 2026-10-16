#!/usr/bin/env python3
"""Build an offline, non-authorizing blinded review page for 15 PNG pairs.

The input manifest is custodian-only and the identity map is written to a
separate private directory. The reviewer page contains only randomized pair
IDs, re-encoded images, and a package hash. It never reads video or annotation
data and cannot authorize clock calibration or scoring.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import stat
from typing import Any

from PIL import Image, UnidentifiedImageError

PAIR_COUNT = 15
ANCHOR_IDS = ("fit_begin", "heldout_quarter", "fit_midpoint",
              "heldout_three_quarter", "fit_end")
MAX_MANIFEST_BYTES = 64_000
MAX_IMAGE_BYTES = 20_000_000
MAX_IMAGE_PIXELS = 25_000_000
MAX_IMAGE_SIDE = 10_000
TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{24}$")
ANCHOR_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
MANIFEST_KEYS = {"schema_version", "purpose", "plan_sha256", "records"}
PLAN_KEYS = {"schema_version", "purpose", "status", "execution_authorization", "records"}
RECORD_KEYS = {"review_token", "anchor_id", "screenshot_crop_file",
               "screenshot_crop_sha256", "decoder_frame_file", "decoder_frame_sha256"}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("manifest_duplicate_key")
            result[key] = value
        return result

    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(
                          ValueError("manifest_nonfinite_number")))


def _private_dir(path: Path, code: str) -> None:
    info = path.lstat()
    if (stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700):
        raise ValueError(code)


def _private_file(path: Path, code: str, *, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        raise ValueError(code) from None
    try:
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_size < 1 or info.st_size > max_bytes):
            raise ValueError(code)
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) != info.st_size:
            raise ValueError(code)
        return raw
    finally:
        os.close(fd)


def _image_png(path: Path, expected_sha: str, label: str) -> tuple[bytes, dict]:
    if not isinstance(expected_sha, str) or not SHA_RE.fullmatch(expected_sha):
        raise ValueError(label + "_expected_hash_invalid")
    raw = _private_file(path, label + "_file_invalid", max_bytes=MAX_IMAGE_BYTES)
    if _sha(raw) != expected_sha:
        raise ValueError(label + "_hash_mismatch")
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.format != "PNG" or source.width < 1 or source.height < 1:
                raise ValueError(label + "_image_invalid")
            if (source.width > MAX_IMAGE_SIDE or source.height > MAX_IMAGE_SIDE
                    or source.width * source.height > MAX_IMAGE_PIXELS):
                raise ValueError(label + "_image_dimensions_exceeded")
            source.load()
            clean = source.convert("RGB")
            out = io.BytesIO()
            clean.save(out, format="PNG", optimize=False)
            png = out.getvalue()
            if len(png) > MAX_IMAGE_BYTES:
                raise ValueError(label + "_reencoded_image_too_large")
            return png, {"width": clean.width, "height": clean.height,
                         "sha256": _sha(png)}
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise ValueError(label + "_image_invalid") from None


def _read_plan(path: Path, expected_plan_sha256: str) -> tuple[bytes, set[tuple[str, str]]]:
    _private_dir(path.parent, "plan_parent_not_private")
    raw = _private_file(path, "plan_file_invalid", max_bytes=2_000_000)
    if _sha(raw) != expected_plan_sha256:
        raise ValueError("plan_sha256_mismatch")
    try:
        plan = _strict_json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("plan_json_invalid") from None
    if (not isinstance(plan, dict) or set(plan) != PLAN_KEYS
            or type(plan.get("schema_version")) is not int or plan["schema_version"] != 1
            or plan.get("purpose") != "predeclared_html5_clock_pts_anchor_plan"
            or plan.get("status") != "predeclared_frozen"
            or plan.get("execution_authorization") != "not_granted"
            or not isinstance(plan.get("records"), list) or len(plan["records"]) != 3):
        raise ValueError("plan_schema_or_authorization_invalid")
    expected = set()
    tokens = set()
    for row in plan["records"]:
        if (not isinstance(row, dict) or set(row) != {
                "review_token", "source_media_sha256", "fit_anchors", "heldout_anchors"}
                or not isinstance(row.get("review_token"), str)
                or not TOKEN_RE.fullmatch(row["review_token"])
                or row["review_token"] in tokens
                or not isinstance(row.get("source_media_sha256"), str)
                or not SHA_RE.fullmatch(row["source_media_sha256"])):
            raise ValueError("plan_record_invalid")
        token = row["review_token"]
        tokens.add(token)
        for field, required_ids in (("fit_anchors", {"fit_begin", "fit_midpoint", "fit_end"}),
                                    ("heldout_anchors", {"heldout_quarter", "heldout_three_quarter"})):
            anchors = row[field]
            if not isinstance(anchors, list) or len(anchors) != len(required_ids):
                raise ValueError("plan_anchor_schema_invalid")
            anchor_ids = set()
            for anchor in anchors:
                if (not isinstance(anchor, dict) or set(anchor) != {"anchor_id", "decoder_pts_seconds"}
                        or not isinstance(anchor.get("anchor_id"), str)
                        or not isinstance(anchor.get("decoder_pts_seconds"), str)
                        or not anchor["decoder_pts_seconds"]):
                    raise ValueError("plan_anchor_schema_invalid")
                anchor_ids.add(anchor["anchor_id"])
            if anchor_ids != required_ids:
                raise ValueError("plan_anchor_ids_invalid")
            expected.update((token, anchor_id) for anchor_id in anchor_ids)
    return raw, expected


def _read_manifest(path: Path, expected_plan_sha256: str,
                   expected_assignments: set[tuple[str, str]]) -> tuple[bytes, dict]:
    _private_dir(path.parent, "manifest_parent_not_private")
    raw = _private_file(path, "manifest_file_invalid", max_bytes=MAX_MANIFEST_BYTES)
    try:
        value = _strict_json(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("manifest_json_invalid") from None
    if (not isinstance(value, dict) or set(value) != MANIFEST_KEYS
            or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("purpose") != "private_blinded_frame_pair_review_input"
            or value.get("plan_sha256") != expected_plan_sha256
            or not isinstance(value.get("plan_sha256"), str)
            or not SHA_RE.fullmatch(value["plan_sha256"])
            or not isinstance(value.get("records"), list)
            or len(value["records"]) != PAIR_COUNT):
        raise ValueError("manifest_schema_or_count_invalid")
    seen = set()
    tokens = set()
    anchors_by_token: dict[str, set[str]] = {}
    screenshot_names, decoder_names = set(), set()
    for row in value["records"]:
        if not isinstance(row, dict) or set(row) != RECORD_KEYS:
            raise ValueError("manifest_record_schema_invalid")
        token, anchor = row["review_token"], row["anchor_id"]
        if (not isinstance(token, str) or not TOKEN_RE.fullmatch(token)
                or not isinstance(anchor, str) or not ANCHOR_RE.fullmatch(anchor)
                or (token, anchor) in seen):
            raise ValueError("manifest_pair_identity_invalid_or_duplicate")
        seen.add((token, anchor))
        tokens.add(token)
        anchors_by_token.setdefault(token, set()).add(anchor)
        for name_key, names in (("screenshot_crop_file", screenshot_names),
                                ("decoder_frame_file", decoder_names)):
            name = row[name_key]
            if (not isinstance(name, str) or not name or Path(name).name != name
                    or name in {".", ".."} or name in names):
                raise ValueError("manifest_image_path_invalid")
            names.add(name)
    if (len(tokens) != 3 or any(anchor_set != set(ANCHOR_IDS)
                                for anchor_set in anchors_by_token.values())):
        raise ValueError("manifest_token_anchor_grid_invalid")
    if seen != expected_assignments:
        raise ValueError("manifest_plan_assignment_mismatch")
    return raw, value


def _write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _html(package_sha: str, package_nonce: str, pairs: list[dict]) -> bytes:
    payload = json.dumps(pairs, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    page = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blinded frame-pair review</title>
<style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#18212b}article{border:1px solid #9aa8b4;border-radius:8px;padding:1rem;margin:1rem 0}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}.side{text-align:center}.side img{max-width:100%;max-height:65vh;object-fit:contain;border:1px solid #555}.controls{display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem}label{display:block}button{font:inherit;padding:.7rem 1rem}</style>
<h1>Blinded frame-pair review</h1><p>For each pair, confirm both images contain only the intended video frame crop, then record whether they match. Any uncertainty or extra content must be reported as ambiguous or impure. This review does not authorize calibration or scoring.</p>
<label>Reviewer pseudonym <input id="reviewer" maxlength="80" autocomplete="off"></label><label><input type="checkbox" id="independent"> I am an independent frame-pair reviewer and did not select, capture, or prepare these pairs.</label>
<main id="rows"></main><button id="export" disabled>Export review JSON</button><p id="status">Complete all 15 rows to export.</p>
<script>
const packageSha=__PACKAGE_SHA__, packageNonce=__PACKAGE_NONCE__, pairs=__PAIRS__;
const rows=document.getElementById('rows'), state=new Map();
function make(tag,attrs={},text=''){const x=document.createElement(tag);for(const [k,v] of Object.entries(attrs))x.setAttribute(k,v);if(text)x.textContent=text;return x}
for(const p of pairs){const card=make('article');card.dataset.id=p.id;card.append(make('h2',{},'Pair '+(pairs.indexOf(p)+1)));const grid=make('div',{class:'pair'});for(const side of ['A','B']){const x=make('div',{class:'side'});x.append(make('h3',{},side));const im=make('img',{alt:'Anonymous image '+side,src:p[side]});x.append(im);grid.append(x)}card.append(grid);const controls=make('div',{class:'controls'});const purity=make('select',{'aria-label':'Crop purity'});for(const [v,t] of [['','Choose crop purity'],['pure','Only intended video frame crop'],['impure','Contains other or uncertain content']]){const o=make('option',{value:v},t);purity.append(o)}const decision=make('select',{'aria-label':'Visual comparison'});for(const [v,t] of [['','Choose comparison'],['match','Match'],['ambiguous','Ambiguous'],['mismatch','Mismatch']])decision.append(make('option',{value:v},t));controls.append(purity,decision);card.append(controls);rows.append(card);state.set(p.id,{purity,decision})}
function check(){const ready=pairs.every(p=>{const s=state.get(p.id);return s.purity.value&&s.decision.value})&&document.getElementById('reviewer').value.trim().length>0&&document.getElementById('independent').checked;document.getElementById('export').disabled=!ready;document.getElementById('status').textContent=ready?'All rows recorded.':'Complete all rows and reviewer attestations to export.'}
for(const s of state.values()){s.purity.addEventListener('change',check);s.decision.addEventListener('change',check)}
document.getElementById('reviewer').addEventListener('input',check);document.getElementById('independent').addEventListener('change',check);
document.getElementById('export').addEventListener('click',()=>{if(pairs.some(p=>{const s=state.get(p.id);return !s.purity.value||!s.decision.value})||!document.getElementById('reviewer').value.trim()||!document.getElementById('independent').checked)return;const records=pairs.map(p=>{const s=state.get(p.id);return {pair_id:p.id,crop_purity_attestation:s.purity.value==='pure',decision:s.decision.value}});const out={schema_version:1,purpose:'private_blinded_frame_pair_review_export',status:'not_authorizing',execution_authorization:'not_granted',package_manifest_sha256:packageSha,package_nonce:packageNonce,reviewer_pseudonym:document.getElementById('reviewer').value.trim(),reviewer_role:'independent_frame_pair_reviewer',independence_attestation:'reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring',reviewer_provenance_status:'self_reported_unauthenticated',signature_status:'none',reviewed_at_utc:new Date().toISOString(),records};const blob=new Blob([JSON.stringify(out,null,2)+'\n'],{type:'application/json'}),a=make('a',{href:URL.createObjectURL(blob),download:'blinded_frame_pair_review.json'});a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)});
</script></html>'''
    page = page.replace("__PACKAGE_SHA__", json.dumps(package_sha))
    page = page.replace("__PACKAGE_NONCE__", json.dumps(package_nonce))
    page = page.replace("__PAIRS__", payload)
    return page.encode("utf-8")


def build_review(manifest_path: Path, screenshot_dir: Path, decoder_dir: Path,
                 viewer_dir: Path, identity_dir: Path,
                 plan_path: Path, expected_plan_sha256: str) -> dict:
    if not isinstance(expected_plan_sha256, str) or not SHA_RE.fullmatch(expected_plan_sha256):
        raise ValueError("expected_plan_sha256_invalid")
    _private_dir(screenshot_dir, "screenshot_directory_not_private")
    _private_dir(decoder_dir, "decoder_directory_not_private")
    if viewer_dir.resolve().parent != identity_dir.resolve().parent:
        raise ValueError("output_directories_must_be_siblings")
    parent = viewer_dir.parent
    _private_dir(parent, "output_parent_not_private")
    viewer_dir = parent / viewer_dir.name
    identity_dir = parent / identity_dir.name
    if (viewer_dir == identity_dir or os.path.lexists(viewer_dir)
            or os.path.lexists(identity_dir) or viewer_dir.name in {"", ".", ".."}
            or identity_dir.name in {"", ".", ".."}):
        raise ValueError("output_directory_collision_or_invalid")
    plan_raw, expected_assignments = _read_plan(plan_path, expected_plan_sha256)
    manifest_raw, manifest = _read_manifest(manifest_path, expected_plan_sha256, expected_assignments)

    # Verify and sanitize every input before creating outputs.
    prepared = []
    identity_rows = []
    for row in manifest["records"]:
        left_raw, left_meta = _image_png(
            screenshot_dir / row["screenshot_crop_file"], row["screenshot_crop_sha256"], "screenshot_crop")
        right_raw, right_meta = _image_png(
            decoder_dir / row["decoder_frame_file"], row["decoder_frame_sha256"], "decoder_frame")
        pair_id = secrets.token_urlsafe(18)
        order = secrets.randbelow(2)
        a_raw, b_raw = (left_raw, right_raw) if order == 0 else (right_raw, left_raw)
        prepared.append({"id": pair_id, "A": "data:image/png;base64," + base64.b64encode(a_raw).decode("ascii"),
                         "B": "data:image/png;base64," + base64.b64encode(b_raw).decode("ascii")})
        identity_rows.append({"pair_id": pair_id, "review_token": row["review_token"],
                              "anchor_id": row["anchor_id"], "A_source": "screenshot_crop" if order == 0 else "decoder_frame",
                              "B_source": "decoder_frame" if order == 0 else "screenshot_crop"})

    # Shuffle all reviewer-facing row order without losing the private join.
    order = list(range(PAIR_COUNT))
    secrets.SystemRandom().shuffle(order)
    prepared = [prepared[i] for i in order]
    identity_rows = [identity_rows[i] for i in order]

    package_nonce = secrets.token_hex(32)
    template_html = _html("__PACKAGE_SHA__", "__PACKAGE_NONCE__", prepared)
    package = {"schema_version": 1, "purpose": "private_blinded_frame_pair_review_package",
               "status": "not_authorizing", "execution_authorization": "not_granted",
               "pair_count": PAIR_COUNT, "package_nonce": package_nonce,
               "plan_sha256": _sha(plan_raw),
               "reviewer_html_template_sha256": _sha(template_html),
               "pairs": [{"pair_id": p["id"], "A_sha256": _sha(base64.b64decode(p["A"].split(",", 1)[1])),
                          "B_sha256": _sha(base64.b64decode(p["B"].split(",", 1)[1]))}
                         for p in prepared]}
    package_raw = _canonical(package)
    package_sha = _sha(package_raw)
    reviewer_html = _html(package_sha, package_nonce, prepared)
    identity = {"schema_version": 1, "purpose": "private_blinded_frame_pair_identity_map",
                "status": "custodian_only", "package_manifest_sha256": package_sha,
                "input_manifest_sha256": _sha(manifest_raw), "records": identity_rows}

    viewer_created = identity_created = False
    try:
        viewer_dir.mkdir(mode=0o700)
        viewer_created = True
        os.chmod(viewer_dir, 0o700)
        identity_dir.mkdir(mode=0o700)
        identity_created = True
        os.chmod(identity_dir, 0o700)
        _write_new(viewer_dir / "index.html", reviewer_html)
        _write_new(viewer_dir / "package_manifest.json", package_raw)
        _write_new(identity_dir / "identity_map.json", _canonical(identity))
    except BaseException:
        if viewer_created:
            shutil.rmtree(viewer_dir, ignore_errors=True)
        if identity_created:
            shutil.rmtree(identity_dir, ignore_errors=True)
        raise
    return {"status": "private_blinded_review_bundle_created", "pair_count": PAIR_COUNT,
            "viewer_directory_mode": "0700", "identity_directory_mode": "0700",
            "file_mode": "0600", "package_manifest_sha256": package_sha,
            "execution_authorization": "not_granted"}


def _check_html_binding(html_raw: bytes, package_raw: bytes, package: dict) -> None:
    try:
        html = html_raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("reviewer_html_invalid") from None
    binding = ("const packageSha=" + json.dumps(_sha(package_raw)) +
               ", packageNonce=" + json.dumps(package["package_nonce"]))
    if html.count(binding) != 1:
        raise ValueError("reviewer_html_package_binding_invalid")
    normalized = html.replace(binding,
        'const packageSha="__PACKAGE_SHA__", packageNonce="__PACKAGE_NONCE__"', 1).encode()
    if _sha(normalized) != package["reviewer_html_template_sha256"]:
        raise ValueError("reviewer_html_hash_mismatch")
    start_marker, end_marker = binding + ", pairs=", ";\nconst rows="
    start = html.find(start_marker)
    end = html.find(end_marker, start + len(start_marker)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise ValueError("reviewer_html_pairs_invalid")
    try:
        pairs = json.loads(html[start + len(start_marker):end])
    except json.JSONDecodeError:
        raise ValueError("reviewer_html_pairs_invalid") from None
    expected = {row["pair_id"]: row for row in package["pairs"]}
    if not isinstance(pairs, list) or len(pairs) != PAIR_COUNT:
        raise ValueError("reviewer_html_pairs_invalid")
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, dict) or set(pair) != {"id", "A", "B"}
                or not isinstance(pair.get("id"), str) or pair["id"] not in expected
                or pair["id"] in seen):
            raise ValueError("reviewer_html_pairs_invalid")
        seen.add(pair["id"])
        for side in ("A", "B"):
            prefix = "data:image/png;base64,"
            value = pair[side]
            if not isinstance(value, str) or not value.startswith(prefix):
                raise ValueError("reviewer_html_image_invalid")
            try:
                image_raw = base64.b64decode(value[len(prefix):], validate=True)
            except (ValueError, base64.binascii.Error):
                raise ValueError("reviewer_html_image_invalid") from None
            if _sha(image_raw) != expected[pair["id"]][side + "_sha256"]:
                raise ValueError("reviewer_html_image_binding_mismatch")
    if seen != set(expected):
        raise ValueError("reviewer_html_pairs_invalid")


def audit_export(export_path: Path, package_manifest_path: Path,
                 reviewer_html_path: Path) -> dict:
    _private_dir(export_path.parent, "review_export_parent_not_private")
    _private_dir(package_manifest_path.parent, "package_manifest_parent_not_private")
    export_raw = _private_file(export_path, "review_export_file_invalid", max_bytes=100_000)
    package_raw = _private_file(package_manifest_path, "package_manifest_file_invalid", max_bytes=100_000)
    if reviewer_html_path.parent != package_manifest_path.parent:
        raise ValueError("reviewer_html_directory_mismatch")
    html_raw = _private_file(reviewer_html_path, "reviewer_html_file_invalid", max_bytes=50_000_000)
    try:
        export, package = _strict_json(export_raw), _strict_json(package_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("review_export_json_invalid") from None
    if (not isinstance(package, dict) or set(package) != {
            "schema_version", "purpose", "status", "execution_authorization", "pair_count",
            "package_nonce", "plan_sha256", "reviewer_html_template_sha256", "pairs"}
            or type(package.get("schema_version")) is not int or package["schema_version"] != 1
            or package.get("purpose") != "private_blinded_frame_pair_review_package"
            or type(package.get("pair_count")) is not int or package.get("pair_count") != PAIR_COUNT
            or not isinstance(package.get("package_nonce"), str)
            or not re.fullmatch(r"[a-f0-9]{64}", package["package_nonce"])
            or not isinstance(package.get("plan_sha256"), str) or not SHA_RE.fullmatch(package["plan_sha256"])
            or not isinstance(package.get("reviewer_html_template_sha256"), str)
            or not SHA_RE.fullmatch(package["reviewer_html_template_sha256"])
            or package.get("status") != "not_authorizing"
            or package.get("execution_authorization") != "not_granted"
            or not isinstance(package.get("pairs"), list) or len(package["pairs"]) != PAIR_COUNT):
        raise ValueError("package_manifest_invalid")
    expected_ids = []
    for row in package["pairs"]:
        if (not isinstance(row, dict) or set(row) != {"pair_id", "A_sha256", "B_sha256"}
                or not isinstance(row.get("pair_id"), str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{24}", row["pair_id"])
                or not isinstance(row.get("A_sha256"), str) or not SHA_RE.fullmatch(row["A_sha256"])
                or not isinstance(row.get("B_sha256"), str) or not SHA_RE.fullmatch(row["B_sha256"])):
            raise ValueError("package_manifest_invalid")
        expected_ids.append(row["pair_id"])
    if len(set(expected_ids)) != PAIR_COUNT:
        raise ValueError("package_manifest_invalid")
    _check_html_binding(html_raw, package_raw, package)
    if (not isinstance(export, dict) or set(export) != {"schema_version", "purpose", "status",
            "execution_authorization", "package_manifest_sha256", "package_nonce", "reviewer_pseudonym",
            "reviewer_role", "independence_attestation", "reviewer_provenance_status", "signature_status",
            "reviewed_at_utc", "records"}
            or type(export.get("schema_version")) is not int or export.get("schema_version") != 1
            or export.get("purpose") != "private_blinded_frame_pair_review_export"
            or export.get("status") != "not_authorizing"
            or export.get("execution_authorization") != "not_granted"
            or export.get("package_manifest_sha256") != _sha(package_raw)
            or export.get("package_nonce") != package["package_nonce"]
            or not isinstance(export.get("reviewer_pseudonym"), str)
            or not export["reviewer_pseudonym"].strip() or len(export["reviewer_pseudonym"]) > 80
            or export.get("reviewer_role") != "independent_frame_pair_reviewer"
            or export.get("independence_attestation") !=
                "reviewer_self_reports_independence_from_capture_custody_pair_selection_and_scoring"
            or export.get("reviewer_provenance_status") != "self_reported_unauthenticated"
            or export.get("signature_status") != "none"
            or not isinstance(export.get("reviewed_at_utc"), str)
            or not export["reviewed_at_utc"].endswith("Z")
            or not isinstance(export.get("records"), list)
            or len(export["records"]) != PAIR_COUNT):
        raise ValueError("review_export_schema_or_package_binding_invalid")
    seen = set()
    for row in export["records"]:
        if (not isinstance(row, dict) or set(row) != {"pair_id", "crop_purity_attestation", "decision"}
                or not isinstance(row.get("pair_id"), str)
                or row["pair_id"] not in expected_ids or row["pair_id"] in seen
                or type(row["crop_purity_attestation"]) is not bool
                or not isinstance(row["decision"], str)
                or row["decision"] not in {"match", "ambiguous", "mismatch"}):
            raise ValueError("review_export_record_invalid")
        seen.add(row["pair_id"])
    if seen != set(expected_ids):
        raise ValueError("review_export_incomplete")
    try:
        reviewed_at = datetime.fromisoformat(export["reviewed_at_utc"].replace("Z", "+00:00"))
    except ValueError:
        raise ValueError("review_export_timestamp_invalid") from None
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() != timezone.utc.utcoffset(reviewed_at):
        raise ValueError("review_export_timestamp_invalid")
    pure = all(row["crop_purity_attestation"] for row in export["records"])
    matches = all(row["decision"] == "match" for row in export["records"])
    return {"status": "candidate_review_complete_not_authorizing" if pure and matches
            else "blocked_review_not_all_pure_and_matching", "pair_count": PAIR_COUNT,
            "crop_purity_pass_count": sum(row["crop_purity_attestation"] for row in export["records"]),
            "match_count": sum(row["decision"] == "match" for row in export["records"]),
            "ambiguous_count": sum(row["decision"] == "ambiguous" for row in export["records"]),
            "mismatch_count": sum(row["decision"] == "mismatch" for row in export["records"]),
            "execution_authorization": "not_granted"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--manifest", required=True, type=Path)
    build.add_argument("--plan", required=True, type=Path)
    build.add_argument("--expected-plan-sha256", required=True)
    build.add_argument("--screenshot-dir", required=True, type=Path)
    build.add_argument("--decoder-dir", required=True, type=Path)
    build.add_argument("--viewer-dir", required=True, type=Path)
    build.add_argument("--identity-dir", required=True, type=Path)
    audit = sub.add_parser("audit-export")
    audit.add_argument("--export", required=True, type=Path)
    audit.add_argument("--package-manifest", required=True, type=Path)
    audit.add_argument("--reviewer-html", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = (build_review(args.manifest, args.screenshot_dir, args.decoder_dir,
                               args.viewer_dir, args.identity_dir, args.plan, args.expected_plan_sha256)
                  if args.command == "build" else audit_export(args.export, args.package_manifest,
                                                                 args.reviewer_html))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
