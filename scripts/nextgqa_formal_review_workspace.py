#!/usr/bin/env python3
"""Local-only, non-authorizing Stage-1 / Stage-2 formal review audit workspace.

The browser is a display surface. All protected bundle reads and cryptographic
checks are performed by the existing Python validators on this machine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
import webbrowser

from audit_qta_stage1_bundle_cli import _load_bundle_dir
from qta_stage1_bundle_audit import audit_qta_stage1_bundle
from nextgqa_prediction_link_attestation import verify_attestation
from nextgqa_prediction_link_review import validate_prediction_link_receipt
from nextgqa_pre_review_integrity_receipt import _open_private_directory
from qta_phase2_plan_manifest import validate_baseline_pq_manifest


MAX_STAGE2_FILE_BYTES = 8 * 1024 * 1024
STAGE2_MANIFEST_KEYS = {"schema_version", "purpose", "baseline_manifest",
                        "prediction_link_receipt", "stage2_attestation", "context"}
STAGE2_ROLES = {"baseline_manifest", "prediction_link_receipt",
                "stage2_attestation", "context"}
DENIAL = {"model_load_permitted": False, "score_ready": False,
          "execution_authorization": "not_granted", "authorized_forwards": 0}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, code: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _value: (_ for _ in ()).throw(
                              ValueError("nonfinite_json")))
    except (UnicodeError, json.JSONDecodeError, ValueError, RecursionError):
        raise ValueError(code) from None


def _report_ok(report: Any) -> bool:
    return (isinstance(report, dict)
            and report.get("status") == "stage1_structurally_valid_non_authorizing"
            and type(report.get("requested_records")) is int
            and report.get("requested_records") == 3
            and type(report.get("effective_records")) is int
            and report.get("effective_records") == 3
            and type(report.get("structurally_valid_records")) is int
            and report.get("structurally_valid_records") == 3
            and type(report.get("failure_count")) is int
            and report.get("failure_count") == 0
            and report.get("failure_codes") == []
            and report.get("model_load_permitted") is False
            and report.get("score_ready") is False
            and report.get("reviewer_identity_verified") is False
            and report.get("semantic_review_authenticated") is False
            and report.get("frame_correspondence_verified") is False
            and report.get("execution_authorization") == "not_granted")


def _tree_signature(root: Path, *, max_entries: int = 10_000) -> tuple:
    """Cheap change detector; authoritative reads still use secure bundle loaders."""
    root = Path(os.path.abspath(root))
    entries = []
    pending = [(root, "")]
    while pending:
        directory, rel = pending.pop()
        before = directory.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ValueError("source_directory_changed_or_unsafe")
        entries.append((rel, before.st_dev, before.st_ino, before.st_mode,
                        before.st_uid, before.st_mtime_ns, before.st_ctime_ns))
        with os.scandir(directory) as scan:
            children = sorted(scan, key=lambda item: item.name)
        for child in children:
            child_rel = f"{rel}/{child.name}" if rel else child.name
            info = child.stat(follow_symlinks=False)
            entries.append((child_rel, info.st_dev, info.st_ino, info.st_mode,
                            info.st_uid, info.st_size, info.st_mtime_ns,
                            info.st_ctime_ns))
            if stat.S_ISDIR(info.st_mode):
                pending.append((Path(child.path), child_rel))
            if len(entries) > max_entries:
                raise ValueError("source_tree_entry_limit_exceeded")
    return tuple(sorted(entries))


def _private_child(root: Path, name: Any, *, limit: int) -> bytes:
    if (not isinstance(name, str) or not name or name in {".", ".."}
            or "/" in name or "\\" in name or "\x00" in name):
        raise ValueError("stage2_filename_invalid")
    root_fd = _open_private_directory(root)
    try:
        try:
            fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), dir_fd=root_fd)
        except OSError:
            raise ValueError("stage2_file_open_failed") from None
        try:
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or stat.S_IMODE(before.st_mode) != 0o600
                    or before.st_size < 1 or before.st_size > limit):
                raise ValueError("stage2_file_permissions_or_size_invalid")
            chunks, size = [], 0
            while True:
                chunk = os.read(fd, min(1024 * 1024, limit + 1 - size))
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
                if size > limit:
                    raise ValueError("stage2_file_too_large")
            after = os.fstat(fd)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                 before.st_ctime_ns) !=
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                     after.st_ctime_ns) or size != before.st_size):
                raise ValueError("stage2_file_changed_during_read")
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(root_fd)


class FormalReviewSession:
    """Stateful controller with fail-closed Stage-1 and Stage-2 transitions."""

    def __init__(self, *, stage1_bundle_dir: Path, stage2_bundle_dir: Path,
                 expected_lineage_sha256: str, expected_custody_sha256: str,
                 expected_root_public_der_sha256: str, expected_ceremony_id: str,
                 expected_stage2_context_sha256: str):
        self.stage1_dir = Path(stage1_bundle_dir)
        self.stage2_dir = Path(stage2_bundle_dir)
        self.expected_stage2_context_sha256 = expected_stage2_context_sha256
        self.stage1_pins = {
            "expected_lineage_sha256": expected_lineage_sha256,
            "expected_custody_sha256": expected_custody_sha256,
            "expected_root_public_der_sha256": expected_root_public_der_sha256,
            "expected_ceremony_id": expected_ceremony_id,
        }
        self.stage1_report = None
        self.stage1_receipt_sha256 = None
        self.stage1_sources = None
        self.stage1_artifacts = None
        self.stage1_signature = None
        self.stage2_report = None
        self.stage2_signature = None
        self.stage2_input_hashes = None
        self.failure = None
        self.lock = threading.RLock()

    def _clear_stage2(self):
        self.stage2_report = None
        self.stage2_signature = None
        self.stage2_input_hashes = None

    def _clear_all(self, failure: str | None = None):
        self.stage1_report = None
        self.stage1_receipt_sha256 = None
        self.stage1_sources = None
        self.stage1_artifacts = None
        self.stage1_signature = None
        self._clear_stage2()
        self.failure = failure

    def reset(self):
        with self.lock:
            self._clear_all()

    def _check_signatures(self):
        try:
            s1 = _tree_signature(self.stage1_dir)
            s2 = _tree_signature(self.stage2_dir)
        except (OSError, ValueError):
            self._clear_all("source_tree_unsafe_or_unavailable")
            return
        if self.stage1_signature is not None and s1 != self.stage1_signature:
            self._clear_all("stage1_source_changed_reload_required")
            return
        if self.stage2_signature is not None and s2 != self.stage2_signature:
            self._clear_stage2()
            self.stage2_signature = None
            self.failure = "stage2_source_changed_reload_required"

    def verify_stage1(self) -> dict[str, Any]:
        with self.lock:
            self._clear_all()
            try:
                before = (_tree_signature(self.stage1_dir), _tree_signature(self.stage2_dir))
                manifest, sources, media_dir, artifacts = _load_bundle_dir(self.stage1_dir)
                report = audit_qta_stage1_bundle(
                    worksheet_raw=sources["worksheet"], lineage_raw=sources["lineage"],
                    review_raw=sources["review"], review_index_raw=sources["review_index"],
                    custody_raw=sources["custody"],
                    pre_review_receipt_raw=sources["pre_review_receipt"],
                    integrity_roster_raw=sources["integrity_roster"],
                    release_roster_raw=sources["release_roster"],
                    inventory_raw=sources["inventory"],
                    release_witness_raw=sources["release_witness"],
                    reviewer_attestation_raw=sources["reviewer_attestation"],
                    media_dir=media_dir,
                    expected_lineage_sha256=self.stage1_pins["expected_lineage_sha256"],
                    expected_custody_sha256=self.stage1_pins["expected_custody_sha256"],
                    root_public_key_pem=sources["root_public_key"],
                    expected_root_public_der_sha256=self.stage1_pins[
                        "expected_root_public_der_sha256"],
                    expected_ceremony_id=self.stage1_pins["expected_ceremony_id"],
                    artifact_sources=artifacts)
                after = (_tree_signature(self.stage1_dir), _tree_signature(self.stage2_dir))
                if before != after:
                    raise ValueError("source_tree_changed_during_audit")
                if not _report_ok(report):
                    raise ValueError("stage1_audit_not_valid")
                receipt_sha = _sha(sources["pre_review_receipt"])
                self.stage1_report = {
                    "status": report["status"],
                    "requested_records": report["requested_records"],
                    "structurally_valid_records": report["structurally_valid_records"],
                    "pre_review_receipt_sha256": receipt_sha,
                    "components": report["components"],
                    "model_load_permitted": False,
                    "reviewer_identity_verified": False,
                    "semantic_review_authenticated": False,
                    "frame_correspondence_verified": False,
                    **DENIAL,
                }
                self.stage1_receipt_sha256 = receipt_sha
                self.stage1_sources = sources
                self.stage1_artifacts = artifacts
                self.stage1_signature, self.stage2_signature = after
                self.failure = None
            except Exception as exc:
                code = str(exc)
                if not (code and code.isascii() and code.replace("_", "").isalnum()):
                    code = "stage1_audit_failed"
                self._clear_all(code)
            return self.state()

    def verify_stage2(self) -> dict[str, Any]:
        with self.lock:
            self._check_signatures()
            if self.stage1_report is None:
                self._clear_stage2()
                self.failure = self.failure or "stage1_must_pass_first"
                return self.state()
            self._clear_stage2()
            try:
                before = _tree_signature(self.stage2_dir)
                manifest_raw = _private_child(self.stage2_dir, "workspace.json", limit=64_000)
                manifest = _strict_json(manifest_raw, "stage2_manifest_invalid")
                if (not isinstance(manifest, dict) or set(manifest) != STAGE2_MANIFEST_KEYS
                        or type(manifest.get("schema_version")) is not int
                        or manifest["schema_version"] != 1
                        or manifest.get("purpose") != "qta_formal_workspace_stage2_bundle_v1"):
                    raise ValueError("stage2_manifest_schema_invalid")
                names = {role: manifest[role] for role in STAGE2_ROLES}
                if any(not isinstance(name, str) or Path(name).name != name for name in names.values()):
                    raise ValueError("stage2_manifest_filename_invalid")
                baseline_raw = _private_child(self.stage2_dir, names["baseline_manifest"],
                                              limit=MAX_STAGE2_FILE_BYTES)
                receipt_raw = _private_child(self.stage2_dir, names["prediction_link_receipt"],
                                             limit=2 * 1024 * 1024)
                attestation_raw = _private_child(self.stage2_dir, names["stage2_attestation"],
                                                 limit=2 * 1024 * 1024)
                context_raw = _private_child(self.stage2_dir, names["context"],
                                             limit=1_000_000)
                if _sha(context_raw) != self.expected_stage2_context_sha256:
                    raise ValueError("stage2_context_external_pin_mismatch")
                context = _strict_json(context_raw, "stage2_context_invalid")
                context_keys = {"schema_version", "purpose", "protocol_sha256",
                                "baseline_manifest_sha256", "cohort_tokens",
                                "source_field_sha256_by_token"}
                if (not isinstance(context, dict) or set(context) != context_keys
                        or type(context.get("schema_version")) is not int
                        or context["schema_version"] != 1
                        or context.get("purpose") != "qta_formal_workspace_expected_stage2_pins_v1"):
                    raise ValueError("stage2_context_schema_invalid")
                baseline_sha = _sha(baseline_raw)
                if context["baseline_manifest_sha256"] != baseline_sha:
                    raise ValueError("stage2_baseline_pin_mismatch")
                rows = validate_baseline_pq_manifest(baseline_raw)
                video_receipt_sha = self.stage1_receipt_sha256
                stage1_receipt_raw = self.stage1_sources["pre_review_receipt"]
                if _sha(stage1_receipt_raw) != video_receipt_sha:
                    raise ValueError("stage1_receipt_snapshot_mismatch")
                receipt_obj = _strict_json(receipt_raw, "stage2_receipt_invalid")
                link_report = validate_prediction_link_receipt(
                    receipt=receipt_obj, baseline_manifest_raw=baseline_raw,
                    expected_baseline_manifest_sha256=baseline_sha,
                    expected_protocol_sha256=context["protocol_sha256"],
                    expected_video_review_receipt_sha256=video_receipt_sha,
                    expected_cohort_tokens=context["cohort_tokens"],
                    expected_source_field_sha256_by_token=
                        context["source_field_sha256_by_token"])
                attestation_report = verify_attestation(
                    attestation_raw=attestation_raw, receipt_raw=receipt_raw,
                    baseline_manifest_raw=baseline_raw,
                    expected_baseline_manifest_sha256=baseline_sha,
                    expected_protocol_sha256=context["protocol_sha256"],
                    expected_video_review_receipt_sha256=video_receipt_sha,
                    expected_cohort_tokens=context["cohort_tokens"],
                    expected_source_field_sha256_by_token=
                        context["source_field_sha256_by_token"],
                    integrity_roster_raw=self.stage1_sources["integrity_roster"],
                    release_roster_raw=self.stage1_sources["release_roster"],
                    root_public_key_pem=self.stage1_sources["root_public_key"],
                    expected_root_public_der_sha256=self.stage1_pins[
                        "expected_root_public_der_sha256"])
                after = _tree_signature(self.stage2_dir)
                if before != after:
                    raise ValueError("stage2_source_changed_during_audit")
                if (link_report.get("status") != "prediction_link_review_structurally_valid_unvalidated"
                        or attestation_report.get("status") != "reviewer_key_attested_unvalidated"
                        or attestation_report.get("execution_authorization") != "not_granted"
                        or attestation_report.get("human_identity_verified") is not False
                        or attestation_report.get("semantic_judgment_authenticated") is not False):
                    raise ValueError("stage2_audit_contract_invalid")
                self.stage2_report = {
                    "status": "stage2_receipt_and_attestation_verified_non_authorizing",
                    "baseline_manifest_sha256": baseline_sha,
                    "stage1_pre_review_receipt_sha256": video_receipt_sha,
                    # Surface the digest checked against the external pin so
                    # operators can audit this trust input from the report.
                    "stage2_context_sha256": _sha(context_raw),
                    "stage2_receipt_sha256": _sha(receipt_raw),
                    "stage2_attestation_sha256": _sha(attestation_raw),
                    "reviewer_key_signature_valid": True,
                    "all_rows_self_report_support_p_over_q":
                        attestation_report["all_rows_self_report_support_p_over_q"],
                    "reviewer_identity_verified": False,
                    "semantic_judgment_authenticated": False,
                    "validated_records": len(rows),
                    **DENIAL,
                }
                self.stage2_input_hashes = {
                    "baseline": baseline_sha,
                    "receipt": _sha(receipt_raw),
                    "attestation": _sha(attestation_raw),
                    "context": _sha(context_raw),
                }
                self.stage2_signature = after
                self.failure = None
            except Exception as exc:
                code = str(exc)
                if not (code and code.isascii() and code.replace("_", "").isalnum()):
                    code = "stage2_audit_failed"
                self._clear_stage2()
                self.failure = code
            return self.state()

    def state(self) -> dict[str, Any]:
        with self.lock:
            self._check_signatures()
            return {
                "stage1": self.stage1_report,
                "stage2": self.stage2_report,
                "failure_code": self.failure,
                "stage1_verified": self.stage1_report is not None,
                "stage2_verified": self.stage2_report is not None,
                "identity_or_semantics_authenticated": False,
                **DENIAL,
            }


PAGE = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; img-src 'none'; media-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
<title>正式レビュー監査ワークスペース</title><style>
body{font:16px/1.55 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#183047;background:#f4f7f9}section{background:white;padding:1.2rem;margin:1rem 0;border:1px solid #ccd8de;border-radius:12px}button{font:inherit;padding:.65rem 1rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f4f6;padding:1rem}.warn{background:#fff0c9;padding:1rem}.status{font-weight:700}
</style></head><body><h1>正式レビューの構造監査</h1>
<p class="warn">この画面はローカルの署名・byte検証結果を表示します。本人性、視聴、意味判断、時計/decoder対応を証明せず、モデル読込・採点を許可しません。ニックネーム簡易チェックとは別の作業画面です。</p>
<section><h2>Stage 1 — レビューbundle</h2><p>ローカルPython検証器がprivate bundleと動画byteを検査します。</p><button id="s1">Stage 1を検証</button><pre id="s1out">未検証</pre></section>
<section><h2>Stage 2 — baseline v3とprediction-link receipt</h2><p>Stage 1のpre-review receipt digest、baseline v3 bytes、receipt、rostered attestationをPython検証器で照合します。</p><button id="s2" disabled>Stage 2を検証</button><pre id="s2out">Stage 1検証後に利用できます</pre></section>
<section><h2>実行状態</h2><p id="deny" class="status"></p><p id="failure"></p><button id="reset">状態を消去</button></section>
<script>'use strict';
const token=__TOKEN__,base='/'+token;const s1=document.getElementById('s1out'),s2=document.getElementById('s2out'),deny=document.getElementById('deny'),failure=document.getElementById('failure');
async function call(path,method='GET'){const r=await fetch(base+path,{method,headers:{'X-Workspace-Token':token},cache:'no-store'});return await r.json()}
function show(x){const a=x.stage1,b=x.stage2;s1.textContent=a?JSON.stringify(a,null,2):'未検証';s2.textContent=b?JSON.stringify(b,null,2):(x.stage1_verified?'未検証':'Stage 1検証後に利用できます');document.getElementById('s2').disabled=!x.stage1_verified;failure.textContent=x.failure_code||'';deny.textContent='本人性・意味判断は未認証 / model_load_permitted=false / execution_authorization=not_granted / authorized_forwards=0'}
async function refresh(){try{show(await call('/api/state'))}catch(_){show({stage1:null,stage2:null,stage1_verified:false,failure_code:'接続が切れました'})}}
document.getElementById('s1').onclick=async()=>{show(await call('/api/verify-stage1','POST'))};document.getElementById('s2').onclick=async()=>{show(await call('/api/verify-stage2','POST'))};document.getElementById('reset').onclick=async()=>{show(await call('/api/reset','POST'))};setInterval(refresh,700);refresh();
</script></body></html>'''


def workspace_page(token: str) -> bytes:
    return PAGE.replace("__TOKEN__", json.dumps(token)).encode()


def local_host_allowed(host: str, port: int) -> bool:
    return host in {f"127.0.0.1:{port}", f"localhost:{port}"}


def local_post_allowed(*, host: str, origin: str | None, supplied_token: str,
                       token: str, port: int) -> bool:
    return (local_host_allowed(host, port)
            and origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}
            and secrets.compare_digest(supplied_token, token))


def workspace_url(token: str, port: int) -> str:
    """Return the only supported browser URL: IPv4 loopback, token-scoped."""
    if (not isinstance(token, str)
            or re.fullmatch(r"[A-Za-z0-9_-]{43}", token) is None
            or type(port) is not int or not 1 <= port <= 65535):
        raise ValueError("invalid_workspace_address")
    return f"http://127.0.0.1:{port}/{token}/"


class WorkspaceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False
    address_family = socket.AF_INET

    def __init__(self, session: FormalReviewSession):
        self.session = session
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", 0), WorkspaceHandler)


class WorkspaceHandler(BaseHTTPRequestHandler):
    server: WorkspaceServer

    def log_message(self, _format, *_args):
        return

    def _json(self, payload: dict[str, Any], status: int = 200):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        self.end_headers()
        self.wfile.write(raw)

    def _local_host(self) -> bool:
        host = self.headers.get("Host", "")
        return local_host_allowed(host, self.server.server_port)

    def _authorized(self) -> bool:
        return (self._local_host() and secrets.compare_digest(
            self.headers.get("X-Workspace-Token", ""), self.server.token))

    def do_GET(self):
        if not self._local_host():
            self._json({"error": "local_host_required"}, 403)
            return
        expected = f"/{self.server.token}"
        if self.path == expected + "/":
            raw = workspace_page(self.server.token)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; "
                             "script-src 'unsafe-inline'; connect-src 'self'; "
                             "img-src 'none'; media-src 'none'; object-src 'none'; "
                             "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
            self.end_headers()
            self.wfile.write(raw)
            return
        if self.path == expected + "/api/state" and self._authorized():
            self._json(self.server.session.state())
            return
        self._json({"error": "not_found"}, 404)

    def do_POST(self):
        if not local_post_allowed(
                host=self.headers.get("Host", ""),
                origin=self.headers.get("Origin"),
                supplied_token=self.headers.get("X-Workspace-Token", ""),
                token=self.server.token, port=self.server.server_port):
            self._json({"error": "local_origin_and_token_required"}, 403)
            return
        if self.headers.get("Content-Length", "0") not in {"0", ""}:
            self._json({"error": "request_body_not_supported"}, 413)
            return
        expected_path = f"/{self.server.token}/api/"
        route = self.path.removeprefix(expected_path)
        if route == "verify-stage1":
            self._json(self.server.session.verify_stage1())
        elif route == "verify-stage2":
            self._json(self.server.session.verify_stage2())
        elif route == "reset":
            self.server.session.reset()
            self._json(self.server.session.state())
        else:
            self._json({"error": "not_found"}, 404)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-bundle-dir", required=True, type=Path)
    parser.add_argument("--stage2-bundle-dir", required=True, type=Path)
    parser.add_argument("--expected-lineage-sha256", required=True)
    parser.add_argument("--expected-custody-sha256", required=True)
    parser.add_argument("--expected-root-public-der-sha256", required=True)
    parser.add_argument("--expected-ceremony-id", required=True)
    parser.add_argument("--expected-stage2-context-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = FormalReviewSession(
        stage1_bundle_dir=args.stage1_bundle_dir,
        stage2_bundle_dir=args.stage2_bundle_dir,
        expected_lineage_sha256=args.expected_lineage_sha256,
        expected_custody_sha256=args.expected_custody_sha256,
        expected_root_public_der_sha256=args.expected_root_public_der_sha256,
        expected_ceremony_id=args.expected_ceremony_id,
        expected_stage2_context_sha256=args.expected_stage2_context_sha256)
    server = WorkspaceServer(session)
    url = workspace_url(server.token, server.server_port)
    print("Formal review workspace bound to 127.0.0.1 only; protected data stay local.")
    print(f"Open this local URL: {url}", flush=True)
    print("Ctrl-C stops the server. Reports remain non-authorizing.")
    try:
        webbrowser.open(url)
    except Exception as exc:
        print(f"Browser opener unavailable ({type(exc).__name__}); use the URL above.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
