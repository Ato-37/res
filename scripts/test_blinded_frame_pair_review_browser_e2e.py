"""Windows Chrome E2E for the synthetic blinded frame-pair artifact.

This test uses only generated PNGs. It exercises all 15 decisions, the
independence gate, browser download, and the production byte-level export
auditor. It does not open protected review material or grant authorization.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import uuid

from PIL import Image
from build_blinded_frame_pair_review import audit_export_bytes
from build_blinded_frame_pair_review import ANCHOR_IDS, build_review


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

# All dynamic values are base64 encoded before inserting into this PowerShell
# source. Chrome runs headless with network requests blocked and a unique temp
# profile. Only a synthetic export is returned to Python, never browser text.
POWERSHELL_CDP = r'''
$ErrorActionPreference='Stop'
$chromePath=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CHROME_B64__'))
$fileUrl=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__URL_B64__'))
$expectedPackage='__PACKAGE_SHA__'
$runId='__RUN_ID__'
$profile=Join-Path ([IO.Path]::GetTempPath()) ('frame-pair-e2e-'+$runId)
$proc=$null; $page=$null; $browser=$null; $port=0; $script:cdpId=0
function Cdp($ws,$method,$params){
  $script:cdpId++;$id=$script:cdpId
  $json=@{id=$id;method=$method;params=$params}|ConvertTo-Json -Compress -Depth 20
  $bytes=[Text.Encoding]::UTF8.GetBytes($json);$seg=[ArraySegment[byte]]::new($bytes)
  $ws.SendAsync($seg,[Net.WebSockets.WebSocketMessageType]::Text,$true,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
  while($true){$buf=New-Object byte[] 65536;$mem=New-Object IO.MemoryStream;do{$part=[ArraySegment[byte]]::new($buf);$res=$ws.ReceiveAsync($part,[Threading.CancellationToken]::None).GetAwaiter().GetResult();if($res.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close){throw 'CDP_CLOSED'};$mem.Write($buf,0,$res.Count)}while(-not $res.EndOfMessage);$msg=[Text.Encoding]::UTF8.GetString($mem.ToArray())|ConvertFrom-Json;if($msg.id -ne $id){continue};if($msg.error){throw 'CDP_ERROR'};return $msg.result}
}
function Eval($ws,$expr){$r=Cdp $ws 'Runtime.evaluate' @{expression=$expr;returnByValue=$true;awaitPromise=$true};if($r.exceptionDetails){throw 'PAGE_EVAL_ERROR'};return $r.result.value}
function WaitFor($ws,$expr,$code){$end=[DateTime]::UtcNow.AddSeconds(20);while([DateTime]::UtcNow -lt $end){if(Eval $ws $expr){return};Start-Sleep -Milliseconds 100};throw $code}
try{
  if(-not $fileUrl.StartsWith('file://wsl.localhost/')){throw 'FILE_URL_INVALID'}
  $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);$listener.Start();$port=$listener.LocalEndpoint.Port;$listener.Stop()
  $args=@('--headless=new','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check',("--remote-debugging-address=127.0.0.1"),("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
  $proc=Start-Process -FilePath $chromePath -ArgumentList $args -PassThru -WindowStyle Hidden
  $version=$null;for($i=0;$i -lt 150;$i++){try{$version=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/version" -TimeoutSec 1;break}catch{Start-Sleep -Milliseconds 100}}
  if(-not $version){throw 'CHROME_CDP_START_FAILED'}
  $target=$null;for($i=0;$i -lt 40;$i++){try{$target=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/new?about%3Ablank" -Method Put -TimeoutSec 2;break}catch{Start-Sleep -Milliseconds 100}}
  if(-not $target){throw 'PAGE_TARGET_FAILED'}
  $pageUri=[Uri]$target.webSocketDebuggerUrl;if($pageUri.Host -ne '127.0.0.1' -or $pageUri.Port -ne $port){throw 'PAGE_CDP_ENDPOINT_INVALID'}
  $page=New-Object Net.WebSockets.ClientWebSocket;$page.ConnectAsync($pageUri,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
  $null=Cdp $page 'Page.enable' @{};$null=Cdp $page 'Runtime.enable' @{};$null=Cdp $page 'Network.enable' @{}
  $null=Cdp $page 'Network.setBlockedURLs' @{urls=@('http://*/*','https://*/*')}
  $null=Cdp $page 'Page.setDownloadBehavior' @{behavior='allow';downloadPath=$profile}
  $null=Cdp $page 'Page.navigate' @{url=$fileUrl}
  WaitFor $page "document.readyState==='complete' && document.querySelectorAll('article').length===15 && !!document.querySelector('#export')" 'PAGE_LOAD_FAILED'
  WaitFor $page "Array.from(document.querySelectorAll('article img')).length===30 && Array.from(document.querySelectorAll('article img')).every(x=>x.complete&&x.naturalWidth===16&&x.naturalHeight===12)" 'SYNTHETIC_IMAGE_DECODE_FAILED'
  if(-not (Eval $page "document.querySelectorAll('input').length===2 && document.querySelector('#export').disabled")){throw 'INITIAL_GATE_INVALID'}
  $null=Eval $page "document.querySelector('#reviewer').value='synthetic-reviewer';document.querySelector('#reviewer').dispatchEvent(new Event('input',{bubbles:true}));for(const a of Array.from(document.querySelectorAll('article')).slice(0,14)){const s=a.querySelectorAll('select');s[0].value='pure';s[0].dispatchEvent(new Event('change',{bubbles:true}));s[1].value='match';s[1].dispatchEvent(new Event('change',{bubbles:true}));}true"
  if(-not (Eval $page "document.querySelector('#export').disabled")){throw 'FOURTEEN_PAIR_GATE_OPEN'}
  $null=Eval $page "const s=document.querySelectorAll('article')[14].querySelectorAll('select');s[0].value='pure';s[0].dispatchEvent(new Event('change',{bubbles:true}));s[1].value='match';s[1].dispatchEvent(new Event('change',{bubbles:true}));true"
  if(-not (Eval $page "document.querySelector('#export').disabled")){throw 'INDEPENDENCE_GATE_OPEN'}
  $null=Eval $page "document.querySelector('#independent').checked=true;document.querySelector('#independent').dispatchEvent(new Event('change',{bubbles:true}));true"
  WaitFor $page "document.querySelector('#export').disabled===false" 'FIFTEEN_PAIR_GATE_NOT_READY'
  $null=Eval $page "document.querySelector('#export').click();true"
  $download=$null;$deadline=[DateTime]::UtcNow.AddSeconds(20);while([DateTime]::UtcNow -lt $deadline){$download=Get-ChildItem -LiteralPath $profile -Filter 'blinded_frame_pair_review.json' -File -ErrorAction SilentlyContinue|Select-Object -First 1;if($download -and -not (Test-Path ($download.FullName+'.crdownload'))){break};Start-Sleep -Milliseconds 100}
  if(-not $download){throw 'JSON_DOWNLOAD_MISSING'}
  $raw=[IO.File]::ReadAllBytes($download.FullName);$text=[Text.Encoding]::UTF8.GetString($raw);$obj=$text|ConvertFrom-Json
  if($obj.schema_version -ne 1 -or $obj.purpose -ne 'private_blinded_frame_pair_review_export' -or $obj.status -ne 'not_authorizing' -or $obj.execution_authorization -ne 'not_granted' -or $obj.package_manifest_sha256 -cne $expectedPackage -or $obj.reviewer_pseudonym -cne 'synthetic-reviewer' -or $obj.records.Count -ne 15){throw 'JSON_EXPORT_CONTRACT_INVALID'}
  if(@($obj.records|Where-Object {$_.decision -ne 'match' -or $_.crop_purity_attestation -ne $true}).Count -ne 0){throw 'JSON_JUDGMENT_ROUNDTRIP_INVALID'}
  Write-Output ('E2E_JSON_B64:'+[Convert]::ToBase64String($raw))
}catch{Write-Output 'E2E_FAIL';exit 1}finally{
  try{if($page -and $page.State -eq [Net.WebSockets.WebSocketState]::Open){$page.Dispose()}}catch{}
  try{if($version -and $version.webSocketDebuggerUrl){$browserUri=[Uri]$version.webSocketDebuggerUrl;$browser=New-Object Net.WebSockets.ClientWebSocket;$browser.ConnectAsync($browserUri,[Threading.CancellationToken]::None).GetAwaiter().GetResult();$script:cdpId=100000;$null=Cdp $browser 'Browser.close' @{}}}catch{}
  try{if($proc){$proc.Refresh();if(-not $proc.HasExited){& "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $proc.Id|Out-Null;$proc.WaitForExit(5000)|Out-Null}}}catch{}
  for($i=0;$i -lt 20 -and (Test-Path -LiteralPath $profile);$i++){Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue;Start-Sleep -Milliseconds 100}
}
if(Test-Path -LiteralPath $profile){Write-Output 'E2E_CLEANUP_FAIL';exit 1}
if($proc){$proc.Refresh();if(-not $proc.HasExited){Write-Output 'E2E_CLEANUP_FAIL';exit 1}}
Write-Output 'E2E_CLEANUP_PASS'
'''


class BlindedFramePairBrowserE2ETests(unittest.TestCase):
    def test_chrome_review_and_download_round_trip_with_synthetic_pairs(self):
        chrome = Path(os.environ.get("FORMAL_REVIEW_CHROME", str(DEFAULT_CHROME)))
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        wslpath = shutil.which("wslpath")
        if not chrome.is_file() or not str(chrome).lower().endswith(".exe"):
            self.skipTest("Windows Chrome through WSL is unavailable")
        if not powershell or not wslpath:
            self.skipTest("Windows PowerShell or wslpath unavailable")

        temp = tempfile.TemporaryDirectory(prefix="frame_pair_e2e_")
        root = Path(temp.name)
        os.chmod(root, 0o700)
        try:
            inputs = root / "inputs"
            crops, frames = inputs / "crop", inputs / "frame"
            crops.mkdir(parents=True, mode=0o700)
            frames.mkdir(mode=0o700)
            os.chmod(inputs, 0o700)
            plan_records = []
            for index in range(3):
                token = f"tok_{index:020d}"
                plan_records.append({
                    "review_token": token, "source_media_sha256": "f" * 64,
                    "fit_anchors": [{"anchor_id": name, "decoder_pts_seconds": str(j)}
                                    for j, name in enumerate(("fit_begin", "fit_midpoint", "fit_end"))],
                    "heldout_anchors": [{"anchor_id": name, "decoder_pts_seconds": str(j + 3)}
                                        for j, name in enumerate(("heldout_quarter", "heldout_three_quarter"))],
                })
            plan_path = inputs / "plan.json"
            plan_raw = json.dumps({
                "schema_version": 1,
                "purpose": "predeclared_html5_clock_pts_anchor_plan",
                "status": "predeclared_frozen", "execution_authorization": "not_granted",
                "records": plan_records,
            }, sort_keys=True, separators=(",", ":")).encode()
            plan_path.write_bytes(plan_raw); os.chmod(plan_path, 0o600)
            plan_sha = hashlib.sha256(plan_raw).hexdigest()
            manifest_rows = []
            for index in range(15):
                token = f"tok_{index // 5:020d}"
                anchor = ANCHOR_IDS[index % 5]
                image = Image.new("RGB", (16, 12), (index * 10, 30, 90))
                buf = io.BytesIO(); image.save(buf, format="PNG")
                image_raw = buf.getvalue()
                crop_name, frame_name = f"crop_{index}.png", f"frame_{index}.png"
                crop_path, frame_path = crops / crop_name, frames / frame_name
                crop_path.write_bytes(image_raw); frame_path.write_bytes(image_raw)
                os.chmod(crop_path, 0o600); os.chmod(frame_path, 0o600)
                manifest_rows.append({
                    "review_token": token, "anchor_id": anchor,
                    "screenshot_crop_file": crop_name,
                    "screenshot_crop_sha256": hashlib.sha256(image_raw).hexdigest(),
                    "decoder_frame_file": frame_name,
                    "decoder_frame_sha256": hashlib.sha256(image_raw).hexdigest(),
                })
            manifest_path = inputs / "pairs.json"
            manifest_raw = json.dumps({
                "schema_version": 1, "purpose": "private_blinded_frame_pair_review_input",
                "plan_sha256": plan_sha, "records": manifest_rows,
            }, sort_keys=True, separators=(",", ":")).encode()
            manifest_path.write_bytes(manifest_raw); os.chmod(manifest_path, 0o600)
            build_review(manifest_path, crops, frames, root / "viewer", root / "identity",
                         plan_path, plan_sha)
            viewer = root / "viewer"
            html_path = viewer / "index.html"
            package_raw = (viewer / "package_manifest.json").read_bytes()
            package_hash = hashlib.sha256(package_raw).hexdigest()
            html_path_win = subprocess.run([wslpath, "-w", str(html_path)],
                                           capture_output=True, text=True,
                                           timeout=5, check=True).stdout.strip()
            chrome_win = subprocess.run([wslpath, "-w", str(chrome)],
                                        capture_output=True, text=True,
                                        timeout=5, check=True).stdout.strip()
            if not html_path_win.startswith("\\\\wsl.localhost\\"):
                self.skipTest("viewer did not map to WSL UNC")
            if not len(chrome_win) >= 3 or chrome_win[1:3] != ":\\":
                self.skipTest("Chrome path did not map to a Windows path")
            file_url = "file:" + html_path_win.replace("\\", "/")
            run_id = uuid.uuid4().hex
            source = (POWERSHELL_CDP.replace("__CHROME_B64__", base64.b64encode(
                        chrome_win.encode()).decode())
                      .replace("__URL_B64__", base64.b64encode(
                        file_url.encode()).decode())
                      .replace("__PACKAGE_SHA__", package_hash)
                      .replace("__RUN_ID__", run_id))
            encoded = base64.b64encode(source.encode("utf-16le")).decode("ascii")
            run = subprocess.run([powershell, "-NoLogo", "-NoProfile",
                                  "-NonInteractive", "-EncodedCommand", encoded],
                                 capture_output=True, text=True, encoding="utf-8",
                                 errors="replace", timeout=90)
            if (run.returncode != 0 or "E2E_JSON_B64:" not in run.stdout
                    or "E2E_CLEANUP_PASS" not in run.stdout):
                markers = [line.strip() for line in (run.stdout + run.stderr).splitlines()
                           if line.strip() and len(line.strip()) < 120]
                self.fail("Chrome E2E failed; short diagnostic: " +
                          "; ".join(markers[-3:]))
            encoded_json = next(line.split(":", 1)[1] for line in run.stdout.splitlines()
                                if line.startswith("E2E_JSON_B64:"))
            export_raw = base64.b64decode(encoded_json, validate=True)
            report = audit_export_bytes(
                export_raw=export_raw, package_raw=package_raw,
                reviewer_html_raw=html_path.read_bytes())
            self.assertEqual(report["status"], "candidate_review_complete_not_authorizing")
            self.assertEqual(report["match_count"], 15)
            self.assertEqual(report["execution_authorization"], "not_granted")
        finally:
            temp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
