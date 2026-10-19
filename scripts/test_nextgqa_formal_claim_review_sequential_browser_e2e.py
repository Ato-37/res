"""Synthetic Windows-Chrome E2E for the formal sequential claim-review page.

Builds a fresh worksheet, three generated MP4s, custody manifest and offline
HTML in a temporary directory. It never opens protected media or responses.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from nextgqa_combined_review_envelope import parse_envelope
from render_nextgqa_claim_review_html import render
from test_build_nextgqa_claim_mapping_packet import encode

DEFAULT_CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

POWERSHELL_CDP = r'''
$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'
$chromePath=__CHROME__; $fileUrl=__URL__
$profile=Join-Path ([IO.Path]::GetTempPath()) ('formal-claim-e2e-'+[Guid]::NewGuid().ToString('N'))
$proc=$null; $ws=$null; $browserWs=$null; $script:cdpId=0
function Cdp($socket,$method,$params) {
  $script:cdpId++; $id=$script:cdpId
  $payload=@{id=$id;method=$method;params=$params}|ConvertTo-Json -Compress -Depth 30
  $bytes=[Text.Encoding]::UTF8.GetBytes($payload);$seg=[ArraySegment[byte]]::new($bytes)
  $socket.SendAsync($seg,[Net.WebSockets.WebSocketMessageType]::Text,$true,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
  while($true){$buf=New-Object byte[] 65536;$mem=New-Object IO.MemoryStream;do{$part=[ArraySegment[byte]]::new($buf);$got=$socket.ReceiveAsync($part,[Threading.CancellationToken]::None).GetAwaiter().GetResult();$mem.Write($buf,0,$got.Count)}while(-not $got.EndOfMessage);$msg=[Text.Encoding]::UTF8.GetString($mem.ToArray())|ConvertFrom-Json;if($msg.id -eq $id){if($msg.error){throw 'CDP_COMMAND_FAILED'};return $msg.result}}
}
function Eval($expr){$r=Cdp $ws 'Runtime.evaluate' @{expression=$expr;returnByValue=$true;awaitPromise=$true};if($r.exceptionDetails){throw 'PAGE_EVALUATION_FAILED'};return $r.result.value}
function Wait($expr,$code){$until=[DateTime]::UtcNow.AddSeconds(25);while([DateTime]::UtcNow -lt $until){if(Eval $expr){return};Start-Sleep -Milliseconds 100};throw $code}
try {
 $tcp=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);$tcp.Start();$port=$tcp.LocalEndpoint.Port;$tcp.Stop()
 $args=@('--headless=new','--autoplay-policy=no-user-gesture-required','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check',("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
 $proc=Start-Process -FilePath $chromePath -ArgumentList $args -PassThru -WindowStyle Hidden
 $version=$null;for($i=0;$i -lt 150;$i++){try{$version=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/version" -TimeoutSec 1;break}catch{Start-Sleep -Milliseconds 100}};if(-not $version){throw 'CDP_START_FAILED'}
 $target=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/new?about%3Ablank" -Method Put -TimeoutSec 5
 $uri=[Uri]$target.webSocketDebuggerUrl;if($uri.Host -ne '127.0.0.1' -or $uri.Port -ne $port -or $uri.AbsolutePath -notmatch '^/devtools/page/[A-Za-z0-9_-]+$'){throw 'CDP_ENDPOINT_INVALID'}
 $ws=New-Object Net.WebSockets.ClientWebSocket;$ws.ConnectAsync($uri,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
 $null=Cdp $ws 'Page.enable' @{};$null=Cdp $ws 'Runtime.enable' @{};$null=Cdp $ws 'Network.enable' @{};$null=Cdp $ws 'Network.setBlockedURLs' @{urls=@('http://*/*','https://*/*')}
 $null=Cdp $ws 'Page.navigate' @{url=$fileUrl};Wait "document.readyState==='complete' && document.querySelectorAll('article').length===3 && document.querySelector('#data').textContent.includes('embedded_media')" 'PAGE_LOAD_TIMEOUT'
 $null=Eval @'
(()=>{window.__downloads=[];window.__blobs=new Map();const create=URL.createObjectURL.bind(URL);URL.createObjectURL=(blob)=>{const url=create(blob);if(blob.type==='application/json')window.__blobs.set(url,blob);return url};HTMLAnchorElement.prototype.click=function(){const blob=window.__blobs.get(this.href);if(blob){window.__downloads.push(blob.text());return}};return true})()
'@
 $null=Eval @'
(async()=>{const wait=async f=>{const end=Date.now()+20000;while(Date.now()<end){if(f())return;await new Promise(r=>setTimeout(r,50))}throw Error('UI_WAIT_TIMEOUT')};const set=(e,v)=>{e.value=v;e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}))};const complete=async name=>{set(document.querySelector('#reviewer'),name);set(document.querySelector('#role'),'independent_human');for(const id of ['#attest','#export-disclosure']){const e=document.querySelector(id);e.checked=true;e.dispatchEvent(new Event('change',{bubbles:true}))}for(const article of document.querySelectorAll('article')){const video=article.querySelector('video');await wait(()=>video.readyState>=1&&article.querySelector('.ok'));article.querySelector('button').click();await wait(()=>!video.paused);video.pause();for(const c of article.querySelectorAll('input[type=checkbox]')){c.checked=true;c.dispatchEvent(new Event('change',{bubbles:true}))}const select=article.querySelector('select');set(select,'partial')}await wait(()=>!document.querySelector('#export').disabled);document.querySelector('#export').click();await wait(()=>document.querySelector('#reviewer-controls').hidden)};await complete('synthetic-reviewer-1');if(document.querySelector('#reviewer-count').textContent!=='確定済み: 1 人')throw Error('FIRST_SESSION_COUNT');document.querySelector('#next-reviewer').click();if(document.querySelector('#snapshot-notice').hidden||document.querySelector('#reviewer').value||Array.from(document.querySelectorAll('article input[type=checkbox]')).some(x=>x.checked)||!document.querySelector('#export').disabled)throw Error('NEXT_REVIEWER_RESET');document.querySelector('#snapshot-download').click();await wait(()=>window.__downloads.length===1);await complete('synthetic-reviewer-2');if(document.querySelector('#reviewer-count').textContent!=='確定済み: 2 人')throw Error('SECOND_SESSION_COUNT');document.querySelector('#download-envelope').click();await wait(()=>window.__downloads.length===2);document.querySelector('#retry-envelope').click();await wait(()=>window.__downloads.length===3);const values=await Promise.all(window.__downloads);document.documentElement.dataset.e2eResult=btoa(unescape(encodeURIComponent(JSON.stringify(values))));return true})()
'@
 Wait "!!document.documentElement.dataset.e2eResult" 'FLOW_TIMEOUT'
 $result=Eval 'document.documentElement.dataset.e2eResult'
 Write-Output ('E2E_RESULT:'+ $result)
 Write-Output 'E2E_PASS: sequential handoff, reset, snapshot, final and retry downloads'
} catch { Write-Output ('E2E_FAIL:'+ $_.Exception.Message); exit 1 }
finally {
 try {if($version.webSocketDebuggerUrl){$browserUri=[Uri]$version.webSocketDebuggerUrl;$browserWs=New-Object Net.WebSockets.ClientWebSocket;$browserWs.ConnectAsync($browserUri,[Threading.CancellationToken]::None).GetAwaiter().GetResult();$script:cdpId=50000;$null=Cdp $browserWs 'Browser.close' @{}}}catch{}
 foreach($s in @($ws,$browserWs)){try{if($s){$s.Dispose()}}catch{}}
 if($proc){try{$proc.Refresh();if(-not $proc.HasExited){& "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $proc.Id|Out-Null;$proc.WaitForExit(5000)|Out-Null}}catch{}}
 for($i=0;$i -lt 20 -and (Test-Path -LiteralPath $profile);$i++){try{Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction Stop}catch{Start-Sleep -Milliseconds 100}}
 if(Test-Path -LiteralPath $profile){Write-Output 'E2E_FAIL:PROFILE_CLEANUP'}
}
'''


def _synthetic_page(temp: Path, ffmpeg: str) -> Path:
    from test_render_nextgqa_claim_review_html import RenderNextgqaClaimReviewHtmlTests

    fixture = RenderNextgqaClaimReviewHtmlTests()
    payload, _, lineage, _ = fixture.payload()
    media: dict[str, bytes] = {}
    for index, row in enumerate(payload["records"]):
        path = temp / f"synthetic-{index}.mp4"
        subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                        f"color=c={'red green blue'.split()[index]}:s=32x32:d=2",
                        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(path)],
                       capture_output=True, timeout=30, check=True)
        video = path.read_bytes()
        media[row["review_token"]] = video
        digest = hashlib.sha256(video).hexdigest()
        line = lineage["records"][index]
        row["video_sha256"] = line["video_sha256"] = digest
        row["video_id"] = line["video_id"] = "video-" + digest
        claim_id = hashlib.sha256((digest + "\0" + line["dataset_question_id"] + "\0"
                                  + row["claim_text_sha256"]).encode()).hexdigest()
        row["claim_id"] = line["claim_id"] = "claim-" + claim_id
    raw, lineage_raw = encode(payload), encode(lineage)
    custody = fixture.custody(payload, raw, lineage_raw)
    for row, entry in zip(payload["records"], custody["records"], strict=True):
        video = media[row["review_token"]]
        entry["source_media_sha256"] = entry["staged_media_sha256"] = row["video_sha256"]
        entry["size_bytes"] = len(video)
        entry["media_check_sha256"] = hashlib.sha256(
            bytes.fromhex(entry["media_check_salt"]) + video).hexdigest()
    custody_raw = encode(custody)
    page = render(payload, raw, lineage, lineage_raw, hashlib.sha256(lineage_raw).hexdigest(),
                  custody, custody_raw, hashlib.sha256(custody_raw).hexdigest(),
                  media_files=media)
    html = page.decode("utf-8").replace("</html>", "<script>window.__e2eReady=true;</script></html>")
    target = temp / "formal_synthetic_review.html"
    target.write_text(html, encoding="utf-8")
    target.chmod(0o600)
    return target


class FormalClaimReviewSequentialBrowserE2E(unittest.TestCase):
    def test_two_reviewers_and_one_file_envelope_in_windows_chrome(self):
        chrome = Path(os.environ.get("FORMAL_REVIEW_CHROME", str(DEFAULT_CHROME)))
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        wslpath = shutil.which("wslpath")
        ffmpeg = shutil.which("ffmpeg")
        if not chrome.is_file() or not powershell or not wslpath or not ffmpeg:
            self.skipTest("Windows Chrome/PowerShell/wslpath/ffmpeg unavailable for browser E2E")
        interop = subprocess.run([powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                                  "-Command", "Write-Output WSL_INTEROP_OK"],
                                 capture_output=True, text=True, errors="replace",
                                 timeout=10, check=False)
        if interop.returncode or "WSL_INTEROP_OK" not in interop.stdout:
            detail = (interop.stdout + interop.stderr).strip().replace("\n", " ")
            self.skipTest("Windows interop unavailable: " + (detail[:180] or "PowerShell probe failed"))
        with tempfile.TemporaryDirectory(prefix="formal_claim_e2e_") as name:
            temp = Path(name)
            os.chmod(temp, 0o700)
            page = _synthetic_page(temp, ffmpeg)
            windows_chrome = subprocess.run([wslpath, "-w", str(chrome)], check=True,
                                            capture_output=True, text=True, timeout=5).stdout.strip()
            unc = subprocess.run([wslpath, "-w", str(page)], check=True,
                                 capture_output=True, text=True, timeout=5).stdout.strip()
            if not unc.startswith("\\\\wsl.localhost\\"):
                self.skipTest("synthetic page did not map to WSL UNC for Windows Chrome")
            file_url = "file:" + unc.replace("\\", "/")
            script = POWERSHELL_CDP.replace("__CHROME__", "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + base64.b64encode(windows_chrome.encode()).decode() + "'))").replace("__URL__", "[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('" + base64.b64encode(file_url.encode()).decode() + "'))")
            encoded = base64.b64encode(script.encode("utf-16le")).decode()
            run = subprocess.run([powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                                  "-EncodedCommand", encoded], capture_output=True,
                                 text=True, errors="replace", timeout=100, check=False)
            output = run.stdout + run.stderr
            if "E2E_PASS:" not in output or run.returncode:
                self.fail("synthetic Chrome flow failed: " + next(
                    (line for line in output.splitlines() if line.startswith("E2E_FAIL:")),
                    "browser result marker missing: " + output[-1200:]))
            marker = next(line.split(":", 1)[1].strip() for line in output.splitlines()
                          if line.startswith("E2E_RESULT:"))
            snapshots, final, retry = json.loads(base64.b64decode(marker).decode("utf-8"))
            self.assertEqual(final, retry, "retry must repeat the exact final download")
            snapshot_sessions = parse_envelope(snapshots.encode("ascii"))
            final_sessions = parse_envelope(final.encode("ascii"))
            self.assertEqual(len(snapshot_sessions), 1)
            self.assertEqual(len(final_sessions), 2)
            self.assertEqual(final_sessions[0]["review_payload"], snapshot_sessions[0]["review_payload"])
            reviewers = [json.loads(session["review_payload"])["human_reviewer_private_id"]
                         for session in final_sessions]
            self.assertEqual(reviewers, ["synthetic-reviewer-1", "synthetic-reviewer-2"])
            for session in final_sessions:
                review = json.loads(session["review_payload"])
                self.assertEqual(review["execution_authorization"], "not_granted")
                self.assertEqual(len(review["records"]), 3)
                index = json.loads(session["review_index"])
                self.assertEqual(index["review_payload_sha256"],
                                 hashlib.sha256(session["review_payload"]).hexdigest())


if __name__ == "__main__":
    unittest.main(verbosity=2)
