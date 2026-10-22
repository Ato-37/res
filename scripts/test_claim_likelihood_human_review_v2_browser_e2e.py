#!/usr/bin/env python3
"""Exercise the exact local review-v2 artifact in isolated Windows Chrome.

Only synthetic nicknames/judgments are entered. The browser profile is temporary;
no page content or response text is printed.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/private_nextgqa/claim_likelihood_human_review_v2_20261021_01.html"
CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

PS = r'''
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$chrome=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CHROME__'))
$url=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__URL__'))
$fingerprint='__FP__';$rid='__RID__';$profile=Join-Path ([IO.Path]::GetTempPath()) ('review-v2-e2e-'+$rid)
$proc=$null;$ws=$null;$browserWs=$null;$port=0;$script:id=0;$script:downloads=0
function Cdp($method,$params,$target){if(-not $target){$target=$ws};$script:id++;$n=$script:id;$payload=@{id=$n;method=$method;params=$params}|ConvertTo-Json -Compress -Depth 20;$bytes=[Text.Encoding]::UTF8.GetBytes($payload);$seg=[ArraySegment[byte]]::new($bytes);$target.SendAsync($seg,[Net.WebSockets.WebSocketMessageType]::Text,$true,[Threading.CancellationToken]::None).GetAwaiter().GetResult();while($true){$buf=New-Object byte[] 65536;$mem=New-Object IO.MemoryStream;do{$part=[ArraySegment[byte]]::new($buf);$res=$target.ReceiveAsync($part,[Threading.CancellationToken]::None).GetAwaiter().GetResult();if($res.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close){throw 'CDP_CLOSED'};$mem.Write($buf,0,$res.Count)}while(-not $res.EndOfMessage);$msg=[Text.Encoding]::UTF8.GetString($mem.ToArray())|ConvertFrom-Json;if($msg.method -eq 'Browser.downloadWillBegin'){$script:downloads++};if($msg.id -ne $n){continue};if($msg.error){throw 'CDP_ERROR'};return $msg.result}}
function Eval($expr){$r=Cdp 'Runtime.evaluate' @{expression=$expr;returnByValue=$true;awaitPromise=$true};if($r.exceptionDetails){throw 'PAGE_EVAL_ERROR'};return $r.result.value}
function WaitFor($expr,$code){$until=[DateTime]::UtcNow.AddSeconds(60);while([DateTime]::UtcNow -lt $until){if(Eval $expr){return};Start-Sleep -Milliseconds 100};throw $code}
function ClickSelector($selector){$doc=Cdp 'DOM.getDocument' @{depth=0};$node=Cdp 'DOM.querySelector' @{nodeId=$doc.root.nodeId;selector=$selector};if(-not $node.nodeId){throw 'DOM_NODE_MISSING'};$null=Cdp 'DOM.scrollIntoViewIfNeeded' @{nodeId=$node.nodeId};$box=Cdp 'DOM.getBoxModel' @{nodeId=$node.nodeId};$q=$box.model.content;$x=($q[0]+$q[2]+$q[4]+$q[6])/4;$y=($q[1]+$q[3]+$q[5]+$q[7])/4;$null=Cdp 'Input.dispatchMouseEvent' @{type='mouseMoved';x=$x;y=$y};$null=Cdp 'Input.dispatchMouseEvent' @{type='mousePressed';x=$x;y=$y;button='left';clickCount=1};$null=Cdp 'Input.dispatchMouseEvent' @{type='mouseReleased';x=$x;y=$y;button='left';clickCount=1}}
try{
 if(-not $url.StartsWith('file://wsl.localhost/')){throw 'FILE_URL_INVALID'}
 $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);$listener.Start();$port=$listener.LocalEndpoint.Port;$listener.Stop()
 $args=@('--headless=new','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1',("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
 $proc=Start-Process -FilePath $chrome -ArgumentList $args -PassThru -WindowStyle Hidden
 $version=$null;for($i=0;$i -lt 200;$i++){try{$version=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/version" -TimeoutSec 1;break}catch{Start-Sleep -Milliseconds 100}};if(-not $version){throw 'CHROME_START_FAILED'}
 $target=$null;for($i=0;$i -lt 50;$i++){try{$target=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/new?about%3Ablank" -Method Put -TimeoutSec 2;break}catch{Start-Sleep -Milliseconds 100}};if(-not $target){throw 'CHROME_TARGET_FAILED'}
 $uri=[Uri]$target.webSocketDebuggerUrl;if($uri.Host -ne '127.0.0.1' -or $uri.Port -ne $port){throw 'CDP_ENDPOINT_INVALID'}
 $browserUri=[Uri]$version.webSocketDebuggerUrl;$browserWs=New-Object Net.WebSockets.ClientWebSocket;$browserWs.ConnectAsync($browserUri,[Threading.CancellationToken]::None).GetAwaiter().GetResult();$null=Cdp 'Browser.setDownloadBehavior' @{behavior='allow';downloadPath=$profile} $browserWs
 $ws=New-Object Net.WebSockets.ClientWebSocket;$ws.ConnectAsync($uri,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
 $null=Cdp 'Page.enable' @{};$null=Cdp 'Page.setDownloadBehavior' @{behavior='allow';downloadPath=$profile};$null=Cdp 'DOM.enable' @{};$null=Cdp 'Runtime.enable' @{};$null=Cdp 'Network.enable' @{};$null=Cdp 'Network.setBlockedURLs' @{urls=@('http://*/*','https://*/*')}
 $null=Cdp 'Page.navigate' @{url=$url};WaitFor "document.readyState==='complete' && document.querySelector('#start') && document.querySelectorAll('#frames img').length===0" 'PAGE_LOAD_FAILED'
 if(-not (Eval "document.querySelector('meta[name=review-artifact-sha256]').content==='$fingerprint' && document.querySelectorAll('input[type=file]').length===0")){throw 'ARTIFACT_UI_INVALID'}
 $null=Eval "(()=>{const original=URL.createObjectURL.bind(URL);window.__reviewJsonBlobs=[];URL.createObjectURL=b=>{if(b.type==='application/json')window.__reviewJsonBlobs.push(b);return original(b)};return true})()"
 $null=Eval "document.querySelector('#nick').value='synthetic-one';document.querySelector('#start').click();true";WaitFor "!document.querySelector('#task').hidden && document.querySelector('#count').textContent==='1 / 3'" 'FIRST_ITEM_FAILED'
 WaitFor "document.querySelectorAll('#frames img').length===32 && Array.from(document.querySelectorAll('#frames img')).every(x=>x.complete&&x.naturalWidth>0)" 'FRAME_RENDER_FAILED'
 $null=Eval "document.querySelector('#next').click();true";if(Eval "document.querySelector('#count').textContent==='2 / 3'"){throw 'EMPTY_JUDGMENT_ACCEPTED'}
 for($i=0;$i -lt 3;$i++){ $null=Eval "Array.from(document.querySelectorAll('#frames img')).length===32";WaitFor "document.querySelectorAll('#frames img').length===32 && Array.from(document.querySelectorAll('#frames img')).every(x=>x.complete&&x.naturalWidth>0)" 'FIRST_ITEM_FRAME_RENDER_FAILED';$value=@('roughly_matches','partly_differs','cannot_tell')[$i];$null=Eval "document.querySelector('[name=judgment][value=$value]').checked=true;document.querySelector('#note').value='synthetic-note';document.querySelector('#next').click();true";if($i -lt 2){$n=$i+2;WaitFor "document.querySelector('#count').textContent==='$n / 3'" 'FIRST_ADVANCE_FAILED'} }
 WaitFor "!document.querySelector('#done').hidden && document.querySelector('#done-count').textContent.includes('1人分')" 'FIRST_REVIEW_FAILED'
 $null=Eval "document.querySelector('#another').click();true";WaitFor "!document.querySelector('#intro').hidden && document.querySelector('#nick').value===''" 'SECOND_NICKNAME_SCREEN_FAILED'
 $null=Eval "document.querySelector('#nick').value='synthetic-two';document.querySelector('#start').click();true";WaitFor "!document.querySelector('#task').hidden && document.querySelector('#count').textContent==='1 / 3'" 'SECOND_ITEM_FAILED'
 for($i=0;$i -lt 3;$i++){ WaitFor "document.querySelectorAll('#frames img').length===32 && Array.from(document.querySelectorAll('#frames img')).every(x=>x.complete&&x.naturalWidth>0)" 'SECOND_ITEM_FRAME_RENDER_FAILED';$null=Eval "document.querySelector('[name=judgment][value=cannot_tell]').checked=true;document.querySelector('#next').click();true";if($i -lt 2){$n=$i+2;WaitFor "document.querySelector('#count').textContent==='$n / 3'" 'SECOND_ADVANCE_FAILED'} }
 WaitFor "!document.querySelector('#done').hidden && document.querySelector('#done-count').textContent.includes('2人分')" 'SECOND_REVIEW_FAILED'
 ClickSelector '#save'
 $finalBlob=Eval "(async()=>{const xs=window.__reviewJsonBlobs;if(xs.length!==1)return false;const x=JSON.parse(await xs[0].text());return x.artifact_fingerprint==='$fingerprint'&&x.respondents.length===2&&x.respondents.every(r=>r.answers.length===3)})()";if(-not $finalBlob){throw 'FINAL_BLOB_INVALID'}
 $until=[DateTime]::UtcNow.AddSeconds(20);$files=@();while([DateTime]::UtcNow -lt $until){$files=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_likelihood_human_review_responses*.json' -File -ErrorAction SilentlyContinue);if($files.Count -ge 1){break};Start-Sleep -Milliseconds 100};if($files.Count -lt 1){throw 'FINAL_DOWNLOAD_MISSING'};Start-Sleep -Milliseconds 500;$files=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_likelihood_human_review_responses*.json' -File -ErrorAction SilentlyContinue);if($files.Count -ne 1){throw 'COMBINED_DOWNLOAD_COUNT_INVALID'}
 $finalFile=$files|Sort-Object LastWriteTimeUtc -Descending|Select-Object -First 1;$finalJson=Get-Content -LiteralPath $finalFile.FullName -Raw|ConvertFrom-Json;if($finalJson.respondents.Count -ne 2 -or $finalJson.artifact_fingerprint -ne $fingerprint -or $finalJson.respondents[0].answers.Count -ne 3 -or $finalJson.respondents[1].answers.Count -ne 3){throw 'FINAL_DOWNLOAD_INVALID'}
 Write-Output 'E2E_PASS: two_reviewer_sessions_six_answers_32_frames'
}catch{$m=$_.Exception.Message;if($m -match '^[A-Z_]+$'){Write-Output ('E2E_FAIL:'+ $m)}else{Write-Output 'E2E_FAIL:SCRIPT'};$jsonCount=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_likelihood_human_review_responses*.json' -File -ErrorAction SilentlyContinue).Count;$partialCount=@(Get-ChildItem -LiteralPath $profile -Filter '*.crdownload' -File -ErrorAction SilentlyContinue).Count;Write-Output ("E2E_DIAG:download_events=$script:downloads;json=$jsonCount;partial=$partialCount");exit 2}
finally{$clean=$true;try{if($ws){$ws.Dispose()}}catch{$clean=$false};try{if($browserWs){$browserWs.Dispose()}}catch{$clean=$false};try{if($proc -and -not $proc.HasExited){Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue;$proc.WaitForExit(10000)};if($proc -and -not $proc.HasExited){$clean=$false}}catch{$clean=$false};try{if(Test-Path -LiteralPath $profile){Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue};if(Test-Path -LiteralPath $profile){$clean=$false}}catch{$clean=$false};if($clean){Write-Output 'E2E_CLEANUP_PASS'}else{Write-Output 'E2E_CLEANUP_FAIL'}}
'''


def main() -> int:
    if not ARTIFACT.is_file() or not CHROME.is_file():
        print("E2E_BLOCKED:browser_or_artifact_missing")
        return 2
    match = re.search(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">',
                      ARTIFACT.read_bytes())
    if not match:
        print("E2E_FAIL:fingerprint_missing")
        return 2
    wslpath = shutil.which("wslpath")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not wslpath or not powershell:
        print("E2E_BLOCKED:windows_interop_missing")
        return 2
    win_chrome = subprocess.run([wslpath, "-w", str(CHROME)], check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    win_artifact = subprocess.run([wslpath, "-w", str(ARTIFACT)], check=True,
                                  capture_output=True, text=True, timeout=5).stdout.strip()
    url = "file:" + win_artifact.replace("\\", "/")
    enc = lambda s: base64.b64encode(s.encode()).decode()
    script = (PS.replace("__CHROME__", enc(win_chrome)).replace("__URL__", enc(url))
              .replace("__FP__", match.group(1).decode()).replace("__RID__", uuid.uuid4().hex))
    with tempfile.TemporaryDirectory(prefix="review-v2-e2e-") as d:
        psfile = Path(d) / "run.ps1"
        psfile.write_bytes(b"\xef\xbb\xbf" + script.encode("utf-8"))
        psfile.chmod(0o600)
        win_ps = subprocess.run([wslpath, "-w", str(psfile)], check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        if not win_ps.startswith("\\\\wsl.localhost\\"):
            print("E2E_FAIL:script_unc_invalid")
            return 2
        bootstrap = f"$s=[IO.File]::ReadAllText('{win_ps.replace(chr(39), chr(39)*2)}'); Invoke-Expression $s"
        run = subprocess.run([powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                              "-Command", bootstrap], capture_output=True, text=True,
                             errors="replace", timeout=180)
    out = run.stdout + "\n" + run.stderr
    result = next((line for line in out.splitlines()
                   if line.startswith(("E2E_PASS:", "E2E_FAIL:"))), None)
    if run.returncode != 0 or result != "E2E_PASS: two_reviewer_sessions_six_answers_32_frames":
        print(result or "E2E_FAIL:result_missing")
        diag = next((line for line in out.splitlines() if line.startswith("E2E_DIAG:")), None)
        if diag:
            print(diag)
        return 2
    if "E2E_CLEANUP_PASS" not in out or "E2E_CLEANUP_FAIL" in out:
        print("E2E_FAIL:cleanup_not_verified")
        return 2
    print(json.dumps({"status": "passed", "reviewer_sessions": 2,
                      "items_per_session": 3, "model_frames_per_item": 32,
                      "combined_json_downloads": 1,
                      "entered_judgments_are_synthetic": True,
                      "semantic_review_performed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
