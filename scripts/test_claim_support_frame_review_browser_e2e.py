#!/usr/bin/env python3
"""Exercise the private frame-review page in Windows Chrome with synthetic answers.

The exact local review artifact is opened, but all entered reviewer names and
judgments are synthetic. The combined JSON and Chrome profile are temporary and
removed. The test prints only fixed status and verification counts.
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

from inspect_claim_support_frame_review_response import inspect

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "data/private_nextgqa/claim_support_frame_review_cohort11_v3_20261021_08.html"
DEFAULT_CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

POWERSHELL = r'''
$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue'
$chrome=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__CHROME__'))
$url=[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__URL__'))
$expected='__FINGERPRINT__';$itemCount=__ITEM_COUNT__;$rid='__RID__';$profile=Join-Path ([IO.Path]::GetTempPath()) ('claim-frame-e2e-'+$rid)
$proc=$null;$ws=$null;$browserWs=$null;$port=0;$step='setup';$script:cdpid=0;$script:downloadBegins=0;$script:downloadStates=@();$script:downloadCheck='none'
function Cdp($method,$params,$target){if(-not $target){$target=$ws};$script:cdpid++;$id=$script:cdpid;$obj=@{id=$id;method=$method;params=$params}|ConvertTo-Json -Compress -Depth 20;$bytes=[Text.Encoding]::UTF8.GetBytes($obj);$seg=[ArraySegment[byte]]::new($bytes);$target.SendAsync($seg,[Net.WebSockets.WebSocketMessageType]::Text,$true,[Threading.CancellationToken]::None).GetAwaiter().GetResult();while($true){$buf=New-Object byte[] 65536;$mem=New-Object IO.MemoryStream;do{$part=[ArraySegment[byte]]::new($buf);$res=$target.ReceiveAsync($part,[Threading.CancellationToken]::None).GetAwaiter().GetResult();if($res.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close){throw 'CDP_CLOSED'};$mem.Write($buf,0,$res.Count)}while(-not $res.EndOfMessage);$msg=[Text.Encoding]::UTF8.GetString($mem.ToArray())|ConvertFrom-Json;if($msg.method -eq 'Browser.downloadWillBegin'){$script:downloadBegins++};if($msg.method -eq 'Browser.downloadProgress'){$script:downloadStates+=@($msg.params.state)};if($msg.id -ne $id){continue};if($msg.error){throw 'CDP_ERROR'};return $msg.result}}
function Eval($expr){$r=Cdp 'Runtime.evaluate' @{expression=$expr;returnByValue=$true;awaitPromise=$true};if($r.exceptionDetails){throw 'PAGE_EVAL_ERROR'};return $r.result.value}
function WaitFor($expr,$code){$until=[DateTime]::UtcNow.AddSeconds(30);while([DateTime]::UtcNow -lt $until){if(Eval $expr){return};Start-Sleep -Milliseconds 100};throw $code}
function ClickSelector($selector){$doc=Cdp 'DOM.getDocument' @{depth=0};$node=Cdp 'DOM.querySelector' @{nodeId=$doc.root.nodeId;selector=$selector};if(-not $node.nodeId){throw 'DOM_NODE_MISSING'};$null=Cdp 'DOM.scrollIntoViewIfNeeded' @{nodeId=$node.nodeId};$box=Cdp 'DOM.getBoxModel' @{nodeId=$node.nodeId};$q=$box.model.content;$x=($q[0]+$q[2]+$q[4]+$q[6])/4;$y=($q[1]+$q[3]+$q[5]+$q[7])/4;$null=Cdp 'Input.dispatchMouseEvent' @{type='mouseMoved';x=$x;y=$y};$null=Cdp 'Input.dispatchMouseEvent' @{type='mousePressed';x=$x;y=$y;button='left';clickCount=1};$null=Cdp 'Input.dispatchMouseEvent' @{type='mouseReleased';x=$x;y=$y;button='left';clickCount=1}}
try{
 if(-not $url.StartsWith('file://wsl.localhost/')){throw 'UNC_URL_INVALID'}
 $listener=[Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback,0);$listener.Start();$port=$listener.LocalEndpoint.Port;$listener.Stop()
 $args=@('--headless=new','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1',("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
 $proc=Start-Process -FilePath $chrome -ArgumentList $args -PassThru -WindowStyle Hidden
 $version=$null;for($i=0;$i -lt 150;$i++){try{$version=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/version" -TimeoutSec 1;break}catch{Start-Sleep -Milliseconds 100}};if(-not $version){throw 'CHROME_CDP_START_FAILED'}
 $target=$null;for($i=0;$i -lt 40;$i++){try{$target=Invoke-RestMethod -Uri "http://127.0.0.1:$port/json/new?about%3Ablank" -Method Put -TimeoutSec 2;break}catch{Start-Sleep -Milliseconds 100}};if(-not $target){throw 'CHROME_TARGET_FAILED'}
 $uri=[Uri]$target.webSocketDebuggerUrl;if($uri.Host -ne '127.0.0.1' -or $uri.Port -ne $port){throw 'CDP_ENDPOINT_INVALID'}
 $browserUri=[Uri]$version.webSocketDebuggerUrl;$browserWs=New-Object Net.WebSockets.ClientWebSocket;$browserWs.ConnectAsync($browserUri,[Threading.CancellationToken]::None).GetAwaiter().GetResult();$null=Cdp 'Browser.setDownloadBehavior' @{behavior='allow';downloadPath=$profile;eventsEnabled=$true} $browserWs
 $ws=New-Object Net.WebSockets.ClientWebSocket;$ws.ConnectAsync($uri,[Threading.CancellationToken]::None).GetAwaiter().GetResult()
 $null=Cdp 'Page.enable' @{};$null=Cdp 'DOM.enable' @{};$null=Cdp 'Runtime.enable' @{};$null=Cdp 'Network.enable' @{};$null=Cdp 'Network.setBlockedURLs' @{urls=@('http://*/*','https://*/*')};$null=Cdp 'Page.setBypassCSP' @{enabled=$true};$null=Cdp 'Page.setDownloadBehavior' @{behavior='allow';downloadPath=$profile}
 $step='navigate';$null=Cdp 'Page.navigate' @{url=$url};WaitFor "document.readyState==='complete' && !!document.querySelector('#welcome-form') && !!document.querySelector('#next-reviewer')" 'PAGE_LOAD_FAILED'
 $hasSamplePlayer=Eval "!!document.querySelector('#player-toggle')"
 if(-not (Eval "document.querySelector('meta[name=review-artifact-sha256]').content==='$expected' && document.querySelectorAll('input[type=file]').length===0 && document.querySelectorAll('input[type=radio]').length===4")){throw 'ARTIFACT_OR_UI_BINDING_INVALID'}
 if(-not (Eval "document.querySelector('#welcome .notice').textContent.includes('processorが作るcrop/patchでもありません')")){throw 'FRAME_DISPLAY_LIMITATION_MISSING'}
 $step='network_policy';if(-not (Eval "document.querySelector('meta[http-equiv=Content-Security-Policy]').content.includes('connect-src ' + String.fromCharCode(39) + 'none' + String.fromCharCode(39))")){throw 'CSP_MISSING'}
 $step='reviewer_one';$null=Eval "document.querySelector('#nickname').value='synthetic-one';document.querySelector('#welcome-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));true";WaitFor "!document.querySelector('#question').classList.contains('hidden') && document.querySelectorAll('#frames img').length===16" 'FIRST_ITEM_FAILED'
 WaitFor "Array.from(document.querySelectorAll('#frames img')).every(x=>x.complete&&x.naturalWidth>0)" 'FRAME_PREVIEW_DECODE_FAILED'
 if($hasSamplePlayer){$step='sampled_frame_player';if(-not (Eval "document.querySelector('#player-position').textContent==='1 / 16 枚目' && document.querySelector('#player-time').textContent.endsWith('秒')")){throw 'PLAYER_INITIAL_FRAME_INVALID'}
 $null=Eval "document.querySelector('#frame-next').click();true";if(-not (Eval "document.querySelector('#player-position').textContent==='2 / 16 枚目'")){throw 'PLAYER_NEXT_FAILED'}
 $null=Eval "document.querySelector('#frame-prev').click();document.querySelector('#player-toggle').click();true";Start-Sleep -Milliseconds 900
 if(-not (Eval "document.querySelector('#player-position').textContent!=='1 / 16 枚目'")){throw 'PLAYER_PLAYBACK_FAILED'}
 $null=Eval "document.querySelector('#player-toggle').click();true";if(-not (Eval "document.querySelector('#player-toggle').textContent==='再生'")){throw 'PLAYER_PAUSE_FAILED'}}
 $decisions=@('supports','contradicts','cannot_judge')
 for($i=0;$i -lt $itemCount;$i++){$step='reviewer_one_item_'+$i;$choice=$decisions[$i];$mark='';if($choice -eq 'supports' -or $choice -eq 'contradicts'){$mark="document.querySelectorAll('#frames input[type=checkbox]')[$i].checked=true;"};if($hasSamplePlayer -and $i -eq 0 -and $i -lt ($itemCount-1)){$null=Eval "document.querySelector('#player-toggle').click();true"};$expr="(()=>{const r=document.querySelector('[name=decision][value=$choice]');r.checked=true;$mark document.querySelector('#next').click();return true})()";$null=Eval $expr;if($i -lt ($itemCount-1)){$n=$i+2;WaitFor "document.querySelector('#counter').textContent.trim().startsWith('$n')" 'ITEM_ADVANCE_FAILED';if($hasSamplePlayer -and $i -eq 0){Start-Sleep -Milliseconds 900;if(-not (Eval "document.querySelector('#player-position').textContent==='1 / 16 枚目' && document.querySelector('#player-toggle').textContent==='再生'")){throw 'PLAYER_TIMER_NOT_STOPPED_ON_ITEM_CHANGE'}}}}
 WaitFor "!document.querySelector('#done').classList.contains('hidden') && document.querySelector('#reviewer-count').textContent.startsWith('1')" 'FIRST_REVIEWER_NOT_DONE'
 $step='first_save_click';$null=Eval "document.querySelector('#save').scrollIntoView({block:'center'});true";$null=ClickSelector '#save'
 $step='first_export_link';WaitFor "!document.querySelector('#saved-file').classList.contains('hidden') && document.querySelector('#saved-file').href.startsWith('blob:')" 'FIRST_EXPORT_MISSING'
 $step='first_export_copy';$null=Eval "window.__oldExport=document.querySelector('#saved-file').href;true";if(-not (Eval "(async()=>{try{const x=JSON.parse(await(await fetch(window.__oldExport)).text());return x.reviewers.length===1}catch{return false}})()")){throw 'FIRST_BLOB_CONTENT_INVALID'}
 $step='first_export_blob_verified';if(-not (Eval "(async()=>{try{const x=JSON.parse(await(await fetch(document.querySelector('#saved-file').href)).text());return x.reviewers.length===1}catch{return false}})()")){throw 'FIRST_BLOB_CONTENT_INVALID'}
 $step='first_download_wait';$firstDownloaded=$null;$until=[DateTime]::UtcNow.AddSeconds(10);while([DateTime]::UtcNow -lt $until){$files=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_support_frame_review_*.json' -File -ErrorAction SilentlyContinue);foreach($f in $files){try{$script:downloadCheck='bytes='+$f.Length;if($f.Length -eq 0){continue};$rawText=[IO.File]::ReadAllText($f.FullName);if($null -eq $rawText){$script:downloadCheck+=';content_null';continue};$firstChar='none';if($rawText.Length -gt 0){$firstChar=[string][char]$rawText[0]};$script:downloadCheck+=';length='+$rawText.Length+';first='+$firstChar+';starts_json='+$rawText.TrimStart().StartsWith('{');$candidate=ConvertFrom-Json -InputObject $rawText -ErrorAction Stop;$rc=0;if($null -ne $candidate.reviewers){$rc=$candidate.reviewers.Count};$script:downloadCheck+=';schema='+$candidate.schema_version+';purpose='+($candidate.purpose -eq 'local_claim_support_frame_review')+';reviewers='+$rc+';fingerprint='+($candidate.artifact_fingerprint -eq $expected);if($candidate.schema_version -eq 1 -and $candidate.purpose -eq 'local_claim_support_frame_review' -and $rc -eq 1 -and $candidate.artifact_fingerprint -eq $expected){$firstDownloaded=$candidate;break}}catch{$script:downloadCheck='bytes='+$f.Length+';parse_error:'+($_.Exception.GetType().Name)}};if($firstDownloaded){break};Start-Sleep -Milliseconds 100};if(-not $firstDownloaded){throw 'FIRST_DOWNLOAD_MISSING'};$firstDownloadCount=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_support_frame_review_*.json' -File -ErrorAction SilentlyContinue).Count
 $null=Eval "document.querySelector('#next-reviewer').click();true";WaitFor "!document.querySelector('#welcome').classList.contains('hidden') && document.querySelector('#welcome-title').textContent.includes('2人目')" 'SECOND_NICKNAME_SCREEN_FAILED'
 if(-not (Eval "document.querySelector('#saved-file').classList.contains('hidden') && document.querySelector('#saved').textContent===''")){throw 'STALE_EXPORT_UI_VISIBLE'}
 if(-not (Eval "(async()=>{try{await fetch(window.__oldExport);return false}catch{return true}})()")){throw 'STALE_BLOB_NOT_REVOKED'}
 $null=Eval "document.querySelector('#saved-file').click();true";Start-Sleep -Milliseconds 250;if(-not (Eval "document.querySelector('#saved-file').classList.contains('hidden')")){throw 'STALE_EXPORT_UI_VISIBLE'}
 $step='reviewer_two';$null=Eval "document.querySelector('#nickname').value='synthetic-two';document.querySelector('#welcome-form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true}));true";WaitFor "!document.querySelector('#question').classList.contains('hidden')" 'SECOND_REVIEW_FAILED'
 for($i=0;$i -lt $itemCount;$i++){$step='reviewer_two_item_'+$i;$null=Eval "document.querySelector('[name=decision][value=cannot_judge]').checked=true;document.querySelector('#next').click();true";if($i -lt ($itemCount-1)){$n=$i+2;WaitFor "document.querySelector('#counter').textContent.trim().startsWith('$n')" 'SECOND_ITEM_ADVANCE_FAILED'}}
 WaitFor "!document.querySelector('#done').classList.contains('hidden') && document.querySelector('#reviewer-count').textContent.startsWith('2')" 'SECOND_REVIEWER_NOT_DONE'
 $step='final_done_state';if(-not (Eval "!document.querySelector('#done').classList.contains('hidden') && document.querySelector('#reviewer-count').textContent.startsWith('2')")){throw 'FINAL_DONE_STATE_INVALID'}
 $step='final_click';$null=Eval "document.querySelector('#save').scrollIntoView({block:'center'});true";$null=ClickSelector '#save'
 $step='final_state_eval';$state=Eval "JSON.stringify({linkVisible:!document.querySelector('#saved-file').classList.contains('hidden'),hasBlob:document.querySelector('#saved-file').href.startsWith('blob:'),savedMessage:document.querySelector('#saved').textContent.startsWith('JSON')})"
 $step='final_state_parse';$stateObj=$state|ConvertFrom-Json;if(-not $stateObj.linkVisible){throw 'FINAL_EXPORT_LINK_HIDDEN'};if(-not $stateObj.hasBlob){throw 'FINAL_EXPORT_BLOB_MISSING'};if(-not $stateObj.savedMessage){throw 'FINAL_EXPORT_STATUS_MISSING'}
 $step='final_blob';$final=Eval "(async()=>await(await fetch(document.querySelector('#saved-file').href)).text())()";if(-not $final){throw 'FINAL_JSON_READ_FAILED'}
 $step='final_json_parse';$obj=$final|ConvertFrom-Json;if($obj.schema_version -ne 1 -or $obj.purpose -ne 'local_claim_support_frame_review' -or $obj.reviewers.Count -ne 2){throw 'FINAL_JSON_HEADER_INVALID'}
 if($obj.reviewers[0].nickname -ne 'synthetic-one' -or $obj.reviewers[1].nickname -ne 'synthetic-two' -or $obj.reviewers[0].answers.Count -ne $itemCount -or $obj.reviewers[1].answers.Count -ne $itemCount){throw 'FINAL_JSON_REVIEWER_COUNTS_INVALID'}
 if($obj.reviewers[0].answers[0].frame_indices.Count -ne 1 -or ($itemCount -gt 1 -and $obj.reviewers[0].answers[1].frame_indices.Count -ne 1)){throw 'FINAL_JSON_EVIDENCE_RULE_INVALID'}
 $step='final_download_wait';$downloaded=$null;$downloadRaw=$null;$until=[DateTime]::UtcNow.AddSeconds(10);while([DateTime]::UtcNow -lt $until){$files=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_support_frame_review_*.json' -File -ErrorAction SilentlyContinue);if($files.Count -gt $firstDownloadCount){foreach($f in ($files|Sort-Object LastWriteTimeUtc -Descending)){try{$rawText=[IO.File]::ReadAllText($f.FullName);$candidate=ConvertFrom-Json -InputObject $rawText -ErrorAction Stop;$rc=0;if($null -ne $candidate.reviewers){$rc=$candidate.reviewers.Count};$script:downloadCheck='final_bytes='+$f.Length+';schema='+$candidate.schema_version+';purpose='+($candidate.purpose -eq 'local_claim_support_frame_review')+';reviewers='+$rc+';fingerprint='+($candidate.artifact_fingerprint -eq $expected);if($candidate.schema_version -eq 1 -and $candidate.purpose -eq 'local_claim_support_frame_review' -and $rc -eq 2 -and $candidate.artifact_fingerprint -eq $expected){$downloaded=$candidate;$downloadRaw=$rawText;break}}catch{$script:downloadCheck='final_parse_error:'+($_.Exception.GetType().Name)}}};if($downloaded){break};Start-Sleep -Milliseconds 100};if(-not $downloaded){throw 'FINAL_DOWNLOAD_MISSING'}
 if(($downloaded|ConvertTo-Json -Depth 20 -Compress) -ne ($obj|ConvertTo-Json -Depth 20 -Compress)){throw 'FINAL_DOWNLOAD_CONTENT_MISMATCH'}
 $b64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($final));$fileB64=[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($downloadRaw));Write-Output ('E2E_JSON:'+ $b64);Write-Output ('E2E_FILE_JSON:'+ $fileB64);Write-Output ('E2E_PASS: two_reviewers_'+(2*$itemCount)+'_items_blob_and_download_verified')
}catch{$failure=$_.Exception.Message;if($failure -match '^(FINAL_[A-Z_]+|FIRST_[A-Z_]+|SECOND_[A-Z_]+|STALE_[A-Z_]+|ITEM_[A-Z_]+|ARTIFACT_[A-Z_]+|FRAME_[A-Z_]+|CSP_[A-Z_]+)$'){Write-Output ('E2E_FAIL:'+ $failure)}else{Write-Output ('E2E_FAIL:'+ $step)};$jsonCount=0;try{$jsonCount=@(Get-ChildItem -LiteralPath $profile -Filter 'claim_support_frame_review_*.json' -File -ErrorAction SilentlyContinue).Count}catch{};Write-Output ('E2E_DOWNLOAD_DIAG:begin='+$script:downloadBegins+';states='+($script:downloadStates -join ',')+';json='+$jsonCount+';check='+$script:downloadCheck);exit 2}finally{try{if($ws){$ws.Dispose()}}catch{};try{if($browserWs){$browserWs.Dispose()}}catch{};try{if($proc -and -not $proc.HasExited){Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue}}catch{};try{if(Test-Path -LiteralPath $profile){Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue}}catch{}}
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--chrome", type=Path, default=DEFAULT_CHROME)
    args = parser.parse_args()
    artifact = args.artifact.resolve(strict=True)
    raw = artifact.read_bytes()
    import re
    match = re.search(rb'<meta name="review-artifact-sha256" content="([0-9a-f]{64})">', raw)
    if not match:
        raise SystemExit("artifact_fingerprint_missing")
    text = raw.decode("utf-8")
    item_match = re.search(r"const items=(.*?);const sourceProvenance=", text, re.S)
    if not item_match:
        raise SystemExit("artifact_items_missing")
    item_count = len(json.loads(item_match.group(1)))
    if not 1 <= item_count <= 3:
        raise SystemExit("artifact_item_count_invalid")
    wslpath = shutil.which("wslpath")
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not wslpath or not powershell or not args.chrome.is_file():
        raise SystemExit("windows_chrome_interop_unavailable")
    win_chrome = subprocess.run([wslpath, "-w", str(args.chrome)], check=True,
                                capture_output=True, text=True, timeout=5).stdout.strip()
    unc = subprocess.run([wslpath, "-w", str(artifact)], check=True,
                         capture_output=True, text=True, timeout=5).stdout.strip()
    file_url = "file:" + unc.replace("\\", "/")
    encode = lambda value: base64.b64encode(value.encode("utf-8")).decode("ascii")
    script = (POWERSHELL.replace("__CHROME__", encode(win_chrome))
              .replace("__URL__", encode(file_url))
              .replace("__FINGERPRINT__", match.group(1).decode("ascii"))
              .replace("__ITEM_COUNT__", str(item_count))
              .replace("__RID__", uuid.uuid4().hex))
    # WSL interop rejects long -EncodedCommand payloads on some hosts. Invoke a
    # private temporary script through the UNC bridge, matching the established
    # browser-E2E harness used elsewhere in this repository.
    with tempfile.TemporaryDirectory(prefix="claim-frame-e2e-") as temp_name:
        ps_path = Path(temp_name) / "browser_e2e.ps1"
        ps_path.write_bytes(b"\xef\xbb\xbf" + script.encode("utf-8"))
        ps_path.chmod(0o600)
        ps_windows = subprocess.run([wslpath, "-w", str(ps_path)], check=True,
                                    capture_output=True, text=True, timeout=5).stdout.strip()
        if not ps_windows.startswith("\\\\wsl.localhost\\"):
            raise SystemExit("powershell_script_unc_invalid")
        ps_windows = ps_windows.replace("'", "''")
        bootstrap = f"$scriptText=[IO.File]::ReadAllText('{ps_windows}'); Invoke-Expression $scriptText"
        run = subprocess.run([powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                              "-Command", bootstrap], capture_output=True, text=True,
                             errors="replace", timeout=120, check=False)
    output = run.stdout + "\n" + run.stderr
    if run.returncode or "E2E_PASS:" not in output:
        marker = next((line for line in output.splitlines() if line.startswith("E2E_FAIL:")),
                      "e2e_result_missing")
        download_diag = next((line for line in output.splitlines()
                              if line.startswith("E2E_DOWNLOAD_DIAG:")), None)
        # PowerShell syntax/startup failures happen before the script's catch block.
        # Surface only a short sanitized diagnostic, never browser/page contents.
        if marker == "e2e_result_missing":
            diagnostic = next((line.strip() for line in reversed(output.splitlines())
                               if line.strip() and not line.startswith("E2E_JSON:")), "")
            print(f"{marker}:powershell_exit_{run.returncode}: {diagnostic[:240]}",
                  file=sys.stderr)
        else:
            print(marker, file=sys.stderr)
        if download_diag:
            print(download_diag, file=sys.stderr)
        return 2
    result_line = next((line for line in output.splitlines() if line.startswith("E2E_JSON:")), None)
    if not result_line:
        print("e2e_json_missing", file=sys.stderr)
        return 2
    response = json.loads(base64.b64decode(result_line.split(":", 1)[1]))
    checked = inspect(response, raw)
    if not checked.get("structurally_valid"):
        print("inspector_rejected_browser_export", file=sys.stderr)
        return 2
    file_line = next((line for line in output.splitlines()
                      if line.startswith("E2E_FILE_JSON:")), None)
    if not file_line:
        print("browser_download_payload_missing", file=sys.stderr)
        return 2
    file_response = json.loads(base64.b64decode(file_line.split(":", 1)[1]))
    downloaded = inspect(file_response, raw)
    if not downloaded.get("structurally_valid") or file_response != response:
        print("browser_download_payload_invalid", file=sys.stderr)
        return 2
    print(json.dumps({"status": "passed", "reviewers": checked["reviewer_count"],
                      "items_per_reviewer": checked["items_per_reviewer"],
                      "browser_export_blob_payload_verified": True,
                      "browser_filesystem_download_verified": True,
                      "stale_blob_revoked": True,
                      "semantic_support_validated": checked["semantic_support_validated"],
                      "reviewer_identity_or_independence_verified": checked[
                          "reviewer_identity_or_independence_verified"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
