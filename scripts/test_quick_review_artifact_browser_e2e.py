"""Windows Chrome E2E for a synthetic quick-review artifact.

The driver uses the production builder with synthetic rows and a tiny synthetic
video. It enters three synthetic reviewers, downloads the combined JSON, verifies
the schema and respondent/item counts without printing response data, and emits
only fixed pass/fail codes. The protected artifact is never opened.
"""
from __future__ import annotations

import base64
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
import uuid

from build_nextgqa_quick_review_artifact import build, translation_key
from inspect_nextgqa_quick_review_response import expected_artifact_fingerprint, expected_item_ids


ROOT = Path(__file__).resolve().parents[1]
PRIVATE_ROOT = ROOT / "data/private_nextgqa"
DEFAULT_CHROME = Path("/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

# Keep all CDP and browser interaction on Windows. The only page source is the
# exact artifact passed in from WSL as a UNC-backed file URL. No page values,
# page source, response objects, or console messages are returned to Python.
POWERSHELL_CDP = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$chromePath = __CHROME_PATH__
$fileUrl = __FILE_URL__
$runId = '__RUN_ID__'
$profile = Join-Path ([IO.Path]::GetTempPath()) ('quick-review-e2e-' + $runId)
$pidMarker = $profile + '.pid'
$proc = $null
$pageSocket = $null
$browserSocket = $null
$port = 0
$step = 'setup'
$script:cdpId = 0
function Invoke-Cdp($ws, $method, $params) {
  $script:cdpId++
  $id = $script:cdpId
  $payload = @{ id = $id; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 20
  $bytes = [Text.Encoding]::UTF8.GetBytes($payload)
  $segment = [ArraySegment[byte]]::new($bytes)
  $sendCts = New-Object Threading.CancellationTokenSource
  $sendCts.CancelAfter(10000)
  $ws.SendAsync($segment, [Net.WebSockets.WebSocketMessageType]::Text, $true, $sendCts.Token).GetAwaiter().GetResult()
  while ($true) {
    $buffer = New-Object byte[] 65536
    $stream = New-Object IO.MemoryStream
    do {
      $part = [ArraySegment[byte]]::new($buffer)
      $recvCts = New-Object Threading.CancellationTokenSource
      $recvCts.CancelAfter(10000)
      $result = $ws.ReceiveAsync($part, $recvCts.Token).GetAwaiter().GetResult()
      if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) { throw 'CDP_CLOSED' }
      $stream.Write($buffer, 0, $result.Count)
    } while (-not $result.EndOfMessage)
    $message = [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
    if ($message.id -ne $id) { continue }
    if ($message.error) { throw 'CDP_COMMAND_FAILED' }
    return $message.result
  }
}
function Evaluate($ws, $expression) {
  $answer = Invoke-Cdp $ws 'Runtime.evaluate' @{ expression = $expression; returnByValue = $true; awaitPromise = $true }
  if ($answer.exceptionDetails) { throw 'PAGE_EVALUATION_FAILED' }
  return $answer.result.value
}
function WaitFor($ws, $expression, $failureCode) {
  $until = [DateTime]::UtcNow.AddSeconds(20)
  while ([DateTime]::UtcNow -lt $until) {
    if (Evaluate $ws $expression) { return }
    Start-Sleep -Milliseconds 100
  }
  throw $failureCode
}
try {
  if (-not $fileUrl) { throw 'FILE_URL_MISSING' }
  if (-not $fileUrl.StartsWith('file://wsl.localhost/')) { throw 'UNC_FILE_URL_INVALID' }
  $step = 'chrome'
  if (-not (Test-Path -LiteralPath $chromePath)) { throw 'WINDOWS_CHROME_PATH_MISSING' }
  $tcp = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)
  $tcp.Start(); $port = $tcp.LocalEndpoint.Port; $tcp.Stop()
  $chromeArgs = @('--headless=new','--disable-gpu','--disable-dev-shm-usage','--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run','--no-default-browser-check','--disable-default-apps','--remote-debugging-address=127.0.0.1',("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
  $step = 'chrome_start'
  $proc = Start-Process -FilePath $chromePath -ArgumentList $chromeArgs -PassThru -WindowStyle Hidden
  [IO.File]::WriteAllText($pidMarker, [string]$proc.Id)
  $step = 'chrome_cdp'
  $version = $null
  for ($i = 0; $i -lt 150; $i++) {
    try { $version = Invoke-RestMethod -Uri ("http://127.0.0.1:$port/json/version") -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 100 }
  }
  if (-not $version) { throw 'CHROME_CDP_START_FAILED' }
  $step = 'target'
  $browserUri = [Uri]$version.webSocketDebuggerUrl
  if ($browserUri.Scheme -ne 'ws' -or $browserUri.Host -ne '127.0.0.1' -or $browserUri.Port -ne $port -or $browserUri.AbsolutePath -notmatch '^/devtools/browser/[A-Za-z0-9_-]+$') { throw 'BROWSER_CDP_ENDPOINT_INVALID' }
  $target = $null
  for ($i = 0; $i -lt 50; $i++) {
    try { $target = Invoke-RestMethod -Uri ("http://127.0.0.1:$port/json/new?about%3Ablank") -Method Put -TimeoutSec 2; break } catch { Start-Sleep -Milliseconds 100 }
  }
  if (-not $target) { throw 'PAGE_TARGET_CREATE_FAILED' }
  $step = 'cdp'
  $pageUri = [Uri]$target.webSocketDebuggerUrl
  if ($pageUri.Scheme -ne 'ws' -or $pageUri.Host -ne '127.0.0.1' -or $pageUri.Port -ne $port -or $pageUri.AbsolutePath -notmatch '^/devtools/page/[A-Za-z0-9_-]+$') { throw 'PAGE_CDP_ENDPOINT_INVALID' }
  $pageSocket = New-Object Net.WebSockets.ClientWebSocket
  $pageSocket.ConnectAsync($pageUri, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
  $null = Invoke-Cdp $pageSocket 'Page.enable' @{}
  $null = Invoke-Cdp $pageSocket 'Page.setDownloadBehavior' @{ behavior = 'allow'; downloadPath = $profile }
  $null = Invoke-Cdp $pageSocket 'Runtime.enable' @{}
  $null = Invoke-Cdp $pageSocket 'Network.enable' @{}
  $null = Invoke-Cdp $pageSocket 'Network.setBlockedURLs' @{ urls = @('http://*/*','https://*/*') }
  $step = 'navigate'
  $null = Invoke-Cdp $pageSocket 'Page.navigate' @{ url = $fileUrl }
  WaitFor $pageSocket "document.readyState === 'complete' && !!document.querySelector('#welcome') && !!document.querySelector('#question') && !!document.querySelector('#done')" 'PAGE_LOAD_TIMEOUT'
  $step = 'file_input'
  if (-not (Evaluate $pageSocket "document.querySelectorAll('input[type=file]').length === 0")) { throw 'FILE_INPUT_PRESENT' }
  $step = 'disclaimer'
  if (-not (Evaluate $pageSocket "document.querySelector('#welcome .notice').textContent.trim().length > 20")) { throw 'INFORMAL_ONLY_NOTICE_MISSING' }
  $step = 'rating_guidance'
  if (-not (Evaluate $pageSocket "document.querySelector('#rating-guide').textContent.includes('②は動画と見比べた答えの正しさ') && document.querySelector('#rating-guide').textContent.includes('③は説明が動画中の出来事を根拠として示しているか') && document.querySelector('#rating-guide').textContent.includes('必要な場合はその順序')")) { throw 'EVIDENCE_RATING_GUIDANCE_MISSING' }
  if (-not (Evaluate $pageSocket "document.querySelector('#rating-guide').textContent.includes('はっきりしないときは推測せず') && document.querySelector('#rating-guide').textContent.includes('②③は「判断できない」を選べます')")) { throw 'NO_GUESS_GUIDANCE_MISSING' }
  if (-not (Evaluate $pageSocket "document.querySelector('#question').textContent.includes('質問を読んで、何を答えればよいか分かりますか？')")) { throw 'QUESTION_CLARITY_PROMPT_MISSING' }
  if (-not (Evaluate $pageSocket "document.querySelector('#question').textContent.includes('動画を見て、説明に書かれた答えは正しいと思いますか？')")) { throw 'ANSWER_CORRECTNESS_PROMPT_MISSING' }
  $step = 'csp'
  if (-not (Evaluate $pageSocket 'document.querySelector("meta[http-equiv=Content-Security-Policy]").content.includes("connect-src " + String.fromCharCode(39) + "none" + String.fromCharCode(39))')) { throw 'CSP_NETWORK_DENIAL_MISSING' }
  $step = 'external_fetch'
  if (-not (Evaluate $pageSocket 'fetch("https://example.invalid/").then(()=>false,()=>true)')) { throw 'EXTERNAL_FETCH_NOT_DENIED' }
  $step = 'initial_state'
  if (-not (Evaluate $pageSocket "!document.querySelector('#welcome').classList.contains('hidden') && document.querySelector('#question').classList.contains('hidden') && document.querySelector('#done').classList.contains('hidden')")) { throw 'NICKNAME_FIRST_STATE_INVALID' }
  $step = 'flow'
  $step = 'save_control'
  if (-not (Evaluate $pageSocket "!!document.querySelector('#save')")) { throw 'SAVE_CONTROL_MISSING' }

  # All injected values are synthetic and are never returned from the page.
  $step = 'enter_nickname'
  $null = Evaluate $pageSocket "document.querySelector('#nickname').value='synthetic-reviewer'; document.querySelector('#start').click(); true"
  WaitFor $pageSocket "!document.querySelector('#welcome').classList.contains('hidden') === false && !document.querySelector('#question').classList.contains('hidden')" 'QUESTION_ONE_TIMEOUT'
  $step = 'video_metadata'
  WaitFor $pageSocket "document.querySelector('#video').readyState >= 1 && document.querySelector('#video').videoWidth > 0 && Number.isFinite(document.querySelector('#video').duration)" 'VIDEO_METADATA_TIMEOUT'
  $step = 'video_frame_decode'
  if (-not (Evaluate $pageSocket "(async()=>{const v=document.querySelector('#video');if(v.readyState<2){v.preload='auto';v.load();await Promise.race([new Promise(resolve=>v.addEventListener('loadeddata',()=>resolve(true),{once:true})),new Promise(resolve=>setTimeout(()=>resolve(false),5000))])}return v.readyState>=2&&v.videoWidth>0})()")) { throw 'VIDEO_FRAME_DECODE_FAILED' }
  for ($i = 0; $i -lt 3; $i++) {
    $step = 'choice_' + ($i + 1)
    $expected = $i + 1
    $seen = Evaluate $pageSocket "document.querySelector('#counter').textContent.trim().startsWith('$expected') && !document.querySelector('#question').classList.contains('hidden')"
    if (-not $seen) { throw ("QUESTION_SEQUENCE_INVALID_" + $expected) }
    if (-not (Evaluate $pageSocket "!document.querySelector('#source-language').open")) { throw 'SOURCE_LANGUAGE_NOT_RESET_ON_ENTRY' }
    $selected = Evaluate $pageSocket "(()=>{for(const name of ['question_clarity','answer_question_fit','video_evidence_sufficiency']){const radios=Array.from(document.querySelectorAll('input[name='+name+']')); if(radios.length!==3) return false; radios[$i].checked=true; radios[$i].dispatchEvent(new Event('change',{bubbles:true}));} return true})()"
    if (-not $selected) { throw 'CHOICE_CONTROL_INVALID' }
    if ($i -lt 2) { $null = Evaluate $pageSocket "document.querySelector('#source-language').open=true; true" }
    $null = Evaluate $pageSocket "document.querySelector('#next').click(); true"
    if ($i -lt 2) {
      $next = $i + 2
      WaitFor $pageSocket "document.querySelector('#counter').textContent.trim().startsWith('$next')" ("QUESTION_" + $next + '_TIMEOUT')
      if (-not (Evaluate $pageSocket "!document.querySelector('#source-language').open")) { throw 'SOURCE_LANGUAGE_DISCLOSURE_STALE' }
    }
  }
  WaitFor $pageSocket "document.querySelector('#question').classList.contains('hidden') && !document.querySelector('#done').classList.contains('hidden') && !!document.querySelector('#save')" 'DONE_STATE_TIMEOUT'
  $step = 'next_reviewer_control'
  if (-not (Evaluate $pageSocket "!!document.querySelector('#next-reviewer') && document.querySelector('#reviewer-count').textContent.startsWith('1')")) { throw 'NEXT_REVIEWER_CONTROL_MISSING' }
  $step = 'first_reviewer_summary'
  if (-not (Evaluate $pageSocket "document.querySelector('#saved').textContent === '' && !document.querySelector('#summary').classList.contains('hidden')")) { throw 'FIRST_REVIEWER_SUMMARY_INVALID' }
  $step = 'second_reviewer'
  $null = Evaluate $pageSocket "document.querySelector('#next-reviewer').click(); true"
  WaitFor $pageSocket "!document.querySelector('#welcome').classList.contains('hidden') && document.querySelector('#welcome-title').textContent.includes('2人目') && document.querySelector('#nickname').value === ''" 'SECOND_REVIEWER_NICKNAME_TIMEOUT'
  if (-not (Evaluate $pageSocket "!document.querySelector('#cancel-next').classList.contains('hidden')")) { throw 'CANCEL_NEXT_CONTROL_MISSING' }
  $null = Evaluate $pageSocket "document.querySelector('#cancel-next').click(); true"
  WaitFor $pageSocket "!document.querySelector('#done').classList.contains('hidden') && document.querySelector('#reviewer-count').textContent.startsWith('1')" 'RETURN_TO_COMPLETION_TIMEOUT'
  $null = Evaluate $pageSocket "document.querySelector('#next-reviewer').click(); true"
  WaitFor $pageSocket "!document.querySelector('#welcome').classList.contains('hidden') && document.querySelector('#welcome-title').textContent.includes('2人目') && document.querySelector('#nickname').value === ''" 'SECOND_REVIEWER_NICKNAME_RETRY_TIMEOUT'
  $null = Evaluate $pageSocket "document.querySelector('#nickname').value='synthetic-reviewer-2'; document.querySelector('#start').click(); true"
  WaitFor $pageSocket "!document.querySelector('#question').classList.contains('hidden') && document.querySelector('#counter').textContent.trim().startsWith('1') && document.querySelector('#who').textContent.includes('synthetic-reviewer-2')" 'SECOND_REVIEWER_QUESTION_TIMEOUT'
  if (-not (Evaluate $pageSocket "Array.from(document.querySelectorAll('input[type=radio]')).every(x=>!x.checked)")) { throw 'SECOND_REVIEWER_STALE_CHOICES' }
  for ($i = 0; $i -lt 3; $i++) {
    $expected = $i + 1
    $selected = Evaluate $pageSocket "(()=>{for(const name of ['question_clarity','answer_question_fit','video_evidence_sufficiency']){const radios=Array.from(document.querySelectorAll('input[name='+name+']')); if(radios.length!==3) return false; radios[$i].checked=true;} return true})()"
    if (-not $selected) { throw 'SECOND_REVIEWER_CHOICE_INVALID' }
    $null = Evaluate $pageSocket "document.querySelector('#next').click(); true"
    if ($i -lt 2) { $next = $i + 2; WaitFor $pageSocket "document.querySelector('#counter').textContent.trim().startsWith('$next')" ("SECOND_REVIEWER_QUESTION_" + $next + '_TIMEOUT') }
  }
  if (-not (Evaluate $pageSocket "document.querySelector('#reviewer-count').textContent.startsWith('2') && !!document.querySelector('#save')")) { throw 'MULTI_REVIEWER_COUNT_INVALID' }
  $step = 'third_reviewer'
  $null = Evaluate $pageSocket "document.querySelector('#next-reviewer').click(); true"
  WaitFor $pageSocket "!document.querySelector('#welcome').classList.contains('hidden') && document.querySelector('#welcome-title').textContent.includes('3人目') && document.querySelector('#nickname').value === ''" 'THIRD_REVIEWER_NICKNAME_TIMEOUT'
  $null = Evaluate $pageSocket "document.querySelector('#nickname').value='synthetic-reviewer-3'; document.querySelector('#start').click(); true"
  WaitFor $pageSocket "!document.querySelector('#question').classList.contains('hidden') && document.querySelector('#counter').textContent.trim().startsWith('1') && document.querySelector('#who').textContent.includes('synthetic-reviewer-3')" 'THIRD_REVIEWER_QUESTION_TIMEOUT'
  if (-not (Evaluate $pageSocket "Array.from(document.querySelectorAll('input[type=radio]')).every(x=>!x.checked)")) { throw 'THIRD_REVIEWER_STALE_CHOICES' }
  for ($i = 0; $i -lt 3; $i++) {
    $choiceIndex = if ($i -eq 0) { "name==='question_clarity'?1:2" } else { '0' }
    $selected = Evaluate $pageSocket "(()=>{for(const name of ['question_clarity','answer_question_fit','video_evidence_sufficiency']){const radios=Array.from(document.querySelectorAll('input[name='+name+']')); if(radios.length!==3) return false; const choice=$choiceIndex; radios[choice].checked=true;} return true})()"
    if (-not $selected) { throw 'THIRD_REVIEWER_CHOICE_INVALID' }
    $null = Evaluate $pageSocket "document.querySelector('#next').click(); true"
    if ($i -lt 2) { $next = $i + 2; WaitFor $pageSocket "document.querySelector('#counter').textContent.trim().startsWith('$next')" ("THIRD_REVIEWER_QUESTION_" + $next + '_TIMEOUT') }
  }
  if (-not (Evaluate $pageSocket "document.querySelector('#reviewer-count').textContent.startsWith('3') && !!document.querySelector('#save')")) { throw 'THREE_REVIEWER_COUNT_INVALID' }
  $step = 'combined_save'
  $null = Evaluate $pageSocket "document.querySelector('#save').click(); true"
  $download = $null
  $until = [DateTime]::UtcNow.AddSeconds(15)
  while ([DateTime]::UtcNow -lt $until) {
    $download = Get-ChildItem -LiteralPath $profile -Filter 'exploratory_video_review_*.json' -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($download) { break }
    Start-Sleep -Milliseconds 100
  }
  if (-not $download) { throw 'COMBINED_JSON_DOWNLOAD_MISSING' }
  $payload = [IO.File]::ReadAllText($download.FullName) | ConvertFrom-Json
  $expectedIds = [string[]](ConvertFrom-Json ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('__EXPECTED_IDS_B64__'))))
  $rootKeys = @($payload.PSObject.Properties.Name | Sort-Object) -join ','
  if ($rootKeys -cne 'artifact_fingerprint,created_at_utc,limitations,purpose,respondents,schema_version,status' -or $payload.schema_version -ne 6 -or $payload.purpose -ne 'exploratory_human_impression_check' -or $payload.status -ne 'informal_observations_only' -or $payload.respondents.Count -ne 3) { throw 'COMBINED_JSON_SCHEMA_INVALID' }
  if ($payload.artifact_fingerprint -cne '__EXPECTED_FINGERPRINT__') { throw 'ARTIFACT_FINGERPRINT_EXPORT_MISMATCH' }
  if ($payload.respondents[0].nickname -cne 'synthetic-reviewer' -or $payload.respondents[1].nickname -cne 'synthetic-reviewer-2' -or $payload.respondents[2].nickname -cne 'synthetic-reviewer-3') { throw 'COMBINED_JSON_REVIEWER_ORDER_INVALID' }
  foreach ($respondent in $payload.respondents) {
    $respondentKeys = @($respondent.PSObject.Properties.Name | Sort-Object) -join ','
    if ($respondentKeys -cne 'answers,nickname' -or $respondent.answers.Count -ne 3) { throw 'COMBINED_JSON_ANSWER_COUNT_INVALID' }
    $actualIds = @()
    $answerIndex = 1
    foreach ($answer in $respondent.answers) {
      $answerKeys = @($answer.PSObject.Properties.Name | Sort-Object) -join ','
      if ($answerKeys -cne 'answer_question_fit,item_id,item_number,question_clarity,video_evidence_sufficiency' -or $answer.item_number -ne $answerIndex -or -not $answer.item_id -or $answer.question_clarity -notin @('clear','unclear','cannot_tell') -or $answer.answer_question_fit -notin @('fits','does_not_fit','cannot_tell') -or $answer.video_evidence_sufficiency -notin @('sufficient','insufficient','cannot_tell')) { throw 'COMBINED_JSON_JUDGMENT_INVALID' }
      $actualIds += [string]$answer.item_id
      $answerIndex++
    }
    if (($actualIds -join ',') -cne ($expectedIds -join ',')) { throw 'COMBINED_JSON_ITEM_ID_ORDER_INVALID' }
  }
  $abstentionAnswer = $payload.respondents[2].answers[0]
  if ($abstentionAnswer.question_clarity -cne 'unclear' -or $abstentionAnswer.answer_question_fit -cne 'cannot_tell' -or $abstentionAnswer.video_evidence_sufficiency -cne 'cannot_tell') { throw 'NO_GUESS_JUDGMENT_NOT_PERSISTED' }
  if (-not (Evaluate $pageSocket "document.querySelector('#saved').textContent.trim().startsWith('3')")) { throw 'COMBINED_SAVE_CONFIRMATION_MISSING' }
  Write-Output 'E2E_PASS: three sequential reviewers; source-language disclosure resets between items; third reviewer used unclear/cannot-tell and abstentions persisted in one combined schema-v6 JSON'
} catch {
  $code = [string]$_
  if ($code -notmatch '^[A-Z0-9_]+$') {
    $typeName = $_.Exception.GetType().Name
    if ($typeName -match '^[A-Za-z]+$') {
      $hr = $_.Exception.HResult.ToString('X8')
      $code = ('PS_' + $step.ToUpperInvariant() + '_' + $typeName.ToUpperInvariant() + '_' + $hr)
    }
    else { $code = 'BROWSER_E2E_FAILED' }
  }
  elseif ($step -and $code -notmatch '^PS_') { $code = $step.ToUpperInvariant() + '_' + $code }
  Write-Output ('E2E_FAIL: ' + $code)
  exit 1
} finally {
  try {
    if ($pageSocket -and $pageSocket.State -eq [Net.WebSockets.WebSocketState]::Open) { $pageSocket.Dispose() }
  } catch { }
  try {
    if ($version -and $version.webSocketDebuggerUrl) {
      $browserSocket = New-Object Net.WebSockets.ClientWebSocket
      $browserSocket.ConnectAsync($browserUri, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
      $script:cdpId = 100000
      $null = Invoke-Cdp $browserSocket 'Browser.close' @{}
    }
  } catch { }
  try { if ($browserSocket -and $browserSocket.State -eq [Net.WebSockets.WebSocketState]::Open) { $browserSocket.Dispose() } } catch { }
  if ($proc) {
    try { $proc.Refresh(); if (-not $proc.HasExited) { & "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $proc.Id | Out-Null; $proc.WaitForExit(5000) | Out-Null } } catch { }
  }
  for ($i = 0; $i -lt 20 -and (Test-Path -LiteralPath $profile); $i++) { try { Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction Stop } catch { Start-Sleep -Milliseconds 100 } }
  Remove-Item -LiteralPath $pidMarker -Force -ErrorAction SilentlyContinue
  if (Test-Path -LiteralPath $profile) { Write-Output 'E2E_FAIL: PROFILE_CLEANUP_FAILED'; exit 1 }
}
'''

CLEANUP_POWERSHELL = r'''
$ErrorActionPreference = 'SilentlyContinue'
$runId = '__RUN_ID__'
$chromePath = __CHROME_PATH__
$profile = Join-Path ([IO.Path]::GetTempPath()) ('quick-review-e2e-' + $runId)
$pidMarker = $profile + '.pid'
if (Test-Path -LiteralPath $pidMarker) {
  $pidText = [IO.File]::ReadAllText($pidMarker).Trim()
  if ($pidText -match '^[0-9]+$') {
    $target = Get-CimInstance Win32_Process -Filter ("ProcessId = " + [int]$pidText)
    if ($target -and $target.CommandLine -and $target.CommandLine.Contains($profile) -and $target.ExecutablePath -and ([IO.Path]::GetFullPath($target.ExecutablePath) -ieq [IO.Path]::GetFullPath($chromePath))) {
      & "$env:SystemRoot\System32\taskkill.exe" /T /F /PID ([int]$pidText) | Out-Null
    }
  }
}
for ($i = 0; $i -lt 30 -and (Test-Path -LiteralPath $profile); $i++) {
  Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction SilentlyContinue
  Start-Sleep -Milliseconds 100
}
Remove-Item -LiteralPath $pidMarker -Force -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $profile) { Write-Output 'CLEANUP_FAIL' } else { Write-Output 'CLEANUP_PASS' }
'''


def _cleanup_windows_profile(powershell: str, run_id: str, chrome_path: str) -> bool:
    """Kill only the Chrome PID whose image and command line match this profile."""
    chrome_b64 = base64.b64encode(chrome_path.encode("utf-8")).decode("ascii")
    source = CLEANUP_POWERSHELL.replace(
        "__RUN_ID__", run_id).replace(
            "__CHROME_PATH__",
            f"[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{chrome_b64}'))")
    encoded = base64.b64encode(source.encode("utf-16le")).decode("ascii")
    try:
        cleanup = subprocess.run(
            [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-OutputFormat",
             "Text", "-EncodedCommand", encoded], capture_output=True,
            timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return cleanup.returncode == 0 and b"CLEANUP_PASS" in cleanup.stdout + cleanup.stderr


class QuickReviewArtifactBrowserE2ETests(unittest.TestCase):
    def test_windows_chrome_wsl_unc_three_reviewer_combined_json_download(self):
        chrome_text = os.environ.get("FORMAL_REVIEW_CHROME")
        chrome = Path(chrome_text) if chrome_text else DEFAULT_CHROME
        if not chrome.is_file():
            self.skipTest("Windows Chrome unavailable")
        if not str(chrome).lower().endswith(".exe"):
            self.skipTest("this E2E specifically requires Windows Chrome through WSL UNC")
        powershell = shutil.which("powershell.exe") or shutil.which("powershell")
        wslpath = shutil.which("wslpath")
        if not powershell or not wslpath:
            self.skipTest("Windows PowerShell or wslpath unavailable")
        ffmpeg = shutil.which("ffmpeg")
        artifact_override = os.environ.get("QUICK_REVIEW_ACTUAL_ARTIFACT")
        if not ffmpeg and not artifact_override:
            self.skipTest("ffmpeg unavailable for synthetic video fixture")
        with tempfile.TemporaryDirectory(prefix="quick_review_e2e_synthetic_",
                                         dir=PRIVATE_ROOT) as temp_name:
            temp_root = Path(temp_name)
            if artifact_override:
                artifact_input = Path(artifact_override)
                if artifact_input.is_symlink():
                    self.fail("actual review artifact symlinks are not accepted")
                artifact = artifact_input.resolve(strict=True)
                try:
                    artifact.relative_to(PRIVATE_ROOT.resolve())
                except ValueError:
                    self.fail("actual review artifact must stay under the private data directory")
                if artifact.is_symlink() or not artifact.is_file():
                    self.fail("actual review artifact must be a regular local file")
            else:
                package = temp_root / "synthetic_package"
                package.mkdir(mode=0o700)
                records = []
                for index in range(3):
                    records.append({
                        "review_token": f"synthetic-token-{index + 1}",
                        "question": f"Synthetic source question {index + 1}?",
                        "claim_text": f"Synthetic source claim {index + 1}.",
                    })
                (package / "index.html").write_text(
                    '<script id="data" type="application/json">'
                    + json.dumps({"records": records}) + "</script>", encoding="utf-8")
                os.chmod(package / "index.html", 0o600)
                video_path = package / "synthetic.mp4"
                try:
                    subprocess.run(
                        [ffmpeg, "-v", "error", "-f", "lavfi", "-i",
                         "color=c=red:s=16x16:d=0.2", "-frames:v", "1", "-an",
                         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags",
                         "+faststart", str(video_path)],
                        capture_output=True, timeout=15, check=True)
                except (OSError, subprocess.SubprocessError):
                    self.skipTest("could not create synthetic video fixture")
                for index in range(3):
                    destination = package / f"synthetic-token-{index + 1}.mp4"
                    destination.write_bytes(video_path.read_bytes())
                    os.chmod(destination, 0o600)
                video_path.unlink()
                translations = {
                    translation_key(row["question"], row["claim_text"]):
                        (f"Synthetic question {i + 1}?", f"Synthetic claim {i + 1}.")
                    for i, row in enumerate(records)
                }
                artifact = temp_root / "synthetic_quick_review.html"
                with redirect_stdout(io.StringIO()):
                    build(package, artifact, translations=translations)
            expected_ids = expected_item_ids(artifact)
            expected_fingerprint = expected_artifact_fingerprint(artifact)
            if not expected_fingerprint:
                self.fail("review artifact fingerprint missing")
            artifact_stat = artifact.stat()
            if (not stat.S_ISREG(artifact_stat.st_mode)
                    or stat.S_IMODE(artifact_stat.st_mode) != 0o600):
                self.fail("review artifact must be owner-only")
            try:
                translated = subprocess.run(
                    [wslpath, "-w", str(artifact)], capture_output=True,
                    text=True, timeout=5, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                self.skipTest("could not translate synthetic artifact path to WSL UNC")
            if not translated.startswith("\\\\wsl.localhost\\"):
                self.skipTest("wslpath did not return the expected WSL UNC path")
            try:
                chrome_windows_path = subprocess.run(
                    [wslpath, "-w", str(chrome)], capture_output=True,
                    text=True, timeout=5, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                self.skipTest("could not translate Windows Chrome executable path")
            if not re.match(r"^[A-Za-z]:\\", chrome_windows_path):
                self.skipTest("Chrome path did not translate to a Windows drive path")
            unc_uri_path = translated.replace("\\", "/")
            file_url = "file:" + unc_uri_path
            if not file_url.startswith("file://wsl.localhost/"):
                self.fail("synthetic artifact path is not a WSL UNC file URL")
            # WSL interop does not reliably forward arbitrary environment
            # variables. Pass only synthetic fixture paths in encoded source.
            chrome_b64 = base64.b64encode(chrome_windows_path.encode("utf-8")).decode("ascii")
            file_b64 = base64.b64encode(file_url.encode("utf-8")).decode("ascii")
            ps_source = POWERSHELL_CDP.replace(
                "__CHROME_PATH__",
                f"[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{chrome_b64}'))",
            ).replace(
                "__FILE_URL__",
                f"[Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('{file_b64}'))",
            ).replace(
                "__EXPECTED_IDS_B64__",
                base64.b64encode(json.dumps(expected_ids).encode("utf-8")).decode("ascii"),
            ).replace(
                "__EXPECTED_FINGERPRINT__", expected_fingerprint,
            )
            run_id = uuid.uuid4().hex
            ps_source = ps_source.replace("__RUN_ID__", run_id)
            ps_file = temp_root / "browser_e2e.ps1"
            ps_file.write_bytes(b"\xef\xbb\xbf" + ps_source.encode("utf-8"))
            os.chmod(ps_file, 0o600)
            try:
                ps_file_windows_path = subprocess.run(
                    [wslpath, "-w", str(ps_file)], capture_output=True,
                    text=True, timeout=5, check=True).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                self.skipTest("could not translate synthetic PowerShell script path")
            if not ps_file_windows_path.startswith("\\\\wsl.localhost\\"):
                self.skipTest("PowerShell script path did not translate to WSL UNC")
            ps_file_windows_path = ps_file_windows_path.replace("'", "''")
            bootstrap = (
                f"$scriptText=[IO.File]::ReadAllText('{ps_file_windows_path}'); "
                "Invoke-Expression $scriptText")
            browser_process = None
            run = None
            timed_out = False
            try:
                browser_process = subprocess.Popen(
                    [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", bootstrap],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    env=os.environ.copy())
                try:
                    stdout_raw, stderr_raw = browser_process.communicate(timeout=100)
                    run = subprocess.CompletedProcess(
                        browser_process.args, browser_process.returncode,
                        stdout_raw.decode("utf-8", "replace"),
                        stderr_raw.decode("utf-8", "replace"))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    browser_process.kill()
                    try:
                        browser_process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            except OSError:
                self.fail("Windows PowerShell could not start")
            finally:
                if not _cleanup_windows_profile(powershell, run_id, chrome_windows_path):
                    self.fail("exact Chrome process/profile cleanup failed")
            if timed_out:
                self.fail("browser E2E exceeded its 100-second outer bound; exact-profile cleanup ran")
            if run is None:
                self.fail("browser E2E did not return a result")
            output = run.stdout + run.stderr
            if "E2E_PASS:" in output and run.returncode == 0:
                return
            if os.environ.get("QUICK_REVIEW_E2E_DEBUG") == "1":
                print("E2E_DEBUG_RETURN_CODE:", run.returncode)
                print("E2E_DEBUG_STDOUT:", run.stdout[-1200:])
                print("E2E_DEBUG_STDERR:", run.stderr[-1200:])
            code = next((line.split(":", 1)[1].strip() for line in output.splitlines()
                         if line.startswith("E2E_FAIL:")), "WINDOWS_BROWSER_UNAVAILABLE")
            if run.returncode == 77:
                self.skipTest(code)
            self.fail(code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
