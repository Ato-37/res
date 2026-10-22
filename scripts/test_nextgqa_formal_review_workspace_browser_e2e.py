"""Synthetic Chrome/CDP E2E for the local formal-review workspace.

Runs only the existing synthetic Stage-1/Stage-2 fixtures. No private bundle or
real review input is opened. Windows Chrome is driven from Windows PowerShell so
CDP stays on Windows loopback; Linux Chrome uses Node 24's built-in WebSocket.
Unavailable runtime/interoperability produces an explicit unittest skip.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest

import test_nextgqa_formal_review_workspace as fixture_module
from nextgqa_formal_review_workspace import WorkspaceServer


DEFAULT_CHROME = Path(
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe")

# Node 24's built-in WebSocket is used directly; no npm packages are needed.
NODE_PAGE_CDP_VALIDATOR = r'''function validatePageWebSocketUrl(endpoint, expectedPort) {
  let parsed;
  try { parsed = new URL(endpoint); } catch (_) { throw new Error('invalid CDP websocket URL'); }
  if (parsed.protocol !== 'ws:' || parsed.hostname !== '127.0.0.1' ||
      parsed.port !== String(expectedPort) ||
      !/^\/devtools\/page\/[A-Za-z0-9_-]+$/.test(parsed.pathname) ||
      parsed.username || parsed.password || parsed.search || parsed.hash) {
    throw new Error('CDP websocket endpoint is not the expected loopback page endpoint');
  }
  return parsed;
}
'''

NODE_CDP = NODE_PAGE_CDP_VALIDATOR + r'''const fs = require('node:fs');
const path = require('node:path');
const {spawn} = require('node:child_process');
const chromePath = process.argv[1];
const profileForWindows = process.argv[2];
const profileForNode = process.argv[3];
const workspaceUrl = process.argv[4];
let chrome;
let socket;
let nextId = 0;
const pending = new Map();
function wait(ms) { return new Promise(resolve => setTimeout(resolve, ms)); }
async function cdp(method, params={}) {
  const id = ++nextId;
  socket.send(JSON.stringify({id, method, params}));
  return await new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`CDP timeout: ${method}`)); }, 10000);
    pending.set(id, {resolve, reject, timer});
  });
}
async function evaluate(expression) {
  const answer = await cdp('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true});
  if (answer.exceptionDetails) throw new Error('browser evaluation failed');
  return answer.result && answer.result.value;
}
async function until(expression, label) {
  const end = Date.now() + 15000;
  while (Date.now() < end) {
    if (await evaluate(expression)) return;
    await wait(100);
  }
  throw new Error(`UI condition timed out: ${label}`);
}
async function main() {
  chrome = spawn(chromePath, [
    '--headless=new', '--disable-gpu', '--disable-dev-shm-usage',
    '--no-first-run', '--no-default-browser-check', '--no-zygote',
    '--remote-debugging-address=127.0.0.1',
    '--remote-debugging-port=0', `--user-data-dir=${profileForWindows}`, 'about:blank'
  ], {stdio:'ignore', windowsHide:true});
  // Node needs the WSL path to read the file; Windows Chrome needs the
  // translated path to create it. They differ for a Windows executable.
  const activePort = path.join(profileForNode, 'DevToolsActivePort');
  let active = null;
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline && chrome.exitCode === null) {
    try { active = fs.readFileSync(activePort, 'utf8').trim().split(/\r?\n/); break; }
    catch (_) { await wait(100); }
  }
  if (!active || !/^\d+$/.test(active[0])) {
    console.log('E2E_UNAVAILABLE: Chrome did not expose a CDP port');
    process.exitCode = 77;
    return;
  }
  const port = Number(active[0]);
  let target;
  try {
    const response = await fetch(`http://127.0.0.1:${port}/json/new?${encodeURIComponent('about:blank')}`, {method:'PUT'});
    if (!response.ok) throw new Error('CDP target creation failed');
    target = await response.json();
  } catch (_) {
    console.log('E2E_UNAVAILABLE: Chrome CDP endpoint is not reachable from WSL');
    process.exitCode = 77;
    return;
  }
  const pageWebSocketUrl = validatePageWebSocketUrl(target.webSocketDebuggerUrl, port);
  socket = new WebSocket(pageWebSocketUrl.href);
  await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('CDP websocket timeout')), 10000);
    socket.addEventListener('open', () => { clearTimeout(timer); resolve(); }, {once:true});
    socket.addEventListener('error', () => { clearTimeout(timer); reject(new Error('CDP websocket failed')); }, {once:true});
  });
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) return;
    const item = pending.get(message.id); pending.delete(message.id); clearTimeout(item.timer);
    if (message.error) item.reject(new Error(message.error.message));
    else item.resolve(message.result || {});
  });
  await cdp('Page.enable');
  await cdp('Runtime.enable');
  await cdp('Page.navigate', {url:workspaceUrl});
  await until("document.readyState === 'complete' && !!document.querySelector('#s1')", 'workspace page loaded');

  await evaluate("document.querySelector('#s1').click()");
  await until("document.querySelector('#s1out').textContent.includes('stage1_structurally_valid_non_authorizing') && !document.querySelector('#s2').disabled", 'Stage 1 passes visibly');
  const stage1Text = await evaluate("document.querySelector('#s1out').textContent");
  const stage1Report = JSON.parse(stage1Text);
  if (stage1Report.status !== 'stage1_structurally_valid_non_authorizing' ||
      stage1Report.model_load_permitted !== false || stage1Report.score_ready !== false ||
      stage1Report.execution_authorization !== 'not_granted' ||
      stage1Report.authorized_forwards !== 0 ||
      stage1Report.reviewer_identity_verified !== false ||
      stage1Report.semantic_review_authenticated !== false) {
    throw new Error('Stage 1 status or denial fields do not match the contract');
  }

  await evaluate("document.querySelector('#s2').click()");
  await until("document.querySelector('#s2out').textContent.includes('stage2_receipt_and_attestation_verified_non_authorizing')", 'Stage 2 passes visibly');
  const stage2Text = await evaluate("document.querySelector('#s2out').textContent");
  const stage2Report = JSON.parse(stage2Text);
  if (stage2Report.status !== 'stage2_receipt_and_attestation_verified_non_authorizing' ||
      stage2Report.reviewer_key_signature_valid !== true ||
      stage2Report.reviewer_identity_verified !== false ||
      stage2Report.semantic_judgment_authenticated !== false ||
      stage2Report.model_load_permitted !== false || stage2Report.score_ready !== false ||
      stage2Report.execution_authorization !== 'not_granted' ||
      stage2Report.authorized_forwards !== 0) {
    throw new Error('Stage 2 status or denial fields do not match the contract');
  }
  const denial = await evaluate("document.querySelector('#deny').textContent");
  if (!denial.includes('model_load_permitted=false') || !denial.includes('authorized_forwards=0')) {
    throw new Error('fixed denial banner is not visible');
  }

  await evaluate("document.querySelector('#reset').click()");
  await until("document.querySelector('#s1out').textContent === '未検証' && document.querySelector('#s2out').textContent === 'Stage 1検証後に利用できます' && document.querySelector('#s2').disabled", 'reset clears visible state');
  const afterReset = await evaluate("JSON.stringify({s1:document.querySelector('#s1out').textContent,s2:document.querySelector('#s2out').textContent,deny:document.querySelector('#deny').textContent,failure:document.querySelector('#failure').textContent})");
  const state = JSON.parse(afterReset);
  if (!state.deny.includes('authorized_forwards=0') || !state.deny.includes('execution_authorization=not_granted')) {
    throw new Error('reset removed fixed denial state');
  }
  console.log('E2E_PASS: Stage 1, Stage 2, visible denial fields, and reset verified in Chrome');
}
main().catch(error => { console.error(error.stack || String(error)); process.exitCode = 1; })
  .finally(async () => {
    try { if (socket) socket.close(); } catch (_) {}
    if (chrome && chrome.pid) {
      try {
        if (chrome.exitCode === null) {
          chrome.kill('SIGTERM');
          await Promise.race([
            new Promise(resolve => chrome.once('close', resolve)),
            wait(3000)
          ]);
          if (chrome.exitCode === null) {
            chrome.kill('SIGKILL');
            await Promise.race([
              new Promise(resolve => chrome.once('close', resolve)),
              wait(1000)
            ]);
          }
        }
      } catch (_) {}
    }
  });
'''


def _terminate_process_group(process: subprocess.Popen, *, grace_seconds: float = 2.0) -> None:
    """Stop only the new session/process group owned by this Popen instance."""
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def _run_node_driver(command: list[str], *, profile: Path,
                     timeout: float) -> subprocess.CompletedProcess:
    """Run the Node/Chrome tree in its own group and always remove its profile."""
    process = None
    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_group(process)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                stdout, stderr = "", ""
            raise subprocess.TimeoutExpired(
                command, timeout, output=stdout, stderr=stderr) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        if process is not None and process.poll() is None:
            _terminate_process_group(process)
        shutil.rmtree(profile, ignore_errors=True)

# Windows Chrome's DevTools HTTP endpoint is not consistently reachable from
# WSL. When Chrome is a Windows executable, launch it and run CDP entirely on
# the Windows side with PowerShell/.NET. Inputs arrive as environment values so
# they are never interpolated into PowerShell source.
POWERSHELL_CDP = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$chromePath = $env:FORMAL_REVIEW_E2E_CHROME
$workspaceUrl = $env:FORMAL_REVIEW_E2E_URL
$profile = Join-Path ([IO.Path]::GetTempPath()) ('formal-workspace-e2e-' + [Guid]::NewGuid().ToString('N'))
$proc = $null
$pageSocket = $null
$browserSocket = $null
function Invoke-Cdp($ws, $method, $params) {
  $script:cdpId++
  $id = $script:cdpId
  $payload = @{ id = $id; method = $method; params = $params } | ConvertTo-Json -Compress -Depth 50
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
      if ($result.MessageType -eq [Net.WebSockets.WebSocketMessageType]::Close) { throw 'CDP websocket closed' }
      $stream.Write($buffer, 0, $result.Count)
    } while (-not $result.EndOfMessage)
    $message = [Text.Encoding]::UTF8.GetString($stream.ToArray()) | ConvertFrom-Json
    if ($message.id -ne $id) { continue }
    if ($message.error) { throw ('CDP error: ' + $message.error.message) }
    return $message.result
  }
}
function Evaluate($ws, $expression) {
  $answer = Invoke-Cdp $ws 'Runtime.evaluate' @{ expression = $expression; returnByValue = $true; awaitPromise = $true }
  if ($answer.exceptionDetails) { throw 'Browser evaluation failed' }
  return $answer.result.value
}
function WaitFor($ws, $expression, $label) {
  $until = [DateTime]::UtcNow.AddSeconds(15)
  while ([DateTime]::UtcNow -lt $until) {
    if (Evaluate $ws $expression) { return }
    Start-Sleep -Milliseconds 100
  }
  throw ('UI condition timed out: ' + $label)
}
try {
  # Chrome must fetch the tokenized workspace from Windows via the explicit
  # WSL localhost-forwarding address. Never broaden either server binding.
  try { $null = Invoke-WebRequest -UseBasicParsing -Uri $workspaceUrl -TimeoutSec 3 }
  catch { Write-Output 'E2E_UNAVAILABLE: Windows cannot reach the 127.0.0.1 workspace URL'; exit 77 }
  $tcp = New-Object Net.Sockets.TcpListener([Net.IPAddress]::Loopback, 0)
  $tcp.Start(); $port = $tcp.LocalEndpoint.Port; $tcp.Stop()
  $args = @('--headless=new','--disable-gpu','--disable-dev-shm-usage','--no-first-run','--no-default-browser-check','--remote-debugging-address=127.0.0.1',("--remote-debugging-port=$port"),("--user-data-dir=$profile"),'about:blank')
  $proc = Start-Process -FilePath $chromePath -ArgumentList $args -PassThru -WindowStyle Hidden
  $version = $null
  for ($i = 0; $i -lt 150; $i++) {
    try { $version = Invoke-RestMethod -Uri ("http://127.0.0.1:$port/json/version") -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 100 }
  }
  if (-not $version) { Write-Output 'E2E_UNAVAILABLE: Windows Chrome CDP did not start'; exit 77 }
  $target = $null
  for ($i = 0; $i -lt 50; $i++) {
    try { $target = Invoke-RestMethod -Uri ("http://127.0.0.1:$port/json/new?about%3Ablank") -Method Put -TimeoutSec 2; break } catch { Start-Sleep -Milliseconds 100 }
  }
  if (-not $target) { throw 'Windows-local CDP target creation failed' }
  $wsUri = [Uri]$target.webSocketDebuggerUrl
  if ($wsUri.Scheme -ne 'ws' -or $wsUri.Host -ne '127.0.0.1' -or $wsUri.Port -ne $port -or $wsUri.AbsolutePath -notmatch '^/devtools/page/[A-Za-z0-9_-]+$') { throw 'Chrome returned a non-loopback or unexpected page CDP endpoint' }
  $pageSocket = New-Object Net.WebSockets.ClientWebSocket
  $pageSocket.ConnectAsync($wsUri, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
  $script:cdpId = 0
  $null = Invoke-Cdp $pageSocket 'Page.enable' @{}
  $null = Invoke-Cdp $pageSocket 'Runtime.enable' @{}
  $null = Invoke-Cdp $pageSocket 'Page.navigate' @{ url = $workspaceUrl }
  WaitFor $pageSocket "document.readyState === 'complete' && !!document.querySelector('#s1')" 'workspace page loaded'
  $null = Evaluate $pageSocket "document.querySelector('#s1').click()"
  WaitFor $pageSocket "document.querySelector('#s1out').textContent.includes('stage1_structurally_valid_non_authorizing') && !document.querySelector('#s2').disabled" 'Stage 1 passes visibly'
  $r1 = (Evaluate $pageSocket "document.querySelector('#s1out').textContent") | ConvertFrom-Json
  if ($r1.status -ne 'stage1_structurally_valid_non_authorizing' -or $r1.model_load_permitted -ne $false -or $r1.score_ready -ne $false -or $r1.execution_authorization -ne 'not_granted' -or $r1.authorized_forwards -ne 0 -or $r1.reviewer_identity_verified -ne $false -or $r1.semantic_review_authenticated -ne $false) { throw 'Stage 1 status/denial contract mismatch' }
  $null = Evaluate $pageSocket "document.querySelector('#s2').click()"
  WaitFor $pageSocket "document.querySelector('#s2out').textContent.includes('stage2_receipt_and_attestation_verified_non_authorizing')" 'Stage 2 passes visibly'
  $r2 = (Evaluate $pageSocket "document.querySelector('#s2out').textContent") | ConvertFrom-Json
  if ($r2.status -ne 'stage2_receipt_and_attestation_verified_non_authorizing' -or $r2.reviewer_key_signature_valid -ne $true -or $r2.reviewer_identity_verified -ne $false -or $r2.semantic_judgment_authenticated -ne $false -or $r2.model_load_permitted -ne $false -or $r2.score_ready -ne $false -or $r2.execution_authorization -ne 'not_granted' -or $r2.authorized_forwards -ne 0) { throw 'Stage 2 status/denial contract mismatch' }
  $deny = Evaluate $pageSocket "document.querySelector('#deny').textContent"
  if (-not $deny.Contains('model_load_permitted=false') -or -not $deny.Contains('authorized_forwards=0')) { throw 'Fixed denial banner missing' }
  $null = Evaluate $pageSocket "document.querySelector('#reset').click()"
  WaitFor $pageSocket "document.querySelector('#s1out').textContent === '未検証' && document.querySelector('#s2out').textContent === 'Stage 1検証後に利用できます' && document.querySelector('#s2').disabled" 'reset clears visible state'
  $state = (Evaluate $pageSocket "JSON.stringify({deny:document.querySelector('#deny').textContent})") | ConvertFrom-Json
  if (-not $state.deny.Contains('authorized_forwards=0') -or -not $state.deny.Contains('execution_authorization=not_granted')) { throw 'Reset removed fixed denial state' }
  Write-Output 'E2E_PASS: Stage 1, Stage 2, visible denial fields, and reset verified in Chrome'
} catch {
  Write-Error $_
  exit 1
} finally {
  # Browser.close is issued on both the success and failure paths whenever the
  # browser endpoint became available; taskkill is a final orphan safeguard.
  try {
    if ($version -and $version.webSocketDebuggerUrl) {
      $browserSocket = New-Object Net.WebSockets.ClientWebSocket
      $browserUri = [Uri]$version.webSocketDebuggerUrl
      if ($browserUri.Scheme -ne 'ws' -or $browserUri.Host -ne '127.0.0.1' -or $browserUri.Port -ne $port -or $browserUri.AbsolutePath -notmatch '^/devtools/browser/[A-Za-z0-9_-]+$') { throw 'Chrome returned a non-loopback or unexpected browser CDP endpoint' }
      $browserSocket.ConnectAsync($browserUri, [Threading.CancellationToken]::None).GetAwaiter().GetResult()
      $script:cdpId = 100000
      $null = Invoke-Cdp $browserSocket 'Browser.close' @{}
    }
  } catch { }
  foreach ($ws in @($pageSocket,$browserSocket)) { try { if ($ws -and $ws.State -eq [Net.WebSockets.WebSocketState]::Open) { $ws.Dispose() } } catch { } }
  if ($proc) {
    try { $proc.Refresh(); if (-not $proc.HasExited) { & "$env:SystemRoot\System32\taskkill.exe" /T /F /PID $proc.Id | Out-Null; $proc.WaitForExit(5000) | Out-Null } } catch { }
  }
  for ($i = 0; $i -lt 20 -and (Test-Path -LiteralPath $profile); $i++) { try { Remove-Item -LiteralPath $profile -Recurse -Force -ErrorAction Stop } catch { Start-Sleep -Milliseconds 100 } }
  if (Test-Path -LiteralPath $profile) { Write-Error 'Temporary Windows Chrome profile cleanup failed' }
}
'''


class FormalWorkspaceBrowserE2ETests(unittest.TestCase):
    def test_node_page_cdp_url_validator_rejects_non_loopback_endpoints(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("Node.js unavailable for the synthetic CDP URL policy check")
        synthetic_checks = r'''
const cases = [
  ['ws://127.0.0.1:9222/devtools/page/page_1', true],
  ['wss://127.0.0.1:9222/devtools/page/page_1', false],
  ['ws://0.0.0.0:9222/devtools/page/page_1', false],
  ['ws://127.0.0.1:9223/devtools/page/page_1', false],
  ['ws://127.0.0.1:9222/devtools/browser/browser_1', false],
  ['ws://user@127.0.0.1:9222/devtools/page/page_1', false],
  ['ws://127.0.0.1:9222/devtools/page/page_1?extra=1', false]
];
for (const [url, shouldPass] of cases) {
  let passed = true;
  try { validatePageWebSocketUrl(url, 9222); } catch (_) { passed = false; }
  if (passed !== shouldPass) process.exit(1);
}
process.stdout.write('NODE_CDP_URL_POLICY_PASS');
'''
        run = subprocess.run(
            [node, "-e", NODE_PAGE_CDP_VALIDATOR + synthetic_checks],
            capture_output=True, text=True, timeout=5)
        self.assertEqual(run.returncode, 0, "synthetic CDP endpoint cases failed")
        self.assertEqual(run.stdout, "NODE_CDP_URL_POLICY_PASS")

    def test_timeout_cleanup_stops_only_owned_process_group_and_profile(self):
        if os.name != "posix" or not Path("/proc").is_dir():
            self.skipTest("process-group cleanup helper is Linux/Unix-specific")
        with tempfile.TemporaryDirectory(prefix="formal-e2e-cleanup-test-") as root_name:
            root = Path(root_name)
            profile = root / "owned-profile"
            profile.mkdir()
            pid_file = root / "child.pid"
            child_code = (
                "import subprocess,sys,time; "
                "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']); "
                "open(sys.argv[1],'w').write(str(child.pid)); time.sleep(30)"
            )
            decoy = subprocess.Popen(
                [sys.executable, "-c", "import time;time.sleep(30)"],
                start_new_session=True)
            try:
                with self.assertRaises(subprocess.TimeoutExpired):
                    _run_node_driver(
                        [sys.executable, "-c", child_code, str(pid_file)],
                        profile=profile, timeout=0.4)
                self.assertFalse(profile.exists(), "owned profile directory was not removed")
                self.assertTrue(pid_file.is_file(), "synthetic child did not start")
                child_pid = int(pid_file.read_text(encoding="ascii"))
                self.assertIsNone(decoy.poll(), "cleanup affected an unrelated process group")
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        break
                    if state == "Z":
                        break
                    time.sleep(0.05)
                else:
                    self.fail("owned child process remained active after group cleanup")
            finally:
                if decoy.poll() is None:
                    decoy.terminate()
                    try:
                        decoy.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        decoy.kill()
                        decoy.wait(timeout=2)

    def test_chrome_stage1_stage2_denial_and_reset(self):
        chrome_text = os.environ.get("FORMAL_REVIEW_CHROME")
        chrome = Path(chrome_text) if chrome_text else DEFAULT_CHROME
        if not chrome.is_file():
            self.skipTest(f"Chrome binary unavailable: {chrome}")
        windows_chrome = str(chrome).lower().endswith(".exe")
        node = None
        powershell = None
        if windows_chrome:
            powershell = shutil.which("powershell.exe") or shutil.which("powershell")
            if not powershell:
                self.skipTest("Windows PowerShell unavailable; Windows-local CDP path cannot run")
        else:
            node = shutil.which("node")
            if not node:
                self.skipTest("Node.js not found; Linux Chrome CDP E2E unavailable")
            try:
                node_ws_type = subprocess.run(
                    [node, "-e", "process.stdout.write(typeof WebSocket)"],
                    capture_output=True, text=True, timeout=5, check=True).stdout
            except (OSError, subprocess.SubprocessError):
                self.skipTest("Node.js runtime probe failed; CDP E2E unavailable")
            if node_ws_type != "function":
                self.skipTest("Node.js built-in WebSocket unavailable; no dependency installation attempted")
            syntax = subprocess.run([node, "--check", "-"], input=NODE_CDP,
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(syntax.returncode, 0,
                             f"embedded Node/CDP driver syntax invalid: {syntax.stderr}")

        fixture = fixture_module.FormalWorkspaceTests()
        fixture.setUp()
        server = WorkspaceServer(fixture.session)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        profile = None
        try:
            if not windows_chrome:
                profile = tempfile.mkdtemp(
                    prefix="formal-workspace-chrome-",
                    dir=os.environ.get("FORMAL_REVIEW_CHROME_PROFILE_DIR") or None)
            try:
                workspace_url = f"http://127.0.0.1:{server.server_port}/{server.token}/"
                if windows_chrome:
                    import base64
                    encoded = base64.b64encode(POWERSHELL_CDP.encode("utf-16le")).decode("ascii")
                    env = os.environ.copy()
                    env["FORMAL_REVIEW_E2E_CHROME"] = str(chrome)
                    env["FORMAL_REVIEW_E2E_URL"] = workspace_url
                    run = subprocess.run(
                        [powershell, "-NoLogo", "-NoProfile", "-NonInteractive",
                         "-EncodedCommand", encoded], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=90, env=env)
                else:
                    run = _run_node_driver(
                        [node, "-e", NODE_CDP, str(chrome), str(profile), str(profile),
                         workspace_url], profile=Path(profile), timeout=75)
            except subprocess.TimeoutExpired:
                self.fail("Chrome/CDP E2E timed out after browser startup")
            if run.returncode == 77 or "E2E_UNAVAILABLE:" in run.stdout:
                self.skipTest((run.stdout + run.stderr).strip() or "Chrome CDP unavailable")
            if windows_chrome and any(marker in (run.stdout + run.stderr)
                                      for marker in ("UtilBindVsockAnyPort", "socket failed 1")):
                self.skipTest("Windows interop unavailable for localhost browser E2E")
            self.assertEqual(run.returncode, 0,
                             f"Chrome/CDP E2E failed:\n{run.stdout}\n{run.stderr}")
            self.assertIn("E2E_PASS:", run.stdout)
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)
            if profile:
                shutil.rmtree(profile, ignore_errors=True)
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main(verbosity=2)
