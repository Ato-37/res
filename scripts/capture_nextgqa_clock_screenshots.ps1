param(
    [Parameter(Mandatory = $true)][string]$PlanPath,
    [Parameter(Mandatory = $true)][string]$PlanProvenancePath,
    [Parameter(Mandatory = $true)][string]$ExpectedPlanSha256,
    [Parameter(Mandatory = $true)][string]$ExpectedPlanProvenanceSha256,
    [Parameter(Mandatory = $true)][string]$OutputDirectory,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

$nativeType = @'
using System;
using System.Runtime.InteropServices;
using System.Text;
public static class NextGqaClockCaptureNative {
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [StructLayout(LayoutKind.Sequential)] public struct POINT { public int X, Y; }
  [StructLayout(LayoutKind.Sequential)] public struct MSG {
    public IntPtr HWnd; public uint Message; public UIntPtr WParam; public IntPtr LParam;
    public uint Time; public POINT Pt;
  }
  [DllImport("user32.dll", SetLastError=true)] public static extern bool RegisterHotKey(IntPtr hWnd, int id, uint modifiers, uint key);
  [DllImport("user32.dll", SetLastError=true)] public static extern bool UnregisterHotKey(IntPtr hWnd, int id);
  [DllImport("user32.dll", SetLastError=true)] public static extern int GetMessage(out MSG msg, IntPtr hWnd, uint min, uint max);
  [DllImport("kernel32.dll")] public static extern uint GetTickCount();
  public static uint Elapsed(uint start, uint end) { unchecked { return end - start; } }
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll", SetLastError=true)] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
  [DllImport("user32.dll", CharSet=CharSet.Unicode)] public static extern int GetWindowText(IntPtr hWnd, StringBuilder text, int count);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint processId);
}
'@
Add-Type -TypeDefinition $nativeType

function Get-Sha256([byte[]]$Bytes) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { return ([BitConverter]::ToString($sha.ComputeHash($Bytes))).Replace('-', '').ToLowerInvariant() }
    finally { $sha.Dispose() }
}

function Read-PrivateJson([string]$Path, [string]$ExpectedHash) {
    if (-not [System.IO.File]::Exists($Path)) { throw 'private_input_file_missing' }
    $bytes = [System.IO.File]::ReadAllBytes($Path)
    if ((Get-Sha256 $bytes) -ne $ExpectedHash) { throw 'private_input_hash_mismatch' }
    return [System.Text.Encoding]::UTF8.GetString($bytes) | ConvertFrom-Json
}

if ($ExpectedPlanSha256 -notmatch '^[0-9a-f]{64}$' -or
    $ExpectedPlanProvenanceSha256 -notmatch '^[0-9a-f]{64}$') { throw 'expected_hash_invalid' }
$plan = Read-PrivateJson $PlanPath $ExpectedPlanSha256
$provenance = Read-PrivateJson $PlanProvenancePath $ExpectedPlanProvenanceSha256
if ($plan.purpose -ne 'predeclared_html5_clock_pts_anchor_plan' -or
    $plan.execution_authorization -ne 'not_granted' -or
    $provenance.purpose -ne 'private_clock_anchor_plan_candidate_provenance' -or
    $provenance.status -ne 'candidate_not_jointly_frozen' -or
    $provenance.plan_sha256 -ne $ExpectedPlanSha256 -or
    $provenance.joint_freeze_receipt_sha256 -ne $null) { throw 'candidate_plan_provenance_invalid' }
if ($plan.records.Count -ne 3) { throw 'candidate_plan_record_count_invalid' }

$anchors = @()
foreach ($record in $plan.records) {
    if ($record.review_token -notmatch '^[A-Za-z0-9_-]{24}$') { throw 'candidate_token_invalid' }
    foreach ($field in @('fit_anchors', 'heldout_anchors')) {
        foreach ($anchor in $record.$field) {
            if ($anchor.anchor_id -notmatch '^[A-Za-z0-9_-]{1,40}$') { throw 'candidate_anchor_invalid' }
            $anchors += [pscustomobject]@{
                review_token = $record.review_token
                anchor_id = $anchor.anchor_id
                decoder_pts_seconds = [string]$anchor.decoder_pts_seconds
            }
        }
    }
}
if ($anchors.Count -ne 15 -or (Test-Path -LiteralPath $OutputDirectory)) {
    throw 'candidate_anchor_count_or_output_collision'
}
$privateRoot = [System.IO.Path]::GetFullPath((Split-Path -Parent $PlanPath))
$targetRoot = [System.IO.Path]::GetFullPath($OutputDirectory)
if ([System.IO.Path]::GetDirectoryName($targetRoot) -ne $privateRoot -or
    [System.IO.Path]::GetFileName($targetRoot) -notmatch '^clock_screenshots_[A-Za-z0-9_-]+$') {
    throw 'screenshot_output_must_be_private_child'
}
if ($ValidateOnly) {
    Write-Host 'Candidate plan and provenance validated; no screenshot capture started.'
    return
}
try {
[void][System.IO.Directory]::CreateDirectory($targetRoot)

$currentSid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
$directoryAcl = New-Object System.Security.AccessControl.DirectorySecurity
$directoryAcl.SetAccessRuleProtection($true, $false)
$directoryAcl.SetOwner($currentSid)
$inheritance = [System.Security.AccessControl.InheritanceFlags]::ContainerInherit -bor
               [System.Security.AccessControl.InheritanceFlags]::ObjectInherit
foreach ($sidText in @($currentSid.Value, 'S-1-5-18', 'S-1-5-32-544')) {
    $sid = New-Object System.Security.Principal.SecurityIdentifier($sidText)
    $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $sid, [System.Security.AccessControl.FileSystemRights]::FullControl, $inheritance,
        [System.Security.AccessControl.PropagationFlags]::None,
        [System.Security.AccessControl.AccessControlType]::Allow)
    $directoryAcl.AddAccessRule($rule)
}
Set-Acl -LiteralPath $targetRoot -AclObject $directoryAcl

$records = [System.Collections.Generic.List[object]]::new()
$hotkeyId = 7331
$modifiers = [uint32](0x0002 -bor 0x0004 -bor 0x4000) # Ctrl+Shift+F11, no-repeat
$virtual = [System.Windows.Forms.SystemInformation]::VirtualScreen
try {
    if (-not [NextGqaClockCaptureNative]::RegisterHotKey([IntPtr]::Zero, $hotkeyId, $modifiers, 0x7A)) {
        throw 'global_capture_hotkey_unavailable'
    }
    Write-Host 'Local-only capture. Follow the browser review rows in order, record each HTML5 clock event, then press Ctrl+Shift+F11 immediately.'
    Write-Host 'A screenshot plus its window/time receipt will be written locally. This is not a human review or clock pass.'
    for ($index = 0; $index -lt $anchors.Count; $index++) {
        $anchor = $anchors[$index]
        Write-Host ("[{0}/15] Follow the next planned anchor in browser row order." -f ($index + 1))
        Write-Host 'Focus the visible Private offline video review Edge window, then press Ctrl+Shift+F11.'
        $message = New-Object NextGqaClockCaptureNative+MSG
        do {
            $messageResult = [NextGqaClockCaptureNative]::GetMessage([ref]$message, [IntPtr]::Zero, 0, 0)
            if ($messageResult -le 0) { throw 'capture_message_loop_failed' }
        } while ($message.Message -ne 0x0312 -or [int]$message.WParam -ne $hotkeyId)

        $window = [NextGqaClockCaptureNative]::GetForegroundWindow()
        if ($window -eq [IntPtr]::Zero) { throw 'foreground_window_missing' }
        [uint32]$processId = 0
        [void][NextGqaClockCaptureNative]::GetWindowThreadProcessId($window, [ref]$processId)
        $process = [System.Diagnostics.Process]::GetProcessById([int]$processId)
        $titleBuffer = New-Object System.Text.StringBuilder 1024
        [void][NextGqaClockCaptureNative]::GetWindowText($window, $titleBuffer, $titleBuffer.Capacity)
        if ($process.ProcessName -ne 'msedge' -or $titleBuffer.ToString() -notmatch 'Private offline video review') {
            throw 'foreground_window_not_private_edge_review'
        }
        $rect = New-Object NextGqaClockCaptureNative+RECT
        if (-not [NextGqaClockCaptureNative]::GetWindowRect($window, [ref]$rect)) { throw 'foreground_window_rect_unavailable' }
        $width, $height = $rect.Right - $rect.Left, $rect.Bottom - $rect.Top
        if ($width -lt 100 -or $height -lt 100 -or $rect.Left -lt $virtual.Left -or
            $rect.Top -lt $virtual.Top -or $rect.Right -gt ($virtual.Left + $virtual.Width) -or
            $rect.Bottom -gt ($virtual.Top + $virtual.Height)) { throw 'foreground_window_not_fully_visible' }

        $hotkeyMessageTick = [uint32]$message.Time
        $bitmap = New-Object System.Drawing.Bitmap $width, $height
        try {
            $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
            try { $graphics.CopyFromScreen($rect.Left, $rect.Top, 0, 0, $bitmap.Size, [System.Drawing.CopyPixelOperation]::SourceCopy) }
            finally { $graphics.Dispose() }
            $captureCompleteTick = [NextGqaClockCaptureNative]::GetTickCount()
            $hotkeyToCaptureMs = [NextGqaClockCaptureNative]::Elapsed($hotkeyMessageTick, $captureCompleteTick)
            if ($hotkeyToCaptureMs -gt 250) { throw 'hotkey_to_capture_delay_exceeded' }
            $capturedAt = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
            $stream = New-Object System.IO.MemoryStream
            try {
                $bitmap.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
                $screenshotBytes = $stream.ToArray()
            }
            finally { $stream.Dispose() }
        }
        finally { $bitmap.Dispose() }

        $filename = $anchor.review_token + '_' + $anchor.anchor_id + '.png'
        $screenshotPath = Join-Path $targetRoot $filename
        $fileStream = New-Object System.IO.FileStream($screenshotPath, [System.IO.FileMode]::CreateNew,
            [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
        try { $fileStream.Write($screenshotBytes, 0, $screenshotBytes.Length); $fileStream.Flush($true) }
        finally { $fileStream.Dispose() }
        $records.Add([pscustomobject]@{
            review_token = $anchor.review_token
            anchor_id = $anchor.anchor_id
            screenshot_file = $filename
            screenshot_sha256 = Get-Sha256 $screenshotBytes
            screenshot_captured_at_utc = $capturedAt
            capture_method = 'windows_gdi_copyfromscreen_foreground_window_v1'
            foreground_process = $process.ProcessName
            foreground_pid = [int]$processId
            foreground_window_title = $titleBuffer.ToString()
            foreground_window_rect_xywh = @($rect.Left, $rect.Top, $width, $height)
            virtual_screen_rect_xywh = @($virtual.Left, $virtual.Top, $virtual.Width, $virtual.Height)
            hotkey_message_tick_ms = [uint64]$hotkeyMessageTick
            capture_complete_tick_ms = [uint64]$captureCompleteTick
            hotkey_to_capture_ms = [uint32]$hotkeyToCaptureMs
        })
    }
}
finally {
    [void][NextGqaClockCaptureNative]::UnregisterHotKey([IntPtr]::Zero, $hotkeyId)
}

$receipt = [ordered]@{
    schema_version = 1
    purpose = 'private_windows_compositor_screenshot_capture_receipt_candidate'
    status = 'self_reported_capture_receipt_not_independently_authenticated'
    execution_authorization = 'not_granted'
    plan_sha256 = $ExpectedPlanSha256
    plan_provenance_sha256 = $ExpectedPlanProvenanceSha256
    capture_method = 'windows_gdi_copyfromscreen_foreground_window_v1'
    capture_helper_file_sha256 = Get-Sha256 ([System.IO.File]::ReadAllBytes($MyInvocation.MyCommand.Path))
    created_at_utc = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ss.fffZ')
    records = $records
}
$receiptRaw = $receipt | ConvertTo-Json -Depth 8
$receiptPath = Join-Path $targetRoot 'capture_receipt.json'
$receiptBytes = (New-Object System.Text.UTF8Encoding($false)).GetBytes($receiptRaw + "`n")
$receiptStream = New-Object System.IO.FileStream($receiptPath, [System.IO.FileMode]::CreateNew,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
try { $receiptStream.Write($receiptBytes, 0, $receiptBytes.Length); $receiptStream.Flush($true) }
finally { $receiptStream.Dispose() }
Write-Host 'Capture receipt created with 15 local screenshots. Its declarations are not authenticated; independently inspect them before use.'
}
catch {
    if ([System.IO.Directory]::Exists($targetRoot)) {
        Remove-Item -LiteralPath $targetRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    throw
}
