# Synthetic formal workspace browser E2E harness (2026-10-19)

## Purpose

Add an isolated browser-level test for the localhost formal-review workspace.
The test uses the existing synthetic Stage-1 and Stage-2 fixtures and checks
the UI's Stage-1 success, Stage-2 success, visible fixed denial fields, and
state clearing after reset. It does not change the controller or UI.

## Implementation

Added `scripts/test_nextgqa_formal_review_workspace_browser_e2e.py`. The Python
harness reuses `FormalWorkspaceTests` to construct only synthetic signed
fixtures and starts `WorkspaceServer` on IPv4 loopback. Windows Chrome is
driven through Windows PowerShell/.NET `ClientWebSocket`, keeping CDP on
Windows loopback because WSL could not reach the Windows CDP endpoint. Linux
Chrome retains Node 24's built-in `WebSocket` path. No npm packages or other
dependencies are used. The default
browser path is the existing Windows Chrome binary at
`/mnt/c/Program Files/Google/Chrome/Application/chrome.exe`; an environment
override is available through `FORMAL_REVIEW_CHROME`. Missing Node/WebSocket or
a Chrome binary that cannot launch from this environment produces an explicit
unittest skip. The Windows path first checks that Windows can reach the exact
tokenized workspace URL over `127.0.0.1`; when forwarding is absent it skips
without trying a WSL IP, port proxy, or non-loopback bind. Chrome uses a unique
Windows Temp profile, attempts CDP `Browser.close` in `finally`, and removes
the profile after a process-tree cleanup fallback.

## Verification

- Harness compilation passed.
- Embedded Node/CDP driver syntax check passed.
- The unprivileged run explicitly skipped because Windows Chrome startup was
  blocked at the WSL vsock boundary (`UtilBindVsockAnyPort:317: socket failed
  1`). Under individually elevated execution, Chrome launched and wrote its
  `DevToolsActivePort` file, but WSL could not reach the Windows CDP HTTP
  endpoint. The harness reported an environment skip; browser interactions
  remain unexecuted.
- After adding the Windows-side CDP fallback, the synthetic harness compiled
  and ran under individually elevated execution. PowerShell/.NET interop
  started, but Windows could not fetch the tokenized workspace URL from WSL's
  `127.0.0.1`; the test explicitly skipped before launching Chrome. Thus the
  fallback avoids the blocked Windows-CDP-from-WSL route but cannot complete
  browser assertions without Windows-to-WSL localhost forwarding.
- Primary audit found and fixed two harness defects before the final elevated
  attempt: Windows Chrome and Node need separate Windows/WSL profile paths, and
  the Windows `--version` probe needs replacement decoding. The current Windows
  path no longer makes that probe from WSL.
- Re-ran `PYTHONPATH=scripts .venv/bin/python -m unittest
  test_nextgqa_formal_review_workspace -v`: **6/6 passed**.
- Re-ran `.venv/bin/python scripts/test_nextgqa_formal_review_workspace_browser_e2e.py`
  with individually elevated execution: **1 test skipped** because Windows
  could not reach the server's tokenized WSL `127.0.0.1` URL. Python compilation
  and the embedded Node driver syntax check passed.
- Independent follow-up hardening moved Linux profile creation inside the
  server-cleanup `try/finally`, suppresses PowerShell progress noise, and checks
  that both page and browser CDP websocket URLs are exactly `ws://127.0.0.1`
  on the allocated Chrome port with expected endpoint paths. `py_compile`,
  `git diff --check`, and the formal controller suite (**6/6**) passed. The
  individually elevated E2E again explicitly skipped before Chrome launch
  because Windows could not reach the WSL loopback workspace URL; no browser
  interactions were performed.
- Current read-only GPU check: RTX 5070 Ti, **15,112/16,303 MiB free**, 0%
  utilization; `.venv` has PyTorch 2.11.0+cu128 and CUDA 12.8. No model forward
  was started: the direct real-data score path remains gated, and the completed
  proxy cohorts are closed to repetition.
- No private review bundle, protected video, real human review, or GPU inference
  was used.

## Limits and next action

The positive browser assertions remain unverified because Windows cannot reach
the local workspace server through the required loopback route in this
environment. Linux Chrome or working Windows-to-WSL localhost forwarding is
still required. This is browser engineering coverage only and adds no
faithfulness evidence or authorization.

## Linux/Node CDP and timeout cleanup hardening

The Linux/Node driver now asks Chrome to bind its remote-debugging endpoint to
`127.0.0.1` and validates the returned page websocket URL before connecting:
the scheme must be `ws`, host exactly `127.0.0.1`, port the allocated CDP port,
and path a page endpoint with no credentials, query, or fragment. The Chrome
child receives a bounded graceful shutdown with a forced-kill fallback.

The Python Node branch starts the Node/Chrome tree in a new process session. On
timeout it terminates only that process group, waits briefly, escalates to
`SIGKILL` if needed, and removes only the exact temporary profile created for
that run. Added synthetic checks reject alternate scheme, host, port, endpoint
path, credentials, and query strings. A process-tree cleanup test creates a
synthetic child and unrelated decoy process; it verifies that the child group
and owned profile are cleaned while the decoy stays alive. The Windows
PowerShell/CDP branch was not changed.

Verification: embedded Node driver syntax check passed; Python compilation and
scoped whitespace check passed; both focused synthetic helper tests passed
(2/2). No Linux Chrome/Chromium executable is installed in this environment,
so a full Linux browser E2E could not be run. The existing Windows E2E remains
blocked by Windows-to-WSL loopback forwarding. No protected bundle, video,
human review, or GPU was used.

Next action: run the Linux browser path in an environment with Linux Chrome or
Chromium and verify the synthetic Stage-1/Stage-2 browser flow, loopback CDP
endpoint, and timeout cleanup end to end.

## Current-worktree recheck (2026-10-19)

The current source tree now contains the formal workspace controller and
synthetic browser harness described above. Re-ran the controller tests:
**8/8 passed**. Python compilation for the controller and both test modules
passed. A normal-sandbox browser-E2E attempt could not bind the required
loopback socket (`PermissionError: Operation not permitted`); the individually
elevated retry ran the harness and explicitly skipped the Chrome interaction
because Windows still could not reach the WSL `127.0.0.1` URL. The two browser
helper/cleanup tests passed. Thus backend fail-closed checks are covered by
synthetic unit tests, but browser rendering and the full HTTP integration are
still not verified in this environment.

This does not add real reviewer, clock/frame, prediction-link, or faithfulness
evidence. The current direct counts remain 0/3 claim reviews, 0/15 clock/frame
pairs, 0/3 prediction-link reviews, and 0/3 direct scores. The next useful
browser check requires an environment where a browser can access the workspace
through the permitted loopback route (for example, Linux Chromium with loopback
socket permission). Do not weaken the loopback binding to work around the WSL
forwarding limitation. The real trust and reviewer inputs remain separately
required.
