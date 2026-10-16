# Candidate clock-frame evidence reader v1

Status: **engineering candidate only; not jointly frozen and not scoring or
clock-calibration authorization**. This reader audits declarations and their
byte bindings after a human has captured and independently reviewed evidence.
The companion `scripts/capture_nextgqa_clock_screenshots.ps1` can capture the
foreground Edge review window with Windows `CopyFromScreen` after a human
hotkey action and write a private receipt. The reader still does not capture
the screen or decide whether two images show the same frame.

The capture helper iterates each token's five candidate anchors in the fixed
chronological order `fit_begin`, `heldout_quarter`, `fit_midpoint`,
`heldout_three_quarter`, `fit_end`, regardless of how anchors are grouped in
the plan JSON. Before each hotkey, its prompt displays the assigned opaque
token, exact `anchor_id`, and planned `decoder_pts_seconds`; the operator must
use that row when positioning the browser review. This prompt reduces row
confusion but does not verify that the browser actually displayed the stated
token, anchor, or time. Before capture, the helper requires three distinct
tokens and validates the fit and heldout anchor sets separately, so an anchor
cannot be silently moved between those roles.

For the candidate three-video plan, the signed evidence manifest has exactly
one row for each of the five planned anchors in each video. Each row binds an
opaque review token and anchor ID to the canonical hash of the raw HTML5 clock
event, screenshot file bytes, a viewport rectangle and its recomputed RGB crop
digest, and a decoder display-order frame index, relative PTS, and RGB digest.
The signed evidence manifest also pins the helper's source hash and binds an
OS capture receipt containing foreground process/title, window and virtual
screen geometry, screenshot hash, and capture timestamp. The reader checks
that receipt against the screenshot and the exact helper source hash.
The reader checks the exact staged video bytes against the plan, decodes those
bytes itself, and verifies the selected frame's PTS and RGB digest. It calls
the production observation converter to validate the raw capture schema and
the plan sidecar before auditing evidence.
It also verifies the helper's uint32 hotkey and capture-completion ticks,
modulo-2^32 elapsed-time arithmetic, and the 250 ms hotkey-to-capture bound.

The signed manifest includes the reviewer's local pseudonym, the required
independent-reader role, an independence attestation, a statement that the
reader personally compared every crop to its candidate decoder frame, a
creation time, and one `match`, `ambiguous`, or `mismatch` decision per anchor.
It uses an Ed25519 signature over canonical JSON. The trusted public-key hash
must be pinned in a separately frozen protocol before any signature can
establish reviewer identity. A signature shows possession of that key; it
does not prove who used it or that the attestations are true.

The screenshot-to-event time tolerance is currently a **250 ms candidate**
parameter. The capture receipt is self-reported: the reader cannot prove that
it came from the actual Edge window or compositor, and there is no trusted
monotonic clock. The viewport rectangle is also a signed declaration. It
recomputes hashes and verifies joins, but does not compare
pixels or infer visual correspondence. A signed `match` therefore remains a
human claim that requires independent review of the actual screenshot crop and
decoded frame. Any missing, duplicate, ambiguous, mismatched, stale, or
modified row blocks the candidate audit.

Even a structurally consistent signed manifest is reported only as
`candidate_signed_evidence_bytes_consistent`. It keeps
`execution_authorization=not_granted` and
`joint_freeze_status=not_jointly_frozen`. Before real use, the protocol must
freeze the reviewer key and capture-helper hashes, the 250 ms tolerance,
capture method and window geometry requirements, and the exact capture/
attestation schema. The helper's current compositor receipt is self-reported;
an independently authenticated receipt and the reviewer's identity/
independence procedure remain to be implemented and audited.

The tool requires a 0600 expected Ed25519 key and helper-source hash, but the
caller-supplied hashes become trust anchors only when they are independently
frozen. The Windows helper accepts only a candidate plan with the same pinned
hashes, requires a new direct child of the plan directory, verifies that the
private parent is Linux mode 0700, then uses the Windows-to-WSL bridge to set
and verify the output directory as 0700 and each PNG/receipt as 0600. It
captures only when the foreground process is Edge with the expected review-page
title. If a caught failure occurs, it removes only the output directory that
the helper created exclusively. The helper's receipt remains a self-report
and cannot authenticate the compositor.

The Windows `ValidateOnly` path was executed successfully against the private
candidate plan and sidecar; it returned before creating output, registering a
hotkey, or capturing pixels. The separate `PermissionSmokeOnly` path checks
WSL POSIX permission enforcement and removes its empty probe afterward. Neither
path tests actual capture. No real screenshots or human review have been
created.

Tests generate synthetic videos, screenshots, keys, and signatures locally.
They exercise the 3×5 joins, ambiguous decisions, missing/duplicate anchors,
tampered files, stale timestamps, key/signature failures, and the limit that a
signed `match` is not an automated image comparison. These tests establish
software behavior only; they are not browser or human-review evidence.
