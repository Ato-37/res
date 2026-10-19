# Formal claim-review sequential browser E2E

## Purpose

Exercise the formal offline renderer through two synthetic sequential reviewers
in Windows Chrome. Check completion of reviewer one, next-reviewer reset, a
snapshot while reviewer two is incomplete, reviewer-two completion, final and
retry downloads, and Python parsing of the exact one-file envelope. No protected
video, response, or private review input is opened.

## Fixture and test

`scripts/test_nextgqa_formal_claim_review_sequential_browser_e2e.py` creates a
fresh three-row worksheet from the existing synthetic packet fixture, generates
three short colored MP4s with ffmpeg, binds their hashes to synthetic lineage
and custody metadata, and renders a temporary owner-only HTML file with the
videos embedded. PowerShell/CDP drives Windows Chrome; JavaScript intercepts
only synthetic JSON download blobs. The test passes final and snapshot bytes
through `nextgqa_combined_review_envelope.parse_envelope`, checks reviewer
separation and exact retry equality, and verifies each inner index hash against
its preserved review bytes.

## Verification and blocker

The generated synthetic page was built successfully (48,146 bytes, mode 0600).
The first ordinary-sandbox attempt skipped before launching Chrome because
Windows interop was unavailable. A per-command elevated rerun initially hit a
PowerShell diagnostic decoding error in the harness, so the subprocess now
uses replacement decoding for non-UTF-8 stderr and stdout. The browser test
then passed under individual elevation:

```bash
PYTHONPATH=scripts .venv/bin/python -m unittest test_nextgqa_formal_claim_review_sequential_browser_e2e -v
```

Observed: **1/1 passed** in Windows Chrome. It completed two synthetic
reviewers, confirmed that the next-reviewer action cleared the in-progress
answers, captured a one-reviewer snapshot while reviewer two was incomplete,
and captured the final and retry downloads. Python `parse_envelope` accepted
both downloads, the retry bytes equaled the final bytes exactly, the final
envelope retained both reviewers in order, and every inner index hash matched
the original review payload bytes. Only generated colored MP4 fixtures and
synthetic judgments were used; no protected media, review response, or GPU
inference was used.

The ordinary-sandbox run may still skip because Windows interop is hidden at
that boundary. The successful elevated E2E verifies browser mechanics and
byte/schema integrity only; it does not authenticate reviewer identity or
independence, validate semantic judgments, authorize scoring, or establish
faithfulness evidence. The local HTML remains a private handoff artifact.
