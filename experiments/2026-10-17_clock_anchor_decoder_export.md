# Candidate decoder-anchor export for clock correspondence

## Purpose

Prepare the decoder-side images required for a later screenshot-to-frame review, after the previously available anchor plan was found not to match the current staged cohort. This CPU-only operation was designed to decode each current candidate video once and export the 5 prespecified anchors per record. It was not a faithfulness score run.

## Setup

The exporter consumed a locally pinned candidate plan and provenance sidecar bound to the current three-record v6 cohort. It checked the exact plan and sidecar byte hashes, current PyAV and FFmpeg library versions, pinned source hashes, staged-media hashes, full decoded PTS-sequence digests, time bases, and the fixed 3×5 token/anchor assignment. Each media file was opened and decoded once. Output stayed in a private local directory. Rights status remains unverified; no raw media, private plan, token, source digest, or frame digest is published here.

## Verified result

The run returned `candidate_artifact_export_only`, with execution authorization explicitly `not_granted`, and wrote 15 requested/effective RGB PNG anchors plus a private crosswalk. Independent verification confirmed exact 3×5 coverage, unique token/anchor pairs, PNG and plan/provenance hash joins, source-media and PTS declarations, valid RGB PNGs, and private file/directory permissions. No mismatch was found. The independent check did not display image content. The exporter synthetic suite passed 5/5; the blinded-review preparation suite passed 10/10.

## Interpretation and limits

This establishes that current decoder-side candidates can be prepared and provenance-joined. It does not establish that a browser displayed the same frames, that any screenshot crop contains only video, that a human judged the frames to match, or that clock calibration passed. No screenshots or human reviews have been produced, and direct claim-linked faithfulness remains 0/3 reviewed and 0/3 scored with no effect estimate, interval, or statistical test. The separate one-video Qwen CUDA smoke is only a runtime diagnostic.

## Next action

Capture the actual review page through the OS/compositor route, provide one bbox for each full-window screenshot to the cropper, and obtain independent human purity/correspondence review before considering clock calibration. The cropper verifies byte joins and bounds; it does not identify the video rectangle. The capture helper now prints the exact token, anchor, and planned decoder PTS in chronological order; its interactive steps and remaining trust limits are documented in `docs/nextgqa_clock_frame_evidence_reader_v1.md` and `docs/nextgqa_clock_screenshot_cropper.md`. Any later scoring still requires a jointly frozen protocol and an audited all-row production gate.
