# Quick-review embedded-media preflight — 2026-10-19

## Purpose

Check that the simplified local review artifact contains browser-compatible
media for all three questions. The lightweight artifact is intended to open
directly at nickname entry and then show each video and question without a
manual file picker.

## Setup and verified result

Read the generated private HTML locally, decoded its embedded media into a
temporary directory, and inspected each stream with `ffprobe`. The check emitted
only the aggregate result and codec class; it did not print question text,
claims, video identifiers, or media bytes. All **3/3** embedded items were
video-only H.264. Temporary decoded copies were removed when the check exited.

The artifact remains local under `data/private_nextgqa/`; it was not staged or
published. No human response was created or read.

## Interpretation and limitations

This supports browser-decoder compatibility of the embedded streams. It does
not prove playback in the user's browser/device, that a person watched the full
clips, or any claim's truth or hallucination status. It is not human-review or
faithfulness evidence. Rights status remains unverified and no protected media
or derived media was published.

Direct faithfulness remains **0/3** claim reviews, **0/15** clock/frame
correspondences, **0/3** prediction-link judgments, and **0/3** direct scores;
no direct `ΔF`, interval, or test exists.

## Next action

Use the local artifact for the requested informal impression check if desired;
keep any response JSON local and exploratory. Formal claim review and direct
model scoring remain separate, and real-data GPU forwards stay gated by the
missing independent review and provenance evidence.
