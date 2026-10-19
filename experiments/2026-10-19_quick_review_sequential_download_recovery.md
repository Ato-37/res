# Quick-review sequential handoff and download recovery (2026-10-19)

## Purpose

Make the three-item informal check easier to run with multiple reviewers in one
session, while avoiding a false claim that a browser download has reached disk.
This is interface and handoff work only; it does not change the formal review
protocol or the faithfulness estimand.

## Changes

- Kept the nickname-first, three-item sequence. After one reviewer completes,
  **次の人へ** opens the next nickname field; Enter starts the next person's
  questions. All completed reviewers remain in one ordered schema-v4 JSON.
- Replaced the success-like save message with a clear notice that the download
  was requested and must be verified in the browser's download list.
- Retained the generated JSON as an on-page retry link instead of revoking its
  object URL after 1.5 seconds.
- Updated the Japanese launcher and runbook to point to the new v7 local-only
  artifact and document the download check and the fact that answers are held
  in the open tab until export.

## Verification

- Builder, response inspector, and private importer unit tests: **28/28 passed**.
- Synthetic Windows Chrome E2E: **1/1 passed**. It used generated fixture
  questions/videos, submitted two synthetic reviewers, downloaded and parsed a
  combined JSON with two ordered respondents and six item judgments, and checked
  the retry link. The protected real artifact was not opened in the browser.
- Python compilation and scoped `git diff --check` passed.
- The local artifact is mode `0600`, 30,652,502 bytes, SHA-256
  `c61ddc2e74c07dc9df38eff4d182d1f650bc8172fddcdb7c0206307a9c98de07`.

## Interpretation and limits

The sequence and combined export are verified as browser behavior using
synthetic data. A clicked download still does not prove that the browser wrote
the file; the user must check the download list, and the retry link remains
available in the page. Responses are held only in page memory until export, so
closing or reloading the tab before saving loses them. The three-item check
remains informal. It does not verify reviewer identity, independence, full
viewing, evidence timestamps, or video/model prediction dependence. The user's
qualitative feedback about question clarity, answer mismatch, and a possibly
incomplete explanation remains an informal observation, not a formal label.

Direct real-video faithfulness remains **0/3** formal claim reviews, **0/15**
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3** direct
scores. This UI change adds no direct `ΔF`, confidence interval, or efficacy
evidence. Protected material remains local; rights status is unverified.

## Next action

Use the v7 artifact for the requested sequential informal check and preserve its
single combined JSON locally. If continuity across browser restarts becomes
necessary, implement and test a privacy-conscious checkpoint path before
relying on it; do not treat the current in-memory session as crash-safe.
