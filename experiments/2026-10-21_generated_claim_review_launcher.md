# GPU-generated claim review launcher (2026-10-21)

## Purpose

Give the next local human check a direct entry to the exact three-item artifact
containing the GPU-generated validation-split explanations. The earlier generic
quick-review launcher opened a different, older item set; its saved response
cannot be joined to these generated claims.

## Change and verification

- Added `docs/nextgqa_generated_claim_review_ja.html`, a small Japanese local
  launcher for the existing generated-claim v4 artifact. It identifies the
  three separate ratings (question clarity, answer fit, and evidence
  sufficiency), retains the nickname-per-reviewer/combined-JSON instructions,
  and labels the activity as preliminary rather than a formal review or a
  faithfulness score.
- Verified the target exists locally, is owner-only (`0600`), and has the
  expected SHA-256 from the local-only validation development log:
  `bf872db7f18e6d0c7a2c87a9647843a639eda70194e301ae2f7a49e41dd9e731`.
- The launcher contains no embedded video, claim, respondent data, or absolute
  path. No protected artifact or response was copied or published.
- Structural verification confirmed the redirect and fallback link target the
  same existing artifact and the disclosure retains the three response axes
  and local-only JSON handling.

## Interpretation and limits

This removes a version mismatch at the handoff: the user can open the generated
claims intended for the next exploratory check without navigating through
multiple files. It does not produce a human response, authenticate reviewers,
establish claim truth, or satisfy the formal Stage-1 gate. The previous legacy
response remains attached only to its unverified older artifact version.

Direct faithfulness remains **0/3** formally reviewed claims, **0/15** verified
clock/frame correspondences, **0/3** prediction-link reviews, and **0/3**
direct scores. There is no direct `DeltaF`, interval, or statistical test.
Dataset/model rights remain unverified; the generated-claim page and response
files stay local.

## Next action

Open the launcher locally and, if available, collect a fresh exploratory
response bound to the displayed artifact fingerprint. Keep that response
private and inspect it without treating it as formal authorization. Continue
the independent clock/frame and trusted-review preparation in parallel.
