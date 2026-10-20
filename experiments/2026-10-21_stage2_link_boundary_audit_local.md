# Stage-2 prediction-link boundary audit (2026-10-21, local)

## Purpose

Strengthen the non-authorizing Stage-2 prediction-link review boundary before
any baseline scoring. Two independent source audits had found the v3 score
binding and release-role separation structurally sound; this check adds tests
for fixed cohort cardinality and for a source digest altered together with its
own binding hash.

## Changes and verification

- Added renderer boundary tests for manifests containing two or four records.
  Both are rejected at the renderer boundary with the generic
  `baseline_manifest_score_binding_invalid` error.
- Added a rehashed-tampering test: change the bound question digest, recompute
  `score_binding_sha256` and the enclosing manifest digest, then submit the
  bytes. The renderer rejects the internally inconsistent source binding.
- First test expectations incorrectly named internal validator error strings;
  the renderer intentionally normalizes those errors. After asserting the
  public boundary code instead, all **26/26** focused renderer, CLI validator,
  and prediction-link tests passed.
- The broader release-order, renderer, prediction-link, provenance, ceremony,
  blinded-frame-pair, workspace, and baseline-candidate suite passed **86/86**.
  The gate CLI's expected report was `blocked` with
  `trust_root_key_file_invalid`, `model_load_permitted=false`, and
  `execution_authorization=not_granted`.
- `git diff --check` passed. Tests use synthetic manifests and fixtures; no
  protected review data, video inference, GPU scoring, or human review was run.

## Independent audit and interpretation

An independent auditor checked the release roster and blinded frame-pair
builder/UI. It found no critical trust-role bypass: `capture_operator` is
distinctly listed, key/fingerprint collisions are rejected, and the release
witness binds the roster digest. The output correctly remains self-reported,
unauthenticated, and non-authorizing. A separate Stage-2 audit confirmed the
v3 binding validation and three-record expectation; the new tests address its
recommended missing cardinality/tampering cases. These checks establish only
fail-closed software behavior, not reviewer identity, review truth, timing
validity, or permission to load a model.

## Faithfulness status and limits

Direct faithfulness remains **0/3 reviewed claims, 0/15 clock/frame
correspondences, 0/3 prediction-link reviews, and 0/3 direct scores**. There is
no direct effect estimate, uncertainty interval, or test statistic. The local
schema-v5 exploratory response is not bound to the formal Stage-1 claim package
and cannot fill these gates. The trust-root check remains blocked because no
valid externally rooted key file is present. Rights status remains unverified;
all protected material stays local.

## Next action

The browser portion is now complete in the separate 2026-10-21 frame-pair
browser E2E log: 30 synthetic embedded images decoded, both the 14/15 and
missing-independence gates blocked export, and all 15 pairs round-tripped
through the byte auditor. The remaining next step is real assigned reviewers
and an independently verified trust-root fingerprint; after those arrive,
integrate their signed evidence into the production pre-load transaction before
any model forward. Do not substitute fabricated identities or synthetic keys.

## Publication status

The first scoped Git staging attempt in this environment was blocked because
`.git` was read-only. That restriction was later bypassed through an
individually approved Git command for the separate frame-pair E2E change,
published as commit `20cd572204aecf80ab087c1c0ef4e26257f9e736`. The Stage-2
v3 changes and this log were outside that earlier commit and are being
published separately after their focused recheck. Private artifacts remain
unstaged.
